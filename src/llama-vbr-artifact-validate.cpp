#include "llama-vbr-artifact-validate.h"

#include "llama-cache-budget.h"
#include "llama-vbr-identity-digest.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>

namespace {

vbr_manifest_validation_result terminal_result(
        vbr_manifest_validation_status status,
        vbr_import_decision decision = vbr_import_decision::reject) {
    vbr_manifest_validation_result result;
    result.status = status;
    result.decision = decision;
    return result;
}

vbr_import_decision fallback_decision(const vbr_adopt_policy & policy) {
    if (policy.allow_rebuild) {
        return vbr_import_decision::rebuild;
    }
    if (policy.allow_cold) {
        return vbr_import_decision::cold;
    }
    return vbr_import_decision::reject;
}

vbr_manifest_validation_status codec_status(vbr_artifact_status status) {
    switch (status) {
        case vbr_artifact_status::ok:
            return vbr_manifest_validation_status::validated;
        case vbr_artifact_status::unsupported_version:
            return vbr_manifest_validation_status::unsupported_artifact_version;
        case vbr_artifact_status::checksum_mismatch:
        case vbr_artifact_status::content_id_mismatch:
            return vbr_manifest_validation_status::checksum_or_digest_mismatch;
        case vbr_artifact_status::topology_mismatch:
            return vbr_manifest_validation_status::topology_mismatch;
        case vbr_artifact_status::generation_mismatch:
            return vbr_manifest_validation_status::generation_mismatch;
        case vbr_artifact_status::accounting_unavailable:
            return vbr_manifest_validation_status::accounting_unavailable;
        case vbr_artifact_status::invalid_argument:
        case vbr_artifact_status::malformed:
        case vbr_artifact_status::out_of_bounds:
            return vbr_manifest_validation_status::malformed;
        case vbr_artifact_status::internal_error:
        case vbr_artifact_status::_count:
            return vbr_manifest_validation_status::internal_error;
    }
    return vbr_manifest_validation_status::internal_error;
}

bool identity_matches(
        const vbr_artifact_reference_manifest & manifest,
        const vbr_adopt_policy & policy) {
    return manifest.identity.execution_identity ==
               policy.identity.execution_identity &&
           manifest.identity.adapter_config_identity ==
               policy.identity.adapter_config_identity &&
           manifest.identity.media_content_identity ==
               policy.identity.media_content_identity &&
           manifest.identity.sequence_epoch ==
               policy.identity.sequence_epoch &&
           manifest.identity.next_position ==
               policy.identity.requested_frontier;
}

bool target_domain_for(
        const vbr_artifact_portable_domain & portable,
        const vbr_adopt_policy & policy,
        llama_cache_acct_resource_domain & output) {
    const auto found = std::find_if(
        policy.domain_bindings.begin(), policy.domain_bindings.end(),
        [&](const llama_vbr_artifact_domain_binding & binding) {
            if (portable.kind ==
                    llama_cache_acct_domain_kind::device_topology) {
                return binding.topology_index == portable.topology_index &&
                       binding.device_ordinal == portable.device_ordinal &&
                       binding.domain.residency == portable.residency;
            }
            return binding.topology_index == UINT32_MAX &&
                   binding.device_ordinal == UINT16_MAX &&
                   binding.domain.residency == portable.residency &&
                   binding.domain.kind == portable.kind;
        });
    if (found == policy.domain_bindings.end()) {
        return false;
    }
    output = found->domain;
    return true;
}

const vbr_target_unit_snapshot * find_target_unit(
        const vbr_target_child_snapshot & child,
        uint32_t logical_unit_id) {
    const auto found = std::find_if(
        child.units.begin(), child.units.end(),
        [&](const vbr_target_unit_snapshot & unit) {
            return unit.logical_unit_id == logical_unit_id;
        });
    return found == child.units.end() ? nullptr : &*found;
}

const vbr_artifact_unit_reference * find_reference(
        const vbr_artifact_reference_manifest & manifest,
        const vbr_artifact_unit_descriptor & descriptor) {
    const auto found = std::find_if(
        manifest.unit_references.begin(),
        manifest.unit_references.end(),
        [&](const vbr_artifact_unit_reference & reference) {
            return reference.lineage_uuid == descriptor.lineage_uuid &&
                   reference.logical_unit_id ==
                       descriptor.logical_unit_id &&
                   reference.repr_gen == descriptor.repr_gen &&
                   reference.unit_version_id.valid();
        });
    return found == manifest.unit_references.end() ? nullptr : &*found;
}

bool authorized_placement_plan(
        const vbr_artifact_reference_manifest & manifest,
        uint32_t child_id,
        const vbr_artifact_unit_reference & reference,
        std::vector<vbr_artifact_stream_placement> & placements,
        std::vector<vbr_authorized_cell_run> & runs) {
    std::vector<uint32_t> cells;
    placements.clear();
    runs.clear();
    for (uint32_t stream_ref : reference.authorized_stream_refs) {
        const auto found = std::find_if(
            manifest.stream_placements.begin(),
            manifest.stream_placements.end(),
            [&](const vbr_artifact_stream_placement & placement) {
                return placement.child_id == child_id &&
                       placement.stream_index == stream_ref;
            });
        if (found == manifest.stream_placements.end()) {
            return false;
        }
        placements.push_back(*found);
        for (const auto & cell : found->cells) {
            cells.push_back(cell.physical_cell);
        }
    }
    std::sort(cells.begin(), cells.end());
    cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
    for (uint32_t cell : cells) {
        if (runs.empty() ||
            uint64_t(runs.back().first_physical_cell) +
                    runs.back().cell_count != cell) {
            runs.push_back({ cell, 1 });
        } else {
            ++runs.back().cell_count;
        }
    }
    return !runs.empty();
}

bool stash_full_prefix(const vbr_artifact_stash_reference & stash) {
    if (stash.valid_rows == 0 ||
        stash.valid_rows > UINT32_MAX ||
        stash.captured_sink_count < stash.valid_rows) {
        return false;
    }
    std::vector<uint32_t> cells;
    for (const auto & page : stash.covered_sink_pages) {
        const uint64_t base =
            uint64_t(page.page_index) * VBR_GENERATION_PAGE_CELLS;
        for (uint32_t bit = 0; bit < VBR_GENERATION_PAGE_CELLS; ++bit) {
            if ((page.covered_mask[bit / 64] &
                 (uint64_t(1) << (bit % 64))) == 0) {
                continue;
            }
            const uint64_t cell = base + bit;
            if (cell > UINT32_MAX) {
                return false;
            }
            cells.push_back(uint32_t(cell));
        }
    }
    std::sort(cells.begin(), cells.end());
    if (cells.size() < stash.valid_rows) {
        return false;
    }
    for (uint32_t i = 0; i < stash.valid_rows; ++i) {
        if (cells[i] != i) {
            return false;
        }
    }
    return true;
}

bool same_geometry(
        const vbr_artifact_unit_descriptor & source,
        const vbr_target_unit_snapshot & target) {
    return source.n_stream == 1 &&
           source.unified == target.unified &&
           source.wm_cells <= target.wm_cells &&
           source.rank == target.rank &&
           source.dimensions == target.dimensions &&
           source.row_alignment == target.row_alignment &&
           source.recoverability == target.recoverability &&
           source.side == target.side &&
           source.layout == target.layout &&
           source.row_codec_version == target.row_codec_version &&
           target.shards.size() == source.shards.size();
}

bool shard_domain_matches(
        const vbr_artifact_shard_descriptor & source,
        const vbr_target_shard_snapshot & target,
        uint64_t source_wm_cells,
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy) {
    llama_cache_acct_resource_domain resolved;
    const vbr_artifact_portable_domain portable {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        source.topology_index,
        source.device_ordinal,
    };
    return target_domain_for(portable, policy, resolved) &&
           target.shard_index == source.shard_index &&
           target.topology_index == source.topology_index &&
           target.device_ordinal == source.device_ordinal &&
           source.topology_index < package.topologies().size() &&
           target.topology_digest ==
               package.topologies()[source.topology_index].digest &&
           target.logical_offset == source.logical_offset &&
           target.row_count == source.row_count &&
           target.domain == resolved &&
           target.pool_cookie != nullptr &&
           target.row_bytes != 0 &&
           source_wm_cells <= UINT64_MAX / target.row_bytes &&
           target.mapped_bytes >= source_wm_cells * target.row_bytes;
}

bool digest_nonzero(const std::array<uint8_t, 32> & digest) {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t value) {
        return value != 0;
    });
}

bool downward_recipe_complete(
        const vbr_artifact_unit_descriptor & source,
        vbr_repr_domain source_domain,
        const vbr_target_unit_snapshot & target) {
    vbr_downward_recipe resolved;
    const auto status = vbr_downward_resolve_recipe(
        static_cast<ggml_type>(source.current_type),
        static_cast<ggml_type>(target.current_type),
        static_cast<ggml_type>(target.controller_floor_type),
        target.downward_movable, resolved);
    return target.downward_supported &&
           target.downward_type == target.current_type &&
           target.downward_domain == target.current_domain &&
           target.downward_recipe_id == VBR_DOWNWARD_RECIPE_ID &&
           target.downward_recipe_version == VBR_DOWNWARD_RECIPE_VERSION &&
           status == vbr_downward_recipe_status::resolved &&
           resolved == target.downward_recipe &&
           digest_nonzero(target.downward_build_identity_digest) &&
           target.downward_row_bytes != 0 &&
           target.downward_mapped_bytes != 0 &&
           target.downward_transfer_bytes != 0 &&
           target.downward_codec_workspace_bytes != 0 &&
           target.downward_meansub_model_id >= 0 &&
           !(source_domain == vbr_repr_domain::tapped &&
             target.downward_domain == vbr_repr_domain::full);
}

bool upward_recipe_complete(
        const vbr_artifact_unit_descriptor & source,
        vbr_repr_domain source_domain,
        const vbr_target_unit_snapshot & target) {
    vbr_upward_recipe resolved;
    const auto status = vbr_upward_resolve_recipe(
        static_cast<ggml_type>(source.current_type),
        static_cast<ggml_type>(target.current_type), resolved);
    const vbr_upward_representation_identity source_identity = {
        source.codebook_digest,
        source.rotation_digest,
        source.meansub_digest,
        source.meansub_model_id,
        source.meansub_layer,
        source.meansub_baked,
        source.representation.codec_id,
        source.representation.codec_version,
        source.representation.reference_digest,
    };
    const bool cross_domain = source_domain != target.current_domain;
    return target.upward_supported &&
           target.upward_type == target.current_type &&
           target.upward_domain == target.current_domain &&
           target.upward_recipe_id == VBR_UPWARD_RECIPE_ID &&
           target.upward_recipe_version == VBR_UPWARD_RECIPE_VERSION &&
           status == vbr_upward_recipe_status::resolved &&
           resolved == target.upward_recipe &&
           digest_nonzero(target.upward_build_identity_digest) &&
           target.upward_row_bytes != 0 &&
           target.upward_mapped_bytes != 0 &&
           target.upward_transfer_bytes != 0 &&
           target.upward_codec_workspace_bytes != 0 &&
           target.upward_meansub_model_id >= 0 &&
           target.upward_source_identity == source_identity &&
           target.upward_source_identity.meansub_model_id ==
               target.upward_meansub_model_id &&
           (!cross_domain ||
            (resolved.edges[0].mean_action ==
                 vbr_upward_mean_action::add_baked_source_mean &&
             target.upward_source_identity.meansub_baked &&
             target.upward_target_identity.meansub_baked &&
             target.upward_source_identity.meansub_model_id > 0 &&
             target.upward_source_identity.meansub_model_id ==
                 target.upward_target_identity.meansub_model_id &&
             target.upward_source_identity.meansub_layer ==
                 target.upward_target_identity.meansub_layer &&
             target.upward_source_identity.meansub_digest ==
                 target.upward_target_identity.meansub_digest &&
             digest_nonzero(
                 target.upward_source_identity.codebook_digest) &&
             digest_nonzero(
                 target.upward_source_identity.rotation_digest) &&
             digest_nonzero(
                 target.upward_source_identity.meansub_digest) &&
             digest_nonzero(
                 target.upward_target_identity.codebook_digest) &&
             digest_nonzero(
                 target.upward_target_identity.rotation_digest) &&
             digest_nonzero(
                 target.upward_target_identity.meansub_digest))) &&
           (source_domain != vbr_repr_domain::tapped ||
            source.promote_hops < 2);
}

bool same_representation(
        const vbr_artifact_unit_descriptor & source,
        const vbr_target_unit_snapshot & target) {
    return source.current_type == target.current_type &&
           source.last_source_type == target.last_source_type &&
           source.promote_hops == target.promote_hops &&
           source.last_transition == target.last_transition &&
           source.representation.kind == target.representation_kind &&
           source.representation.codec_id == target.codec_id &&
           source.representation.codec_version == target.codec_version &&
           source.representation.reference_digest ==
               target.representation_reference_digest &&
           source.representation.source_loss_history ==
               target.source_loss_history &&
           source.representation.checkpoint_codec_hops ==
               target.checkpoint_codec_hops &&
           source.codebook_digest == target.codebook_digest &&
           source.rotation_digest == target.rotation_digest &&
           source.meansub_digest == target.meansub_digest;
}

bool add_checked(uint64_t a, uint64_t b, uint64_t & output) {
    if (b > UINT64_MAX - a) {
        return false;
    }
    output = a + b;
    return true;
}

bool price_plan(
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_budget_config & config,
        const llama_cache_budget_plan & plan,
        llama_cache_budget_fit_state & state) {
    llama_cache_budget_coordinator coordinator;
    if (!coordinator.reset(snapshot, config)) {
        return false;
    }
    const auto result = coordinator.fits(plan);
    state = result.state;
    return result.accounting_serial == snapshot.serial;
}

bool accounting_plan_source(
        const vbr_artifact_reference_manifest & manifest,
        const std::vector<vbr_artifact_unit_view> & units,
        const std::vector<vbr_artifact_allocation_view> &
            reference_allocations,
        llama_cache_acct_artifact_id source_artifact,
        const vbr_adopt_policy & policy,
        std::vector<llama_cache_transaction_leaf> & leaves,
        llama_cache_budget_plan & plan) {
    plan.accounting_serial = policy.accounting_snapshot->serial;
    using domain_key = std::tuple<uint8_t, uint8_t, uint16_t, uint64_t>;
    using accounting_key =
        std::tuple<uint8_t, uint8_t, uint8_t, uint16_t, uint64_t>;
    std::map<domain_key, uint64_t> totals;
    std::map<accounting_key, std::pair<uint64_t, uint64_t>> expected_existing;
    std::map<accounting_key, std::pair<uint64_t, uint64_t>> actual_existing;
    std::set<uint64_t> existing_allocations;
    const auto append_existing = [&](const vbr_artifact_allocation_view & value) {
        if (!value.allocation || value.artifact.v == 0 ||
            value.content.v == 0 || value.lineage.v == 0 ||
            value.logical == 0 || value.resident == 0) {
            return false;
        }
        if (!existing_allocations.insert(value.allocation.v).second) {
            return true;
        }
        const accounting_key key {
            uint8_t(value.category), uint8_t(value.domain.residency),
            uint8_t(value.domain.kind), value.domain.device_ordinal.v,
            value.domain.topology.v,
        };
        auto & actual = actual_existing[key];
        if (!add_checked(actual.first, value.logical, actual.first) ||
            !add_checked(actual.second, value.resident, actual.second)) {
            return false;
        }
        llama_cache_transaction_leaf leaf;
        leaf.category = value.category;
        leaf.domain = value.domain;
        leaf.attribution = {
            llama_cache_acct_attr_kind::artifact,
            -1,
            value.artifact,
        };
        leaf.expected_logical = value.logical;
        leaf.reserve_resident = 0;
        leaf.stage_resident = value.resident;
        leaf.artifact = value.artifact;
        leaf.content = value.content;
        leaf.lineage = value.lineage;
        leaf.existing_allocation = value.allocation;
        leaves.push_back(leaf);
        return true;
    };
    for (const auto & unit : units) {
        for (const auto & allocation : unit.payload_allocations) {
            // A singleton catalog stores descriptor metadata on its sole
            // blob owner rather than on the reference owner.  Metadata is
            // minted afresh below and must not be counted as immutable
            // payload merely because its physical owner is the blob.
            if (allocation.category ==
                    llama_cache_acct_category::artifact_descriptor_metadata ||
                allocation.category ==
                    llama_cache_acct_category::artifact_reference_metadata) {
                continue;
            }
            if (!append_existing(allocation)) {
                return false;
            }
        }
        for (const auto & allocation : unit.stash_allocations) {
            if (!append_existing(allocation)) {
                return false;
            }
        }
    }
    for (const auto & allocation : reference_allocations) {
        if (allocation.category ==
                llama_cache_acct_category::artifact_descriptor_metadata ||
            allocation.category ==
                llama_cache_acct_category::artifact_reference_metadata) {
            continue;
        }
        if (!append_existing(allocation)) {
            return false;
        }
    }

    for (const auto & row : manifest.accounting) {
        if (row.role ==
                vbr_artifact_accounting_role::descriptor_metadata ||
            row.role ==
                vbr_artifact_accounting_role::reference_metadata) {
            continue;
        }
        llama_cache_acct_resource_domain domain;
        if (!target_domain_for(row.domain, policy, domain)) {
            return false;
        }
        const accounting_key key {
            uint8_t(vbr_artifact_accounting_category(row.role)),
            uint8_t(domain.residency), uint8_t(domain.kind),
            domain.device_ordinal.v, domain.topology.v,
        };
        auto & expected = expected_existing[key];
        if (!add_checked(
                expected.first, row.logical_bytes, expected.first) ||
            !add_checked(
                expected.second, row.resident_bytes, expected.second)) {
            return false;
        }
    }
    if (actual_existing != expected_existing) {
        return false;
    }

    // Descriptor/reference receipts are destination-local metadata. Unlike
    // immutable payload/stash/companion storage they deliberately mint fresh
    // allocations during  adoption.
    for (const auto & row : manifest.accounting) {
        if (row.role !=
                vbr_artifact_accounting_role::descriptor_metadata &&
            row.role !=
                vbr_artifact_accounting_role::reference_metadata) {
            continue;
        }
        llama_cache_acct_resource_domain domain;
        if (!target_domain_for(row.domain, policy, domain)) {
            return false;
        }
        const auto category = vbr_artifact_accounting_category(row.role);
        const vbr_artifact_allocation_view * source_receipt = nullptr;
        const auto consider_source_receipt =
                [&](const vbr_artifact_allocation_view & value) {
            if (value.category != category || value.domain != domain ||
                value.logical != row.logical_bytes ||
                value.resident != row.resident_bytes ||
                value.content.v == 0 || value.lineage.v == 0) {
                return true;
            }
            if (source_receipt != nullptr &&
                source_receipt->allocation != value.allocation) {
                return false;
            }
            source_receipt = &value;
            return true;
        };
        for (const auto & allocation : reference_allocations) {
            if (!consider_source_receipt(allocation)) {
                return false;
            }
        }
        for (const auto & unit : units) {
            for (const auto & allocation : unit.payload_allocations) {
                if (!consider_source_receipt(allocation)) {
                    return false;
                }
            }
        }
        if (source_receipt == nullptr) {
            return false;
        }
        llama_cache_transaction_leaf leaf;
        leaf.category = category;
        leaf.domain = domain;
        leaf.attribution = {
            row.attribution, -1, source_artifact,
        };
        leaf.expected_logical = row.logical_bytes;
        leaf.reserve_resident = row.resident_bytes;
        leaf.stage_resident = row.resident_bytes;
        leaf.artifact = source_artifact;
        leaf.content = source_receipt->content;
        leaf.lineage = source_receipt->lineage;
        leaves.push_back(leaf);
        const auto key = std::make_tuple(
            uint8_t(domain.residency), uint8_t(domain.kind),
            domain.device_ordinal.v,
            domain.topology.v);
        uint64_t next;
        if (!add_checked(totals[key], leaf.reserve_resident, next)) {
            return false;
        }
        totals[key] = next;
    }
    for (const auto & total : totals) {
        llama_cache_budget_plan_entry entry;
        const auto found = std::find_if(
            leaves.begin(), leaves.end(),
            [&](const llama_cache_transaction_leaf & leaf) {
                return std::make_tuple(
                           uint8_t(leaf.domain.residency),
                           uint8_t(leaf.domain.kind),
                           leaf.domain.device_ordinal.v,
                           leaf.domain.topology.v) == total.first;
            });
        if (found == leaves.end()) {
            return false;
        }
        entry.domain = found->domain;
        entry.reserve_bytes = total.second;
        plan.entries.push_back(entry);
    }
    return true;
}

bool accounting_plan(
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy,
        std::vector<llama_cache_transaction_leaf> & leaves,
        llama_cache_budget_plan & plan) {
    return accounting_plan_source(
        package.manifest(), package.units(), package.reference_allocations(),
        package.reference_artifact(), policy, leaves, plan);
}

vbr_manifest_validation_status configure_transform_plan(
        const vbr_artifact_unit_descriptor & descriptor,
        vbr_repr_domain source_domain,
        const vbr_target_unit_snapshot & target,
        uint32_t child_id,
        uint32_t logical_unit,
        const vbr_adopt_policy & policy,
        vbr_validated_child_plan & plan,
        bool & needs_downward,
        bool & needs_upward,
        bool & needs_cross_domain_upward) {
    vbr_import_transform_kind kind = vbr_import_transform_kind::none;
    const bool representation_matches =
        same_representation(descriptor, target);
    if (representation_matches) {
        if (target.current_domain != source_domain) {
            return vbr_manifest_validation_status::representation_mismatch;
        }
    } else {
        if (descriptor.current_type == target.current_type) {
            return descriptor.codebook_digest != target.codebook_digest ||
                   descriptor.rotation_digest != target.rotation_digest ||
                   descriptor.meansub_digest != target.meansub_digest
                ? vbr_manifest_validation_status::codebook_mismatch
                : vbr_manifest_validation_status::representation_mismatch;
        }
        const bool upward = policy.schedule_quote != nullptr &&
            (policy.schedule_quote->status() ==
                 vbr_import_schedule_status::upward_same_domain ||
             policy.schedule_quote->status() ==
                 vbr_import_schedule_status::upward_cross_domain);
        if (upward) {
            if (!policy.allow_upward ||
                !upward_recipe_complete(descriptor, source_domain, target)) {
                return vbr_manifest_validation_status::
                    representation_mismatch;
            }
            const bool cross = source_domain != target.current_domain;
            kind = cross
                ? vbr_import_transform_kind::upward_cross_domain
                : vbr_import_transform_kind::upward_same_domain;
            needs_upward = true;
            needs_cross_domain_upward |= cross;
        } else {
            if (descriptor.codebook_digest != target.codebook_digest ||
                descriptor.rotation_digest != target.rotation_digest ||
                descriptor.meansub_digest != target.meansub_digest) {
                return vbr_manifest_validation_status::codebook_mismatch;
            }
            if (!policy.allow_downward ||
                !downward_recipe_complete(
                    descriptor, source_domain, target)) {
                return vbr_manifest_validation_status::
                    representation_mismatch;
            }
            kind = vbr_import_transform_kind::downward;
            needs_downward = true;
        }
    }

    plan.transform_kind = kind;
    plan.selected_target_type = target.current_type;
    plan.source_domain = source_domain;
    plan.selected_target_domain = target.current_domain;
    plan.target_last_source_type = kind == vbr_import_transform_kind::none
        ? descriptor.last_source_type : plan.selected_target_type;
    plan.target_promote_hops = kind == vbr_import_transform_kind::none
        ? descriptor.promote_hops : 0;
    if (kind == vbr_import_transform_kind::upward_same_domain &&
        source_domain == vbr_repr_domain::tapped) {
        plan.target_last_source_type = descriptor.current_type;
        plan.target_promote_hops = uint8_t(descriptor.promote_hops + 1);
    } else if (kind == vbr_import_transform_kind::upward_cross_domain) {
        plan.target_last_source_type = descriptor.current_type;
        plan.target_promote_hops = uint8_t(descriptor.promote_hops + 1);
    }

    if (kind == vbr_import_transform_kind::downward) {
        if (policy.downward_projection == nullptr ||
            child_id >= policy.downward_projection->final_types.size() ||
            child_id >=
                policy.downward_projection->child_type_digests.size() ||
            logical_unit >=
                policy.downward_projection->final_types[child_id].size() ||
            policy.downward_projection->final_types[child_id][logical_unit] !=
                static_cast<ggml_type>(target.current_type) ||
            !digest_nonzero(policy.downward_projection->tree_digest)) {
            return vbr_manifest_validation_status::policy_mismatch;
        }
        plan.transcode_recipe_id = target.downward_recipe_id;
        plan.transcode_recipe_version = target.downward_recipe_version;
        plan.transcode_build_identity_digest =
            target.downward_build_identity_digest;
        plan.transcode_recipe = target.downward_recipe;
        plan.transcode_policy_digest =
            policy.downward_projection->child_type_digests[child_id];
        plan.transcode_tree_digest =
            policy.downward_projection->tree_digest;
        plan.transcode_meansub_model_id =
            target.downward_meansub_model_id;
        const auto identity = vbr_downward_build_identity(
            target.downward_recipe, target.downward_meansub_model_id,
            target.meansub_digest, plan.transcode_policy_digest,
            plan.transcode_tree_digest);
        if (!digest_nonzero(identity) ||
            identity != target.downward_build_identity_digest) {
            return vbr_manifest_validation_status::codebook_mismatch;
        }
        plan.target_row_bytes = target.downward_row_bytes;
        plan.target_mapped_bytes = target.downward_mapped_bytes;
        plan.transfer_bytes = target.downward_transfer_bytes;
        plan.codec_workspace_bytes =
            target.downward_codec_workspace_bytes;
    } else if (kind == vbr_import_transform_kind::upward_same_domain ||
               kind == vbr_import_transform_kind::upward_cross_domain) {
        const auto * destination = policy.schedule_quote
            ? &policy.schedule_quote->destination() : nullptr;
        if (!destination || !destination->feasible() ||
            child_id >= destination->final_types.size() ||
            child_id >= destination->child_type_digests.size() ||
            logical_unit >= destination->final_types[child_id].size() ||
            destination->final_types[child_id][logical_unit] !=
                static_cast<ggml_type>(target.current_type) ||
            !digest_nonzero(destination->tree_digest)) {
            return vbr_manifest_validation_status::policy_mismatch;
        }
        plan.transcode_recipe_id = target.upward_recipe_id;
        plan.transcode_recipe_version = target.upward_recipe_version;
        plan.transcode_build_identity_digest =
            target.upward_build_identity_digest;
        plan.upward_recipe = target.upward_recipe;
        plan.transcode_policy_digest =
            destination->child_type_digests[child_id];
        plan.transcode_tree_digest = destination->tree_digest;
        plan.transcode_meansub_model_id = target.upward_meansub_model_id;
        plan.transcode_source_identity = target.upward_source_identity;
        plan.transcode_target_identity = target.upward_target_identity;
        const auto identity = vbr_upward_build_identity(
            target.upward_recipe, target.upward_source_identity,
            target.upward_target_identity, plan.transcode_policy_digest,
            plan.transcode_tree_digest);
        if (!digest_nonzero(identity) ||
            identity != target.upward_build_identity_digest) {
            return vbr_manifest_validation_status::codebook_mismatch;
        }
        plan.target_row_bytes = target.upward_row_bytes;
        plan.target_mapped_bytes = target.upward_mapped_bytes;
        plan.transfer_bytes = target.upward_transfer_bytes;
        plan.codec_workspace_bytes = target.upward_codec_workspace_bytes;
    } else {
        plan.target_row_bytes = target.shards.front().row_bytes;
        plan.target_mapped_bytes = target.shards.front().mapped_bytes;
    }
    return vbr_manifest_validation_status::validated;
}

} // namespace

const char * vbr_import_schedule_status_name(
        vbr_import_schedule_status status) noexcept {
    switch (status) {
        case vbr_import_schedule_status::exact: return "exact";
        case vbr_import_schedule_status::downward: return "downward";
        case vbr_import_schedule_status::upward_same_domain:
            return "upward_same_domain";
        case vbr_import_schedule_status::upward_cross_domain:
            return "upward_cross_domain";
        case vbr_import_schedule_status::mixed_direction_unsupported:
            return "mixed_direction_unsupported";
        case vbr_import_schedule_status::unavailable: return "unavailable";
        case vbr_import_schedule_status::_count: break;
    }
    return "invalid";
}

vbr_import_schedule_status vbr_classify_import_schedule_units(
        const std::vector<vbr_import_schedule_unit> & units) noexcept {
    if (units.empty()) {
        return vbr_import_schedule_status::unavailable;
    }
    bool has_downward = false;
    bool has_upward = false;
    bool has_cross_domain_upward = false;
    for (const auto & unit : units) {
        if (unit.source_type < 0 || unit.target_type < 0) {
            return vbr_import_schedule_status::unavailable;
        }
        const auto source = static_cast<ggml_type>(unit.source_type);
        const auto target = static_cast<ggml_type>(unit.target_type);
        if (unit.source_domain != vbr_downward_tier_domain(source) ||
            unit.target_domain != vbr_downward_tier_domain(target)) {
            return vbr_import_schedule_status::unavailable;
        }
        vbr_downward_recipe recipe;
        const auto relation = vbr_downward_resolve_recipe(
            source, target, GGML_TYPE_TURBO1_TCQ, true, recipe);
        if (relation == vbr_downward_recipe_status::resolved) {
            has_downward = true;
        } else if (relation ==
                   vbr_downward_recipe_status::upward_forbidden) {
            has_upward = true;
            has_cross_domain_upward |=
                unit.source_domain != unit.target_domain;
        } else if (relation != vbr_downward_recipe_status::equal_tier) {
            return vbr_import_schedule_status::unavailable;
        }
    }
    if (has_downward && has_upward) {
        return vbr_import_schedule_status::mixed_direction_unsupported;
    }
    if (has_upward) {
        return has_cross_domain_upward
            ? vbr_import_schedule_status::upward_cross_domain
            : vbr_import_schedule_status::upward_same_domain;
    }
    return has_downward
        ? vbr_import_schedule_status::downward
        : vbr_import_schedule_status::exact;
}

bool vbr_quote_import_schedule(
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & package,
        vbr_import_schedule_quote & output) noexcept {
    output = {};
    try {
        if (!package || package.validate() != vbr_artifact_status::ok ||
            !package.manifest().manifest_digest.valid() ||
            target.memory_instance_cookie == 0 ||
            target.target_state_serial == 0 ||
            target.tree_shape_digest == 0 || target.policy_epoch == 0 ||
            target.children.empty() || package.units().empty()) {
            return false;
        }
        output.manifest_digest_ = package.manifest().manifest_digest;
        output.memory_instance_cookie_ = target.memory_instance_cookie;
        output.target_state_serial_ = target.target_state_serial;
        output.accounting_serial_ = target.accounting_serial;
        output.tree_shape_digest_ = target.tree_shape_digest;
        output.policy_epoch_ = target.policy_epoch;

        for (size_t child_index = 0;
             child_index < target.children.size(); ++child_index) {
            const auto & child = target.children[child_index];
            if (child.child_id != child_index) {
                output = {};
                return false;
            }
            for (size_t unit_index = 0;
                 unit_index < child.units.size(); ++unit_index) {
                if (child.units[unit_index].logical_unit_id != unit_index) {
                    output = {};
                    return false;
                }
            }
        }
        output.units_.reserve(package.units().size());
        for (const auto & source : package.units()) {
            if (source.descriptor.child_id >= target.children.size()) {
                output = {};
                return false;
            }
            const auto & child =
                target.children[source.descriptor.child_id];
            if (source.descriptor.logical_unit_id >= child.units.size() ||
                source.descriptor.current_type < 0 ||
                child.units[source.descriptor.logical_unit_id].current_type < 0) {
                output = {};
                return false;
            }
            const auto & unit =
                child.units[source.descriptor.logical_unit_id];
            const auto source_type = static_cast<ggml_type>(
                source.descriptor.current_type);
            const auto target_type = static_cast<ggml_type>(
                unit.current_type);
            const auto source_domain = vbr_downward_tier_domain(source_type);
            const auto target_domain = vbr_downward_tier_domain(target_type);
            if (unit.current_domain != target_domain) {
                output = {};
                return false;
            }
            for (const auto & shard : source.descriptor.shards) {
                uint64_t next = 0;
                if (!add_checked(
                        output.source_payload_bytes_,
                        shard.payload_bytes, next)) {
                    output = {};
                    return false;
                }
                output.source_payload_bytes_ = next;
            }
            for (const auto & shard : unit.shards) {
                uint64_t next = 0;
                if (!add_checked(
                        output.target_mapped_bytes_,
                        shard.mapped_bytes, next)) {
                    output = {};
                    return false;
                }
                output.target_mapped_bytes_ = next;
            }
            output.units_.push_back({
                source.descriptor.child_id,
                source.descriptor.logical_unit_id,
                source.descriptor.current_type,
                unit.current_type,
                source_domain,
                target_domain,
            });
        }
        output.status_ = vbr_classify_import_schedule_units(output.units_);
        return output.status_ != vbr_import_schedule_status::unavailable;
    } catch (...) {
        output = {};
        return false;
    }
}

bool vbr_import_schedule_quote_matches_source(
        const vbr_import_schedule_quote & quote,
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_reference_manifest & manifest,
        const std::vector<vbr_artifact_unit_view> & units) noexcept {
    if (quote.status_ == vbr_import_schedule_status::unavailable ||
        quote.status_ == vbr_import_schedule_status::_count ||
        quote.manifest_digest_ != manifest.manifest_digest ||
        quote.memory_instance_cookie_ != target.memory_instance_cookie ||
        quote.target_state_serial_ != target.target_state_serial ||
        quote.accounting_serial_ != target.accounting_serial ||
        quote.tree_shape_digest_ != target.tree_shape_digest ||
        quote.policy_epoch_ != target.policy_epoch) {
        return false;
    }
    if (quote.units_.size() != units.size()) {
        return false;
    }
    uint64_t source_payload_bytes = 0;
    uint64_t target_mapped_bytes = 0;
    for (size_t i = 0; i < quote.units_.size(); ++i) {
        const auto & expected = quote.units_[i];
        const auto & source = units[i].descriptor;
        if (expected.child_id != source.child_id ||
            expected.logical_unit_id != source.logical_unit_id ||
            expected.source_type != source.current_type ||
            source.child_id >= target.children.size() ||
            target.children[source.child_id].child_id != source.child_id ||
            source.logical_unit_id >=
                target.children[source.child_id].units.size()) {
            return false;
        }
        const auto & unit = target.children[source.child_id].units[
            source.logical_unit_id];
        if (unit.logical_unit_id != source.logical_unit_id ||
            expected.target_type != unit.current_type ||
            expected.source_domain != vbr_downward_tier_domain(
                static_cast<ggml_type>(source.current_type)) ||
            expected.target_domain != unit.current_domain ||
            expected.target_domain != vbr_downward_tier_domain(
                static_cast<ggml_type>(unit.current_type))) {
            return false;
        }
        for (const auto & shard : source.shards) {
            uint64_t next = 0;
            if (!add_checked(
                    source_payload_bytes, shard.payload_bytes, next)) {
                return false;
            }
            source_payload_bytes = next;
        }
        for (const auto & shard : unit.shards) {
            uint64_t next = 0;
            if (!add_checked(
                    target_mapped_bytes, shard.mapped_bytes, next)) {
                return false;
            }
            target_mapped_bytes = next;
        }
    }
    if (source_payload_bytes != quote.source_payload_bytes_ ||
        target_mapped_bytes != quote.target_mapped_bytes_ ||
        vbr_classify_import_schedule_units(quote.units_) != quote.status_) {
        return false;
    }
    if (!quote.destination_.feasible()) {
        return true;
    }
    if (quote.destination_.prefix.size() >
            VBR_IMPORT_DESTINATION_MAX_STEPS ||
        quote.destination_.initial_types.size() != target.children.size() ||
        quote.destination_.initial_cursors.size() != target.children.size() ||
        quote.destination_.final_types.size() != target.children.size() ||
        quote.destination_.final_cursors.size() != target.children.size() ||
        quote.destination_.child_type_digests.size() !=
            target.children.size() ||
        vbr_type_tree_digest(
            quote.destination_.child_type_digests,
            VBR_DOWNWARD_RECIPE_VERSION) != quote.destination_.tree_digest) {
        return false;
    }
    for (size_t child_index = 0; child_index < target.children.size();
         ++child_index) {
        const auto & child = target.children[child_index];
        if (child.child_id != child_index ||
            child.controller_policy.cursor !=
                quote.destination_.final_cursors[child_index] ||
            child.controller_policy.current_type_vector_digest !=
                quote.destination_.child_type_digests[child_index] ||
            child.units.size() !=
                quote.destination_.final_types[child_index].size()) {
            return false;
        }
        for (const auto & unit : child.units) {
            if (unit.logical_unit_id >=
                    quote.destination_.final_types[child_index].size() ||
                unit.current_type != int32_t(
                    quote.destination_.final_types[child_index]
                        [unit.logical_unit_id])) {
                return false;
            }
        }
    }
    return true;
}

bool vbr_import_schedule_quote_matches(
        const vbr_import_schedule_quote & quote,
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & package) noexcept {
    return package && vbr_import_schedule_quote_matches_source(
        quote, target, package.manifest(), package.units());
}

bool vbr_rebind_import_schedule_quote(
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & package,
        const vbr_import_destination_projection & destination,
        vbr_import_schedule_quote & output) noexcept {
    if (!package || !destination.feasible() ||
        output.manifest_digest_ != package.manifest().manifest_digest ||
        output.units_.size() != package.units().size()) {
        return false;
    }
    output.target_mapped_bytes_ = 0;
    for (size_t i = 0; i < output.units_.size(); ++i) {
        auto & quoted = output.units_[i];
        const auto & source = package.units()[i].descriptor;
        if (quoted.child_id != source.child_id ||
            quoted.logical_unit_id != source.logical_unit_id ||
            quoted.source_type != source.current_type ||
            source.child_id >= target.children.size() ||
            source.logical_unit_id >=
                target.children[source.child_id].units.size()) {
            return false;
        }
        const auto & selected = target.children[source.child_id].units[
            source.logical_unit_id];
        quoted.target_type = selected.current_type;
        quoted.target_domain = selected.current_domain;
        for (const auto & shard : selected.shards) {
            if (shard.mapped_bytes > UINT64_MAX -
                    output.target_mapped_bytes_) {
                return false;
            }
            output.target_mapped_bytes_ += shard.mapped_bytes;
        }
    }
    output.status_ = vbr_classify_import_schedule_units(output.units_);
    if (output.status_ == vbr_import_schedule_status::unavailable) {
        return false;
    }
    output.destination_ = destination;
    // The caller immediately hands this private, friend-minted capability to
    // the manifest validator, which performs the one authoritative full
    // source/target shard match. Repeating it here doubles max-shape restore
    // work without adding an intervening mutation boundary.
    return true;
}

vbr_validated_manifest::vbr_validated_manifest(
        vbr_validated_manifest && other) noexcept {
    *this = std::move(other);
}
vbr_validated_manifest & vbr_validated_manifest::operator=(
        vbr_validated_manifest && other) noexcept {
    if (this == &other) {
        return *this;
    }
    const bool internal_tokens =
        other.authenticated_identity_.tokens == &other.token_block_.tokens;
    source_lease_ = std::move(other.source_lease_);
    source_projection_ = std::move(other.source_projection_);
    occupied_replacement_ = std::move(other.occupied_replacement_);
    decision_ = other.decision_;
    target_ = std::move(other.target_);
    manifest_digest_ = other.manifest_digest_;
    capture_generation_id_ = other.capture_generation_id_;
    authenticated_identity_ = std::move(other.authenticated_identity_);
    token_block_ = std::move(other.token_block_);
    if (internal_tokens) {
        authenticated_identity_.tokens = &token_block_.tokens;
    }
    children_ = std::move(other.children_);
    companions_ = std::move(other.companions_);
    accounting_leaves_ = std::move(other.accounting_leaves_);
    tracker_install_ = std::move(other.tracker_install_);
    source_controllers_ = std::move(other.source_controllers_);
    adoption_nonce_ = other.adoption_nonce_;
    recheck_context_ = other.recheck_context_;
    recheck_target_empty_ = other.recheck_target_empty_;
    read_accounting_serial_ = other.read_accounting_serial_;
    read_policy_epoch_ = other.read_policy_epoch_;
    read_transform_tree_digest_ = other.read_transform_tree_digest_;
    occupied_representation_context_ =
        other.occupied_representation_context_;
    occupied_representation_identity_ =
        other.occupied_representation_identity_;
    return *this;
}
vbr_validated_manifest::~vbr_validated_manifest() = default;
vbr_parsed_companion_image::~vbr_parsed_companion_image() = default;

vbr_manifest_validation_result vbr_validate_unit_manifest_snapshot(
        const vbr_target_validation_snapshot & target,
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy) noexcept {
    try {
        if (!package) {
            return terminal_result(
                vbr_manifest_validation_status::unsupported_artifact_version);
        }
        if (policy.schedule_quote != nullptr) {
            if (!vbr_import_schedule_quote_matches(
                    *policy.schedule_quote, target, package)) {
                return terminal_result(
                    vbr_manifest_validation_status::unavailable);
            }
            switch (policy.schedule_quote->status()) {
                case vbr_import_schedule_status::exact:
                case vbr_import_schedule_status::downward:
                case vbr_import_schedule_status::upward_same_domain:
                case vbr_import_schedule_status::upward_cross_domain:
                    break;
                case vbr_import_schedule_status::mixed_direction_unsupported:
                    return terminal_result(
                        vbr_manifest_validation_status::unavailable);
                case vbr_import_schedule_status::unavailable:
                case vbr_import_schedule_status::_count:
                    return terminal_result(
                        vbr_manifest_validation_status::unavailable);
            }
        } else {
            const auto codec = package.validate();
            if (codec != vbr_artifact_status::ok) {
                return terminal_result(codec_status(codec));
            }
        }
        const auto & manifest = package.manifest();
        const bool occupied_replacement =
            policy.occupied_replacement != nullptr;
        if (occupied_replacement &&
            (!policy.occupied_replacement->ready() ||
             policy.occupied_replacement->destination() !=
                 policy.destination_sequence ||
             policy.occupied_replacement->incoming_artifact() !=
                 package.reference_artifact() ||
             !policy.occupied_replacement->recovery_package() ||
             policy.occupied_representation_identity == nullptr)) {
            return terminal_result(
                vbr_manifest_validation_status::unavailable);
        }
        if (manifest.version <
                VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT) {
            return terminal_result(
                vbr_manifest_validation_status::restore_metadata_missing,
                fallback_decision(policy));
        }
        if (manifest.version > VBR_UNIT_ARTIFACT_FORMAT_VERSION) {
            return terminal_result(
                vbr_manifest_validation_status::unsupported_artifact_version);
        }
        if (!policy.authorized) {
            return terminal_result(vbr_manifest_validation_status::unauthorized);
        }
        if (!identity_matches(manifest, policy)) {
            return terminal_result(vbr_manifest_validation_status::identity_mismatch);
        }
        if (policy.identity.tokens == nullptr ||
            manifest.token_block.tokens != *policy.identity.tokens ||
            manifest.token_block.tokens.size() !=
                size_t(manifest.identity.token_count)) {
            return terminal_result(
                vbr_manifest_validation_status::token_block_mismatch);
        }
        if (target.memory_instance_cookie == 0 ||
            target.tree_shape_digest == 0 || target.children.empty() ||
            target.children.size() !=
                manifest.generation.controllers.size()) {
            return terminal_result(
                vbr_manifest_validation_status::memory_tree_mismatch);
        }
        for (size_t i = 0; i < target.children.size(); ++i) {
            if (target.children[i].child_id !=
                    manifest.generation.controllers[i].child_id ||
                target.children[i].dependency_mode !=
                    manifest.generation.controllers[i].dependency_mode) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
        }
        if (!target.scheduler_idle) {
            return terminal_result(vbr_manifest_validation_status::target_not_idle);
        }
        if (!target.destination_sequence_absent && !occupied_replacement) {
            return terminal_result(vbr_manifest_validation_status::target_not_empty);
        }
        std::set<uint32_t> target_child_ids;
        std::set<const void *> target_memory_cookies;
        std::vector<vbr_controller_instance_id> target_instances;
        for (const auto & child : target.children) {
            const bool duplicate_instance = std::any_of(
                target_instances.begin(), target_instances.end(),
                [&](vbr_controller_instance_id instance) {
                    return instance == child.instance_id;
                });
            if (child.memory_cookie == nullptr ||
                !target_child_ids.insert(child.child_id).second ||
                !target_memory_cookies.insert(child.memory_cookie).second ||
                duplicate_instance) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            target_instances.push_back(child.instance_id);
            if (!child.empty && !occupied_replacement) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_empty);
            }
            if (!child.dedicated) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_dedicated);
            }
            if (!child.armed ||
                !vbr_controller_instance_id_is_set(child.instance_id)) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_armed);
            }
            if (child.policy_epoch != target.policy_epoch ||
                std::any_of(
                    child.units.begin(), child.units.end(),
                    [](const vbr_target_unit_snapshot & unit) {
                        return unit.n_stream != 1 || unit.v_trans;
                    })) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_armed);
            }
        }

        std::vector<vbr_validated_child_plan> child_plans;
        vbr_tracker_install_plan tracker;
        bool needs_live_rebase =
            manifest.consistency.kind ==
                vbr_artifact_consistency_kind::live_rebased;
        bool needs_downward = false;
        bool needs_upward = false;
        bool needs_cross_domain_upward = false;
        for (size_t controller_index = 0;
             controller_index < manifest.generation.controllers.size();
             ++controller_index) {
            const auto & controller =
                manifest.generation.controllers[controller_index];
            const auto * target_child = &target.children[controller_index];
            if (target_child->dependency_mode !=
                    controller.dependency_mode) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            const size_t source_units = size_t(std::count_if(
                package.units().begin(), package.units().end(),
                [&](const vbr_artifact_unit_view & unit) {
                    return unit.descriptor.child_id == controller.child_id;
                }));
            if (target_child->units.size() != source_units ||
                controller.units.size() != source_units) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            if (target_child->previously_observed) {
                needs_live_rebase = true;
            }
            if (controller.child_id >= manifest.controller_policy.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::policy_mismatch);
            }
            const auto & controller_policy =
                manifest.controller_policy[controller.child_id];
            const auto * selected_destination = policy.schedule_quote &&
                    policy.schedule_quote->destination().feasible()
                ? &policy.schedule_quote->destination() : nullptr;
            const bool negotiated_projected_policy =
                selected_destination != nullptr &&
                controller_index <
                    selected_destination->child_type_digests.size() &&
                controller_index <
                    selected_destination->final_types.size() &&
                controller_index <
                    selected_destination->final_cursors.size() &&
                target_child->controller_policy.current_type_vector_digest ==
                    selected_destination->child_type_digests[controller_index] &&
                target_child->controller_policy.cursor ==
                    selected_destination->final_cursors[controller_index];
            // Compatibility for direct validator callers that predate
            // negotiated destinations. Production imports authenticate the
            // generic selected tuple through schedule_quote; this fallback is
            // deliberately restricted to an explicitly authorized downward
            // projection.
            const bool legacy_downward_projected_policy =
                selected_destination == nullptr &&
                policy.allow_downward &&
                policy.downward_projection != nullptr &&
                policy.downward_projection->status ==
                    vbr_downward_policy_status::coherent &&
                controller_index <
                    policy.downward_projection->child_type_digests.size() &&
                target_child->controller_policy.current_type_vector_digest ==
                    policy.downward_projection->child_type_digests[
                        controller_index];
            const bool projected_policy = negotiated_projected_policy ||
                legacy_downward_projected_policy;
            if (controller_policy.child_id != controller.child_id ||
                controller_policy.dependency_mode !=
                    controller.dependency_mode ||
                target_child->controller_policy.degrade_order_digest !=
                    controller_policy.degrade_order_digest ||
                target_child->controller_policy.policy_digest !=
                    controller_policy.policy_digest ||
                target_child->controller_policy.floor_type !=
                    controller_policy.floor_type ||
                target_child->controller_policy.pressure_independent_settings !=
                    controller_policy.pressure_independent_settings ||
                target_child->controller_policy.n_stream !=
                    controller_policy.n_stream ||
                target_child->controller_policy.unified !=
                    controller_policy.unified ||
                target_child->controller_policy.wm_cells !=
                    controller_policy.wm_cells ||
                (!projected_policy &&
                 (target_child->controller_policy.cursor !=
                      controller_policy.cursor ||
                  target_child->controller_policy.current_type_vector_digest !=
                      controller_policy.current_type_vector_digest)) ||
                target_child->controller_policy.completed_wave !=
                    controller_policy.completed_wave) {
                return terminal_result(
                    vbr_manifest_validation_status::policy_mismatch);
            }
            if (!target_child->generation_compatible) {
                return terminal_result(
                    vbr_manifest_validation_status::generation_mismatch);
            }
            if (!target_child->ownership_compatible) {
                return terminal_result(
                    vbr_manifest_validation_status::ownership_mismatch);
            }
            if (!target_child->stash_compatible) {
                return terminal_result(
                    vbr_manifest_validation_status::stash_inconsistent);
            }
            vbr_tracker_install_child tracker_child;
            tracker_child.child_id = controller.child_id;
            tracker_child.transition =
                vbr_tracker_install_transition::native_clone;
            tracker_child.lineage_uuid = controller.lineage_uuid;
            tracker_child.target_instance = target_child->instance_id;
            tracker_child.global_generation =
                controller.global_generation;
            tracker_child.units = controller.units;
            tracker.children.push_back(std::move(tracker_child));
        }

        for (const auto & unit : package.units()) {
            const auto & descriptor = unit.descriptor;
            if (descriptor.child_id >= target.children.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            const auto * target_child =
                &target.children[descriptor.child_id];
            const auto * target_unit = find_target_unit(
                *target_child, descriptor.logical_unit_id);
            if (target_unit == nullptr) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            if (!same_geometry(descriptor, *target_unit)) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            if (descriptor.shards.empty()) {
                return terminal_result(
                    vbr_manifest_validation_status::topology_mismatch);
            }
            for (size_t shard_index = 0;
                 shard_index < descriptor.shards.size(); ++shard_index) {
                const auto & shard = descriptor.shards[shard_index];
                const auto & target_shard =
                    target_unit->shards[shard_index];
                if (!shard_domain_matches(
                        shard, target_shard, descriptor.wm_cells,
                        package, policy)) {
                    return terminal_result(
                        vbr_manifest_validation_status::topology_mismatch);
                }
            }
            const auto & controller =
                manifest.generation.controllers[descriptor.child_id];
            if (descriptor.logical_unit_id >= controller.units.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::generation_mismatch);
            }
            const vbr_repr_domain source_domain =
                controller.units[descriptor.logical_unit_id].domain;
            vbr_validated_child_plan plan;
            const auto transform_status = configure_transform_plan(
                descriptor, source_domain, *target_unit,
                descriptor.child_id, descriptor.logical_unit_id, policy,
                plan, needs_downward, needs_upward,
                needs_cross_domain_upward);
            if (transform_status !=
                    vbr_manifest_validation_status::validated) {
                return terminal_result(transform_status);
            }
            const auto transform_kind = plan.transform_kind;
            if (transform_kind == vbr_import_transform_kind::none) {
                for (size_t i = 0; i < descriptor.shards.size(); ++i) {
                    if (target_unit->shards[i].row_bytes !=
                            descriptor.shards[i].row_bytes ||
                        target_unit->shards[i].mapped_bytes <
                            descriptor.shards[i].payload_bytes) {
                        return terminal_result(
                            vbr_manifest_validation_status::geometry_mismatch);
                    }
                }
            }
            const auto * reference = find_reference(manifest, descriptor);
            if (reference == nullptr) {
                return terminal_result(
                    vbr_manifest_validation_status::generation_mismatch);
            }
            std::vector<vbr_artifact_stream_placement> placements;
            std::vector<vbr_authorized_cell_run> runs;
            if (!authorized_placement_plan(
                    manifest, descriptor.child_id, *reference,
                    placements, runs)) {
                return terminal_result(
                    vbr_manifest_validation_status::ownership_mismatch);
            }
            for (const auto & placement : placements) {
                for (const auto & cell : placement.cells) {
                    if (cell.physical_cell >= descriptor.wm_cells) {
                        return terminal_result(
                            vbr_manifest_validation_status::ownership_mismatch);
                    }
                }
            }

            vbr_validated_stash_action stash_action =
                vbr_validated_stash_action::_count;
            switch (descriptor.clean_stash_state) {
                case vbr_artifact_clean_stash_state::absent_at_source:
                    stash_action =
                        vbr_validated_stash_action::none_at_source;
                    // A full-domain F16/raw unit has no clean-stash by
                    // construction; that honest absence is capture-exact.
                    // A tapped unit without its source-present clean prefix
                    // cannot preserve native generations and must rebase.
                    if (source_domain != vbr_repr_domain::full) {
                        needs_live_rebase = true;
                    }
                    break;
                case vbr_artifact_clean_stash_state::omitted_source_present:
                    stash_action =
                        vbr_validated_stash_action::omit_live_rebased;
                    needs_live_rebase = true;
                    break;
                case vbr_artifact_clean_stash_state::present:
                    if (!reference->has_stash_reference) {
                        return terminal_result(
                            vbr_manifest_validation_status::stash_inconsistent);
                    }
                    if (stash_full_prefix(reference->stash_reference)) {
                        stash_action =
                            vbr_validated_stash_action::restore_exact;
                    } else {
                        stash_action =
                            vbr_validated_stash_action::omit_live_rebased;
                        needs_live_rebase = true;
                    }
                    break;
                case vbr_artifact_clean_stash_state::_count:
                    return terminal_result(
                        vbr_manifest_validation_status::stash_inconsistent);
            }

            plan.child_id = descriptor.child_id;
            plan.dependency_mode = target_child->dependency_mode;
            plan.logical_unit_id = descriptor.logical_unit_id;
            plan.target_pool_cookie = target_unit->shards[0].pool_cookie;
            plan.descriptor = descriptor;
            plan.authorized_runs = runs;
            plan.placements = std::move(placements);
            // The degrade cursor is controller-wide. A mixed projection may
            // transcode only some units, but every unit publishes under the
            // same projected cursor.
            plan.target_controller_cursor =
                target_child->controller_policy.cursor;
            // Downward import regenerates the target-tier sink stash before
            // the canonical outgoing tapped edge. A source-tier stash is not a
            // target-tier byte image and must never be copied as if exact.
            // Same-domain tapped upward reconstruction keeps an authenticated
            // clean tapped-domain prefix: it remains valid for future retiering.
            // Full-domain upward has no such source stash.
            plan.stash_action = transform_kind ==
                    vbr_import_transform_kind::downward ||
                (transform_kind ==
                     vbr_import_transform_kind::upward_same_domain &&
                 source_domain == vbr_repr_domain::full)
                ? vbr_validated_stash_action::omit_live_rebased
                : stash_action;
            if (transform_kind ==
                    vbr_import_transform_kind::upward_cross_domain) {
                needs_live_rebase = true;
                plan.stash_action = stash_action ==
                        vbr_validated_stash_action::restore_exact
                    ? vbr_validated_stash_action::consume_exact_then_drop
                    : vbr_validated_stash_action::omit_live_rebased;
            }
            plan.unit_reference = *reference;
            plan.controller_policy =
                manifest.controller_policy[descriptor.child_id];
            plan.operation_target.instance_id = target_child->instance_id;
            plan.operation_target.operation_class =
                vbr_operation_class::state_api;
            plan.operation_target.registrant_mask =
                vbr_registrant_bit(
                    vbr_mutation_registrant::whole_import);
            plan.operation_target.child_phase =
                vbr_operation_phase::mutate;
            plan.operation_target.stream = VBR_STREAM_ANY;
            plan.operation_target.seq_id = policy.destination_sequence;
            plan.operation_target.range = {
                0, std::numeric_limits<llama_pos>::max(),
            };
            if (unit.payload_shards.size() != descriptor.shards.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::malformed);
            }
            for (size_t i = 0; i < descriptor.shards.size(); ++i) {
                vbr_validated_shard_plan shard;
                shard.shard_index = uint32_t(i);
                shard.target_pool_cookie =
                    target_unit->shards[i].pool_cookie;
                if (!target_domain_for(
                        { llama_cache_acct_residency::device,
                          llama_cache_acct_domain_kind::device_topology,
                          descriptor.shards[i].topology_index,
                          descriptor.shards[i].device_ordinal },
                        policy, shard.domain)) {
                    return terminal_result(
                        vbr_manifest_validation_status::topology_mismatch);
                }
                shard.logical_offset =
                    descriptor.shards[i].logical_offset;
                shard.row_count = descriptor.shards[i].row_count;
                shard.row_bytes = descriptor.shards[i].row_bytes;
                shard.target_row_bytes = target_unit->shards[i].row_bytes;
                shard.target_mapped_bytes = target_unit->shards[i].mapped_bytes;
                shard.payload_bytes =
                    descriptor.shards[i].payload_bytes;
                shard.source = unit.payload_shards[i];
                if (!shard.source ||
                    shard.source->size() != shard.payload_bytes) {
                    return terminal_result(
                        vbr_manifest_validation_status::malformed);
                }
                plan.shards.push_back(std::move(shard));
            }
            child_plans.push_back(std::move(plan));
        }

        std::vector<vbr_validated_companion_plan> companion_plans;
        if (package.companions().size() != manifest.companions.size()) {
            return terminal_result(
                vbr_manifest_validation_status::required_companion_unavailable);
        }
        for (size_t companion_index = 0;
             companion_index < package.companions().size();
             ++companion_index) {
            const auto & companion = package.companions()[companion_index];
            const auto target_companion = std::find_if(
                target.companions.begin(), target.companions.end(),
                [&](const vbr_target_companion_snapshot & value) {
                    return value.kind == companion.descriptor.kind &&
                           value.format_version ==
                               companion.descriptor.format_version &&
                           value.build_identity_digest ==
                               companion.descriptor.build_identity_digest;
                });
            if (target_companion == target.companions.end() ||
                !target_companion->available ||
                target_companion->target_cookie == nullptr ||
                !companion.payload ||
                companion.payload->size() !=
                    companion.descriptor.payload_bytes) {
                return terminal_result(
                    vbr_manifest_validation_status::required_companion_unavailable);
            }
            std::unique_ptr<vbr_parsed_companion_image> parsed;
            if (policy.parse_companion == nullptr ||
                !policy.parse_companion(
                    policy.context, companion.descriptor,
                    *companion.payload, *target_companion, parsed) ||
                !parsed || parsed->kind() != companion.descriptor.kind ||
                parsed->format_version() !=
                    companion.descriptor.format_version) {
                return terminal_result(
                    vbr_manifest_validation_status::required_companion_unavailable);
            }
            vbr_validated_companion_plan plan;
            plan.descriptor = companion.descriptor;
            plan.target_cookie = target_companion->target_cookie;
            plan.source = companion.payload;
            plan.parsed = std::move(parsed);
            if (occupied_replacement &&
                (companion.descriptor.kind ==
                     vbr_artifact_companion_kind::recurrent ||
                 companion.descriptor.kind ==
                     vbr_artifact_companion_kind::required_spec_payload ||
                 companion.descriptor.kind ==
                     vbr_artifact_companion_kind::typed_accelerator ||
                 companion.descriptor.kind ==
                     vbr_artifact_companion_kind::qsa_index)) {
                const auto & recovery =
                    policy.occupied_replacement->recovery_package();
                if (companion_index >= recovery.companions().size()) {
                    return terminal_result(
                        vbr_manifest_validation_status::
                            required_companion_unavailable);
                }
                const auto & old = recovery.companions()[companion_index];
                if (old.descriptor.kind != companion.descriptor.kind ||
                    old.descriptor.format_version !=
                        companion.descriptor.format_version ||
                    old.descriptor.build_identity_digest !=
                        companion.descriptor.build_identity_digest ||
                    !old.payload || old.payload->size() !=
                        old.descriptor.payload_bytes) {
                    return terminal_result(
                        vbr_manifest_validation_status::
                            required_companion_unavailable);
                }
                std::unique_ptr<vbr_parsed_companion_image> recovery_parsed;
                if (!policy.parse_companion(
                        policy.context, old.descriptor, *old.payload,
                        *target_companion, recovery_parsed) ||
                    !recovery_parsed ||
                    recovery_parsed->kind() != old.descriptor.kind ||
                    recovery_parsed->format_version() !=
                        old.descriptor.format_version) {
                    return terminal_result(
                        vbr_manifest_validation_status::
                            required_companion_unavailable);
                }
                plan.recovery_source = old.payload;
                plan.recovery_parsed = std::move(recovery_parsed);
            }
            companion_plans.push_back(std::move(plan));
        }

        if (policy.accounting_snapshot == nullptr ||
            policy.budget_config == nullptr ||
            policy.accounting_snapshot->schema_version !=
                LLAMA_CACHE_ACCT_SCHEMA_VERSION ||
            policy.accounting_snapshot->serial !=
                target.accounting_serial ||
            policy.accounting_snapshot->completeness_manifest !=
                llama_cache_acct_known::known) {
            return terminal_result(
                vbr_manifest_validation_status::accounting_unavailable);
        }
        std::vector<llama_cache_transaction_leaf> leaves;
        llama_cache_budget_plan native_plan;
        if (!accounting_plan(
                package, policy, leaves, native_plan)) {
            return terminal_result(
                vbr_manifest_validation_status::accounting_unavailable);
        }
        llama_cache_budget_fit_state native_fit;
        if (!price_plan(
                *policy.accounting_snapshot, *policy.budget_config,
                native_plan, native_fit) ||
            native_fit == llama_cache_budget_fit_state::unavailable) {
            return terminal_result(
                vbr_manifest_validation_status::budget_unavailable);
        }
        if (needs_downward && needs_upward) {
            return terminal_result(
                vbr_manifest_validation_status::representation_mismatch);
        }
        const bool needs_transform = needs_downward || needs_upward;
        if (needs_transform) {
            const auto * transform_tree = needs_downward &&
                    policy.downward_projection != nullptr
                ? &policy.downward_projection->tree_digest
                : policy.schedule_quote != nullptr
                    ? &policy.schedule_quote->destination().tree_digest
                    : nullptr;
            if (policy.transform_budget_plan == nullptr ||
                policy.read_transform_tree_digest == nullptr ||
                transform_tree == nullptr || !digest_nonzero(*transform_tree) ||
                (needs_downward && policy.downward_projection == nullptr)) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            std::array<uint8_t, 32> live_transform_tree = {};
            if (!policy.read_transform_tree_digest(
                    policy.context, live_transform_tree) ||
                live_transform_tree != *transform_tree) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            // An empty transform plan uses the already-priced native claim as
            // its classification citation. The authenticated destination quote
            // owns current physical feasibility; stage's transform reservation
            // remains the final exact growth/workspace admission before H2D.
            llama_cache_budget_fit_state transform_fit = native_fit;
            if (policy.transform_budget_plan->accounting_serial !=
                    target.accounting_serial ||
                (!policy.transform_budget_plan->entries.empty() &&
                 !price_plan(
                     *policy.accounting_snapshot, *policy.budget_config,
                     *policy.transform_budget_plan, transform_fit)) ||
                transform_fit ==
                    llama_cache_budget_fit_state::unavailable) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            if (transform_fit == llama_cache_budget_fit_state::exceeds) {
                return terminal_result(
                    vbr_manifest_validation_status::validated,
                    fallback_decision(policy));
            }
        } else if (native_fit == llama_cache_budget_fit_state::exceeds) {
            return terminal_result(
                vbr_manifest_validation_status::validated,
                fallback_decision(policy));
        }

        vbr_import_decision decision;
        if (policy.schedule_quote != nullptr &&
            ((needs_downward && policy.schedule_quote->status() !=
                                  vbr_import_schedule_status::downward) ||
             (needs_upward && policy.schedule_quote->status() !=
                                  (needs_cross_domain_upward
                                      ? vbr_import_schedule_status::upward_cross_domain
                                      : vbr_import_schedule_status::upward_same_domain)) ||
             (!needs_transform && policy.schedule_quote->status() !=
                                   vbr_import_schedule_status::exact))) {
            return terminal_result(
                vbr_manifest_validation_status::unavailable);
        }
        if (needs_downward) {
            decision = vbr_import_decision::downward_rebase;
        } else if (needs_upward) {
            decision = vbr_import_decision::upward_reconstruct;
        } else if (needs_live_rebase || !policy.allow_native) {
            if (!policy.allow_live_rebased) {
                decision = fallback_decision(policy);
            } else {
                decision = vbr_import_decision::live_rebased;
            }
        } else {
            if (!policy.native_instance_available) {
                return terminal_result(
                    vbr_manifest_validation_status::native_lineage_unavailable);
            }
            decision = vbr_import_decision::native_import;
        }
        if (decision == vbr_import_decision::rebuild ||
            decision == vbr_import_decision::cold ||
            decision == vbr_import_decision::reject) {
            return terminal_result(
                vbr_manifest_validation_status::validated, decision);
        }

        vbr_target_empty_fingerprint fingerprint;
        fingerprint.memory_instance_cookie = target.memory_instance_cookie;
        fingerprint.target_state_serial = target.target_state_serial;
        fingerprint.accounting_serial = target.accounting_serial;
        fingerprint.tree_shape_digest = target.tree_shape_digest;
        fingerprint.policy_epoch = target.policy_epoch;
        for (const auto & child : target.children) {
            fingerprint.previously_observed |= child.previously_observed;
            fingerprint.children.push_back({
                child.child_id, child.memory_cookie,
                child.state_serial, child.instance_id,
            });
        }
        if ((!occupied_replacement &&
             (policy.recheck_target_empty == nullptr ||
              !policy.recheck_target_empty(policy.context, fingerprint))) ||
            policy.read_accounting_serial == nullptr ||
            policy.read_policy_epoch == nullptr ||
            policy.read_accounting_serial(policy.context) !=
                target.accounting_serial ||
            policy.read_policy_epoch(policy.context) !=
                target.policy_epoch) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        if (policy.adoption_nonce == 0) {
            return terminal_result(vbr_manifest_validation_status::internal_error);
        }
        if (occupied_replacement) {
            if (target.destination_sequence_absent || target.children.size() != 1 ||
                target.children.front().empty ||
                manifest.stream_placements.size() != 1 ||
                child_plans.empty()) {
                return terminal_result(
                    vbr_manifest_validation_status::target_not_empty);
            }
            const auto & mappings =
                policy.occupied_replacement->cell_mapping();
            const auto & placement = manifest.stream_placements.front();
            if (mappings.empty() || mappings.size() != placement.cells.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            for (size_t i = 0; i < mappings.size(); ++i) {
                const auto & mapping = mappings[i];
                const auto & source = placement.cells[i];
                if (mapping.source_stream != placement.stream_index ||
                    mapping.logical_position != source.logical_position ||
                    mapping.source_physical_cell != source.physical_cell ||
                    mapping.ext_x != source.ext_x || mapping.ext_y != source.ext_y) {
                    return terminal_result(
                        vbr_manifest_validation_status::ownership_mismatch);
                }
            }
            const auto & guard_runs =
                policy.occupied_replacement->relocation_runs();
            const auto & recovery_runs =
                policy.occupied_replacement->recovery_runs();
            const auto strategy = policy.occupied_replacement->strategy();
            if (guard_runs.empty() || guard_runs.size() >
                    VBR_OCCUPIED_REPLACEMENT_MAX_RUNS ||
                (strategy ==
                     vbr_occupied_replacement_strategy::provisional_free_cells
                    ? !recovery_runs.empty()
                    : strategy ==
                          vbr_occupied_replacement_strategy::recycle_incumbent_cells
                        ? recovery_runs.empty()
                        : true)) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            if (needs_transform &&
                strategy !=
                    vbr_occupied_replacement_strategy::recycle_incumbent_cells) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            size_t mapping_index = 0;
            for (const auto & run : guard_runs) {
                if (run.cell_count == 0 ||
                    run.first_source_packed_row == UINT64_MAX ||
                    run.first_destination_physical_cell == UINT32_MAX ||
                    run.cell_count > mappings.size()-mapping_index) {
                    return terminal_result(
                        vbr_manifest_validation_status::geometry_mismatch);
                }
                for (uint32_t offset = 0; offset < run.cell_count; ++offset) {
                    const auto & mapping = mappings[mapping_index++];
                    if (run.first_source_packed_row > UINT64_MAX-offset ||
                        run.first_source_packed_row+offset !=
                            mapping.source_packed_row ||
                        uint64_t(run.first_destination_physical_cell)+offset !=
                            mapping.destination_physical_cell) {
                        return terminal_result(
                            vbr_manifest_validation_status::geometry_mismatch);
                    }
                }
            }
            if (mapping_index != mappings.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            uint64_t shard_count = 0;
            for (const auto & plan : child_plans) {
                shard_count += plan.shards.size();
            }
            if (shard_count == 0 ||
                recovery_runs.size() >
                    VBR_OCCUPIED_REPLACEMENT_MAX_RUNS-guard_runs.size() ||
                guard_runs.size()+recovery_runs.size() > 4096/shard_count) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            // Exact replacement publishes a fresh live-rebased lineage. A
            // transformed replacement retains the already-authenticated
            // direction so staging reserves and adoption executes the shared
            // codec transaction before the same no-fail metadata swap.
            if (!needs_transform) {
                decision = vbr_import_decision::live_rebased;
            }
        }
        if (decision != vbr_import_decision::native_import) {
            for (auto & child : tracker.children) {
                if (child.child_id >= target.children.size()) {
                    return terminal_result(
                        vbr_manifest_validation_status::internal_error);
                }
                child.transition =
                    vbr_tracker_install_transition::whole_import;
                child.lineage_uuid =
                    target.children[child.child_id].lineage_uuid;
                child.global_generation = 1;
                child.units.assign(
                    target.children[child.child_id].units.size(), {});
                std::vector<bool> initialized(child.units.size(), false);
                for (const auto & plan : child_plans) {
                    if (plan.child_id != child.child_id) {
                        continue;
                    }
                    if (plan.logical_unit_id >= child.units.size() ||
                        initialized[plan.logical_unit_id]) {
                        return terminal_result(
                            vbr_manifest_validation_status::internal_error);
                    }
                    auto & fresh = child.units[plan.logical_unit_id];
                    fresh.repr_gen = 1;
                    fresh.current_type = plan.selected_target_type;
                    fresh.last_source_type = plan.target_last_source_type;
                    fresh.domain = plan.selected_target_domain;
                    fresh.promote_hops = plan.target_promote_hops;
                    fresh.last_transition =
                        vbr_repr_transition::whole_import;
                    initialized[plan.logical_unit_id] = true;
                }
                if (std::find(initialized.begin(), initialized.end(), false) !=
                        initialized.end()) {
                    return terminal_result(
                        vbr_manifest_validation_status::internal_error);
                }
            }
        }
        vbr_artifact_package_view retained;
        if (package.retain(retained) != vbr_artifact_resolve_status::ok) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        auto proof = std::unique_ptr<vbr_validated_manifest>(
            new vbr_validated_manifest());
        proof->source_lease_ = std::move(retained);
        if (occupied_replacement) {
            proof->occupied_replacement_ =
                std::make_unique<vbr_occupied_replacement_guard>(
                    std::move(*policy.occupied_replacement));
        }
        proof->decision_ = decision;
        proof->target_ = std::move(fingerprint);
        proof->manifest_digest_ = manifest.manifest_digest;
        proof->capture_generation_id_ =
            manifest.capture_generation_id;
        proof->authenticated_identity_ = policy.identity;
        proof->token_block_ = manifest.token_block;
        proof->children_ = std::move(child_plans);
        proof->companions_ = std::move(companion_plans);
        proof->accounting_leaves_ = std::move(leaves);
        proof->tracker_install_ = std::move(tracker);
        proof->source_controllers_ = manifest.generation.controllers;
        proof->adoption_nonce_ = policy.adoption_nonce;
        proof->recheck_context_ = policy.context;
        proof->recheck_target_empty_ = policy.recheck_target_empty;
        proof->read_accounting_serial_ = policy.read_accounting_serial;
        proof->read_policy_epoch_ = policy.read_policy_epoch;
        proof->read_transform_tree_digest_ =
            policy.read_transform_tree_digest;
        proof->occupied_representation_context_ =
            policy.occupied_representation_context;
        proof->occupied_representation_identity_ =
            policy.occupied_representation_identity;

        vbr_manifest_validation_result result;
        result.status = vbr_manifest_validation_status::validated;
        result.decision = decision;
        result.proof = std::move(proof);
        return result;
    } catch (...) {
        return terminal_result(vbr_manifest_validation_status::internal_error);
    }
}

vbr_manifest_validation_result vbr_validate_attention_prefix_projection(
        const vbr_target_validation_snapshot & target,
        vbr_artifact_attention_prefix_projection && projection_value,
        const vbr_adopt_policy & policy) noexcept {
    vbr_artifact_attention_prefix_projection projection(
        std::move(projection_value));
    try {
        vbr_artifact_prefix_validation_source source;
        if (!projection.validation_source(source) ||
            source.artifact.v == 0 || source.topologies == nullptr ||
            source.manifest == nullptr || source.units == nullptr ||
            source.companions == nullptr ||
            source.reference_allocations == nullptr) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        const auto & manifest = *source.manifest;
        const auto & units = *source.units;
        const auto prefix_tokens = projection.prefix_tokens().size();
        const bool occupied_replacement =
            policy.occupied_replacement != nullptr;
        if (!policy.authorized) {
            return terminal_result(
                vbr_manifest_validation_status::unauthorized);
        }
        if (prefix_tokens == 0 || prefix_tokens > UINT32_MAX ||
            projection.parent_artifact() != source.artifact ||
            projection.parent_manifest_digest() != manifest.manifest_digest ||
            !projection.digest().valid() ||
            projection.identity().token_count != int64_t(prefix_tokens) ||
            projection.identity().next_position != llama_pos(prefix_tokens) ||
            projection.identity().execution_identity !=
                policy.identity.execution_identity ||
            projection.identity().adapter_config_identity !=
                policy.identity.adapter_config_identity ||
            projection.identity().media_content_identity !=
                policy.identity.media_content_identity ||
            projection.identity().sequence_epoch !=
                policy.identity.sequence_epoch ||
            policy.identity.requested_frontier != llama_pos(prefix_tokens)) {
            return terminal_result(
                vbr_manifest_validation_status::identity_mismatch);
        }
        if (policy.identity.tokens == nullptr ||
            *policy.identity.tokens != projection.prefix_tokens()) {
            return terminal_result(
                vbr_manifest_validation_status::token_block_mismatch);
        }
        if (policy.schedule_quote != nullptr &&
            !vbr_import_schedule_quote_matches_source(
                *policy.schedule_quote, target, manifest, units)) {
            return terminal_result(
                vbr_manifest_validation_status::unavailable);
        }
        if (occupied_replacement &&
            (!policy.occupied_replacement->ready() ||
             policy.occupied_replacement->destination() !=
                 policy.destination_sequence ||
             policy.occupied_replacement->incoming_artifact() !=
                 projection.parent_artifact() ||
             !policy.occupied_replacement->recovery_package())) {
            return terminal_result(
                vbr_manifest_validation_status::ownership_mismatch);
        }
        if (manifest.version <
                VBR_UNIT_ARTIFACT_FORMAT_VERSION_REFERENCE_PLACEMENT ||
            manifest.version > VBR_UNIT_ARTIFACT_FORMAT_VERSION ||
            manifest.manifest_digest != projection.parent_manifest_digest() ||
            manifest.identity.execution_identity !=
                projection.identity().execution_identity ||
            manifest.identity.adapter_config_identity !=
                projection.identity().adapter_config_identity ||
            manifest.identity.media_content_identity !=
                projection.identity().media_content_identity ||
            manifest.identity.sequence_epoch !=
                projection.identity().sequence_epoch ||
            !manifest.capture_generation_id.valid() ||
            manifest.generation.status !=
                vbr_checkpoint_generation_status::complete ||
            manifest.generation.controllers.size() != 1 ||
            manifest.controller_policy.size() != 1 ||
            manifest.stream_placements.size() != 1 ||
            manifest.unit_references.size() != units.size() ||
            !source.companions->empty() || !manifest.companions.empty()) {
            return terminal_result(
                vbr_manifest_validation_status::restore_metadata_missing);
        }
        if (target.memory_instance_cookie == 0 ||
            target.target_state_serial == 0 ||
            target.tree_shape_digest == 0 || target.policy_epoch == 0 ||
            target.children.size() != 1 || !target.scheduler_idle ||
            target.destination_sequence_absent == occupied_replacement) {
            return terminal_result(
                vbr_manifest_validation_status::memory_tree_mismatch);
        }
        const auto & controller = manifest.generation.controllers.front();
        const auto & source_policy = manifest.controller_policy.front();
        const auto & source_placement = manifest.stream_placements.front();
        const auto & target_child = target.children.front();
        if (controller.child_id != 0 || target_child.child_id != 0 ||
            source_policy.child_id != 0 || source_placement.child_id != 0 ||
            controller.dependency_mode !=
                checkpoint_child_dependency_mode::live_guarded ||
            source_policy.dependency_mode != controller.dependency_mode ||
            target_child.dependency_mode != controller.dependency_mode ||
            source_policy.n_stream != 1 || !source_policy.unified ||
            !source_policy.completed_wave || source_placement.stream_index != 0 ||
            source_placement.computation_frontier !=
                manifest.identity.next_position ||
            target_child.memory_cookie == nullptr ||
            target_child.empty == occupied_replacement ||
            !target_child.dedicated || !target_child.armed ||
            !vbr_controller_instance_id_is_set(target_child.instance_id) ||
            target_child.policy_epoch != target.policy_epoch ||
            !target_child.generation_compatible ||
            !target_child.ownership_compatible ||
            !target_child.stash_compatible) {
            return terminal_result(
                vbr_manifest_validation_status::target_not_armed);
        }
        const auto & target_policy = target_child.controller_policy;
        const auto * selected_destination = policy.schedule_quote &&
                policy.schedule_quote->destination().feasible()
            ? &policy.schedule_quote->destination() : nullptr;
        const bool projected_policy = selected_destination != nullptr &&
            !selected_destination->final_types.empty() &&
            !selected_destination->final_cursors.empty() &&
            !selected_destination->child_type_digests.empty() &&
            target_policy.cursor == selected_destination->final_cursors[0] &&
            target_policy.current_type_vector_digest ==
                selected_destination->child_type_digests[0];
        if (target_policy.child_id != source_policy.child_id ||
            target_policy.dependency_mode != source_policy.dependency_mode ||
            target_policy.degrade_order_digest !=
                source_policy.degrade_order_digest ||
            target_policy.policy_digest != source_policy.policy_digest ||
            target_policy.floor_type != source_policy.floor_type ||
            target_policy.pressure_independent_settings !=
                source_policy.pressure_independent_settings ||
            target_policy.n_stream != 1 || !target_policy.unified ||
            target_policy.wm_cells < prefix_tokens ||
            (!projected_policy &&
             (target_policy.cursor != source_policy.cursor ||
              target_policy.current_type_vector_digest !=
                  source_policy.current_type_vector_digest)) ||
            target_policy.completed_wave != source_policy.completed_wave ||
            target_child.units.size() != units.size() ||
            controller.units.size() != units.size() || units.empty()) {
            return terminal_result(
                vbr_manifest_validation_status::policy_mismatch);
        }

        std::vector<const vbr_artifact_cell_placement *> source_cells(
            prefix_tokens, nullptr);
        std::vector<uint64_t> source_packed_rows(prefix_tokens, UINT64_MAX);
        for (const auto & cell : source_placement.cells) {
            if (cell.logical_position >= 0 &&
                uint64_t(cell.logical_position) < prefix_tokens) {
                auto & slot = source_cells[size_t(cell.logical_position)];
                if (slot != nullptr) {
                    return terminal_result(
                        vbr_manifest_validation_status::ownership_mismatch);
                }
                slot = &cell;
            }
        }
        if (std::find(source_cells.begin(), source_cells.end(), nullptr) !=
                source_cells.end()) {
            return terminal_result(
                vbr_manifest_validation_status::ownership_mismatch);
        }
        uint64_t cell_count = 0;
        for (const auto & run : projection.cell_runs()) {
            if (run.cell_count == 0 || run.first_logical_position < 0 ||
                uint64_t(run.first_logical_position) != cell_count ||
                run.cell_count > prefix_tokens - cell_count) {
                return terminal_result(
                    vbr_manifest_validation_status::ownership_mismatch);
            }
            for (uint32_t i = 0; i < run.cell_count; ++i) {
                const uint64_t physical =
                    uint64_t(run.first_physical_cell) + i;
                if (physical > UINT32_MAX ||
                    source_cells[size_t(cell_count + i)]->physical_cell !=
                        physical ||
                    run.first_packed_row > UINT64_MAX - i) {
                    return terminal_result(
                        vbr_manifest_validation_status::ownership_mismatch);
                }
                source_packed_rows[size_t(cell_count + i)] =
                    run.first_packed_row + i;
            }
            cell_count += run.cell_count;
        }
        if (cell_count != prefix_tokens) {
            return terminal_result(
                vbr_manifest_validation_status::ownership_mismatch);
        }

        // The target image owns one placement shared by every unit. Empty
        // imports use the dense logical prefix; occupied imports consume the
        // guard's already-authenticated destination mapping.
        // Repeating this million-cell vector on each unit would add no
        // authority: build_live_image() already unions plan placements.
        vbr_artifact_stream_placement dense_placement;
        dense_placement.child_id = 0;
        dense_placement.stream_index = 0;
        dense_placement.source_sequence = source_placement.source_sequence;
        dense_placement.computation_frontier = llama_pos(prefix_tokens);
        dense_placement.cells.reserve(prefix_tokens);
        const auto * occupied_mappings = occupied_replacement
            ? &policy.occupied_replacement->cell_mapping() : nullptr;
        if (occupied_mappings && occupied_mappings->size() != prefix_tokens) {
            return terminal_result(
                vbr_manifest_validation_status::ownership_mismatch);
        }
        for (uint32_t i = 0; i < prefix_tokens; ++i) {
            const uint32_t destination = occupied_mappings
                ? (*occupied_mappings)[i].destination_physical_cell : i;
            if (occupied_mappings &&
                ((*occupied_mappings)[i].logical_position != llama_pos(i) ||
                 (*occupied_mappings)[i].source_physical_cell !=
                     source_cells[i]->physical_cell ||
                 (*occupied_mappings)[i].source_packed_row !=
                     source_packed_rows[i])) {
                return terminal_result(
                    vbr_manifest_validation_status::ownership_mismatch);
            }
            dense_placement.cells.push_back({
                destination, llama_pos(i), source_cells[i]->ext_x,
                source_cells[i]->ext_y,
            });
        }

        std::vector<vbr_validated_child_plan> child_plans;
        child_plans.reserve(units.size());
        std::vector<const vbr_artifact_unit_reference *> references(
            units.size(), nullptr);
        for (const auto & reference : manifest.unit_references) {
            if (reference.logical_unit_id >= references.size() ||
                references[reference.logical_unit_id] != nullptr) {
                return terminal_result(
                    vbr_manifest_validation_status::generation_mismatch);
            }
            references[reference.logical_unit_id] = &reference;
        }
        std::vector<size_t> binding_order(policy.domain_bindings.size());
        std::iota(binding_order.begin(), binding_order.end(), size_t(0));
        const auto binding_key = [&](size_t index) {
            const auto & binding = policy.domain_bindings[index];
            return std::make_tuple(
                uint8_t(binding.domain.residency), binding.topology_index,
                binding.device_ordinal);
        };
        std::sort(binding_order.begin(), binding_order.end(),
            [&](size_t lhs, size_t rhs) {
                return std::tie(
                    policy.domain_bindings[lhs].domain.residency,
                    policy.domain_bindings[lhs].topology_index,
                    policy.domain_bindings[lhs].device_ordinal, lhs) <
                std::tie(
                    policy.domain_bindings[rhs].domain.residency,
                    policy.domain_bindings[rhs].topology_index,
                    policy.domain_bindings[rhs].device_ordinal, rhs);
            });
        uint64_t selected_bytes = 0;
        size_t proof_cursor = 0;
        size_t source_run_cursor = 0;
        bool needs_downward = false;
        bool needs_upward = false;
        bool needs_cross_domain_upward = false;
        for (size_t unit_index = 0; unit_index < units.size(); ++unit_index) {
            const auto & unit = units[unit_index];
            const auto & descriptor = unit.descriptor;
            const auto * reference = references[unit_index];
            if (descriptor.child_id != 0 ||
                descriptor.logical_unit_id != unit_index ||
                descriptor.n_stream != 1 || !descriptor.unified ||
                descriptor.wm_cells < prefix_tokens ||
                descriptor.shards.empty() ||
                descriptor.clean_stash_state !=
                    vbr_artifact_clean_stash_state::absent_at_source ||
                !descriptor.clean_stash.shards.empty() ||
                !unit.stash_shards.empty() || reference == nullptr ||
                reference->lineage_uuid != descriptor.lineage_uuid ||
                reference->logical_unit_id != unit_index ||
                reference->repr_gen != descriptor.repr_gen ||
                !reference->unit_version_id.valid() ||
                reference->authorized_stream_refs.size() != 1 ||
                reference->authorized_stream_refs.front() != 0 ||
                reference->has_stash_reference ||
                unit.payload_shards.size() != descriptor.shards.size() ||
                unit_index >= controller.units.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
            const auto & source_generation = controller.units[unit_index];
            const auto & target_unit = target_child.units[unit_index];
            const auto source_domain = source_generation.domain;
            if (target_unit.logical_unit_id != unit_index ||
                target_unit.n_stream != 1 || !target_unit.unified ||
                target_unit.v_trans ||
                target_unit.wm_cells < prefix_tokens ||
                target_unit.recoverability != descriptor.recoverability ||
                target_unit.side != descriptor.side ||
                target_unit.layout != descriptor.layout ||
                target_unit.row_codec_version !=
                    descriptor.row_codec_version ||
                target_unit.rank != descriptor.rank ||
                target_unit.dimensions != descriptor.dimensions ||
                target_unit.row_alignment != descriptor.row_alignment ||
                source_generation.current_type != descriptor.current_type ||
                source_generation.repr_gen != descriptor.repr_gen ||
                source_generation.last_source_type !=
                    descriptor.last_source_type ||
                source_generation.promote_hops != descriptor.promote_hops ||
                source_generation.last_transition !=
                    descriptor.last_transition ||
                target_unit.shards.size() != descriptor.shards.size()) {
                return terminal_result(
                    vbr_manifest_validation_status::representation_mismatch);
            }

            vbr_validated_child_plan plan;
            const auto transform_status = configure_transform_plan(
                descriptor, source_domain, target_unit, 0,
                uint32_t(unit_index), policy, plan, needs_downward,
                needs_upward, needs_cross_domain_upward);
            if (transform_status !=
                    vbr_manifest_validation_status::validated) {
                return terminal_result(transform_status);
            }
            // The shared transform owner supplies representation/recipe and
            // per-row workspace authority. Prefix aggregate bytes are
            // recomputed below from the selected rows rather than inheriting
            // the full-frontier totals.
            plan.target_mapped_bytes = 0;
            plan.transfer_bytes = 0;
            const auto transform_kind = plan.transform_kind;
            plan.child_id = 0;
            plan.dependency_mode = target_child.dependency_mode;
            plan.logical_unit_id = uint32_t(unit_index);
            plan.target_pool_cookie = target_unit.shards.front().pool_cookie;
            plan.descriptor = descriptor;
            plan.descriptor.wm_cells = prefix_tokens;
            plan.authorized_runs.push_back({ 0, uint32_t(prefix_tokens) });
            if (unit_index == 0) {
                plan.placements.push_back(std::move(dense_placement));
            }
            plan.target_controller_cursor = target_policy.cursor;
            plan.stash_action = transform_kind ==
                    vbr_import_transform_kind::none
                ? vbr_validated_stash_action::none_at_source
                : vbr_validated_stash_action::omit_live_rebased;
            plan.unit_reference = *reference;
            plan.unit_reference.authorized_stream_refs = { 0 };
            plan.controller_policy = target_policy;
            plan.controller_policy.wm_cells = prefix_tokens;
            plan.operation_target.instance_id = target_child.instance_id;
            plan.operation_target.operation_class =
                vbr_operation_class::state_api;
            plan.operation_target.registrant_mask = vbr_registrant_bit(
                vbr_mutation_registrant::whole_import);
            plan.operation_target.child_phase = vbr_operation_phase::mutate;
            plan.operation_target.stream = VBR_STREAM_ANY;
            plan.operation_target.seq_id = policy.destination_sequence;
            plan.operation_target.range = { 0, llama_pos(prefix_tokens) };
            plan.shards.reserve(descriptor.shards.size());

            for (size_t shard_index = 0;
                 shard_index < descriptor.shards.size(); ++shard_index) {
                const auto & source_shard = descriptor.shards[shard_index];
                const auto & target_shard = target_unit.shards[shard_index];
                llama_cache_acct_resource_domain domain;
                const auto domain_key = std::make_tuple(
                    uint8_t(llama_cache_acct_residency::device),
                    source_shard.topology_index,
                    source_shard.device_ordinal);
                const auto binding = std::lower_bound(
                    binding_order.begin(), binding_order.end(), domain_key,
                    [&](size_t index, const auto & key) {
                        return binding_key(index) < key;
                    });
                const bool binding_found =
                    binding != binding_order.end() &&
                    binding_key(*binding) == domain_key;
                if (binding_found) {
                    domain = policy.domain_bindings[*binding].domain;
                }
                if (source_shard.shard_index != shard_index ||
                    target_shard.shard_index != shard_index ||
                    source_shard.row_bytes == 0 ||
                    target_shard.row_bytes == 0 ||
                    target_shard.row_bytes != plan.target_row_bytes ||
                    (transform_kind == vbr_import_transform_kind::none &&
                     source_shard.row_bytes != target_shard.row_bytes) ||
                    source_shard.topology_index >= source.topologies->size() ||
                    target_shard.topology_index !=
                        source_shard.topology_index ||
                    target_shard.device_ordinal !=
                        source_shard.device_ordinal ||
                    target_shard.topology_digest !=
                        (*source.topologies)[source_shard.topology_index].digest ||
                    target_shard.logical_offset !=
                        source_shard.logical_offset ||
                    target_shard.row_count < prefix_tokens ||
                    target_shard.pool_cookie == nullptr ||
                    prefix_tokens > UINT64_MAX/source_shard.row_bytes ||
                    target_shard.mapped_bytes <
                        prefix_tokens*target_shard.row_bytes ||
                    !unit.payload_shards[shard_index] ||
                    !binding_found ||
                    domain != target_shard.domain) {
                    return terminal_result(
                        vbr_manifest_validation_status::topology_mismatch);
                }
                if (proof_cursor >= projection.proofs().size()) {
                    return terminal_result(
                        vbr_manifest_validation_status::checksum_or_digest_mismatch);
                }
                const auto & proof = projection.proofs()[proof_cursor++];
                if (proof.unit_index != unit_index ||
                    proof.shard_index != shard_index || !proof.proof ||
                    proof.proof.root() != source_shard.section_checksum ||
                    proof.proof.total_bytes() !=
                        unit.payload_shards[shard_index]->size()) {
                    return terminal_result(
                        vbr_manifest_validation_status::checksum_or_digest_mismatch);
                }
                uint64_t logical = 0;
                uint64_t shard_bytes = 0;
                vbr_validated_shard_plan shard;
                shard.shard_index = uint32_t(shard_index);
                shard.domain = domain;
                shard.target_pool_cookie = target_shard.pool_cookie;
                shard.logical_offset = source_shard.logical_offset;
                shard.row_count = prefix_tokens;
                shard.row_bytes = source_shard.row_bytes;
                shard.target_row_bytes = target_shard.row_bytes;
                shard.target_mapped_bytes =
                    prefix_tokens*target_shard.row_bytes;
                shard.source = unit.payload_shards[shard_index];
                shard.projection_proof = proof.proof;
                shard.projection_runs.reserve(projection.cell_runs().size());
                for (size_t run_index = 0;
                     run_index < projection.cell_runs().size(); ++run_index) {
                    if (source_run_cursor >= projection.source_runs().size()) {
                        return terminal_result(
                            vbr_manifest_validation_status::ownership_mismatch);
                    }
                    const auto & run =
                        projection.source_runs()[source_run_cursor++];
                    uint64_t size = 0;
                    uint64_t destination = 0;
                    if (run.unit_index != unit_index ||
                        run.shard_index != shard_index ||
                        run.cell_count == 0 ||
                        run.first_logical_position < 0 ||
                        uint64_t(run.first_logical_position) != logical ||
                        run.cell_count > prefix_tokens - logical ||
                        uint64_t(run.cell_count) >
                            UINT64_MAX/source_shard.row_bytes) {
                        return terminal_result(
                            vbr_manifest_validation_status::ownership_mismatch);
                    }
                    size = uint64_t(run.cell_count)*source_shard.row_bytes;
                    if (run.size != size ||
                        run.first_physical_cell !=
                            source_cells[size_t(logical)]->physical_cell ||
                        source_packed_rows[size_t(logical)] >
                            UINT64_MAX/source_shard.row_bytes ||
                        run.source_offset !=
                            source_packed_rows[size_t(logical)]*
                                source_shard.row_bytes ||
                        logical > UINT64_MAX/source_shard.row_bytes) {
                        return terminal_result(
                            vbr_manifest_validation_status::geometry_mismatch);
                    }
                    // H2D targets the source-typed alias. The transaction maps
                    // target growth first, uploads compact source rows using
                    // their own stride, then expands/re-encodes in place.
                    destination = logical*source_shard.row_bytes;
                    const auto & ranges = proof.proof.ranges();
                    auto containing = std::upper_bound(
                        ranges.begin(), ranges.end(), run.source_offset,
                        [](uint64_t offset, const auto & range) {
                            return offset < range.offset;
                        });
                    if (containing == ranges.begin() ||
                        (--containing)->offset > run.source_offset ||
                        run.source_offset - containing->offset >
                            containing->size ||
                        size > containing->size -
                            (run.source_offset - containing->offset) ||
                        run.source_offset > shard.source->size() ||
                        size > shard.source->size() - run.source_offset) {
                        return terminal_result(
                            vbr_manifest_validation_status::ownership_mismatch);
                    }
                    shard.projection_runs.push_back({
                        run.source_offset, destination, size,
                    });
                    logical += run.cell_count;
                    shard_bytes += size;
                }
                if (logical != prefix_tokens || shard_bytes == 0 ||
                    shard_bytes > UINT64_MAX - selected_bytes) {
                    return terminal_result(
                        vbr_manifest_validation_status::ownership_mismatch);
                }
                shard.payload_bytes = shard_bytes;
                plan.descriptor.shards[shard_index].row_count = prefix_tokens;
                plan.descriptor.shards[shard_index].payload_bytes = shard_bytes;
                selected_bytes += shard_bytes;
                if (plan.target_mapped_bytes > UINT64_MAX-
                        shard.target_mapped_bytes ||
                    plan.transfer_bytes > UINT64_MAX-shard_bytes) {
                    return terminal_result(
                        vbr_manifest_validation_status::geometry_mismatch);
                }
                plan.target_mapped_bytes += shard.target_mapped_bytes;
                plan.transfer_bytes += shard_bytes;
                plan.shards.push_back(std::move(shard));
            }
            child_plans.push_back(std::move(plan));
        }
        if (selected_bytes != projection.selected_bytes() ||
            proof_cursor != projection.proofs().size() ||
            source_run_cursor != projection.source_runs().size()) {
            return terminal_result(
                vbr_manifest_validation_status::ownership_mismatch);
        }
        if (occupied_replacement) {
            uint64_t shard_count = 0;
            for (const auto & plan : child_plans) {
                if (plan.shards.size() > UINT64_MAX-shard_count) {
                    return terminal_result(
                        vbr_manifest_validation_status::geometry_mismatch);
                }
                shard_count += plan.shards.size();
            }
            const auto & incoming_runs =
                policy.occupied_replacement->relocation_runs();
            const auto & recovery_runs =
                policy.occupied_replacement->recovery_runs();
            if (shard_count == 0 || incoming_runs.empty() ||
                recovery_runs.size() >
                    VBR_OCCUPIED_REPLACEMENT_MAX_RUNS-incoming_runs.size() ||
                incoming_runs.size()+recovery_runs.size() >
                    4096/shard_count) {
                return terminal_result(
                    vbr_manifest_validation_status::geometry_mismatch);
            }
        }

        if (policy.accounting_snapshot == nullptr ||
            policy.budget_config == nullptr ||
            policy.accounting_snapshot->schema_version !=
                LLAMA_CACHE_ACCT_SCHEMA_VERSION ||
            policy.accounting_snapshot->serial != target.accounting_serial ||
            policy.accounting_snapshot->completeness_manifest !=
                llama_cache_acct_known::known) {
            return terminal_result(
                vbr_manifest_validation_status::accounting_unavailable);
        }
        std::vector<llama_cache_transaction_leaf> leaves;
        llama_cache_budget_plan budget_plan;
        if (!accounting_plan_source(
                manifest, units, *source.reference_allocations,
                source.artifact, policy, leaves, budget_plan)) {
            return terminal_result(
                vbr_manifest_validation_status::accounting_unavailable);
        }
        llama_cache_budget_fit_state fit;
        if (!price_plan(
                *policy.accounting_snapshot, *policy.budget_config,
                budget_plan, fit) ||
            fit == llama_cache_budget_fit_state::unavailable) {
            return terminal_result(
                vbr_manifest_validation_status::budget_unavailable);
        }
        if (needs_downward && needs_upward) {
            return terminal_result(
                vbr_manifest_validation_status::representation_mismatch);
        }
        const bool needs_transform = needs_downward || needs_upward;
        if (needs_transform) {
            const auto * transform_tree = needs_downward &&
                    policy.downward_projection != nullptr
                ? &policy.downward_projection->tree_digest
                : selected_destination
                    ? &selected_destination->tree_digest : nullptr;
            if (policy.transform_budget_plan == nullptr ||
                policy.read_transform_tree_digest == nullptr ||
                transform_tree == nullptr || !digest_nonzero(*transform_tree) ||
                policy.transform_budget_plan->accounting_serial !=
                    target.accounting_serial) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            std::array<uint8_t, 32> live_tree = {};
            if (!policy.read_transform_tree_digest(
                    policy.context, live_tree) || live_tree != *transform_tree) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            llama_cache_budget_fit_state transform_fit = fit;
            if (!policy.transform_budget_plan->entries.empty() &&
                (!price_plan(
                    *policy.accounting_snapshot, *policy.budget_config,
                    *policy.transform_budget_plan, transform_fit) ||
                 transform_fit == llama_cache_budget_fit_state::unavailable)) {
                return terminal_result(
                    vbr_manifest_validation_status::budget_unavailable);
            }
            if (transform_fit == llama_cache_budget_fit_state::exceeds) {
                return terminal_result(
                    vbr_manifest_validation_status::validated,
                    fallback_decision(policy));
            }
        } else if (fit == llama_cache_budget_fit_state::exceeds) {
            return terminal_result(
                vbr_manifest_validation_status::validated,
                fallback_decision(policy));
        }
        if (policy.schedule_quote != nullptr &&
            ((needs_downward && policy.schedule_quote->status() !=
                                  vbr_import_schedule_status::downward) ||
             (needs_upward && policy.schedule_quote->status() !=
                                  (needs_cross_domain_upward
                                      ? vbr_import_schedule_status::
                                            upward_cross_domain
                                      : vbr_import_schedule_status::
                                            upward_same_domain)) ||
             (!needs_transform && policy.schedule_quote->status() !=
                                   vbr_import_schedule_status::exact))) {
            return terminal_result(
                vbr_manifest_validation_status::unavailable);
        }
        if (!policy.allow_live_rebased && !needs_transform) {
            return terminal_result(
                vbr_manifest_validation_status::validated,
                fallback_decision(policy));
        }

        vbr_target_empty_fingerprint fingerprint;
        fingerprint.memory_instance_cookie = target.memory_instance_cookie;
        fingerprint.target_state_serial = target.target_state_serial;
        fingerprint.accounting_serial = target.accounting_serial;
        fingerprint.tree_shape_digest = target.tree_shape_digest;
        fingerprint.policy_epoch = target.policy_epoch;
        fingerprint.previously_observed = target_child.previously_observed;
        fingerprint.children.push_back({
            target_child.child_id, target_child.memory_cookie,
            target_child.state_serial, target_child.instance_id,
        });
        if ((!occupied_replacement &&
             (policy.recheck_target_empty == nullptr ||
              !policy.recheck_target_empty(policy.context, fingerprint))) ||
            policy.read_accounting_serial == nullptr ||
            policy.read_policy_epoch == nullptr ||
            policy.read_accounting_serial(policy.context) !=
                target.accounting_serial ||
            policy.read_policy_epoch(policy.context) != target.policy_epoch) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        if (policy.adoption_nonce == 0) {
            return terminal_result(
                vbr_manifest_validation_status::internal_error);
        }

        vbr_tracker_install_plan tracker;
        vbr_tracker_install_child tracker_child;
        tracker_child.child_id = 0;
        tracker_child.transition =
            vbr_tracker_install_transition::whole_import;
        tracker_child.lineage_uuid = target_child.lineage_uuid;
        tracker_child.target_instance = target_child.instance_id;
        tracker_child.global_generation = 1;
        tracker_child.units.resize(child_plans.size());
        for (const auto & plan : child_plans) {
            auto & fresh = tracker_child.units[plan.logical_unit_id];
            fresh.repr_gen = 1;
            fresh.current_type = plan.selected_target_type;
            fresh.last_source_type = plan.target_last_source_type;
            fresh.domain = plan.selected_target_domain;
            fresh.promote_hops = plan.target_promote_hops;
            fresh.last_transition = vbr_repr_transition::whole_import;
        }
        tracker.children.push_back(std::move(tracker_child));

        const vbr_import_decision decision = needs_downward
            ? vbr_import_decision::downward_rebase
            : needs_upward
                ? vbr_import_decision::upward_reconstruct
                : vbr_import_decision::live_rebased;

        llama_sha256_writer manifest_hash;
        static constexpr char MANIFEST_DOMAIN[] =
            "buun.vbr.attention-prefix-validated-manifest/v2";
        manifest_hash.string(MANIFEST_DOMAIN, sizeof(MANIFEST_DOMAIN) - 1);
        manifest_hash.bytes(
            projection.parent_manifest_digest().bytes().data(), 32);
        manifest_hash.bytes(projection.digest().bytes().data(), 32);
        manifest_hash.u32(uint32_t(decision));
        if (needs_transform) {
            const auto & tree = needs_downward
                ? policy.downward_projection->tree_digest
                : selected_destination->tree_digest;
            manifest_hash.bytes(tree.data(), tree.size());
        }
        const auto derived_manifest =
            vbr_manifest_digest::from_sha256(manifest_hash.finish());
        llama_sha256_writer generation_hash;
        static constexpr char GENERATION_DOMAIN[] =
            "buun.vbr.attention-prefix-capture-generation/v1";
        generation_hash.string(
            GENERATION_DOMAIN, sizeof(GENERATION_DOMAIN) - 1);
        generation_hash.bytes(manifest.capture_generation_id.bytes().data(), 32);
        generation_hash.bytes(projection.digest().bytes().data(), 32);
        const auto derived_generation =
            vbr_capture_generation_id::from_sha256(generation_hash.finish());
        llama_sha256_writer token_hash;
        static constexpr char TOKEN_DOMAIN[] =
            "buun.vbr.attention-prefix-validated-token/v1";
        token_hash.string(TOKEN_DOMAIN, sizeof(TOKEN_DOMAIN) - 1);
        token_hash.bytes(projection.token_digest().bytes().data(), 32);
        token_hash.bytes(projection.digest().bytes().data(), 32);
        const auto derived_token =
            vbr_token_block_digest::from_sha256(token_hash.finish());
        if (!derived_manifest.valid() || !derived_generation.valid() ||
            !derived_token.valid()) {
            return terminal_result(
                vbr_manifest_validation_status::internal_error);
        }

        auto proof = std::unique_ptr<vbr_validated_manifest>(
            new vbr_validated_manifest());
        proof->source_projection_ = std::move(projection);
        if (occupied_replacement) {
            proof->occupied_replacement_ =
                std::make_unique<vbr_occupied_replacement_guard>(
                    std::move(*policy.occupied_replacement));
        }
        proof->decision_ = decision;
        proof->target_ = std::move(fingerprint);
        proof->manifest_digest_ = derived_manifest;
        proof->capture_generation_id_ = derived_generation;
        proof->authenticated_identity_ = policy.identity;
        proof->token_block_.codec_version =
            VBR_ARTIFACT_TOKEN_BLOCK_CODEC_VERSION;
        proof->token_block_.tokens = proof->source_projection_.prefix_tokens();
        proof->token_block_.digest = derived_token;
        proof->authenticated_identity_.tokens = &proof->token_block_.tokens;
        proof->children_ = std::move(child_plans);
        proof->accounting_leaves_ = std::move(leaves);
        proof->tracker_install_ = std::move(tracker);
        auto derived_controller = controller;
        if (derived_controller.streams.size() != 1 ||
            derived_controller.streams.front().stream_index != 0) {
            return terminal_result(
                vbr_manifest_validation_status::generation_mismatch);
        }
        derived_controller.streams.front().computation_frontier =
            llama_pos(prefix_tokens);
        derived_controller.streams.front().captured_dependency_count =
            uint32_t(prefix_tokens);
        // Whole-import assigns fresh page generations to the dense target;
        // retaining parent physical-page witnesses would falsely bind the
        // projected image back to its sparse source cells.
        derived_controller.streams.front().pages.clear();
        proof->source_controllers_.push_back(std::move(derived_controller));
        proof->adoption_nonce_ = policy.adoption_nonce;
        proof->recheck_context_ = policy.context;
        proof->recheck_target_empty_ = policy.recheck_target_empty;
        proof->read_accounting_serial_ = policy.read_accounting_serial;
        proof->read_policy_epoch_ = policy.read_policy_epoch;
        proof->read_transform_tree_digest_ =
            policy.read_transform_tree_digest;
        proof->occupied_representation_context_ =
            policy.occupied_representation_context;
        proof->occupied_representation_identity_ =
            policy.occupied_representation_identity;

        vbr_manifest_validation_result result;
        result.status = vbr_manifest_validation_status::validated;
        result.decision = decision;
        result.proof = std::move(proof);
        return result;
    } catch (...) {
        return terminal_result(vbr_manifest_validation_status::internal_error);
    }
}

vbr_manifest_validation_result vbr_validate_unit_manifest(
        llama_memory_i & target,
        const vbr_artifact_package_view & package,
        const vbr_adopt_policy & policy) noexcept {
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&target, tree)) {
            return terminal_result(
                vbr_manifest_validation_status::memory_tree_mismatch);
        }
        if (policy.inspect_target == nullptr) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        vbr_target_validation_snapshot snapshot;
        if (!policy.inspect_target(
                policy.context, target, tree, snapshot)) {
            return terminal_result(vbr_manifest_validation_status::unavailable);
        }
        // The snapshot carries one child per ATTENTION tree child; recurrent
        // children travel as companions (checked by the snapshot core against
        // the manifest's companion set). Pairing against tree.size() would make
        // every hybrid (attention+recurrent) import unsatisfiable: the snapshot
        // core requires children == controllers == attention count.
        size_t snapshot_index = 0;
        for (const auto & child : tree) {
            if (child.attention == nullptr) {
                continue;
            }
            if (snapshot_index >= snapshot.children.size() ||
                snapshot.children[snapshot_index].child_id != child.child_id ||
                snapshot.children[snapshot_index].dependency_mode !=
                    child.dependency_mode) {
                return terminal_result(
                    vbr_manifest_validation_status::memory_tree_mismatch);
            }
            ++snapshot_index;
        }
        if (snapshot_index != snapshot.children.size()) {
            return terminal_result(
                vbr_manifest_validation_status::memory_tree_mismatch);
        }
        return vbr_validate_unit_manifest_snapshot(
            snapshot, package, policy);
    } catch (...) {
        return terminal_result(vbr_manifest_validation_status::internal_error);
    }
}

const char * vbr_import_decision_name(
        vbr_import_decision decision) noexcept {
    switch (decision) {
        case vbr_import_decision::native_import: return "native_import";
        case vbr_import_decision::live_rebased: return "live_rebased";
        case vbr_import_decision::downward_rebase: return "downward_rebase";
        case vbr_import_decision::upward_reconstruct:
            return "upward_reconstruct";
        case vbr_import_decision::rebuild: return "rebuild";
        case vbr_import_decision::cold: return "cold";
        case vbr_import_decision::reject: return "reject";
        case vbr_import_decision::_count: break;
    }
    return "invalid";
}

const char * vbr_manifest_validation_status_name(
        vbr_manifest_validation_status status) noexcept {
    switch (status) {
        case vbr_manifest_validation_status::validated: return "validated";
        case vbr_manifest_validation_status::unauthorized: return "unauthorized";
        case vbr_manifest_validation_status::unsupported_artifact_version: return "unsupported_artifact_version";
        case vbr_manifest_validation_status::restore_metadata_missing: return "restore_metadata_missing";
        case vbr_manifest_validation_status::malformed: return "malformed";
        case vbr_manifest_validation_status::checksum_or_digest_mismatch: return "checksum_or_digest_mismatch";
        case vbr_manifest_validation_status::identity_mismatch: return "identity_mismatch";
        case vbr_manifest_validation_status::token_block_mismatch: return "token_block_mismatch";
        case vbr_manifest_validation_status::memory_tree_mismatch: return "memory_tree_mismatch";
        case vbr_manifest_validation_status::target_not_idle: return "target_not_idle";
        case vbr_manifest_validation_status::target_not_empty: return "target_not_empty";
        case vbr_manifest_validation_status::target_not_dedicated: return "target_not_dedicated";
        case vbr_manifest_validation_status::target_not_armed: return "target_not_armed";
        case vbr_manifest_validation_status::geometry_mismatch: return "geometry_mismatch";
        case vbr_manifest_validation_status::topology_mismatch: return "topology_mismatch";
        case vbr_manifest_validation_status::representation_mismatch: return "representation_mismatch";
        case vbr_manifest_validation_status::codebook_mismatch: return "codebook_mismatch";
        case vbr_manifest_validation_status::policy_mismatch: return "policy_mismatch";
        case vbr_manifest_validation_status::generation_mismatch: return "generation_mismatch";
        case vbr_manifest_validation_status::ownership_mismatch: return "ownership_mismatch";
        case vbr_manifest_validation_status::stash_inconsistent: return "stash_inconsistent";
        case vbr_manifest_validation_status::required_companion_unavailable: return "required_companion_unavailable";
        case vbr_manifest_validation_status::accounting_unavailable: return "accounting_unavailable";
        case vbr_manifest_validation_status::budget_unavailable: return "budget_unavailable";
        case vbr_manifest_validation_status::native_lineage_unavailable: return "native_lineage_unavailable";
        case vbr_manifest_validation_status::unavailable: return "unavailable";
        case vbr_manifest_validation_status::internal_error: return "internal_error";
        case vbr_manifest_validation_status::_count: break;
    }
    return "invalid";
}
