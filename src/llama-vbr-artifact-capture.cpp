#include "llama-vbr-artifact-capture.h"

#include "llama-vbr-identity-digest.h"
#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <tuple>
#include <utility>

namespace {

struct capture_projection_placement {
    uint64_t manifest_id = 0;
    const vbr_artifact_stream_placement * placement = nullptr;
};

struct capture_projection_cursor {
    const capture_projection_placement * source = nullptr;
    size_t cell = 0;
};

bool capture_checked_add(uint64_t a, uint64_t b, uint64_t & output) {
    if (b > UINT64_MAX - a) {
        return false;
    }
    output = a + b;
    return true;
}

bool capture_generation_equal(
        const vbr_unit_generation & lhs,
        const vbr_unit_generation & rhs) noexcept {
    return lhs.repr_gen == rhs.repr_gen &&
           lhs.publish_seq == rhs.publish_seq &&
           lhs.current_type == rhs.current_type &&
           lhs.last_source_type == rhs.last_source_type &&
           lhs.domain == rhs.domain &&
           lhs.promote_hops == rhs.promote_hops &&
           lhs.last_transition == rhs.last_transition &&
           lhs.flags == rhs.flags;
}

bool capture_generation_valid(
        const vbr_unit_generation & generation) noexcept {
    return generation.repr_gen != 0 &&
           (generation.publish_seq & 1u) == 0 &&
           generation.current_type >= 0 &&
           generation.current_type < GGML_TYPE_COUNT &&
           generation.last_source_type >= 0 &&
           generation.last_source_type < GGML_TYPE_COUNT &&
           generation.domain <= vbr_repr_domain::tapped &&
           generation.last_transition <=
               vbr_repr_transition::recovery_invalidate &&
           generation.flags == 0;
}

bool capture_digest_nonzero(
        const std::array<uint8_t, 32> & digest) noexcept {
    return std::any_of(digest.begin(), digest.end(),
        [](uint8_t value) { return value != 0; });
}

bool capture_policy_equal_impl(
        const vbr_artifact_controller_policy & lhs,
        const vbr_artifact_controller_policy & rhs) noexcept {
    return lhs.child_id == rhs.child_id &&
           lhs.dependency_mode == rhs.dependency_mode &&
           lhs.degrade_order_digest == rhs.degrade_order_digest &&
           lhs.policy_digest == rhs.policy_digest &&
           lhs.cursor == rhs.cursor &&
           lhs.floor_type == rhs.floor_type &&
           lhs.pressure_independent_settings ==
               rhs.pressure_independent_settings &&
           lhs.n_stream == rhs.n_stream && lhs.unified == rhs.unified &&
           lhs.wm_cells == rhs.wm_cells &&
           lhs.current_type_vector_digest ==
               rhs.current_type_vector_digest &&
           lhs.completed_wave == rhs.completed_wave;
}

bool capture_shard_schema_equal(
        const vbr_artifact_shard_descriptor & lhs,
        const vbr_artifact_shard_descriptor & rhs) noexcept {
    return lhs.shard_index == rhs.shard_index &&
           lhs.topology_index == rhs.topology_index &&
           lhs.device_ordinal == rhs.device_ordinal &&
           lhs.logical_offset == rhs.logical_offset &&
           lhs.row_count == rhs.row_count &&
           lhs.column_count == rhs.column_count &&
           lhs.row_bytes == rhs.row_bytes &&
           lhs.payload_bytes == rhs.payload_bytes &&
           lhs.section_checksum == rhs.section_checksum;
}

bool capture_descriptor_schema_equal(
        const vbr_artifact_unit_descriptor & lhs,
        const vbr_artifact_unit_descriptor & rhs) {
    if (lhs.child_id != rhs.child_id ||
        lhs.logical_unit_id != rhs.logical_unit_id ||
        lhs.lineage_uuid != rhs.lineage_uuid ||
        lhs.repr_gen != rhs.repr_gen ||
        lhs.current_type != rhs.current_type ||
        lhs.last_source_type != rhs.last_source_type ||
        lhs.promote_hops != rhs.promote_hops ||
        lhs.last_transition != rhs.last_transition ||
        lhs.representation.kind != rhs.representation.kind ||
        lhs.representation.codec_id != rhs.representation.codec_id ||
        lhs.representation.codec_version != rhs.representation.codec_version ||
        lhs.representation.reference_digest != rhs.representation.reference_digest ||
        lhs.representation.source_loss_history != rhs.representation.source_loss_history ||
        lhs.representation.checkpoint_codec_hops != rhs.representation.checkpoint_codec_hops ||
        lhs.recoverability != rhs.recoverability || lhs.side != rhs.side ||
        lhs.layout != rhs.layout || lhs.n_stream != rhs.n_stream ||
        lhs.unified != rhs.unified || lhs.wm_cells != rhs.wm_cells ||
        lhs.rank != rhs.rank || lhs.dimensions != rhs.dimensions ||
        lhs.row_alignment != rhs.row_alignment ||
        lhs.row_codec_version != rhs.row_codec_version ||
        lhs.codebook_digest != rhs.codebook_digest ||
        lhs.rotation_digest != rhs.rotation_digest ||
        lhs.meansub_digest != rhs.meansub_digest ||
        lhs.meansub_model_id != rhs.meansub_model_id ||
        lhs.meansub_layer != rhs.meansub_layer ||
        lhs.meansub_baked != rhs.meansub_baked ||
        lhs.clean_stash_state != rhs.clean_stash_state ||
        lhs.clean_stash.valid_rows != rhs.clean_stash.valid_rows ||
        lhs.clean_stash.domain != rhs.clean_stash.domain ||
        lhs.clean_stash.layout != rhs.clean_stash.layout ||
        lhs.clean_stash.row_count != rhs.clean_stash.row_count ||
        lhs.clean_stash.column_count != rhs.clean_stash.column_count ||
        lhs.clean_stash.row_bytes != rhs.clean_stash.row_bytes ||
        lhs.clean_stash.payload_id != rhs.clean_stash.payload_id ||
        lhs.shards.size() != rhs.shards.size() ||
        lhs.clean_stash.shards.size() != rhs.clean_stash.shards.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.shards.size(); ++i) {
        if (!capture_shard_schema_equal(lhs.shards[i], rhs.shards[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < lhs.clean_stash.shards.size(); ++i) {
        if (!capture_shard_schema_equal(
                lhs.clean_stash.shards[i],
                rhs.clean_stash.shards[i])) {
            return false;
        }
    }
    return true;
}

bool capture_cursor_after(
        const capture_projection_cursor & lhs,
        const capture_projection_cursor & rhs) {
    const auto & lhs_cell = lhs.source->placement->cells[lhs.cell];
    const auto & rhs_cell = rhs.source->placement->cells[rhs.cell];
    return std::tie(lhs_cell.physical_cell, lhs.source->manifest_id) >
           std::tie(rhs_cell.physical_cell, rhs.source->manifest_id);
}

} // namespace

bool vbr_artifact_controller_policy_equal(
        const vbr_artifact_controller_policy & lhs,
        const vbr_artifact_controller_policy & rhs) noexcept {
    return capture_policy_equal_impl(lhs, rhs);
}

bool vbr_capture_controller_representation_equal(
        const vbr_capture_controller_target & lhs,
        const vbr_capture_controller_target & rhs) noexcept {
    if (lhs.child_id != rhs.child_id ||
        lhs.lineage_uuid != rhs.lineage_uuid ||
        lhs.controller_generation != rhs.controller_generation ||
        !vbr_artifact_controller_policy_equal(lhs.policy, rhs.policy) ||
        lhs.units.size() != rhs.units.size() ||
        lhs.unit_descriptors.size() != rhs.unit_descriptors.size() ||
        lhs.units.size() != lhs.unit_descriptors.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.units.size(); ++i) {
        if (!capture_generation_equal(lhs.units[i], rhs.units[i]) ||
            !capture_descriptor_schema_equal(
                lhs.unit_descriptors[i], rhs.unit_descriptors[i])) {
            return false;
        }
    }
    return true;
}

vbr_capture_projection::vbr_capture_projection(
        std::shared_ptr<const vbr_capture_projection_plan> plan) noexcept
    : plan_(std::move(plan)) {}

const vbr_capture_projection_plan *
vbr_capture_projection::operator->() const noexcept {
    return plan_.get();
}

const vbr_capture_projection_plan &
vbr_capture_projection::operator*() const noexcept {
    return *plan_;
}

vbr_capture_projection::operator bool() const noexcept {
    return bool(plan_);
}

bool vbr_capture_projection::operator==(
        const vbr_capture_projection & other) const noexcept {
    return plan_ == other.plan_;
}

bool vbr_artifact_project_capture_union(
        const vbr_capture_projection_batch & batch,
        const vbr_capture_projection_limits & limits,
        vbr_capture_projection & output) noexcept {
    output = {};
    try {
        const auto & manifests = batch.manifests;
        if (batch.source_namespace == 0 || manifests.empty() ||
            manifests.size() > limits.max_manifests ||
            limits.max_manifests == 0 || limits.max_placements == 0 ||
            limits.max_input_cells == 0 || limits.max_union_cells == 0 ||
            limits.max_segments == 0 ||
            limits.max_dependency_references == 0 ||
            limits.max_token_ids == 0 || limits.max_string_bytes == 0 ||
            limits.max_generation_controllers == 0 ||
            limits.max_generation_units == 0 ||
            limits.max_generation_streams == 0 ||
            limits.max_generation_pages == 0 ||
            limits.max_companions == 0 ||
            limits.max_companion_payload_bytes == 0 ||
            limits.max_semantic_metadata_bytes == 0) {
            return false;
        }

        uint64_t placement_count = 0;
        uint64_t input_cells = 0;
        uint64_t token_ids = 0;
        uint64_t string_bytes = 0;
        uint64_t generation_controllers = 0;
        uint64_t generation_units = 0;
        uint64_t generation_streams = 0;
        uint64_t generation_pages = 0;
        uint64_t companions = 0;
        uint64_t companion_payload_bytes = 0;
        uint64_t semantic_metadata_bytes = 0;
        const auto add_semantic = [&](uint64_t count, uint64_t element) {
            return element == 0 || count <= UINT64_MAX/element
                ? capture_checked_add(
                      semantic_metadata_bytes, count*element,
                      semantic_metadata_bytes) &&
                      semantic_metadata_bytes <=
                          limits.max_semantic_metadata_bytes
                : false;
        };
        std::vector<uint64_t> manifest_ids;
        manifest_ids.reserve(manifests.size());
        std::vector<capture_projection_placement> placements;
        for (const auto & manifest : manifests) {
            if (manifest.manifest_id == 0 ||
                (manifest.dependencies_available !=
                    !manifest.placements.empty()) ||
                !capture_checked_add(
                    placement_count, manifest.placements.size(),
                    placement_count) ||
                placement_count > limits.max_placements) {
                return false;
            }
            uint64_t manifest_strings = 0;
            if (!capture_checked_add(
                    manifest_strings,
                    manifest.identity.execution_identity.size(),
                    manifest_strings) ||
                !capture_checked_add(
                    manifest_strings,
                    manifest.identity.adapter_config_identity.size(),
                    manifest_strings) ||
                !capture_checked_add(
                    manifest_strings,
                    manifest.identity.media_content_identity.size(),
                    manifest_strings)) {
                return false;
            }
            if (!capture_checked_add(
                    token_ids, manifest.token_block.tokens.size(), token_ids) ||
                token_ids > limits.max_token_ids ||
                !capture_checked_add(
                    string_bytes, manifest_strings, string_bytes) ||
                string_bytes > limits.max_string_bytes ||
                !capture_checked_add(
                    generation_controllers,
                    manifest.generation.controllers.size(),
                    generation_controllers) ||
                generation_controllers >
                    limits.max_generation_controllers ||
                !capture_checked_add(
                    companions, manifest.companions.size(), companions) ||
                companions > limits.max_companions ||
                !add_semantic(
                    manifest.token_block.tokens.size(),
                    sizeof(llama_token)) ||
                !add_semantic(manifest_strings, 1) ||
                !add_semantic(
                    manifest.generation.controllers.size(),
                    sizeof(vbr_checkpoint_generation_controller)) ||
                !add_semantic(
                    manifest.companions.size(),
                    sizeof(vbr_artifact_companion_payload)) ||
                !add_semantic(
                    manifest.placements.size(),
                    sizeof(vbr_artifact_stream_placement))) {
                return false;
            }
            for (const auto & companion : manifest.companions) {
                if (!capture_checked_add(
                        companion_payload_bytes, companion.payload_bytes,
                        companion_payload_bytes) ||
                    companion_payload_bytes >
                        limits.max_companion_payload_bytes) {
                    return false;
                }
            }
            for (const auto & controller :
                 manifest.generation.controllers) {
                if (!capture_checked_add(
                        generation_units, controller.units.size(),
                        generation_units) ||
                    generation_units > limits.max_generation_units ||
                    !capture_checked_add(
                        generation_streams, controller.streams.size(),
                        generation_streams) ||
                    generation_streams > limits.max_generation_streams ||
                    !add_semantic(
                        controller.units.size(),
                        sizeof(vbr_checkpoint_unit_generation)) ||
                    !add_semantic(
                        controller.streams.size(),
                        sizeof(vbr_checkpoint_generation_stream))) {
                    return false;
                }
                for (const auto & stream : controller.streams) {
                    if (!capture_checked_add(
                            generation_pages, stream.pages.size(),
                            generation_pages) ||
                        generation_pages > limits.max_generation_pages ||
                        !add_semantic(
                            stream.pages.size(),
                            sizeof(vbr_generation_page_ref))) {
                        return false;
                    }
                }
            }
            manifest_ids.push_back(manifest.manifest_id);
            std::vector<std::pair<llama_seq_id, llama_pos>> logical_positions;
            for (const auto & placement : manifest.placements) {
                if (placement.child_id == UINT32_MAX ||
                    placement.stream_index == UINT32_MAX ||
                    placement.source_sequence < 0 ||
                    placement.computation_frontier <= 0 ||
                    placement.cells.empty() ||
                    !capture_checked_add(
                        input_cells, placement.cells.size(), input_cells) ||
                    input_cells > limits.max_input_cells) {
                    return false;
                }
                placements.push_back({ manifest.manifest_id, &placement });
                if (!add_semantic(
                        placement.cells.size(),
                        sizeof(vbr_artifact_cell_placement))) {
                    return false;
                }
                for (size_t i = 0; i < placement.cells.size(); ++i) {
                    const auto & cell = placement.cells[i];
                    if (cell.physical_cell == UINT32_MAX ||
                        cell.logical_position < 0 ||
                        cell.logical_position >=
                            placement.computation_frontier ||
                        (i != 0 &&
                         placement.cells[i - 1].physical_cell >=
                             cell.physical_cell)) {
                        return false;
                    }
                    logical_positions.push_back({
                        placement.source_sequence,
                        cell.logical_position,
                    });
                }
            }
            std::sort(logical_positions.begin(), logical_positions.end());
            if (std::adjacent_find(
                    logical_positions.begin(), logical_positions.end()) !=
                    logical_positions.end()) {
                return false;
            }
        }
        std::sort(manifest_ids.begin(), manifest_ids.end());
        if (std::adjacent_find(manifest_ids.begin(), manifest_ids.end()) !=
                manifest_ids.end()) {
            return false;
        }
        std::sort(placements.begin(), placements.end(),
            [](const auto & lhs, const auto & rhs) {
                return std::tie(lhs.placement->child_id,
                                lhs.placement->stream_index,
                                lhs.manifest_id) <
                       std::tie(rhs.placement->child_id,
                                rhs.placement->stream_index,
                                rhs.manifest_id);
            });
        if (std::adjacent_find(
                placements.begin(), placements.end(),
                [](const auto & lhs, const auto & rhs) {
                    return lhs.manifest_id == rhs.manifest_id &&
                           lhs.placement->child_id ==
                               rhs.placement->child_id &&
                           lhs.placement->stream_index ==
                               rhs.placement->stream_index;
                }) != placements.end()) {
            return false;
        }

        vbr_capture_projection_plan plan;
        plan.source_namespace = batch.source_namespace;
        plan.manifest_count = uint32_t(manifests.size());
        plan.placement_count = uint32_t(placement_count);
        plan.input_cell_references = input_cells;
        plan.manifests = batch.manifests;
        std::sort(plan.manifests.begin(), plan.manifests.end(),
            [](const auto & lhs, const auto & rhs) {
                return lhs.manifest_id < rhs.manifest_id;
            });
        uint64_t segment_count = 0;
        std::vector<capture_projection_cursor> heap;
        std::vector<uint64_t> dependencies;
        for (size_t group_begin = 0; group_begin < placements.size();) {
            size_t group_end = group_begin + 1;
            while (group_end < placements.size() &&
                   placements[group_end].placement->child_id ==
                       placements[group_begin].placement->child_id &&
                   placements[group_end].placement->stream_index ==
                       placements[group_begin].placement->stream_index) {
                ++group_end;
            }
            plan.streams.push_back({
                placements[group_begin].placement->child_id,
                placements[group_begin].placement->stream_index,
                {},
            });
            auto & stream = plan.streams.back();
            heap.clear();
            dependencies.clear();
            heap.reserve(group_end - group_begin);
            dependencies.reserve(group_end - group_begin);
            for (size_t i = group_begin; i < group_end; ++i) {
                heap.push_back({ &placements[i], 0 });
            }
            std::make_heap(heap.begin(), heap.end(), capture_cursor_after);

            while (!heap.empty()) {
                const uint32_t cell =
                    heap.front().source->placement->cells[
                        heap.front().cell].physical_cell;
                dependencies.clear();
                while (!heap.empty() &&
                       heap.front().source->placement->cells[
                           heap.front().cell].physical_cell == cell) {
                    std::pop_heap(
                        heap.begin(), heap.end(), capture_cursor_after);
                    auto cursor = heap.back();
                    heap.pop_back();
                    if (!dependencies.empty() &&
                        dependencies.back() == cursor.source->manifest_id) {
                        return false;
                    }
                    dependencies.push_back(cursor.source->manifest_id);
                    ++cursor.cell;
                    if (cursor.cell <
                            cursor.source->placement->cells.size()) {
                        heap.push_back(cursor);
                        std::push_heap(
                            heap.begin(), heap.end(), capture_cursor_after);
                    }
                }

                if (plan.union_cell_count == limits.max_union_cells) {
                    return false;
                }
                const auto dependencies_equal = [&]() {
                    if (stream.segments.empty()) {
                        return false;
                    }
                    const auto & prior = stream.segments.back();
                    return prior.dependency_count == dependencies.size() &&
                        std::equal(
                            dependencies.begin(), dependencies.end(),
                            plan.dependent_manifest_ids.begin() +
                                prior.first_dependency);
                };
                const bool extend = !stream.segments.empty() &&
                    uint64_t(stream.segments.back().first_physical_cell) +
                        stream.segments.back().cell_count == cell &&
                    dependencies_equal();
                if (extend) {
                    if (stream.segments.back().cell_count == UINT32_MAX) {
                        return false;
                    }
                    ++stream.segments.back().cell_count;
                } else {
                    if (segment_count == limits.max_segments ||
                        plan.dependency_references >
                            limits.max_dependency_references ||
                        dependencies.size() >
                            limits.max_dependency_references -
                                plan.dependency_references ||
                        plan.dependent_manifest_ids.size() > UINT32_MAX ||
                        dependencies.size() > UINT32_MAX) {
                        return false;
                    }
                    const uint32_t first_dependency = uint32_t(
                        plan.dependent_manifest_ids.size());
                    plan.dependent_manifest_ids.insert(
                        plan.dependent_manifest_ids.end(),
                        dependencies.begin(), dependencies.end());
                    ++segment_count;
                    plan.dependency_references += dependencies.size();
                    stream.segments.push_back({
                        cell, 1, first_dependency,
                        uint32_t(dependencies.size()),
                    });
                }
                ++plan.union_cell_count;
            }
            uint64_t packed_first_row = 0;
            for (auto & segment : stream.segments) {
                segment.packed_first_row = packed_first_row;
                packed_first_row += segment.cell_count;
            }
            group_begin = group_end;
        }
        if (plan.streams.empty() && std::any_of(
                plan.manifests.begin(), plan.manifests.end(),
                [](const auto & manifest) {
                    return manifest.dependencies_available;
                })) {
            return false;
        }
        output = vbr_capture_projection(
            std::make_shared<const vbr_capture_projection_plan>(
                std::move(plan)));
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

const char * vbr_capture_stream_status_name(
        vbr_capture_stream_status status) noexcept {
    switch (status) {
        case vbr_capture_stream_status::ok:                  return "ok";
        case vbr_capture_stream_status::invalid_argument:    return "invalid_argument";
        case vbr_capture_stream_status::ring_unavailable:    return "ring_unavailable";
        case vbr_capture_stream_status::cancelled:           return "cancelled";
        case vbr_capture_stream_status::transfer_failed:     return "transfer_failed";
        case vbr_capture_stream_status::short_read:          return "short_read";
        case vbr_capture_stream_status::duplicate_segment:   return "duplicate_segment";
        case vbr_capture_stream_status::missing_segment:     return "missing_segment";
        case vbr_capture_stream_status::late_segment:        return "late_segment";
        case vbr_capture_stream_status::hash_mismatch:       return "hash_mismatch";
        case vbr_capture_stream_status::format_rejected:      return "format_rejected";
        case vbr_capture_stream_status::accounting_unavailable: return "accounting_unavailable";
        case vbr_capture_stream_status::accounting_refused:  return "accounting_refused";
        case vbr_capture_stream_status::stage_failed:         return "stage_failed";
        case vbr_capture_stream_status::commit_failed:        return "commit_failed";
        case vbr_capture_stream_status::publication_failed:  return "publication_failed";
        case vbr_capture_stream_status::projection_invalid:   return "projection_invalid";
        case vbr_capture_stream_status::snapshot_unavailable: return "snapshot_unavailable";
        case vbr_capture_stream_status::snapshot_changed:     return "snapshot_changed";
        case vbr_capture_stream_status::internal_error:      return "internal_error";
        case vbr_capture_stream_status::_count:              break;
    }
    return "invalid";
}

const char * vbr_capture_reservation_group_name(
        vbr_capture_reservation_group group) noexcept {
    switch (group) {
        case vbr_capture_reservation_group::none: return "none";
        case vbr_capture_reservation_group::transfer_staging: return "transfer_staging";
        case vbr_capture_reservation_group::durable_artifact: return "durable_artifact";
        case vbr_capture_reservation_group::_count: break;
    }
    return "invalid";
}

const char * vbr_capture_ring_create_failure_name(
        vbr_capture_ring_create_failure failure) noexcept {
    switch (failure) {
        case vbr_capture_ring_create_failure::none:
            return "none";
        case vbr_capture_ring_create_failure::invalid_geometry:
            return "invalid_geometry";
        case vbr_capture_ring_create_failure::invalid_accounting_binding:
            return "invalid_accounting_binding";
        case vbr_capture_ring_create_failure::existing_ring_charge:
            return "existing_ring_charge";
        case vbr_capture_ring_create_failure::accounting_update_failed:
            return "accounting_update_failed";
        case vbr_capture_ring_create_failure::budget_reset_failed:
            return "budget_reset_failed";
        case vbr_capture_ring_create_failure::budget_unavailable:
            return "budget_unavailable";
        case vbr_capture_ring_create_failure::budget_exceeded:
            return "budget_exceeded";
        case vbr_capture_ring_create_failure::global_capacity_exceeded:
            return "global_capacity_exceeded";
        case vbr_capture_ring_create_failure::invalid_lane_binding:
            return "invalid_lane_binding";
        case vbr_capture_ring_create_failure::duplicate_device_lane:
            return "duplicate_device_lane";
        case vbr_capture_ring_create_failure::host_buffer_type_unavailable:
            return "host_buffer_type_unavailable";
        case vbr_capture_ring_create_failure::host_buffer_allocation_failed:
            return "host_buffer_allocation_failed";
        case vbr_capture_ring_create_failure::host_buffer_too_small:
            return "host_buffer_too_small";
        case vbr_capture_ring_create_failure::host_buffer_base_unavailable:
            return "host_buffer_base_unavailable";
        case vbr_capture_ring_create_failure::lane_underprovisioned:
            return "lane_underprovisioned";
        case vbr_capture_ring_create_failure::accounting_charge_failed:
            return "accounting_charge_failed";
        case vbr_capture_ring_create_failure::internal_error:
            return "internal_error";
        case vbr_capture_ring_create_failure::_count:
            break;
    }
    return "invalid";
}

namespace {

std::array<uint8_t, 32> capture_range_leaf_digest(
    uint64_t index,
    uint32_t size,
    const std::array<uint8_t, 32> & payload) noexcept;

constexpr char CAPTURE_STREAM_DIGEST_DOMAIN[] =
    "buun.vbr.capture.segment-stream";

void capture_stream_digest_begin(
        llama_sha256_writer & hash, uint64_t expected_bytes) {
    hash.string(
        CAPTURE_STREAM_DIGEST_DOMAIN,
        sizeof(CAPTURE_STREAM_DIGEST_DOMAIN) - 1);
    hash.u64(expected_bytes);
}

} // namespace

struct artifact_segment_chain::impl {
    std::vector<artifact_segment> segments;
    std::vector<uint64_t> segment_ends;
    uint64_t total = 0;
    size_t max_segment = 0;
    uint32_t authenticated_chunk_bytes = 0;
    uint32_t max_authenticated_chunks = 0;
    uint32_t authenticated_current_bytes = 0;
    bool authenticated_closed = false;
    llama_sha256_writer authenticated_current_hash;
    std::vector<std::array<uint8_t, 32>> authenticated_leaves;
    bool stream_digest_enabled = false;
    bool stream_digest_complete = false;
    uint64_t stream_digest_expected = 0;
    llama_sha256_writer stream_digest_hash;
    std::array<uint8_t, 32> stream_digest = {};
};

artifact_segment_chain::artifact_segment_chain()
    : impl_(new impl) {}
artifact_segment_chain::artifact_segment_chain(uint64_t expected_stream_bytes)
    : impl_(new impl) {
    impl_->stream_digest_enabled = true;
    impl_->stream_digest_expected = expected_stream_bytes;
    capture_stream_digest_begin(
        impl_->stream_digest_hash, expected_stream_bytes);
    if (expected_stream_bytes == 0) {
        impl_->stream_digest = impl_->stream_digest_hash.finish();
        impl_->stream_digest_complete = true;
    }
}
artifact_segment_chain::artifact_segment_chain(
        uint32_t authenticated_chunk_bytes,
        uint32_t max_authenticated_chunks)
    : impl_(new impl) {
    if (authenticated_chunk_bytes == VBR_CAPTURE_RANGE_CHUNK_BYTES &&
        max_authenticated_chunks != 0) {
        impl_->authenticated_chunk_bytes = authenticated_chunk_bytes;
        impl_->max_authenticated_chunks = max_authenticated_chunks;
    }
}
artifact_segment_chain::~artifact_segment_chain() = default;
artifact_segment_chain::artifact_segment_chain(
        artifact_segment_chain &&) noexcept = default;
artifact_segment_chain & artifact_segment_chain::operator=(
        artifact_segment_chain &&) noexcept = default;

bool artifact_segment_chain::append(
        const uint8_t * data, size_t size) noexcept {
    if ((!data && size != 0) || impl_->authenticated_closed ||
        size > std::numeric_limits<uint64_t>::max() - impl_->total ||
        (impl_->stream_digest_enabled &&
         size > impl_->stream_digest_expected - impl_->total)) {
        return false;
    }
    try {
        std::vector<uint8_t> bytes;
        if (size != 0) {
            bytes.assign(data, data + size);
        }
        return append_owned(std::move(bytes));
    } catch (...) {
        return false;
    }
}

bool artifact_segment_chain::append_owned(
        std::vector<uint8_t> data) noexcept {
    try {
        return append_storage(
            std::make_shared<std::vector<uint8_t>>(std::move(data)));
    } catch (...) {
        return false;
    }
}

bool artifact_segment_chain::append_storage(
        std::shared_ptr<std::vector<uint8_t>> bytes) noexcept {
    try {
        if (!bytes || impl_->authenticated_closed ||
            bytes->size() > std::numeric_limits<uint64_t>::max() -
                impl_->total ||
            (impl_->stream_digest_enabled &&
             bytes->size() >
                impl_->stream_digest_expected - impl_->total)) {
            return false;
        }
        const size_t size = bytes->size();
        const uint8_t * data = bytes->data();
        if (impl_->authenticated_chunk_bytes != 0) {
            const uint64_t new_total = impl_->total + size;
            const uint64_t chunks = new_total == 0 ? 0 :
                (new_total - 1)/impl_->authenticated_chunk_bytes + 1;
            if (chunks > impl_->max_authenticated_chunks) {
                return false;
            }
            const uint64_t completed =
                new_total/impl_->authenticated_chunk_bytes;
            if (completed > impl_->authenticated_leaves.capacity()) {
                const uint64_t doubled =
                    impl_->authenticated_leaves.capacity() == 0 ? 1 :
                    std::min<uint64_t>(
                        impl_->max_authenticated_chunks,
                        uint64_t(impl_->authenticated_leaves.capacity())*2);
                impl_->authenticated_leaves.reserve(size_t(std::max(
                    completed, doubled)));
            }
        }
        if (impl_->segments.size() == impl_->segments.capacity()) {
            const size_t next = impl_->segments.empty() ? 1 :
                impl_->segments.size() <= SIZE_MAX/2 ?
                    impl_->segments.size()*2 : SIZE_MAX;
            if (next == SIZE_MAX) {
                return false;
            }
            impl_->segments.reserve(next);
        }
        if (impl_->segment_ends.size() == impl_->segment_ends.capacity()) {
            const size_t next = impl_->segment_ends.empty() ? 1 :
                impl_->segment_ends.size() <= SIZE_MAX/2 ?
                    impl_->segment_ends.size()*2 : SIZE_MAX;
            if (next == SIZE_MAX) {
                return false;
            }
            impl_->segment_ends.reserve(next);
        }
        impl_->segments.push_back({
            std::move(bytes), 0, uint64_t(size),
        });
        impl_->total += size;
        impl_->segment_ends.push_back(impl_->total);
        impl_->max_segment =
            std::max(impl_->max_segment, size);
        if (impl_->authenticated_chunk_bytes != 0) {
            size_t consumed = 0;
            while (consumed < size) {
                const size_t take = std::min<size_t>(
                    size - consumed,
                    impl_->authenticated_chunk_bytes -
                        impl_->authenticated_current_bytes);
                impl_->authenticated_current_hash.bytes(
                    data + consumed, take);
                impl_->authenticated_current_bytes += uint32_t(take);
                consumed += take;
                if (impl_->authenticated_current_bytes ==
                        impl_->authenticated_chunk_bytes) {
                    const auto payload =
                        impl_->authenticated_current_hash.finish();
                    impl_->authenticated_leaves.push_back(
                        capture_range_leaf_digest(
                            impl_->authenticated_leaves.size(),
                            impl_->authenticated_current_bytes,
                            payload));
                    impl_->authenticated_current_hash = {};
                    impl_->authenticated_current_bytes = 0;
                }
            }
        }
        if (impl_->stream_digest_enabled &&
            !impl_->stream_digest_complete) {
            impl_->stream_digest_hash.bytes(data, size);
            if (impl_->total == impl_->stream_digest_expected) {
                impl_->stream_digest =
                    impl_->stream_digest_hash.finish();
                impl_->stream_digest_complete = true;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

uint64_t artifact_segment_chain::size() const noexcept {
    return impl_->total;
}

size_t artifact_segment_chain::segment_count() const noexcept {
    return impl_->segments.size();
}

size_t artifact_segment_chain::max_segment_size() const noexcept {
    return impl_->max_segment;
}

bool artifact_segment_chain::authenticated() const noexcept {
    return impl_->authenticated_chunk_bytes != 0;
}

bool artifact_segment_chain::read(
        uint64_t offset, uint8_t * destination,
        size_t size) const noexcept {
    if ((!destination && size != 0) ||
        offset > impl_->total ||
        size > impl_->total - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    const auto first = std::upper_bound(
        impl_->segment_ends.begin(), impl_->segment_ends.end(), offset);
    size_t segment_index = size_t(first - impl_->segment_ends.begin());
    uint64_t cursor = segment_index == 0 ? 0 :
        impl_->segment_ends[segment_index - 1];
    size_t remaining = size;
    for (; segment_index < impl_->segments.size(); ++segment_index) {
        const auto & segment = impl_->segments[segment_index];
        const uint64_t end = impl_->segment_ends[segment_index];
        const uint64_t within = offset > cursor
            ? offset - cursor : 0;
        const size_t available =
            size_t(segment.length - within);
        const size_t take = std::min(available, remaining);
        if (take != 0) {
            if (!segment.storage ||
                segment.offset + within >
                    segment.storage->size() ||
                take > segment.storage->size() -
                    size_t(segment.offset + within)) {
                return false;
            }
            std::memcpy(
                destination + (size - remaining),
                segment.storage->data() +
                    size_t(segment.offset + within),
                take);
            remaining -= take;
            offset += take;
            if (remaining == 0) {
                return true;
            }
        }
        cursor = end;
    }
    return remaining == 0;
}

namespace {

bool segment_source_read(
        const void * context, uint64_t offset,
        uint8_t * destination, size_t size) noexcept {
    const auto * chain =
        static_cast<const artifact_segment_chain *>(context);
    return chain &&
           chain->read(offset, destination, size);
}

} // namespace

vbr_artifact_byte_source artifact_segment_chain::source() const noexcept {
    return { size(), this, segment_source_read };
}

std::array<uint8_t, 32> vbr_capture_stream_digest(
        const artifact_segment_chain & chain) noexcept {
    if (chain.impl_->stream_digest_enabled) {
        return chain.impl_->stream_digest_complete &&
               chain.impl_->total == chain.impl_->stream_digest_expected
            ? chain.impl_->stream_digest
            : std::array<uint8_t, 32> {};
    }
    llama_sha256_writer hash;
    capture_stream_digest_begin(hash, chain.size());
    std::array<uint8_t, 64*1024> scratch;
    for (uint64_t offset = 0; offset < chain.size();) {
        const size_t count = size_t(std::min<uint64_t>(
            scratch.size(), chain.size() - offset));
        if (!chain.read(offset, scratch.data(), count)) {
            return {};
        }
        hash.bytes(scratch.data(), count);
        offset += count;
    }
    return hash.finish();
}

namespace {

using capture_range_digest = std::array<uint8_t, 32>;

// Conservative charge for the shared-owner control block, the tree data
// object, and its small vector headers. Digest arenas are charged separately.
static constexpr uint64_t CAPTURE_RANGE_TREE_OWNER_BYTES = 256;
static constexpr uint64_t CAPTURE_SHARED_OWNER_BYTES = 64;

capture_range_digest capture_range_leaf_digest(
        uint64_t index, uint32_t size,
        const capture_range_digest & payload) noexcept {
    llama_sha256_writer hash;
    static constexpr char digest_domain[] =
        "buun.vbr.capture/range-leaf/v1";
    hash.string(digest_domain, sizeof(digest_domain) - 1);
    hash.u64(index);
    hash.u32(size);
    hash.bytes(payload.data(), payload.size());
    return hash.finish();
}

capture_range_digest capture_range_node_digest(
        uint32_t level, const capture_range_digest & left,
        const capture_range_digest & right) noexcept {
    llama_sha256_writer hash;
    static constexpr char digest_domain[] =
        "buun.vbr.capture/range-node/v1";
    hash.string(digest_domain, sizeof(digest_domain) - 1);
    hash.u32(level);
    hash.bytes(left.data(), left.size());
    hash.bytes(right.data(), right.size());
    return hash.finish();
}

capture_range_digest capture_range_root_digest(
        uint64_t total_bytes, uint32_t chunk_bytes,
        uint32_t chunk_count, uint32_t padded_count,
        const capture_range_digest & tree_root) noexcept {
    llama_sha256_writer hash;
    static constexpr char digest_domain[] =
        "buun.vbr.capture/range-root/v1";
    hash.string(digest_domain, sizeof(digest_domain) - 1);
    hash.u64(total_bytes);
    hash.u32(chunk_bytes);
    hash.u32(chunk_count);
    hash.u32(padded_count);
    hash.bytes(tree_root.data(), tree_root.size());
    return hash.finish();
}

bool capture_range_metadata_add(
        uint64_t count, uint64_t item_bytes,
        uint64_t & total) noexcept {
    return item_bytes == 0 ||
        (count <= (UINT64_MAX - total)/item_bytes &&
         (total += count*item_bytes, true));
}

bool capture_range_tree_metadata_bytes(
        uint64_t chunk_count, uint64_t & metadata_bytes,
        uint32_t * padded_count = nullptr) noexcept {
    metadata_bytes = CAPTURE_RANGE_TREE_OWNER_BYTES;
    if (chunk_count == 0 || chunk_count > UINT32_MAX) {
        return false;
    }
    uint32_t padded = 1;
    while (padded < chunk_count) {
        if (padded > UINT32_MAX/2) {
            return false;
        }
        padded *= 2;
    }
    uint64_t node_count = 0;
    for (uint64_t count = padded;; count /= 2) {
        if (!capture_checked_add(node_count, count, node_count)) {
            return false;
        }
        if (count == 1) {
            break;
        }
    }
    if (!capture_range_metadata_add(
            node_count, sizeof(capture_range_digest),
            metadata_bytes)) {
        return false;
    }
    if (padded_count) {
        *padded_count = padded;
    }
    return true;
}

} // namespace

struct vbr_capture_range_tree::data {
    uint64_t total_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint32_t chunk_bytes = 0;
    uint32_t chunk_count = 0;
    uint32_t padded_count = 0;
    capture_range_digest root = {};
    std::vector<std::vector<capture_range_digest>> levels;
};

struct capture_range_proof_node {
    uint32_t level = 0;
    uint32_t index = 0;
    capture_range_digest digest = {};
};

struct vbr_capture_range_proof::data {
    uint64_t total_bytes = 0;
    uint64_t metadata_bytes = 0;
    uint32_t chunk_bytes = 0;
    uint32_t chunk_count = 0;
    uint32_t padded_count = 0;
    capture_range_digest root = {};
    std::vector<vbr_capture_authenticated_range> ranges;
    std::vector<uint32_t> selected_chunks;
    // Tree-authenticated digests for exactly selected_chunks. Retaining these
    // manifest-local leaves permits a narrower proof to be derived without
    // rereading omitted payload or retaining the broader range tree.
    std::vector<capture_range_digest> selected_leaf_digests;
    std::vector<capture_range_proof_node> proof_nodes;
};

vbr_capture_range_tree::vbr_capture_range_tree(
        std::shared_ptr<const data> data) noexcept
    : data_(std::move(data)) {}

vbr_capture_range_tree::operator bool() const noexcept {
    return bool(data_);
}

uint64_t vbr_capture_range_tree::total_bytes() const noexcept {
    return data_ ? data_->total_bytes : 0;
}

uint32_t vbr_capture_range_tree::chunk_bytes() const noexcept {
    return data_ ? data_->chunk_bytes : 0;
}

uint32_t vbr_capture_range_tree::chunk_count() const noexcept {
    return data_ ? data_->chunk_count : 0;
}

uint64_t vbr_capture_range_tree::metadata_bytes() const noexcept {
    return data_ ? data_->metadata_bytes : 0;
}

const capture_range_digest & vbr_capture_range_tree::root() const noexcept {
    static const capture_range_digest empty = {};
    return data_ ? data_->root : empty;
}

vbr_capture_range_proof::vbr_capture_range_proof(
        std::shared_ptr<const data> data) noexcept
    : data_(std::move(data)) {}

vbr_capture_range_proof::operator bool() const noexcept {
    return bool(data_);
}

const capture_range_digest & vbr_capture_range_proof::root() const noexcept {
    static const capture_range_digest empty = {};
    return data_ ? data_->root : empty;
}

uint64_t vbr_capture_range_proof::total_bytes() const noexcept {
    return data_ ? data_->total_bytes : 0;
}

uint32_t vbr_capture_range_proof::chunk_bytes() const noexcept {
    return data_ ? data_->chunk_bytes : 0;
}

const std::vector<vbr_capture_authenticated_range> &
vbr_capture_range_proof::ranges() const noexcept {
    static const std::vector<vbr_capture_authenticated_range> empty;
    return data_ ? data_->ranges : empty;
}

uint32_t vbr_capture_range_proof::selected_chunk_count() const noexcept {
    return data_ ? uint32_t(data_->selected_chunks.size()) : 0;
}

uint32_t vbr_capture_range_proof::proof_node_count() const noexcept {
    return data_ ? uint32_t(data_->proof_nodes.size()) : 0;
}

uint64_t vbr_capture_range_proof::metadata_bytes() const noexcept {
    return data_ ? data_->metadata_bytes : 0;
}

bool vbr_capture_range_seal(
        artifact_segment_chain & chain,
        uint64_t max_metadata_bytes,
        vbr_capture_range_tree & output) noexcept {
    output = {};
    auto & state = *chain.impl_;
    if (state.authenticated_chunk_bytes !=
            VBR_CAPTURE_RANGE_CHUNK_BYTES ||
        state.max_authenticated_chunks == 0 ||
        state.authenticated_closed || state.total == 0 ||
        max_metadata_bytes == 0) {
        return false;
    }
    state.authenticated_closed = true;
    try {
        if (state.authenticated_current_bytes != 0) {
            if (state.authenticated_leaves.size() >=
                    state.max_authenticated_chunks) {
                return false;
            }
            state.authenticated_leaves.reserve(
                state.authenticated_leaves.size() + 1);
            const auto payload =
                state.authenticated_current_hash.finish();
            state.authenticated_leaves.push_back(
                capture_range_leaf_digest(
                    state.authenticated_leaves.size(),
                    state.authenticated_current_bytes, payload));
            state.authenticated_current_bytes = 0;
        }
        if (state.authenticated_leaves.empty() ||
            state.authenticated_leaves.size() > UINT32_MAX) {
            return false;
        }
        uint64_t metadata_bytes = 0;
        uint32_t padded = 0;
        if (!capture_range_tree_metadata_bytes(
                state.authenticated_leaves.size(), metadata_bytes,
                &padded) ||
            metadata_bytes > max_metadata_bytes) {
            return false;
        }
        auto result = std::make_shared<vbr_capture_range_tree::data>();
        result->total_bytes = state.total;
        result->metadata_bytes = metadata_bytes;
        result->chunk_bytes = state.authenticated_chunk_bytes;
        result->chunk_count = uint32_t(state.authenticated_leaves.size());
        result->padded_count = padded;
        result->levels.push_back(
            std::move(state.authenticated_leaves));
        auto & leaves = result->levels.back();
        leaves.reserve(padded);
        llama_sha256_writer empty_payload_hash;
        const auto empty_payload = empty_payload_hash.finish();
        while (leaves.size() < padded) {
            leaves.push_back(capture_range_leaf_digest(
                leaves.size(), 0, empty_payload));
        }
        uint32_t level = 0;
        while (result->levels.back().size() > 1) {
            const auto & prior = result->levels.back();
            std::vector<capture_range_digest> next;
            next.reserve(prior.size()/2);
            for (size_t i = 0; i < prior.size(); i += 2) {
                next.push_back(capture_range_node_digest(
                    level, prior[i], prior[i + 1]));
            }
            result->levels.push_back(std::move(next));
            ++level;
        }
        result->root = capture_range_root_digest(
            result->total_bytes, result->chunk_bytes,
            result->chunk_count, result->padded_count,
            result->levels.back().front());
        if (!capture_digest_nonzero(result->root)) {
            return false;
        }
        output = vbr_capture_range_tree(std::move(result));
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

bool vbr_capture_range_prove(
        const vbr_capture_range_tree & tree,
        const std::vector<vbr_capture_authenticated_range> & ranges,
        const vbr_capture_range_proof_limits & limits,
        vbr_capture_range_proof & output) noexcept {
    output = {};
    try {
        if (!tree.data_ || ranges.empty() || limits.max_ranges == 0 ||
            ranges.size() > limits.max_ranges ||
            limits.max_selected_chunks == 0 ||
            limits.max_proof_nodes == 0 ||
            limits.max_metadata_bytes == 0) {
            return false;
        }
        auto result = std::make_shared<vbr_capture_range_proof::data>();
        result->total_bytes = tree.data_->total_bytes;
        result->chunk_bytes = tree.data_->chunk_bytes;
        result->chunk_count = tree.data_->chunk_count;
        result->padded_count = tree.data_->padded_count;
        result->root = tree.data_->root;
        result->ranges = ranges;
        uint64_t prior_end = 0;
        for (const auto & range : ranges) {
            if (range.size == 0 || range.offset < prior_end ||
                range.offset >= result->total_bytes ||
                range.size > result->total_bytes - range.offset) {
                return false;
            }
            const uint64_t end = range.offset + range.size;
            const uint32_t first = uint32_t(
                range.offset/result->chunk_bytes);
            const uint32_t last = uint32_t(
                (end - 1)/result->chunk_bytes);
            for (uint32_t chunk = first;; ++chunk) {
                if (result->selected_chunks.empty() ||
                    result->selected_chunks.back() != chunk) {
                    if (result->selected_chunks.size() >=
                            limits.max_selected_chunks) {
                        return false;
                    }
                    result->selected_chunks.push_back(chunk);
                    result->selected_leaf_digests.push_back(
                        tree.data_->levels.front()[chunk]);
                }
                if (chunk == last) {
                    break;
                }
            }
            prior_end = end;
        }
        std::vector<uint32_t> known = result->selected_chunks;
        for (uint32_t level = 0;
             level + 1 < tree.data_->levels.size(); ++level) {
            std::vector<uint32_t> parents;
            parents.reserve((known.size() + 1)/2);
            for (const uint32_t index : known) {
                const uint32_t sibling = index ^ 1u;
                if (!std::binary_search(
                        known.begin(), known.end(), sibling)) {
                    if (result->proof_nodes.size() >=
                            limits.max_proof_nodes ||
                        sibling >= tree.data_->levels[level].size()) {
                        return false;
                    }
                    result->proof_nodes.push_back({
                        level, sibling,
                        tree.data_->levels[level][sibling],
                    });
                }
                const uint32_t parent = index/2;
                if (parents.empty() || parents.back() != parent) {
                    parents.push_back(parent);
                }
            }
            known = std::move(parents);
        }
        if (known.size() != 1 || known.front() != 0) {
            return false;
        }
        uint64_t metadata_bytes =
            sizeof(vbr_capture_range_proof::data) +
            CAPTURE_SHARED_OWNER_BYTES;
        if (!capture_range_metadata_add(
                result->ranges.capacity(),
                sizeof(vbr_capture_authenticated_range),
                metadata_bytes) ||
            !capture_range_metadata_add(
                result->selected_chunks.capacity(), sizeof(uint32_t),
                metadata_bytes) ||
            !capture_range_metadata_add(
                result->selected_leaf_digests.capacity(),
                sizeof(capture_range_digest), metadata_bytes) ||
            !capture_range_metadata_add(
                result->proof_nodes.capacity(),
                sizeof(capture_range_proof_node), metadata_bytes) ||
            metadata_bytes > limits.max_metadata_bytes) {
            return false;
        }
        result->metadata_bytes = metadata_bytes;
        output = vbr_capture_range_proof(std::move(result));
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

vbr_capture_range_restrict_status vbr_capture_range_restrict(
        const vbr_capture_range_proof & parent,
        const std::vector<vbr_capture_authenticated_range> & ranges,
        const vbr_capture_range_proof_limits & limits,
        vbr_capture_range_proof & output) noexcept {
    using status = vbr_capture_range_restrict_status;
    output = {};
    try {
        const auto * source = parent.data_.get();
        if (!source) {
            return status::parent_invalid;
        }
        if (ranges.empty()) {
            return status::invalid_argument;
        }
        if (limits.max_ranges == 0 ||
            ranges.size() > limits.max_ranges ||
            limits.max_selected_chunks == 0 ||
            limits.max_proof_nodes == 0 ||
            limits.max_metadata_bytes == 0) {
            return status::limit_exceeded;
        }
        if (source->total_bytes == 0 ||
            source->chunk_bytes != VBR_CAPTURE_RANGE_CHUNK_BYTES ||
            source->chunk_count == 0 || source->padded_count == 0 ||
            (source->padded_count & (source->padded_count - 1)) != 0 ||
            source->chunk_count > source->padded_count ||
            source->selected_chunks.empty() ||
            source->selected_chunks.size() !=
                source->selected_leaf_digests.size() ||
            source->ranges.empty() ||
            !capture_digest_nonzero(source->root)) {
            return status::parent_invalid;
        }

        // Validate the requested canonical ranges and prove that every byte is
        // covered by the parent's manifest-local authority. Adjacent parent
        // ranges may jointly cover one child range; gaps may not.
        size_t parent_range = 0;
        uint64_t prior_end = 0;
        for (const auto & range : ranges) {
            if (range.size == 0 || range.offset < prior_end ||
                range.offset >= source->total_bytes ||
                range.size > source->total_bytes - range.offset) {
                return status::invalid_argument;
            }
            const uint64_t end = range.offset + range.size;
            uint64_t cursor = range.offset;
            while (cursor < end) {
                while (parent_range < source->ranges.size()) {
                    const auto & authorized = source->ranges[parent_range];
                    if (authorized.size == 0 ||
                        authorized.offset >= source->total_bytes ||
                        authorized.size >
                            source->total_bytes - authorized.offset) {
                        return status::parent_invalid;
                    }
                    const uint64_t authorized_end =
                        authorized.offset + authorized.size;
                    if (authorized_end > cursor) {
                        break;
                    }
                    ++parent_range;
                }
                if (parent_range == source->ranges.size()) {
                    return status::range_unauthorized;
                }
                const auto & authorized = source->ranges[parent_range];
                if (authorized.offset > cursor) {
                    return status::range_unauthorized;
                }
                cursor = std::min(
                    end, authorized.offset + authorized.size);
                if (cursor < end) {
                    ++parent_range;
                }
            }
            prior_end = end;
        }

        using indexed_digest =
            std::pair<uint32_t, capture_range_digest>;
        uint32_t height = 0;
        for (uint32_t width = source->padded_count;
             width > 1; width /= 2) {
            ++height;
        }
        std::vector<std::vector<indexed_digest>> available(height + 1);
        available[0].reserve(source->selected_chunks.size());
        uint32_t prior_chunk = 0;
        bool first_chunk = true;
        for (size_t i = 0; i < source->selected_chunks.size(); ++i) {
            const uint32_t chunk = source->selected_chunks[i];
            if (chunk >= source->chunk_count ||
                (!first_chunk && chunk <= prior_chunk) ||
                !capture_digest_nonzero(source->selected_leaf_digests[i])) {
                return status::parent_invalid;
            }
            first_chunk = false;
            prior_chunk = chunk;
            available[0].push_back({
                chunk, source->selected_leaf_digests[i],
            });
        }
        for (const auto & node : source->proof_nodes) {
            if (node.level >= height ||
                node.index >= (source->padded_count >> node.level) ||
                !capture_digest_nonzero(node.digest)) {
                return status::parent_invalid;
            }
            available[node.level].push_back({ node.index, node.digest });
        }

        // Reconstruct the parent's sparse authenticated frontier once. Every
        // pair must be complete; a missing, duplicate, or conflicting node is
        // malformed authority and cannot mint a child proof.
        for (uint32_t level = 0; level < height; ++level) {
            auto & nodes = available[level];
            std::sort(nodes.begin(), nodes.end(),
                [](const indexed_digest & lhs, const indexed_digest & rhs) {
                    return lhs.first < rhs.first;
                });
            if (nodes.empty() || (nodes.size() & 1u) != 0) {
                return status::parent_invalid;
            }
            auto & parents = available[level + 1];
            parents.reserve(parents.size() + nodes.size()/2);
            for (size_t i = 0; i < nodes.size(); i += 2) {
                if ((nodes[i].first & 1u) != 0 ||
                    nodes[i + 1].first != nodes[i].first + 1) {
                    return status::parent_invalid;
                }
                parents.push_back({
                    nodes[i].first/2,
                    capture_range_node_digest(
                        level, nodes[i].second, nodes[i + 1].second),
                });
            }
        }
        auto & root_nodes = available[height];
        std::sort(root_nodes.begin(), root_nodes.end(),
            [](const indexed_digest & lhs, const indexed_digest & rhs) {
                return lhs.first < rhs.first;
            });
        if (root_nodes.size() != 1 || root_nodes.front().first != 0 ||
            capture_range_root_digest(
                source->total_bytes, source->chunk_bytes,
                source->chunk_count, source->padded_count,
                root_nodes.front().second) != source->root) {
            return status::parent_invalid;
        }

        auto result = std::make_shared<vbr_capture_range_proof::data>();
        result->total_bytes = source->total_bytes;
        result->chunk_bytes = source->chunk_bytes;
        result->chunk_count = source->chunk_count;
        result->padded_count = source->padded_count;
        result->root = source->root;
        result->ranges = ranges;
        for (const auto & range : ranges) {
            const uint64_t end = range.offset + range.size;
            const uint32_t first = uint32_t(
                range.offset/result->chunk_bytes);
            const uint32_t last = uint32_t(
                (end - 1)/result->chunk_bytes);
            for (uint32_t chunk = first;; ++chunk) {
                if (result->selected_chunks.empty() ||
                    result->selected_chunks.back() != chunk) {
                    if (result->selected_chunks.size() >=
                            limits.max_selected_chunks) {
                        return status::limit_exceeded;
                    }
                    const auto leaf = std::lower_bound(
                        available[0].begin(), available[0].end(), chunk,
                        [](const indexed_digest & value, uint32_t index) {
                            return value.first < index;
                        });
                    if (leaf == available[0].end() ||
                        leaf->first != chunk) {
                        return status::parent_invalid;
                    }
                    result->selected_chunks.push_back(chunk);
                    result->selected_leaf_digests.push_back(leaf->second);
                }
                if (chunk == last) {
                    break;
                }
            }
        }

        std::vector<uint32_t> known = result->selected_chunks;
        for (uint32_t level = 0; level < height; ++level) {
            std::vector<uint32_t> parents;
            parents.reserve((known.size() + 1)/2);
            for (const uint32_t index : known) {
                const uint32_t sibling = index ^ 1u;
                if (!std::binary_search(known.begin(), known.end(), sibling)) {
                    if (result->proof_nodes.size() >=
                            limits.max_proof_nodes) {
                        return status::limit_exceeded;
                    }
                    const auto found = std::lower_bound(
                        available[level].begin(), available[level].end(),
                        sibling,
                        [](const indexed_digest & value, uint32_t target) {
                            return value.first < target;
                        });
                    if (found == available[level].end() ||
                        found->first != sibling) {
                        return status::parent_invalid;
                    }
                    result->proof_nodes.push_back({
                        level, sibling, found->second,
                    });
                }
                const uint32_t next = index/2;
                if (parents.empty() || parents.back() != next) {
                    parents.push_back(next);
                }
            }
            known = std::move(parents);
        }
        if (known.size() != 1 || known.front() != 0) {
            return status::parent_invalid;
        }

        uint64_t metadata_bytes =
            sizeof(vbr_capture_range_proof::data) +
            CAPTURE_SHARED_OWNER_BYTES;
        if (!capture_range_metadata_add(
                result->ranges.capacity(),
                sizeof(vbr_capture_authenticated_range), metadata_bytes) ||
            !capture_range_metadata_add(
                result->selected_chunks.capacity(), sizeof(uint32_t),
                metadata_bytes) ||
            !capture_range_metadata_add(
                result->selected_leaf_digests.capacity(),
                sizeof(capture_range_digest), metadata_bytes) ||
            !capture_range_metadata_add(
                result->proof_nodes.capacity(),
                sizeof(capture_range_proof_node), metadata_bytes) ||
            metadata_bytes > limits.max_metadata_bytes) {
            return status::limit_exceeded;
        }
        result->metadata_bytes = metadata_bytes;
        output = vbr_capture_range_proof(std::move(result));
        return status::restricted;
    } catch (...) {
        output = {};
        return status::internal_error;
    }
}

bool vbr_capture_range_verify(
        const vbr_capture_range_proof & proof,
        const vbr_artifact_byte_source & source,
        uint64_t * bytes_read) noexcept {
    if (bytes_read) {
        *bytes_read = 0;
    }
    try {
        if (!proof.data_ || !source.read ||
            source.size != proof.data_->total_bytes ||
            proof.data_->chunk_bytes != VBR_CAPTURE_RANGE_CHUNK_BYTES ||
            proof.data_->selected_chunks.empty() ||
            proof.data_->padded_count == 0) {
            return false;
        }
        using indexed_digest =
            std::pair<uint32_t, capture_range_digest>;
        std::vector<indexed_digest> known;
        known.reserve(proof.data_->selected_chunks.size());
        std::array<uint8_t, VBR_CAPTURE_RANGE_CHUNK_BYTES> scratch;
        uint64_t read_total = 0;
        if (proof.data_->selected_leaf_digests.size() !=
                proof.data_->selected_chunks.size()) {
            return false;
        }
        for (size_t i = 0; i < proof.data_->selected_chunks.size(); ++i) {
            const uint32_t chunk = proof.data_->selected_chunks[i];
            if (chunk >= proof.data_->chunk_count) {
                return false;
            }
            const uint64_t offset = uint64_t(chunk)*proof.data_->chunk_bytes;
            const uint32_t size = uint32_t(std::min<uint64_t>(
                proof.data_->chunk_bytes,
                proof.data_->total_bytes - offset));
            if (!source.read(
                    source.context, offset, scratch.data(), size)) {
                return false;
            }
            llama_sha256_writer payload_hash;
            payload_hash.bytes(scratch.data(), size);
            const auto leaf = capture_range_leaf_digest(
                chunk, size, payload_hash.finish());
            if (leaf != proof.data_->selected_leaf_digests[i]) {
                return false;
            }
            known.push_back({ chunk, leaf });
            read_total += size;
        }
        size_t proof_cursor = 0;
        uint32_t level = 0;
        for (uint32_t width = proof.data_->padded_count;
             width > 1; width /= 2, ++level) {
            std::vector<indexed_digest> parents;
            parents.reserve((known.size() + 1)/2);
            for (size_t i = 0; i < known.size();) {
                const auto current = known[i];
                const uint32_t sibling_index = current.first ^ 1u;
                capture_range_digest sibling;
                bool sibling_known = false;
                if (i + 1 < known.size() &&
                    known[i + 1].first == sibling_index) {
                    sibling = known[i + 1].second;
                    sibling_known = true;
                } else {
                    if (proof_cursor >= proof.data_->proof_nodes.size()) {
                        return false;
                    }
                    const auto & node =
                        proof.data_->proof_nodes[proof_cursor++];
                    if (node.level != level ||
                        node.index != sibling_index) {
                        return false;
                    }
                    sibling = node.digest;
                }
                const auto & left = (current.first & 1u) == 0
                    ? current.second : sibling;
                const auto & right = (current.first & 1u) == 0
                    ? sibling : current.second;
                parents.push_back({
                    current.first/2,
                    capture_range_node_digest(level, left, right),
                });
                i += sibling_known ? 2 : 1;
            }
            known = std::move(parents);
        }
        if (proof_cursor != proof.data_->proof_nodes.size() ||
            known.size() != 1 || known.front().first != 0) {
            return false;
        }
        const auto root = capture_range_root_digest(
            proof.data_->total_bytes, proof.data_->chunk_bytes,
            proof.data_->chunk_count, proof.data_->padded_count,
            known.front().second);
        if (root != proof.data_->root) {
            return false;
        }
        if (bytes_read) {
            *bytes_read = read_total;
        }
        return true;
    } catch (...) {
        if (bytes_read) {
            *bytes_read = 0;
        }
        return false;
    }
}

struct vbr_pinned_chunk_ring::impl {
    std::shared_ptr<vbr_bounded_pinned_ring_core> core;
};

vbr_pinned_chunk_ring::vbr_pinned_chunk_ring(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

vbr_pinned_chunk_ring::~vbr_pinned_chunk_ring() = default;

std::unique_ptr<vbr_pinned_chunk_ring>
vbr_pinned_chunk_ring::create(
        const std::vector<vbr_capture_lane> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_capture_stream_status & status,
        const vbr_capture_ring_accounting * accounting,
        vbr_capture_ring_create_failure * failure) noexcept {
    status = vbr_capture_stream_status::ring_unavailable;
    vbr_capture_ring_create_failure reason =
        vbr_capture_ring_create_failure::none;
    try {
        std::unique_ptr<impl> state(new impl);
        state->core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
            vbr_bounded_pinned_ring_core::create(
                lanes, total_bytes, chunk_bytes, accounting, reason));
        if (!state->core) {
            status = vbr_capture_ring_failure_status(reason);
            if (failure) {
                *failure = reason;
            }
            return nullptr;
        }
        status = vbr_capture_stream_status::ok;
        if (failure) {
            *failure = reason;
        }
        return std::unique_ptr<vbr_pinned_chunk_ring>(
            new vbr_pinned_chunk_ring(std::move(state)));
    } catch (...) {
        status = vbr_capture_stream_status::internal_error;
        if (failure) {
            *failure = vbr_capture_ring_create_failure::internal_error;
        }
        return nullptr;
    }
}

std::unique_ptr<vbr_pinned_chunk_ring>
vbr_pinned_chunk_ring::attach(
        std::shared_ptr<vbr_bounded_pinned_ring_core> core) noexcept {
    try {
        if (!core) {
            return nullptr;
        }
        std::unique_ptr<impl> state(new impl);
        state->core = std::move(core);
        return std::unique_ptr<vbr_pinned_chunk_ring>(
            new vbr_pinned_chunk_ring(std::move(state)));
    } catch (...) {
        return nullptr;
    }
}

uint64_t vbr_pinned_chunk_ring::capacity_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->capacity_bytes() : 0;
}

size_t vbr_pinned_chunk_ring::chunk_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->chunk_bytes() : 0;
}

size_t vbr_pinned_chunk_ring::lane_count() const noexcept {
    return impl_ && impl_->core ? impl_->core->lane_count() : 0;
}

vbr_pinned_ring_operation
vbr_pinned_chunk_ring::try_begin_operation() noexcept {
    if (!impl_ || !impl_->core) {
        return {};
    }
    auto operation = impl_->core->try_begin_operation();
    if (operation) {
        operation.keepalive_ = impl_->core;
    }
    return operation;
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream(
        const vbr_capture_stream_source & source,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    const vbr_capture_stream_range range { 0, source.size };
    return stream_ranges_impl(
        nullptr, source, &range, 1, destination, stats);
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream_ranges(
        const vbr_capture_stream_source & source,
        const std::vector<vbr_capture_stream_range> & ranges,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    return stream_ranges_impl(
        nullptr, source, ranges.data(), ranges.size(), destination, stats);
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream_ranges_reserved(
        const vbr_pinned_ring_operation & operation,
        const vbr_capture_stream_source & source,
        const std::vector<vbr_capture_stream_range> & ranges,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    return stream_ranges_impl(
        &operation, source, ranges.data(), ranges.size(), destination, stats);
}

vbr_capture_stream_status vbr_pinned_chunk_ring::stream_ranges_impl(
        const vbr_pinned_ring_operation * operation,
        const vbr_capture_stream_source & source,
        const vbr_capture_stream_range * ranges,
        size_t range_count,
        artifact_segment_chain & destination,
        vbr_capture_stream_stats & stats) noexcept {
    stats = {};
    if (!impl_ || !impl_->core ||
        source.lane >= impl_->core->lane_count() || source.size == 0 ||
        ranges == nullptr || range_count == 0 || destination.size() != 0) {
        return vbr_capture_stream_status::invalid_argument;
    }
    uint64_t transfer_bytes = 0;
    uint64_t prior_end = 0;
    for (size_t i = 0; i < range_count; ++i) {
        const auto & range = ranges[i];
        if (range.size == 0 || range.source_offset < prior_end ||
            range.source_offset > source.size ||
            range.size > source.size - range.source_offset ||
            !capture_checked_add(
                transfer_bytes, range.size, transfer_bytes)) {
            return vbr_capture_stream_status::invalid_argument;
        }
        prior_end = range.source_offset + range.size;
    }
    const auto * lane = impl_->core->lane_binding(source.lane);
    const bool tensor_source = source.tensor != nullptr;
    if (tensor_source) {
        // The store may be constructed before VBR lazily creates its dedicated
        // side-stream backend. Events and pinned buffers are device-scoped, so
        // bind the lane to the physical device rather than one backend handle.
        if (!source.backend || !source.device ||
            !lane || source.device != lane->device ||
            ggml_backend_get_device(source.backend) !=
                lane->device ||
            source.tensor_offset >
                std::numeric_limits<uint64_t>::max() -
                    source.size ||
            source.tensor_offset > ggml_nbytes(source.tensor) ||
            source.size >
                ggml_nbytes(source.tensor) -
                    source.tensor_offset) {
            return vbr_capture_stream_status::invalid_argument;
        }
    } else if (!source.read) {
        return vbr_capture_stream_status::invalid_argument;
    }
    const bool legacy_digest = !destination.authenticated();
    llama_sha256_writer hash;
    static constexpr char domain_label[] =
        "buun.vbr.capture.segment-stream";
    if (legacy_digest) {
        hash.string(domain_label, sizeof(domain_label) - 1);
        hash.u64(transfer_bytes);
    }

    struct capture_pump_context {
        const vbr_capture_stream_source * source = nullptr;
        const vbr_capture_stream_range * ranges = nullptr;
        size_t range_count = 0;
        size_t range_index = 0;
        uint64_t range_offset = 0;
        bool tensor_source = false;
        artifact_segment_chain * destination = nullptr;
        llama_sha256_writer * hash = nullptr;
        bool legacy_digest = false;
    } pump_context {
        &source, ranges, range_count, 0, 0, tensor_source,
        &destination, &hash, legacy_digest,
    };
    vbr_bounded_pinned_ring_core::pump_callbacks callbacks;
    callbacks.context = &pump_context;
    callbacks.ok = uint32_t(vbr_capture_stream_status::ok);
    callbacks.ring_unavailable =
        uint32_t(vbr_capture_stream_status::ring_unavailable);
    callbacks.submit_failed =
        uint32_t(vbr_capture_stream_status::internal_error);
    callbacks.wait_failed =
        uint32_t(vbr_capture_stream_status::internal_error);
    callbacks.internal_error =
        uint32_t(vbr_capture_stream_status::internal_error);
    callbacks.serialize_submissions = source.continue_transfer != nullptr;
    callbacks.more = [](void * opaque) noexcept {
        const auto & context =
            *static_cast<capture_pump_context *>(opaque);
        return context.range_index < context.range_count;
    };
    callbacks.fill = [](void * opaque, uint8_t * output,
                        size_t capacity,
                        vbr_bounded_pinned_ring_core::pump_step & step)
            noexcept -> uint32_t {
        auto & context = *static_cast<capture_pump_context *>(opaque);
        const auto & source = *context.source;
        try {
            if (source.continue_transfer &&
                !source.continue_transfer(source.continue_context)) {
                return uint32_t(vbr_capture_stream_status::cancelled);
            }
            size_t filled = 0;
            while (filled < capacity &&
                   context.range_index < context.range_count) {
                const auto & range = context.ranges[context.range_index];
                const size_t count = size_t(std::min<uint64_t>(
                    capacity - filled,
                    range.size - context.range_offset));
                const uint64_t source_offset =
                    range.source_offset + context.range_offset;
                if (context.tensor_source) {
                    // Multiple gets are queued on the same backend stream;
                    // submit records one completion event after the complete
                    // packed chunk rather than one event per logical range.
                    ggml_backend_tensor_get_async(
                        source.backend, source.tensor,
                        output + filled,
                        size_t(source.tensor_offset + source_offset),
                        count);
                } else if (!source.read(
                               source.context, source_offset,
                               output + filled, count)) {
                    return uint32_t(
                        vbr_capture_stream_status::short_read);
                }
                filled += count;
                context.range_offset += count;
                if (context.range_offset == range.size) {
                    ++context.range_index;
                    context.range_offset = 0;
                }
            }
            step.valid = filled;
            step.backend = context.tensor_source ? source.backend : nullptr;
            return uint32_t(vbr_capture_stream_status::ok);
        } catch (...) {
            return uint32_t(vbr_capture_stream_status::internal_error);
        }
    };
    callbacks.consume = [](void * opaque, const uint8_t * input,
                           size_t size, uint64_t, bool,
                           uint64_t ordinal, bool &) noexcept -> uint32_t {
        auto & context = *static_cast<capture_pump_context *>(opaque);
        // KNOWN LIMITATION: ggml's asynchronous copy/event APIs return no
        // transfer result. The synthetic seam can report transfer_failed,
        // while a real device error can only surface later as a length or
        // digest mismatch; the capture hardware gate must account for that.
        if (ordinal == context.source->fail_completion_at) {
            return uint32_t(vbr_capture_stream_status::transfer_failed);
        }
        if (!context.destination->append(input, size)) {
            return uint32_t(vbr_capture_stream_status::internal_error);
        }
        if (context.legacy_digest) {
            context.hash->bytes(input, size);
        }
        return uint32_t(vbr_capture_stream_status::ok);
    };

    vbr_bounded_pinned_ring_core::pump_stats pump_stats;
    const auto pumped = vbr_capture_stream_status(operation
        ? impl_->core->pump_reserved(
              *operation, source.lane, callbacks, pump_stats)
        : impl_->core->pump(source.lane, callbacks, pump_stats));
    stats.bytes = pump_stats.bytes;
    stats.chunks = pump_stats.chunks;
    stats.submitted_bytes = pump_stats.submitted_bytes;
    stats.submitted_chunks = pump_stats.submitted_chunks;
    stats.backpressure_waits = pump_stats.backpressure_waits;
    stats.event_completions = pump_stats.event_completions;
    stats.synchronous_fallbacks = pump_stats.synchronous_fallbacks;
    if (pumped != vbr_capture_stream_status::ok) {
        return pumped;
    }
    if (stats.bytes != transfer_bytes) {
        return vbr_capture_stream_status::short_read;
    }
    stats.max_segment_size = destination.max_segment_size();
    if (legacy_digest) {
        stats.streaming_digest = hash.finish();
    }
    return vbr_capture_stream_status::ok;
}

bool vbr_capture_projected_shard_topology(
        const std::vector<vbr_capture_projected_shard_source> & sources,
        uint32_t & shard_count,
        std::array<uint8_t, 32> & digest) noexcept {
    shard_count = 0;
    digest = {};
    try {
        if (sources.empty() || sources.size() > UINT32_MAX) {
            return false;
        }
        std::vector<const vbr_capture_projected_shard_source *> ordered;
        ordered.reserve(sources.size());
        for (const auto & source : sources) {
            ordered.push_back(&source);
        }
        std::sort(ordered.begin(), ordered.end(),
            [](const auto * lhs, const auto * rhs) {
                return lhs->shard_index < rhs->shard_index;
            });
        llama_sha256_writer hash;
        static constexpr char digest_domain[] =
            "buun.vbr.capture/projected-shard-topology";
        hash.string(digest_domain, sizeof(digest_domain) - 1);
        hash.u32(uint32_t(ordered.size()));
        for (uint32_t i = 0; i < ordered.size(); ++i) {
            const auto & source = *ordered[i];
            if (source.shard_index != i || source.source_identity == 0 ||
                source.row_count == 0 || source.row_bytes == 0 ||
                source.source.size == 0) {
                return false;
            }
            hash.u32(source.shard_index);
            hash.u32(source.row_count);
            hash.u64(source.row_bytes);
            hash.u64(source.source_identity);
            hash.u64(source.source.size);
            hash.u32(source.source.lane);
            // This digest is deliberately process-local: bind the exact
            // provider-issued byte-source capability as well as its stable
            // identity so an accidental callback/tensor substitution cannot
            // reuse otherwise identical geometry.
            hash.bytes(&source.source.context,
                       sizeof(source.source.context));
            hash.bytes(&source.source.read,
                       sizeof(source.source.read));
            hash.bytes(&source.source.backend,
                       sizeof(source.source.backend));
            hash.bytes(&source.source.device,
                       sizeof(source.source.device));
            hash.bytes(&source.source.tensor,
                       sizeof(source.source.tensor));
            hash.u64(source.source.tensor_offset);
        }
        shard_count = uint32_t(ordered.size());
        digest = hash.finish();
        return std::any_of(
            digest.begin(), digest.end(), [](uint8_t value) {
                return value != 0;
            });
    } catch (...) {
        shard_count = 0;
        digest = {};
        return false;
    }
}

struct vbr_capture_projected_unit::data {
    vbr_capture_projection projection;
    vbr_capture_unit_snapshot snapshot;
    uint32_t child_id = UINT32_MAX;
    uint32_t stream_index = UINT32_MAX;
    uint32_t logical_unit_id = UINT32_MAX;
    uint64_t packed_bytes = 0;
    vbr_capture_stream_stats transfer;
    std::vector<vbr_capture_projected_shard> shards;
};

vbr_capture_projected_unit::vbr_capture_projected_unit(
        std::shared_ptr<const data> data) noexcept
    : data_(std::move(data)) {}

vbr_capture_projected_unit::operator bool() const noexcept {
    return bool(data_);
}

const vbr_capture_projection &
vbr_capture_projected_unit::projection() const noexcept {
    static const vbr_capture_projection empty;
    return data_ ? data_->projection : empty;
}

const vbr_capture_unit_snapshot &
vbr_capture_projected_unit::snapshot() const noexcept {
    static const vbr_capture_unit_snapshot empty;
    return data_ ? data_->snapshot : empty;
}

uint32_t vbr_capture_projected_unit::child_id() const noexcept {
    return data_ ? data_->child_id : UINT32_MAX;
}

uint32_t vbr_capture_projected_unit::stream_index() const noexcept {
    return data_ ? data_->stream_index : UINT32_MAX;
}

uint32_t vbr_capture_projected_unit::logical_unit_id() const noexcept {
    return data_ ? data_->logical_unit_id : UINT32_MAX;
}

uint64_t vbr_capture_projected_unit::packed_bytes() const noexcept {
    return data_ ? data_->packed_bytes : 0;
}

const vbr_capture_stream_stats &
vbr_capture_projected_unit::transfer() const noexcept {
    static const vbr_capture_stream_stats empty;
    return data_ ? data_->transfer : empty;
}

const std::vector<vbr_capture_projected_shard> &
vbr_capture_projected_unit::shards() const noexcept {
    static const std::vector<vbr_capture_projected_shard> empty;
    return data_ ? data_->shards : empty;
}

namespace {

bool projected_snapshot_valid(
        const vbr_capture_unit_snapshot & snapshot) noexcept {
    return snapshot.source_namespace != 0 &&
           snapshot.child_id != UINT32_MAX &&
           snapshot.logical_unit_id != UINT32_MAX &&
           vbr_lineage_uuid_is_set(snapshot.lineage_uuid) &&
           snapshot.controller_generation != 0 &&
           (snapshot.mutation_serial & 1u) == 0 &&
           capture_generation_valid(snapshot.generation) &&
           snapshot.shard_count != 0 &&
           capture_digest_nonzero(snapshot.shard_topology_digest);
}

} // namespace

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
        vbr_capture_stream_stats * attempted,
        const vbr_pinned_ring_operation * operation) noexcept {
    output = {};
    if (attempted) {
        *attempted = {};
    }
    try {
        if (!projection || projection->source_namespace == 0 ||
            child_id == UINT32_MAX || stream_index == UINT32_MAX ||
            logical_unit_id == UINT32_MAX || sources.empty() ||
            limits.max_shards == 0 ||
            sources.size() > limits.max_shards ||
            limits.max_shard_segment_references == 0 ||
            limits.max_source_operations == 0 ||
            limits.max_total_packed_bytes == 0 ||
            limits.max_authenticated_chunks == 0 ||
            limits.max_authenticated_metadata_bytes == 0 ||
            !snapshots.acquire || !snapshots.recheck ||
            !snapshots.release ||
            projection->dependency_references !=
                projection->dependent_manifest_ids.size()) {
            return vbr_capture_stream_status::projection_invalid;
        }
        const vbr_capture_projection_stream * selected = nullptr;
        for (const auto & stream : projection->streams) {
            if (stream.child_id == child_id &&
                stream.stream_index == stream_index) {
                if (selected != nullptr) {
                    return vbr_capture_stream_status::projection_invalid;
                }
                selected = &stream;
            }
        }
        if (!selected || selected->segments.empty() ||
            selected->segments.size() > UINT32_MAX) {
            return vbr_capture_stream_status::projection_invalid;
        }

        uint64_t prior_end = 0;
        for (const auto & segment : selected->segments) {
            if (segment.cell_count == 0 ||
                segment.first_dependency >
                    projection->dependent_manifest_ids.size() ||
                segment.dependency_count == 0 ||
                segment.dependency_count >
                    projection->dependent_manifest_ids.size() -
                        segment.first_dependency ||
                uint64_t(segment.first_physical_cell) < prior_end) {
                return vbr_capture_stream_status::projection_invalid;
            }
            const uint64_t end =
                uint64_t(segment.first_physical_cell) +
                segment.cell_count;
            if (end > uint64_t(UINT32_MAX) + 1) {
                return vbr_capture_stream_status::projection_invalid;
            }
            const auto dependency_begin =
                projection->dependent_manifest_ids.begin() +
                segment.first_dependency;
            const auto dependency_end =
                dependency_begin + segment.dependency_count;
            if (*dependency_begin == 0 ||
                std::adjacent_find(
                    dependency_begin, dependency_end) != dependency_end ||
                !std::is_sorted(dependency_begin, dependency_end)) {
                return vbr_capture_stream_status::projection_invalid;
            }
            prior_end = end;
        }

        std::vector<const vbr_capture_projected_shard_source *> ordered;
        ordered.reserve(sources.size());
        for (const auto & source : sources) {
            ordered.push_back(&source);
        }
        std::sort(ordered.begin(), ordered.end(),
            [](const auto * lhs, const auto * rhs) {
                return lhs->shard_index < rhs->shard_index;
            });
        uint32_t topology_count = 0;
        std::array<uint8_t, 32> topology_digest = {};
        if (!vbr_capture_projected_shard_topology(
                sources, topology_count, topology_digest) ||
            topology_count != ordered.size() ||
            selected->segments.size() >
                limits.max_shard_segment_references/sources.size()) {
            return vbr_capture_stream_status::projection_invalid;
        }

        struct cell_range { uint32_t first = 0; uint32_t count = 0; };
        std::vector<cell_range> cell_ranges;
        cell_ranges.reserve(selected->segments.size());
        uint64_t packed_rows = 0;
        for (uint32_t i = 0; i < selected->segments.size(); ++i) {
            const auto & segment = selected->segments[i];
            if (segment.packed_first_row != packed_rows) {
                return vbr_capture_stream_status::projection_invalid;
            }
            packed_rows += segment.cell_count;
            if (!cell_ranges.empty() &&
                uint64_t(cell_ranges.back().first) +
                        cell_ranges.back().count ==
                    segment.first_physical_cell &&
                segment.cell_count <=
                    UINT32_MAX - cell_ranges.back().count) {
                cell_ranges.back().count += segment.cell_count;
            } else {
                cell_ranges.push_back({
                    segment.first_physical_cell, segment.cell_count,
                });
            }
        }
        std::vector<uint64_t> shard_packed_bytes;
        shard_packed_bytes.reserve(ordered.size());
        std::vector<uint32_t> shard_authenticated_chunks;
        shard_authenticated_chunks.reserve(ordered.size());
        std::vector<uint64_t> shard_authenticated_metadata;
        shard_authenticated_metadata.reserve(ordered.size());
        uint64_t total_packed_bytes = 0;
        uint64_t source_operations = 0;
        uint64_t authenticated_chunks = 0;
        uint64_t authenticated_metadata_bytes = 0;
        const uint64_t chunk_bytes = ring.chunk_bytes();
        if (chunk_bytes == 0) {
            return vbr_capture_stream_status::projection_invalid;
        }
        for (const auto * source : ordered) {
            if (source->row_count == 0 || source->row_bytes == 0 ||
                source->row_count > UINT64_MAX/source->row_bytes ||
                uint64_t(source->row_count)*source->row_bytes >
                    source->source.size) {
                return vbr_capture_stream_status::projection_invalid;
            }
            for (const auto & segment : selected->segments) {
                const uint64_t end =
                    uint64_t(segment.first_physical_cell) +
                    segment.cell_count;
                if (end > source->row_count ||
                    segment.first_physical_cell >
                        UINT64_MAX/source->row_bytes ||
                    segment.cell_count >
                        UINT64_MAX/source->row_bytes) {
                    return vbr_capture_stream_status::projection_invalid;
                }
                if (uint64_t(segment.first_physical_cell)*
                            source->row_bytes > source->source.size ||
                    uint64_t(segment.cell_count)*source->row_bytes >
                        source->source.size -
                            uint64_t(segment.first_physical_cell)*
                                source->row_bytes) {
                    return vbr_capture_stream_status::projection_invalid;
                }
            }
            if (packed_rows > UINT64_MAX/source->row_bytes) {
                return vbr_capture_stream_status::projection_invalid;
            }
            const uint64_t packed_bytes = packed_rows*source->row_bytes;
            if (packed_bytes == 0 ||
                packed_bytes > limits.max_total_packed_bytes -
                    total_packed_bytes) {
                return vbr_capture_stream_status::projection_invalid;
            }
            total_packed_bytes += packed_bytes;
            shard_packed_bytes.push_back(packed_bytes);
            const uint64_t chunk_count =
                (packed_bytes - 1)/VBR_CAPTURE_RANGE_CHUNK_BYTES + 1;
            uint64_t tree_metadata_bytes = 0;
            if (chunk_count >
                    limits.max_authenticated_chunks -
                        authenticated_chunks ||
                !capture_range_tree_metadata_bytes(
                    chunk_count, tree_metadata_bytes) ||
                tree_metadata_bytes >
                    limits.max_authenticated_metadata_bytes -
                        authenticated_metadata_bytes) {
                return vbr_capture_stream_status::projection_invalid;
            }
            authenticated_chunks += chunk_count;
            authenticated_metadata_bytes += tree_metadata_bytes;
            shard_authenticated_chunks.push_back(uint32_t(chunk_count));
            shard_authenticated_metadata.push_back(tree_metadata_bytes);

            uint64_t packed_cursor = 0;
            for (const auto & range : cell_ranges) {
                const uint64_t bytes =
                    uint64_t(range.count)*source->row_bytes;
                const uint64_t first_capacity =
                    chunk_bytes - packed_cursor%chunk_bytes;
                uint64_t operations = 1;
                if (bytes > first_capacity) {
                    const uint64_t remaining = bytes - first_capacity;
                    operations += remaining/chunk_bytes;
                    operations += remaining%chunk_bytes != 0;
                }
                if (operations >
                        limits.max_source_operations - source_operations) {
                    return vbr_capture_stream_status::projection_invalid;
                }
                source_operations += operations;
                packed_cursor += bytes;
            }
            if (packed_cursor != packed_bytes) {
                return vbr_capture_stream_status::projection_invalid;
            }
        }

        // One reusable byte-range workspace is allocated before the unit
        // lease. Segment boundaries remain in the shared slice map; adjacent
        // physical runs and small disjoint runs are packed by the ring.
        std::vector<vbr_capture_stream_range> ranges(cell_ranges.size());

        // Hash the immutable projection before acquiring the unit-version
        // lease. Only bounded source reads and snapshot-dependent sealing
        // remain inside the lease interval.
        llama_sha256_writer layout_hash;
        static constexpr char LAYOUT_DOMAIN[] =
            "buun.vbr.capture/projected-layout";
        layout_hash.string(LAYOUT_DOMAIN, sizeof(LAYOUT_DOMAIN) - 1);
        layout_hash.u64(projection->source_namespace);
        layout_hash.u32(child_id);
        layout_hash.u32(stream_index);
        layout_hash.u32(logical_unit_id);
        layout_hash.bytes(topology_digest.data(), topology_digest.size());
        layout_hash.u64(selected->segments.size());
        for (const auto & segment : selected->segments) {
            layout_hash.u32(segment.first_physical_cell);
            layout_hash.u32(segment.cell_count);
            layout_hash.u64(segment.packed_first_row);
            layout_hash.u32(segment.dependency_count);
            for (uint32_t i = 0; i < segment.dependency_count; ++i) {
                layout_hash.u64(projection->dependent_manifest_ids[
                    segment.first_dependency + i]);
            }
        }
        const auto layout_digest = layout_hash.finish();

        vbr_capture_unit_snapshot snapshot;
        if (!snapshots.acquire(
                snapshots.context, projection->source_namespace,
                child_id, logical_unit_id, snapshot)) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        struct release_guard {
            const vbr_capture_unit_snapshot_provider * provider = nullptr;
            const vbr_capture_unit_snapshot * snapshot = nullptr;
            bool active = false;
            ~release_guard() {
                if (active) {
                    provider->release(provider->context, *snapshot);
                }
            }
        } release { &snapshots, &snapshot, true };
        if (!projected_snapshot_valid(snapshot) ||
            snapshot.source_namespace != projection->source_namespace ||
            snapshot.child_id != child_id ||
            snapshot.logical_unit_id != logical_unit_id ||
            snapshot.shard_count != topology_count ||
            snapshot.shard_topology_digest != topology_digest) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }

        vbr_capture_projected_unit::data result;
        result.projection = std::move(projection);
        result.snapshot = snapshot;
        result.child_id = child_id;
        result.stream_index = stream_index;
        result.logical_unit_id = logical_unit_id;
        result.packed_bytes = total_packed_bytes;
        result.shards.reserve(ordered.size());
        llama_sha256_writer unit_hash;
        static constexpr char UNIT_DOMAIN[] =
            "buun.vbr.capture/projected-unit";
        unit_hash.string(UNIT_DOMAIN, sizeof(UNIT_DOMAIN) - 1);
        unit_hash.bytes(layout_digest.data(), layout_digest.size());
        unit_hash.u64(snapshot.lineage_uuid.hi);
        unit_hash.u64(snapshot.lineage_uuid.lo);
        unit_hash.u64(snapshot.controller_generation);
        unit_hash.u64(snapshot.mutation_serial);
        unit_hash.u64(snapshot.generation.repr_gen);
        unit_hash.u64(snapshot.generation.publish_seq);
        unit_hash.u32(uint32_t(snapshot.generation.current_type));
        unit_hash.u32(uint32_t(snapshot.generation.last_source_type));
        unit_hash.u32(uint32_t(snapshot.generation.domain));
        unit_hash.u32(snapshot.generation.promote_hops);
        unit_hash.u32(uint32_t(snapshot.generation.last_transition));
        for (size_t shard_index = 0;
             shard_index < ordered.size(); ++shard_index) {
            const auto * shard = ordered[shard_index];
            for (size_t i = 0; i < cell_ranges.size(); ++i) {
                ranges[i] = {
                    uint64_t(cell_ranges[i].first)*shard->row_bytes,
                    uint64_t(cell_ranges[i].count)*shard->row_bytes,
                };
            }
            auto chain = std::make_shared<artifact_segment_chain>(
                VBR_CAPTURE_RANGE_CHUNK_BYTES,
                shard_authenticated_chunks[shard_index]);
            vbr_capture_stream_stats stats;
            const auto streamed = operation
                ? ring.stream_ranges_reserved(
                      *operation, shard->source, ranges, *chain, stats)
                : ring.stream_ranges(
                      shard->source, ranges, *chain, stats);
            if (attempted) {
                if (stats.bytes > UINT64_MAX - attempted->bytes ||
                    stats.chunks > UINT64_MAX - attempted->chunks ||
                    stats.submitted_bytes >
                        UINT64_MAX - attempted->submitted_bytes ||
                    stats.submitted_chunks >
                        UINT64_MAX - attempted->submitted_chunks ||
                    stats.backpressure_waits >
                        UINT64_MAX - attempted->backpressure_waits ||
                    stats.event_completions >
                        UINT64_MAX - attempted->event_completions ||
                    stats.synchronous_fallbacks >
                        UINT64_MAX - attempted->synchronous_fallbacks) {
                    return vbr_capture_stream_status::internal_error;
                }
                attempted->bytes += stats.bytes;
                attempted->chunks += stats.chunks;
                attempted->submitted_bytes += stats.submitted_bytes;
                attempted->submitted_chunks += stats.submitted_chunks;
                attempted->backpressure_waits += stats.backpressure_waits;
                attempted->event_completions += stats.event_completions;
                attempted->synchronous_fallbacks +=
                    stats.synchronous_fallbacks;
                attempted->max_segment_size = std::max(
                    attempted->max_segment_size,
                    stats.max_segment_size);
            }
            if (streamed != vbr_capture_stream_status::ok) {
                return streamed;
            }
            if (stats.bytes != shard_packed_bytes[shard_index] ||
                stats.bytes > UINT64_MAX - result.transfer.bytes ||
                stats.chunks > UINT64_MAX - result.transfer.chunks ||
                stats.backpressure_waits >
                    UINT64_MAX - result.transfer.backpressure_waits ||
                stats.event_completions >
                    UINT64_MAX - result.transfer.event_completions ||
                stats.synchronous_fallbacks >
                    UINT64_MAX - result.transfer.synchronous_fallbacks) {
                return vbr_capture_stream_status::internal_error;
            }
            result.transfer.bytes += stats.bytes;
            result.transfer.chunks += stats.chunks;
            result.transfer.submitted_bytes += stats.submitted_bytes;
            result.transfer.submitted_chunks += stats.submitted_chunks;
            result.transfer.backpressure_waits += stats.backpressure_waits;
            result.transfer.event_completions += stats.event_completions;
            result.transfer.synchronous_fallbacks +=
                stats.synchronous_fallbacks;
            result.transfer.max_segment_size = std::max(
                result.transfer.max_segment_size,
                stats.max_segment_size);
            unit_hash.u32(shard->shard_index);
            unit_hash.u32(shard->row_count);
            unit_hash.u64(shard->row_bytes);
            unit_hash.u64(shard->source_identity);
            vbr_capture_range_tree authenticated_ranges;
            if (!vbr_capture_range_seal(
                    *chain, shard_authenticated_metadata[shard_index],
                    authenticated_ranges) ||
                authenticated_ranges.total_bytes() != chain->size()) {
                return vbr_capture_stream_status::internal_error;
            }
            stats.streaming_digest = authenticated_ranges.root();
            unit_hash.bytes(
                stats.streaming_digest.data(),
                stats.streaming_digest.size());
            result.shards.push_back({
                shard->shard_index,
                shard->row_count,
                shard->row_bytes,
                shard->source_identity,
                std::move(chain),
                stats.streaming_digest,
                std::move(authenticated_ranges),
            });
        }
        result.transfer.streaming_digest = unit_hash.finish();
        if (!snapshots.recheck(snapshots.context, snapshot)) {
            return vbr_capture_stream_status::snapshot_changed;
        }
        snapshots.release(snapshots.context, snapshot);
        release.active = false;
        output = vbr_capture_projected_unit(
            std::make_shared<const vbr_capture_projected_unit::data>(
                std::move(result)));
        return vbr_capture_stream_status::ok;
    } catch (...) {
        output = {};
        return vbr_capture_stream_status::internal_error;
    }
}

vbr_capture_sealed_companion::operator bool() const noexcept {
    return companion_index_ != UINT32_MAX && bytes_ != nullptr &&
           capture_digest_nonzero(streaming_digest_);
}

uint32_t vbr_capture_sealed_companion::companion_index() const noexcept {
    return companion_index_;
}

uint64_t vbr_capture_sealed_companion::size() const noexcept {
    return bytes_ ? bytes_->size() : 0;
}

const std::array<uint8_t, 32> &
vbr_capture_sealed_companion::streaming_digest() const noexcept {
    return streaming_digest_;
}

bool vbr_capture_seal_companion(
        uint32_t companion_index,
        std::unique_ptr<artifact_segment_chain> bytes,
        vbr_capture_sealed_companion & output) noexcept {
    output = {};
    try {
        if (companion_index == UINT32_MAX || !bytes || bytes->size() == 0) {
            return false;
        }
        const auto digest = vbr_capture_stream_digest(*bytes);
        if (!capture_digest_nonzero(digest)) {
            return false;
        }
        output.companion_index_ = companion_index;
        output.streaming_digest_ = digest;
        output.bytes_ = std::shared_ptr<const artifact_segment_chain>(
            std::move(bytes));
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

struct vbr_capture_manifest_assembly::data {
    vbr_capture_projection projection;
    std::vector<vbr_capture_controller_target> controller_targets;
    std::vector<vbr_capture_projected_unit> projected_units;
    std::vector<uint32_t> controller_references;
    std::vector<uint32_t> unit_references;
    std::vector<vbr_capture_manifest_range_proof> range_proofs;
    std::vector<vbr_capture_manifest_result> manifests;
};

vbr_capture_manifest_assembly::vbr_capture_manifest_assembly(
        std::shared_ptr<const data> data) noexcept
    : data_(std::move(data)) {}

vbr_capture_manifest_assembly::operator bool() const noexcept {
    return bool(data_);
}

const vbr_capture_projection &
vbr_capture_manifest_assembly::projection() const noexcept {
    static const vbr_capture_projection empty;
    return data_ ? data_->projection : empty;
}

const std::vector<vbr_capture_controller_target> &
vbr_capture_manifest_assembly::controller_targets() const noexcept {
    static const std::vector<vbr_capture_controller_target> empty;
    return data_ ? data_->controller_targets : empty;
}

const std::vector<vbr_capture_projected_unit> &
vbr_capture_manifest_assembly::projected_units() const noexcept {
    static const std::vector<vbr_capture_projected_unit> empty;
    return data_ ? data_->projected_units : empty;
}

const std::vector<uint32_t> &
vbr_capture_manifest_assembly::controller_references() const noexcept {
    static const std::vector<uint32_t> empty;
    return data_ ? data_->controller_references : empty;
}

const std::vector<uint32_t> &
vbr_capture_manifest_assembly::unit_references() const noexcept {
    static const std::vector<uint32_t> empty;
    return data_ ? data_->unit_references : empty;
}

const std::vector<vbr_capture_manifest_range_proof> &
vbr_capture_manifest_assembly::range_proofs() const noexcept {
    static const std::vector<vbr_capture_manifest_range_proof> empty;
    return data_ ? data_->range_proofs : empty;
}

const std::vector<vbr_capture_manifest_result> &
vbr_capture_manifest_assembly::manifests() const noexcept {
    static const std::vector<vbr_capture_manifest_result> empty;
    return data_ ? data_->manifests : empty;
}

bool vbr_capture_assemble_manifests(
        vbr_capture_projection projection,
        std::vector<vbr_capture_controller_target> && controller_targets,
        std::vector<vbr_capture_projected_unit> && projected_units,
        const vbr_capture_controller_target_provider & targets,
        const vbr_capture_manifest_assembly_limits & limits,
        vbr_capture_manifest_assembly & output) noexcept {
    output = {};
    try {
        if (!projection || !targets.recheck ||
            limits.max_controller_targets == 0 ||
            limits.max_projected_units == 0 ||
            limits.max_unit_descriptor_shards == 0 ||
            limits.max_unit_descriptor_metadata_bytes == 0 ||
            limits.max_manifests == 0 ||
            limits.max_controller_references == 0 ||
            limits.max_unit_references == 0 ||
            limits.max_range_proofs == 0 ||
            limits.max_range_proof_metadata_bytes == 0 ||
            limits.range_proof.max_ranges == 0 ||
            limits.range_proof.max_selected_chunks == 0 ||
            limits.range_proof.max_proof_nodes == 0 ||
            limits.range_proof.max_metadata_bytes == 0 ||
            limits.max_controller_references > UINT32_MAX ||
            limits.max_unit_references > UINT32_MAX ||
            limits.max_range_proofs > UINT32_MAX ||
            controller_targets.size() > limits.max_controller_targets ||
            controller_targets.size() >
                limits.max_controller_references ||
            projected_units.size() > limits.max_projected_units ||
            projection->manifest_count == 0 ||
            projection->manifest_count > limits.max_manifests ||
            projection->source_namespace == 0 ||
            projection->dependency_references !=
                projection->dependent_manifest_ids.size()) {
            return false;
        }

        using target_key = std::pair<uint64_t, uint32_t>;
        using stream_key = std::pair<uint32_t, uint32_t>;
        using manifest_stream_key =
            std::tuple<uint64_t, uint32_t, uint32_t>;
        using unit_key =
            std::tuple<uint32_t, uint32_t, uint64_t, uint32_t>;
        std::sort(controller_targets.begin(), controller_targets.end(),
            [](const auto & lhs, const auto & rhs) {
                return std::tie(lhs.manifest_id, lhs.child_id) <
                       std::tie(rhs.manifest_id, rhs.child_id);
            });
        std::sort(projected_units.begin(), projected_units.end(),
            [](const auto & lhs, const auto & rhs) {
                return std::make_tuple(
                           lhs.child_id(), lhs.stream_index(),
                           lhs.snapshot().controller_generation,
                           lhs.logical_unit_id()) <
                       std::make_tuple(
                           rhs.child_id(), rhs.stream_index(),
                           rhs.snapshot().controller_generation,
                           rhs.logical_unit_id());
            });

        std::map<stream_key, const vbr_capture_projection_stream *>
            projection_streams;
        for (const auto & stream : projection->streams) {
            if (stream.segments.empty() ||
                !projection_streams.emplace(
                    stream_key { stream.child_id, stream.stream_index },
                    &stream).second) {
                return false;
            }
        }

        std::map<target_key, uint32_t> target_by_manifest_child;
        std::map<std::pair<uint32_t, uint64_t>, uint32_t>
            target_by_representation;
        uint64_t target_unit_references = 0;
        uint64_t descriptor_shards = 0;
        uint64_t descriptor_metadata_bytes = 0;
        for (uint32_t i = 0; i < controller_targets.size(); ++i) {
            const auto & target = controller_targets[i];
            if (target.units.empty() ||
                target.unit_descriptors.size() != target.units.size() ||
                target.units.size() > limits.max_projected_units ||
                target.units.size() >
                    limits.max_unit_references - target_unit_references) {
                return false;
            }
            target_unit_references += target.units.size();
            std::vector<ggml_type> types;
            types.reserve(target.units.size());
            bool generations_valid = !target.units.empty();
            for (uint32_t unit = 0; unit < target.units.size(); ++unit) {
                const auto & generation = target.units[unit];
                const auto & descriptor = target.unit_descriptors[unit];
                if (!capture_checked_add(
                        descriptor_shards, descriptor.shards.size(),
                        descriptor_shards) ||
                    descriptor_shards >
                        limits.max_unit_descriptor_shards ||
                    !capture_range_metadata_add(
                        1, sizeof(vbr_artifact_unit_descriptor),
                        descriptor_metadata_bytes) ||
                    !capture_range_metadata_add(
                        descriptor.shards.size(),
                        sizeof(vbr_artifact_shard_descriptor),
                        descriptor_metadata_bytes) ||
                    descriptor_metadata_bytes >
                        limits.max_unit_descriptor_metadata_bytes) {
                    return false;
                }
                generations_valid = generations_valid &&
                    capture_generation_valid(generation) &&
                    descriptor.child_id == target.child_id &&
                    descriptor.logical_unit_id == unit &&
                    descriptor.lineage_uuid == target.lineage_uuid &&
                    descriptor.repr_gen == generation.repr_gen &&
                    descriptor.current_type == generation.current_type &&
                    descriptor.last_source_type == generation.last_source_type &&
                    descriptor.promote_hops == generation.promote_hops &&
                    descriptor.last_transition == generation.last_transition &&
                    descriptor.n_stream == target.policy.n_stream &&
                    descriptor.unified == target.policy.unified &&
                    descriptor.wm_cells == target.policy.wm_cells &&
                    !descriptor.shards.empty();
                types.push_back(static_cast<ggml_type>(
                    generation.current_type));
            }
            if (target.manifest_id == 0 ||
                target.source_namespace != projection->source_namespace ||
                target.child_id == UINT32_MAX ||
                !vbr_lineage_uuid_is_set(target.lineage_uuid) ||
                target.controller_generation == 0 ||
                target.policy.child_id != target.child_id ||
                target.policy.dependency_mode !=
                    checkpoint_child_dependency_mode::live_guarded ||
                !capture_digest_nonzero(
                    target.policy.degrade_order_digest) ||
                !capture_digest_nonzero(target.policy.policy_digest) ||
                target.policy.floor_type < 0 ||
                target.policy.floor_type >= GGML_TYPE_COUNT ||
                target.policy.n_stream == 0 ||
                target.policy.unified != (target.policy.n_stream == 1) ||
                target.policy.wm_cells == 0 ||
                !target.policy.completed_wave || !generations_valid ||
                target.policy.current_type_vector_digest !=
                    vbr_type_vector_digest(types) ||
                !target_by_manifest_child.emplace(target_key {
                    target.manifest_id, target.child_id }, i).second) {
                return false;
            }
            const auto representation = target_by_representation.emplace(
                std::pair<uint32_t, uint64_t> {
                    target.child_id, target.controller_generation }, i);
            if (!representation.second) {
                const auto & prior =
                    controller_targets[representation.first->second];
                if (!vbr_capture_controller_representation_equal(
                        prior, target)) {
                    return false;
                }
            }
        }

        std::map<uint64_t, std::set<stream_key>> manifest_streams;
        std::map<uint64_t, bool> manifest_preflight_available;
        std::map<manifest_stream_key,
                 std::vector<vbr_capture_authenticated_range>>
            manifest_packed_ranges;
        for (const auto & manifest : projection->manifests) {
            if (!manifest_preflight_available.emplace(
                    manifest.manifest_id,
                    manifest.dependencies_available).second) {
                return false;
            }
            manifest_streams.emplace(
                manifest.manifest_id, std::set<stream_key> {});
        }
        for (const auto & stream : projection->streams) {
            for (const auto & segment : stream.segments) {
                if (segment.dependency_count == 0 ||
                    segment.first_dependency >
                        projection->dependent_manifest_ids.size() ||
                    segment.dependency_count >
                        projection->dependent_manifest_ids.size() -
                            segment.first_dependency) {
                    return false;
                }
                for (uint32_t i = 0; i < segment.dependency_count; ++i) {
                    const uint64_t manifest_id =
                        projection->dependent_manifest_ids[
                            segment.first_dependency + i];
                    if (manifest_id == 0) {
                        return false;
                    }
                    manifest_streams[manifest_id].insert({
                        stream.child_id, stream.stream_index,
                    });
                    auto & ranges = manifest_packed_ranges[
                        manifest_stream_key {
                            manifest_id, stream.child_id,
                            stream.stream_index,
                        }];
                    if (!ranges.empty() &&
                        ranges.back().offset + ranges.back().size ==
                            segment.packed_first_row) {
                        ranges.back().size += segment.cell_count;
                    } else {
                        ranges.push_back({
                            segment.packed_first_row,
                            segment.cell_count,
                        });
                    }
                }
            }
        }
        if (manifest_streams.size() != projection->manifest_count) {
            return false;
        }

        std::map<uint64_t, std::vector<uint32_t>> manifest_targets;
        std::vector<bool> manifest_reachable;
        manifest_reachable.reserve(manifest_streams.size());
        uint64_t referenced_targets = 0;
        for (const auto & entry : manifest_streams) {
            auto & refs = manifest_targets[entry.first];
            const auto preflight =
                manifest_preflight_available.find(entry.first);
            if (preflight == manifest_preflight_available.end()) {
                return false;
            }
            bool complete = preflight->second;
            std::map<uint32_t, std::set<uint32_t>> streams_by_child;
            for (const auto & stream : entry.second) {
                streams_by_child[stream.first].insert(stream.second);
            }
            for (const auto & child : streams_by_child) {
                const auto target = target_by_manifest_child.find({
                    entry.first, child.first,
                });
                if (target == target_by_manifest_child.end()) {
                    complete = false;
                    continue;
                }
                const auto & policy =
                    controller_targets[target->second].policy;
                for (const uint32_t stream : child.second) {
                    if (stream >= policy.n_stream) {
                        return false;
                    }
                }
                refs.push_back(target->second);
            }
            referenced_targets += refs.size();
            if (complete) {
                if (refs.empty() ||
                    refs.front() >= controller_targets.size() ||
                    refs.size() > controller_targets.size() - refs.front()) {
                    return false;
                }
                for (uint32_t i = 0; i < refs.size(); ++i) {
                    if (refs[i] != refs.front() + i) {
                        return false;
                    }
                }
            }
            manifest_reachable.push_back(
                complete && targets.recheck(
                    targets.context, entry.first,
                    controller_targets.data() + refs.front(), refs.size()));
        }
        if (referenced_targets != controller_targets.size()) {
            return false;
        }

        std::map<unit_key, uint32_t> input_unit_by_key;
        std::vector<bool> unit_valid(projected_units.size(), false);
        for (uint32_t i = 0; i < projected_units.size(); ++i) {
            const auto & unit = projected_units[i];
            const auto & snapshot = unit.snapshot();
            const unit_key key {
                unit.child_id(), unit.stream_index(),
                snapshot.controller_generation,
                unit.logical_unit_id(),
            };
            if (!unit || !(unit.projection() == projection) ||
                !input_unit_by_key.emplace(key, i).second) {
                return false;
            }
            const auto stream = projection_streams.find({
                unit.child_id(), unit.stream_index(),
            });
            const auto representation = target_by_representation.find({
                unit.child_id(), snapshot.controller_generation,
            });
            const bool named_by_target =
                representation != target_by_representation.end() &&
                unit.logical_unit_id() < controller_targets[
                    representation->second].units.size();
            const vbr_unit_generation * expected = nullptr;
            if (named_by_target) {
                expected = &controller_targets[representation->second].units[
                    unit.logical_unit_id()];
            }
            if (stream == projection_streams.end() ||
                snapshot.source_namespace != projection->source_namespace ||
                snapshot.child_id != unit.child_id() ||
                snapshot.logical_unit_id != unit.logical_unit_id() ||
                !vbr_lineage_uuid_is_set(snapshot.lineage_uuid) ||
                snapshot.controller_generation == 0 ||
                (snapshot.mutation_serial & 1u) != 0 ||
                !capture_generation_valid(snapshot.generation) ||
                snapshot.shard_count != unit.shards().size() ||
                snapshot.shard_count == 0 ||
                !capture_digest_nonzero(
                    snapshot.shard_topology_digest) ||
                unit.packed_bytes() == 0 ||
                unit.transfer().bytes != unit.packed_bytes() ||
                !capture_digest_nonzero(
                    unit.transfer().streaming_digest)) {
                return false;
            }
            for (uint32_t j = 0; j < unit.shards().size(); ++j) {
                const auto & shard = unit.shards()[j];
                if (shard.shard_index != j || shard.bytes == nullptr ||
                    shard.bytes->size() == 0 ||
                    !capture_digest_nonzero(shard.streaming_digest) ||
                    !shard.authenticated_ranges ||
                    shard.authenticated_ranges.total_bytes() !=
                        shard.bytes->size() ||
                    !capture_digest_nonzero(
                        shard.authenticated_ranges.root())) {
                    return false;
                }
            }
            unit_valid[i] = named_by_target && expected != nullptr &&
                snapshot.lineage_uuid == controller_targets[
                    representation->second].lineage_uuid &&
                capture_generation_equal(snapshot.generation, *expected);
            if (unit_valid[i]) {
                const auto & descriptor = controller_targets[
                    representation->second].unit_descriptors[
                        unit.logical_unit_id()];
                if (descriptor.shards.size() != unit.shards().size()) {
                    unit_valid[i] = false;
                }
                for (uint32_t shard = 0;
                     unit_valid[i] && shard < unit.shards().size(); ++shard) {
                    unit_valid[i] =
                        descriptor.shards[shard].shard_index == shard &&
                        descriptor.shards[shard].row_bytes ==
                            unit.shards()[shard].row_bytes &&
                        descriptor.shards[shard].row_count ==
                            unit.shards()[shard].source_row_count;
                }
            }
        }

        auto result = std::make_shared<vbr_capture_manifest_assembly::data>();
        result->projection = projection;
        result->controller_targets = std::move(controller_targets);
        result->projected_units.reserve(projected_units.size());
        std::map<unit_key, uint32_t> valid_unit_by_key;
        for (uint32_t i = 0; i < projected_units.size(); ++i) {
            if (!unit_valid[i]) {
                continue;
            }
            const auto & unit = projected_units[i];
            const uint32_t next = uint32_t(result->projected_units.size());
            valid_unit_by_key.emplace(unit_key {
                unit.child_id(), unit.stream_index(),
                unit.snapshot().controller_generation,
                unit.logical_unit_id(),
            }, next);
            result->projected_units.push_back(std::move(projected_units[i]));
        }

        result->manifests.reserve(manifest_targets.size());
        uint32_t manifest_index = 0;
        uint64_t range_proof_metadata_bytes = 0;
        for (const auto & entry : manifest_targets) {
            const uint64_t manifest_id = entry.first;
            const auto & controller_refs = entry.second;
            const size_t controller_mark =
                result->controller_references.size();
            const size_t unit_mark = result->unit_references.size();
            const size_t range_proof_mark = result->range_proofs.size();
            bool ready = manifest_reachable[manifest_index++];
            for (const uint32_t target_index : controller_refs) {
                if (!ready) {
                    break;
                }
                if (result->controller_references.size() >=
                        limits.max_controller_references) {
                    return false;
                }
                result->controller_references.push_back(target_index);
            }
            for (const auto & stream : manifest_streams.at(manifest_id)) {
                if (!ready) {
                    break;
                }
                const auto target_index = target_by_manifest_child.find({
                    manifest_id, stream.first,
                });
                if (target_index == target_by_manifest_child.end()) {
                    ready = false;
                    break;
                }
                const auto & target =
                    result->controller_targets[target_index->second];
                for (uint32_t unit = 0; unit < target.units.size(); ++unit) {
                    const auto found = valid_unit_by_key.find(unit_key {
                        target.child_id, stream.second,
                        target.controller_generation, unit,
                    });
                    if (found == valid_unit_by_key.end() ||
                        !capture_generation_equal(
                            result->projected_units[found->second].
                                snapshot().generation,
                            target.units[unit])) {
                        ready = false;
                        break;
                    }
                    if (result->unit_references.size() >=
                            limits.max_unit_references) {
                        return false;
                    }
                    result->unit_references.push_back(found->second);
                    const auto projection_stream =
                        projection_streams.find(stream);
                    if (projection_stream == projection_streams.end()) {
                        return false;
                    }
                    const auto & segments =
                        projection_stream->second->segments;
                    if (segments.empty()) {
                        return false;
                    }
                    const uint64_t packed_rows =
                        segments.back().packed_first_row +
                        segments.back().cell_count;
                    if (packed_rows == 0) {
                        return false;
                    }
                    const auto & captured =
                        result->projected_units[found->second];
                    for (const auto & shard : captured.shards()) {
                        if (!shard.bytes ||
                            shard.bytes->size()%packed_rows != 0) {
                            return false;
                        }
                        const uint64_t row_bytes =
                            shard.bytes->size()/packed_rows;
                        if (row_bytes == 0) {
                            return false;
                        }
                        const auto planned_ranges =
                            manifest_packed_ranges.find({
                                manifest_id, stream.first, stream.second,
                            });
                        if (planned_ranges == manifest_packed_ranges.end() ||
                            planned_ranges->second.empty() ||
                            planned_ranges->second.size() >
                                limits.range_proof.max_ranges) {
                            return false;
                        }
                        std::vector<vbr_capture_authenticated_range> ranges;
                        ranges.reserve(planned_ranges->second.size());
                        for (const auto & rows : planned_ranges->second) {
                            if (rows.offset > UINT64_MAX/row_bytes ||
                                rows.size > UINT64_MAX/row_bytes) {
                                return false;
                            }
                            ranges.push_back({
                                rows.offset*row_bytes,
                                rows.size*row_bytes,
                            });
                        }
                        if (ranges.empty() ||
                            result->range_proofs.size() >=
                                limits.max_range_proofs) {
                            return false;
                        }
                        vbr_capture_range_proof proof;
                        if (!vbr_capture_range_prove(
                                shard.authenticated_ranges, ranges,
                                limits.range_proof, proof)) {
                            return false;
                        }
                        const uint64_t proof_charge =
                            proof.metadata_bytes() +
                            2*sizeof(vbr_capture_manifest_range_proof);
                        if (proof_charge < proof.metadata_bytes() ||
                            proof_charge >
                                limits.max_range_proof_metadata_bytes -
                                    range_proof_metadata_bytes) {
                            return false;
                        }
                        range_proof_metadata_bytes +=
                            proof_charge;
                        result->range_proofs.push_back({
                            found->second, shard.shard_index,
                            std::move(proof),
                        });
                    }
                }
            }
            if (!ready) {
                result->controller_references.resize(controller_mark);
                result->unit_references.resize(unit_mark);
                while (result->range_proofs.size() > range_proof_mark) {
                    range_proof_metadata_bytes -=
                        result->range_proofs.back().proof.metadata_bytes() +
                        2*sizeof(vbr_capture_manifest_range_proof);
                    result->range_proofs.pop_back();
                }
            }
            vbr_capture_manifest_result manifest;
            manifest.manifest_id = manifest_id;
            manifest.state = ready
                ? vbr_capture_manifest_state::ready
                : vbr_capture_manifest_state::dependency_unavailable;
            manifest.first_controller = uint32_t(controller_mark);
            manifest.controller_count = ready
                ? uint32_t(result->controller_references.size() -
                    controller_mark) : 0;
            manifest.first_unit = uint32_t(unit_mark);
            manifest.unit_count = ready
                ? uint32_t(result->unit_references.size() - unit_mark) : 0;
            manifest.first_range_proof = uint32_t(range_proof_mark);
            manifest.range_proof_count = ready
                ? uint32_t(result->range_proofs.size() -
                    range_proof_mark) : 0;
            result->manifests.push_back(manifest);
        }

        // Do not keep unique payload segments alive for manifests that could
        // not be certified. Units shared with any ready manifest survive and
        // every flat reference is remapped to the compact owned vector.
        std::vector<bool> referenced_units(
            result->projected_units.size(), false);
        for (const uint32_t unit : result->unit_references) {
            referenced_units[unit] = true;
        }
        std::vector<uint32_t> unit_remap(
            result->projected_units.size(), UINT32_MAX);
        std::vector<vbr_capture_projected_unit> retained_units;
        retained_units.reserve(result->projected_units.size());
        for (uint32_t i = 0; i < result->projected_units.size(); ++i) {
            if (!referenced_units[i]) {
                continue;
            }
            unit_remap[i] = uint32_t(retained_units.size());
            retained_units.push_back(std::move(result->projected_units[i]));
        }
        for (uint32_t & unit : result->unit_references) {
            unit = unit_remap[unit];
        }
        for (auto & proof : result->range_proofs) {
            if (proof.unit_index >= unit_remap.size() ||
                unit_remap[proof.unit_index] == UINT32_MAX) {
                return false;
            }
            proof.unit_index = unit_remap[proof.unit_index];
        }
        result->projected_units = std::move(retained_units);
        output = vbr_capture_manifest_assembly(std::move(result));
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}
