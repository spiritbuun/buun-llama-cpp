#include "llama-vbr-artifact-adopt.h"
#include "llama-bit-ops.h"

#include "common.h"
#include "speculative.h"
#include "llama-context.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-vbr-generation.h"
#include "llama-vbr-operation.h"
#include "llama-memory-tree.h"
#include "llama-vbr-artifact-catalog.h"
#include "llama-vbr-explicit-capture.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-upward.h"

#include "server-cache-authority.h"
#include "server-retention-sidecar.h"
#include "server-task.h"
#include "server-vbr-artifact-store.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", \
                         __FILE__, __LINE__, #cond); \
            ++failures; \
        } \
    } while (0)

static void test_epoch_capacity_preflight() {
    CHECK(vbr_artifact_epoch_capacity(0, 0, 0, 0));
    CHECK(!vbr_artifact_epoch_capacity(0, UINT64_MAX, 0, 0));
    CHECK(!vbr_artifact_epoch_capacity(0, 0, UINT64_MAX, 0));
    CHECK(vbr_artifact_epoch_capacity(UINT64_MAX, 0, 0, 0));
    CHECK(!vbr_artifact_epoch_capacity(UINT64_MAX, 0, 0, 1));
    CHECK(vbr_artifact_epoch_capacity(UINT64_MAX - 1, 0, 0, 1));
    CHECK(!vbr_artifact_epoch_capacity(UINT64_MAX - 1, 0, 0, 2));
}

// Friend-only harness view for post-adopt checks and deliberate live-degrade
// setup. Import inspection and reservation use the production public doors.
struct llama_kv_cache_vbr_epoch_test {
    static bool snapshot_sequence_rows(
            const llama_kv_cache * cache,
            llama_seq_id sequence,
            uint32_t rows,
            std::vector<uint8_t> & output) {
        output.clear();
        if (!cache || cache->v_cells.size() != 1 || rows == 0) {
            return false;
        }
        std::vector<uint32_t> physical(rows, UINT32_MAX);
        const auto & cells = cache->v_cells.front();
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            if (cells.seq_has(cell, sequence)) {
                const llama_pos position = cells.pos_get(cell);
                if (position >= 0 && uint64_t(position) < rows) {
                    if (physical[size_t(position)] != UINT32_MAX) {
                        return false;
                    }
                    physical[size_t(position)] = cell;
                }
            }
        }
        if (std::find(physical.begin(), physical.end(), UINT32_MAX) !=
                physical.end()) {
            return false;
        }
        try {
            for (const auto & layer : cache->layers) {
                for (const auto * tensor : { layer.k, layer.v }) {
                    if (!tensor || tensor->ne[1] <= 0) {
                        return false;
                    }
                    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
                    const size_t mark = output.size();
                    output.resize(mark + size_t(rows)*row_bytes);
                    for (uint32_t position = 0; position < rows; ++position) {
                        const uint32_t cell = physical[position];
                        if (cell >= uint64_t(tensor->ne[1]) ||
                            cell > SIZE_MAX/tensor->nb[1]) {
                            return false;
                        }
                        ggml_backend_tensor_get(
                            tensor, output.data()+mark+size_t(position)*row_bytes,
                            size_t(cell)*tensor->nb[1], row_bytes);
                    }
                }
            }
            return true;
        } catch (...) {
            output.clear();
            return false;
        }
    }

    static bool adopted_matches(
            const llama_kv_cache * cache,
            const vbr_artifact_package_view & package,
            const vbr_artifact_reference_manifest & manifest,
            llama_seq_id destination) noexcept {
        if (!cache || !cache->vbr_generation_tracker_get() ||
            !cache->vbr_generation_tracker_get()->stable()) {
            return false;
        }
        const auto controller = std::find_if(
            manifest.generation.controllers.begin(),
            manifest.generation.controllers.end(),
            [](const vbr_checkpoint_generation_controller & value) {
                return value.child_id == 0;
            });
        if (controller == manifest.generation.controllers.end() ||
            cache->vbr_generation_tracker_get()->lineage_identity() !=
                controller->lineage_uuid) {
            return false;
        }
        for (const auto & placement : manifest.stream_placements) {
            if (placement.child_id != 0 ||
                placement.stream_index >= cache->v_cells.size()) {
                continue;
            }
            const auto & cells = cache->v_cells[placement.stream_index];
            std::vector<uint32_t> owned;
            if (!cache->vbr_ownership_ ||
                !cache->vbr_ownership_->enumerate_owned(
                    placement.stream_index, destination, owned)) {
                return false;
            }
            std::vector<uint32_t> expected_owned;
            for (const auto & cell : placement.cells) {
                if (cell.physical_cell >= cells.size() ||
                    cells.pos_get(cell.physical_cell) != cell.logical_position ||
                    !cells.seq_has(cell.physical_cell, destination) ||
                    cells.ext_get(cell.physical_cell).x != cell.ext_x ||
                    cells.ext_get(cell.physical_cell).y != cell.ext_y) {
                    return false;
                }
                expected_owned.push_back(cell.physical_cell);
            }
            std::sort(expected_owned.begin(), expected_owned.end());
            if (owned != expected_owned) {
                return false;
            }
        }
        for (const auto & unit : package.units()) {
            const auto & descriptor = unit.descriptor;
            const size_t ikv = descriptor.logical_unit_id/2;
            const bool is_v = (descriptor.logical_unit_id & 1u) != 0;
            if (ikv >= cache->layers.size()) {
                return false;
            }
            const auto * tensor = is_v ? cache->layers[ikv].v : cache->layers[ikv].k;
            if (!tensor || tensor->type != descriptor.current_type) {
                return false;
            }
            const auto generation = cache->vbr_generation_tracker_get()->
                unit_generation(descriptor.logical_unit_id);
            const auto source_controller = std::find_if(
                manifest.generation.controllers.begin(),
                manifest.generation.controllers.end(),
                [&](const vbr_checkpoint_generation_controller & value) {
                    return value.child_id == descriptor.child_id;
                });
            if (source_controller == manifest.generation.controllers.end() ||
                descriptor.logical_unit_id >= source_controller->units.size() ||
                generation.domain !=
                    source_controller->units[descriptor.logical_unit_id].domain ||
                generation.current_type != descriptor.current_type) {
                return false;
            }
        }
        for (const auto & policy : manifest.controller_policy) {
            if (policy.child_id == 0 &&
                (cache->vbr_degrade_cursor_ != policy.cursor ||
                 std::any_of(cache->vbr_pools_.begin(), cache->vbr_pools_.end(),
                    [&](const llama_kv_cache::vbr_pool & pool) {
                        return pool.wm_cells != policy.wm_cells;
                    }))) {
                return false;
            }
        }
        return true;
    }

    static bool tracker_identity(
            const llama_kv_cache * cache,
            vbr_controller_instance_id & instance,
            vbr_lineage_uuid & lineage,
            uint64_t & generation,
            vbr_repr_transition & unit_transition) noexcept {
        const auto * tracker = cache ? cache->vbr_generation_tracker_get() : nullptr;
        if (!tracker || !tracker->stable() || tracker->unit_count() == 0) {
            return false;
        }
        instance = tracker->runtime_instance();
        lineage = tracker->lineage_identity();
        generation = tracker->controller_generation();
        unit_transition = tracker->unit_generation(0).last_transition;
        return true;
    }

    static vbr_generation_teardown_state tracker_teardown_state(
            const llama_kv_cache * cache) noexcept {
        const auto * tracker = cache ? cache->vbr_generation_tracker_get() : nullptr;
        return tracker ? tracker->teardown_state()
                       : vbr_generation_teardown_state::instance_owner_mismatch;
    }

    static bool representation_types(
            const llama_kv_cache * cache,
            const std::vector<ggml_type> & expected) noexcept {
        if (!cache || expected.size() != cache->layers.size()*2) {
            return false;
        }
        for (size_t layer = 0; layer < cache->layers.size(); ++layer) {
            if (!cache->layers[layer].k || !cache->layers[layer].v ||
                cache->layers[layer].k->type != expected[layer*2] ||
                cache->layers[layer].v->type != expected[layer*2+1]) {
                return false;
            }
        }
        return true;
    }

    static bool representation_bytes_equal(
            const llama_kv_cache * lhs,
            const llama_kv_cache * rhs,
            const vbr_artifact_reference_manifest & manifest) {
        if (!lhs || !rhs || lhs->layers.size() != rhs->layers.size()) {
            return false;
        }
        const auto placement = std::find_if(
            manifest.stream_placements.begin(),
            manifest.stream_placements.end(),
            [](const vbr_artifact_stream_placement & value) {
                return value.child_id == 0;
            });
        if (placement == manifest.stream_placements.end() ||
            placement->cells.empty()) {
            return false;
        }
        for (size_t layer = 0; layer < lhs->layers.size(); ++layer) {
            for (bool is_v : { false, true }) {
                const auto * a = is_v ? lhs->layers[layer].v :
                                       lhs->layers[layer].k;
                const auto * b = is_v ? rhs->layers[layer].v :
                                       rhs->layers[layer].k;
                if (!a || !b || a->type != b->type ||
                    a->ne[0] != b->ne[0]) {
                    return false;
                }
                const size_t row_bytes = ggml_row_size(a->type, a->ne[0]);
                std::vector<uint8_t> a_row(row_bytes);
                std::vector<uint8_t> b_row(row_bytes);
                for (const auto & cell : placement->cells) {
                    if (cell.physical_cell >= uint64_t(a->ne[1]) ||
                        cell.physical_cell >= uint64_t(b->ne[1]) ||
                        cell.physical_cell > SIZE_MAX/a->nb[1] ||
                        cell.physical_cell > SIZE_MAX/b->nb[1]) {
                        return false;
                    }
                    ggml_backend_tensor_get(
                        a, a_row.data(), cell.physical_cell*a->nb[1],
                        row_bytes);
                    ggml_backend_tensor_get(
                        b, b_row.data(), cell.physical_cell*b->nb[1],
                        row_bytes);
                    if (a_row != b_row) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    static bool degrade_to(
            llama_kv_cache * cache, ggml_type target) noexcept {
        if (!cache || cache->layers.empty()) {
            return false;
        }
        const auto at_target = [&]() {
            return std::all_of(
                cache->layers.begin(), cache->layers.end(),
                [&](const auto & layer) {
                    return layer.k && layer.v &&
                           layer.k->type == target &&
                           layer.v->type == target;
                });
        };
        for (size_t i = 0; i <= cache->vbr_degrade_order_.size(); ++i) {
            if (at_target()) {
                return cache->vbr_capture_settle();
            }
            uint32_t watermark = 0;
            for (const auto & pool : cache->vbr_pools_) {
                watermark = std::max(watermark, pool.wm_cells);
            }
            if (watermark == 0 ||
                cache->vbr_degrade_next(watermark) !=
                    llama_kv_cache::vbr_degrade_result::applied) {
                return false;
            }
        }
        return false;
    }

    // Live-schedule target constructor: the source schedule is produced by the
    // shipped controller, then a separate context repeats it.  A normal empty
    // boundary deliberately full-resets tiers to F16, so it cannot construct
    // the same-tier native-import target.  Retire the repeated context's VMM
    // backing while preserving only its controller-selected type vector and
    // cursor.  This friend lives in the test binary; no production trigger or
    // environment door can invoke it.
    static bool make_construction_empty_preserve_tiers(
            llama_kv_cache * cache) noexcept {
        if (!cache || !cache->vbr_vmm_active() ||
            std::any_of(cache->v_cells.begin(), cache->v_cells.end(),
                [](const llama_kv_cells & cells) {
                    return cells.get_used() != 0;
                })) {
            return false;
        }
        cache->vbr_flush_deferred_unmaps();
        for (auto & pool : cache->vbr_pools_) {
            if (!pool.vmm || !pool.be || pool.gran == 0) {
                return false;
            }
            if (pool.backend) {
                ggml_backend_synchronize(pool.backend);
            }
            pool.be->sync_device(pool.device);
            for (size_t layer = 0; layer < cache->layers.size(); ++layer) {
                for (bool is_v : { false, true }) {
                    auto & extent = is_v ? pool.v[layer] : pool.k[layer];
                    if (!extent.t) {
                        continue;
                    }
                    const size_t logical =
                        size_t(ggml_row_size(GGML_TYPE_F16, extent.t->ne[0]))*
                        size_t(extent.t->ne[1])*size_t(extent.t->ne[2]);
                    const size_t span = GGML_PAD(logical, pool.gran);
                    pool.be->vmm_pool_unmap(
                        pool.vmm, extent.byte_off, span);
                    extent.stash_valid = 0;
                }
            }
            pool.wm_cells = 0;
        }
        return true;
    }
};

struct model_adoption_harness_target {
    llama_memory_i * memory = nullptr;
    llama_cache_acct_ledger * ledger = nullptr;
    std::array<uint8_t, 32> downward_tree_digest = {};
    const vbr_artifact_package_view * package = nullptr;
    const std::vector<llama_vbr_artifact_domain_binding> * bindings = nullptr;
    const vbr_import_schedule_quote * schedule_quote = nullptr;
    const void * representation_context = nullptr;
    vbr_explicit_capture_request::representation_identity_fn
        representation_identity = nullptr;
    llama_seq_id destination = 0;

    static uint64_t accounting_serial(const void * context) noexcept {
        const auto * self = static_cast<const model_adoption_harness_target *>(context);
        return self && self->ledger ? self->ledger->snapshot().serial : 0;
    }
    static uint64_t policy_serial(const void * context) noexcept {
        const auto * self = static_cast<const model_adoption_harness_target *>(context);
        return self && self->memory
            ? vbr_explicit_import_policy_epoch(*self->memory) : 0;
    }
    static bool transform_tree(
            const void * context,
            std::array<uint8_t, 32> & output) noexcept {
        const auto * self = static_cast<const model_adoption_harness_target *>(context);
        if (!self || !self->memory || !self->package ||
            !self->bindings || !self->schedule_quote) {
            return false;
        }
        return vbr_explicit_import_transform_projection_recheck(
                *self->memory, self->destination, *self->package,
                *self->bindings, *self->schedule_quote,
                self->representation_context, self->representation_identity,
                output) && output == self->downward_tree_digest;
    }
    static bool recheck(
            const void * context,
            const vbr_target_empty_fingerprint & expected) noexcept {
        const auto * self = static_cast<const model_adoption_harness_target *>(context);
        return self && self->memory && self->ledger &&
            expected.accounting_serial == self->ledger->snapshot().serial &&
            vbr_explicit_import_target_recheck(
                *self->memory, self->destination, expected);
    }
};

struct model_adoption_representation_drift_context {
    enum class kind : uint8_t {
        codec_version = 0,
        codebook_digest,
        rotation_digest,
    } drift = kind::codec_version;
    const vbr_explicit_representation_policy * policy = nullptr;
    int32_t source_type = GGML_TYPE_COUNT;
};

static bool model_adoption_representation_identity_with_source_drift(
        const void * context,
        int32_t current_type,
        bool value_side,
        int32_t meansub_model_id,
        vbr_explicit_representation_identity & output) noexcept {
    const auto * drift =
        static_cast<const model_adoption_representation_drift_context *>(context);
    if (!drift || !drift->policy ||
        !vbr_explicit_capture_representation_identity(
            drift->policy, current_type, value_side,
            meansub_model_id, output)) {
        return false;
    }
    if (current_type == drift->source_type) {
        switch (drift->drift) {
            case model_adoption_representation_drift_context::kind::codec_version:
                ++output.codec_version;
                break;
            case model_adoption_representation_drift_context::kind::codebook_digest:
                output.codebook_digest[0] ^= 0x5a;
                break;
            case model_adoption_representation_drift_context::kind::rotation_digest:
                output.rotation_digest[0] ^= 0xa5;
                break;
        }
    }
    return true;
}

static bool model_adoption_reserve_transform(
        void * context,
        const std::vector<vbr_validated_child_plan> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept {
    auto * memory = static_cast<llama_memory_i *>(context);
    return memory && vbr_explicit_import_reserve_transform(
        *memory, plans, ledger, budget, output);
}

static bool model_adoption_owner_token(
        const void * context,
        const void * token,
        const llama_memory_i * target) noexcept {
    return context != nullptr && token == context && target == context;
}

static bool model_adoption_decode(
        llama_context * context,
        const std::vector<llama_token> & tokens,
        llama_pos first_position) {
    if (!context || tokens.empty()) {
        return false;
    }
    const size_t n_batch = llama_n_batch(context);
    if (n_batch == 0) {
        return false;
    }
    for (size_t offset = 0; offset < tokens.size();) {
        const size_t count = std::min(n_batch, tokens.size()-offset);
        llama_batch batch = llama_batch_init(int32_t(count), 0, 1);
        for (size_t i = 0; i < count; ++i) {
            common_batch_add(batch, tokens[offset+i],
                             first_position+llama_pos(offset+i), { 0 },
                             offset+i+1 == tokens.size());
        }
        const bool ok = llama_decode(context, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            return false;
        }
        offset += count;
    }
    return true;
}

static bool model_adoption_initialize_accounting(
        llama_cache_acct_ledger & ledger,
        const std::vector<llama_cache_acct_resource_domain> & domains) {
    std::vector<llama_cache_acct_completeness_requirement> required;
    for (const auto & domain : domains) {
        required.push_back({
            domain,
            domain.residency == llama_cache_acct_residency::device
                ? llama_cache_acct_producer::live_memory
                : llama_cache_acct_producer::retention_sidecar,
        });
    }
    if (!ledger.configure_required_producers(
            required.data(), required.size())) {
        return false;
    }
    for (const auto & domain : domains) {
        for (uint8_t raw = 0;
             raw < uint8_t(llama_cache_acct_category::_count); ++raw) {
            const auto category = llama_cache_acct_category(raw);
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                ledger.gauge_set(category, domain, measure, 0);
            }
        }
        const auto producer =
            domain.residency == llama_cache_acct_residency::device
                ? llama_cache_acct_producer::live_memory
                : llama_cache_acct_producer::retention_sidecar;
        if (!ledger.certify_complete(domain, producer)) {
            return false;
        }
    }
    return true;
}

static llama_cache_budget_config model_adoption_budget(
        ggml_backend_dev_t device,
        const llama_cache_acct_resource_domain & domain) {
    llama_cache_budget_config budget;
    llama_cache_budget_device_input input;
    input.backend_device = device;
    input.domain = domain;
    input.physical_total = 1ull << 40;
    // The catalog's device-resident artifact receipts participate in capacity accounting.
    // Keep the synthetic physical sample internally consistent with that
    // nonzero resident baseline; total==free would correctly make fits()
    // unavailable once a live capture has been charged.
    input.physical_free = (1ull << 40) - (1ull << 30);
    input.phys_state = llama_cache_budget_capacity_state::known;
    input.current_compute_allocated = 0;
    input.configured_compute_reserve = 0;
    input.compute_state = llama_cache_budget_capacity_state::known;
    input.cache_cap_state = llama_cache_budget_capacity_state::unbounded;
    budget.devices.push_back(input);
    budget.host.pageable_state = llama_cache_budget_capacity_state::unbounded;
    budget.host.pinned_state = llama_cache_budget_capacity_state::unbounded;
    budget.host.total_state = llama_cache_budget_capacity_state::unbounded;
    return budget;
}

struct occupied_budget_source {
    llama_cache_budget_config budget;
};

static bool occupied_sample_budget(
        void * opaque, llama_cache_budget_config & output) noexcept {
    try {
        output = static_cast<occupied_budget_source *>(opaque)->budget;
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

static vbr_checkpoint_generation_controller source_controller(
        vbr_lineage_uuid lineage) {
    vbr_checkpoint_generation_controller source;
    source.child_id = 0;
    source.dependency_mode =
        checkpoint_child_dependency_mode::payload_complete;
    source.lineage_uuid = lineage;
    source.global_generation = 9;
    source.units.push_back({
        17, GGML_TYPE_F16, GGML_TYPE_F16,
        vbr_repr_domain::full, 0, vbr_repr_transition::initial,
    });
    vbr_checkpoint_generation_stream stream;
    stream.stream_index = 0;
    stream.dependency_seq_id = 0;
    stream.computation_frontier = 11;
    stream.captured_dependency_count = 1;
    vbr_generation_page_ref page;
    page.page_index = 0;
    page.captured_page_gen = 23;
    page.covered_mask[0] = uint64_t(1) << 5;
    stream.pages.push_back(page);
    source.streams.push_back(stream);
    return source;
}

static vbr_artifact_stream_placement one_cell_placement() {
    vbr_artifact_stream_placement placement;
    placement.child_id = 0;
    placement.stream_index = 0;
    placement.source_sequence = 0;
    placement.cells.push_back({ 5, 10, 1, 1 });
    return placement;
}

static vbr_operation_binding import_binding(
        vbr_controller_instance_id instance) {
    vbr_operation_binding binding;
    binding.kind = vbr_operation_kind::state_import;
    binding.child_phase = vbr_operation_phase::mutate;
    CHECK(vbr_binding_add_instance_target(
        binding, vbr_operation_kind::state_import,
        vbr_operation_class::state_api, instance, VBR_STREAM_ANY,
        1, 0, std::numeric_limits<llama_pos>::max()));
    return binding;
}

static void install_and_check(
        vbr_generation_tracker & target,
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        bool native) {
    const auto runtime = target.runtime_instance();
    const auto target_lineage = target.lineage_identity();
    vbr_tracker_import_image image;
    CHECK(target.prepare_import_image(
        plan, source, 1, { one_cell_placement() }, image));
    CHECK(image.ready());
    CHECK(image.stable());

    auto binding = import_binding(runtime);
    vbr_scoped_operation operation(binding);
    CHECK(bool(operation));
    CHECK(target.import_image_installable(image, operation.id()));
    target.install_import_image_swap(image);
    CHECK(!image.ready());
    CHECK(target.runtime_instance() == runtime);
    CHECK(vbr_controller_instance_owned_by(runtime, &target));
    CHECK(target.stable());
    CHECK(target.controller_generation() == plan.global_generation);
    CHECK(target.lineage_identity() ==
          (native ? source.lineage_uuid : target_lineage));
    CHECK(target.page_generation(0, 0) == (native ? 23u : 1u));
    CHECK(target.dependency_generation(0, 5) ==
          (native ? 23u : 1u));
    CHECK(target.membership_generation(0, 5) ==
          (native ? 23u : 1u));
    CHECK(target.last_membership_seq(0, 5) == 1);
    const uint16_t import_provenance =
        uint16_t(vbr_mutation_family::import) |
        (uint16_t(vbr_operation_class::state_api) << 8);
    CHECK(target.dependency_provenance(0, 5) == import_provenance);
    CHECK(target.membership_provenance(0, 5) == import_provenance);
    const auto dep = target.dependency_extent(0, 5);
    const auto mem = target.membership_extent(0, 5);
    const auto * dep_extent = target.extent_store().lookup_committed(dep);
    const auto * mem_extent = target.extent_store().lookup_committed(mem);
    CHECK(dep_extent != nullptr);
    CHECK(mem_extent != nullptr);
    if (dep_extent) {
        CHECK(dep_extent->family == vbr_mutation_family::import);
        CHECK(dep_extent->operation_class == vbr_operation_class::state_api);
        CHECK(dep_extent->stream == 0);
        CHECK(dep_extent->seq_id == 1);
        CHECK(dep_extent->p0 == 0);
        CHECK(dep_extent->p1 == 11);
    }
    if (mem_extent) {
        CHECK(mem_extent->family == vbr_mutation_family::import);
        CHECK(mem_extent->operation_class == vbr_operation_class::state_api);
        CHECK(mem_extent->stream == 0);
        CHECK(mem_extent->seq_id == 1);
        CHECK(mem_extent->p0 == 0);
        CHECK(mem_extent->p1 == 11);
    }
    CHECK(target.unit_generation(0).repr_gen == plan.units[0].repr_gen);
    CHECK(target.unit_generation(0).last_transition ==
          plan.units[0].last_transition);
    operation.close(vbr_operation_outcome::committed);
}

static void test_native_tracker_image_preserves_lineage_and_runtime() {
    const vbr_lineage_uuid source_lineage { 0x1234, 0x5678 };
    const auto source = source_controller(source_lineage);
    vbr_generation_tracker target(1, 256, 1,
                                  vbr_lineage_uuid { 0xa1, 0xb2 });
    CHECK(target.active());
    CHECK(target.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full));

    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::native_clone;
    plan.lineage_uuid = source_lineage;
    plan.target_instance = target.runtime_instance();
    plan.global_generation = source.global_generation;
    plan.units = source.units;
    install_and_check(target, plan, source, true);
}

static void test_real_tracker_import_settle_and_teardown() {
    const auto source = source_controller(
        vbr_lineage_uuid { 0x1234, 0x5678 });
    auto target = std::make_unique<vbr_generation_tracker>(
        1, 256, 1, vbr_lineage_uuid { 0xa3, 0xb4 });
    CHECK(target->active());
    CHECK(target->initialize_unit(
        0, GGML_TYPE_F16, vbr_repr_domain::full));

    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::native_clone;
    plan.lineage_uuid = source.lineage_uuid;
    plan.target_instance = target->runtime_instance();
    plan.global_generation = source.global_generation;
    plan.units = source.units;

    vbr_tracker_import_image image;
    CHECK(target->prepare_import_image(
        plan, source, 1, { one_cell_placement() }, image));
    auto binding = import_binding(target->runtime_instance());
    vbr_scoped_operation operation(binding);
    CHECK(bool(operation));
    const int32_t recovery = vbr_recovery_reserve(
        operation.id(), target->runtime_instance());
    CHECK(recovery >= 0);
    CHECK(target->import_image_installable(image, operation.id()));
    target->install_import_image_swap(image);
    CHECK(target->stable());
    CHECK(vbr_recovery_release_unused(recovery, operation.id()));
    CHECK(operation.close(vbr_operation_outcome::committed));

    // Model the first post-import append under a normal decode operation.  Its
    // event must settle mutation parity before the imported tracker dies.
    vbr_operation_binding decode;
    decode.kind = vbr_operation_kind::decode;
    decode.child_phase = vbr_operation_phase::mutate;
    CHECK(vbr_binding_add_instance_target(
        decode, vbr_operation_kind::decode,
        vbr_operation_class::ordinary_decode,
        target->runtime_instance(), 0, 1, 11,
        std::numeric_limits<llama_pos>::max()));
    vbr_scoped_operation decode_operation(decode);
    CHECK(bool(decode_operation));
    const int32_t decode_recovery = vbr_recovery_reserve(
        decode_operation.id(), target->runtime_instance());
    CHECK(decode_recovery >= 0);
    {
        auto event = target->begin_event(
            vbr_mutation_registrant::apply_ubatch_append,
            vbr_operation_class::ordinary_decode, 0,
            vbr_generation_stamp_kind::dependency,
            decode_operation.id());
        CHECK(bool(event));
        CHECK(target->stamp_cell(event, 6, 1, 11));
    }
    CHECK(target->stable());
    vbr_operation_binding live_decode;
    CHECK(vbr_operation_registry_binding(
        decode_operation.id(), live_decode));
    CHECK(live_decode.kind == vbr_operation_kind::decode);
    CHECK(live_decode.n_targets == 1);
    CHECK(live_decode.targets[0].operation_class ==
          vbr_operation_class::ordinary_decode);
    CHECK(live_decode.targets[0].instance_id ==
          target->runtime_instance());
    CHECK(target->teardown_state() ==
          vbr_generation_teardown_state::operation_live);
    CHECK(vbr_recovery_release_unused(
        decode_recovery, decode_operation.id()));
    CHECK(decode_operation.close(vbr_operation_outcome::committed));
    CHECK(target->teardown_state() ==
          vbr_generation_teardown_state::clean);
    target.reset();
}

static void test_live_rebased_tracker_image_is_fresh() {
    const auto source = source_controller(vbr_lineage_uuid { 0x1234, 0x5678 });
    const vbr_lineage_uuid target_lineage { 0xc1, 0xd2 };
    vbr_generation_tracker target(1, 256, 1, target_lineage);
    CHECK(target.active());
    CHECK(target.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full));

    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::whole_import;
    plan.lineage_uuid = target_lineage;
    plan.target_instance = target.runtime_instance();
    plan.global_generation = 1;
    plan.units.push_back({
        1, GGML_TYPE_F16, GGML_TYPE_F16,
        vbr_repr_domain::full, 0, vbr_repr_transition::whole_import,
    });
    install_and_check(target, plan, source, false);
}

static void test_native_tracker_rejects_uncovered_cell_and_tuple_splice() {
    const vbr_lineage_uuid source_lineage { 0x1234, 0x5678 };
    const auto source = source_controller(source_lineage);
    vbr_generation_tracker target(1, 256, 1,
                                  vbr_lineage_uuid { 0x91, 0x92 });
    CHECK(target.initialize_unit(0, GGML_TYPE_F16, vbr_repr_domain::full));
    vbr_tracker_install_child plan;
    plan.child_id = 0;
    plan.transition = vbr_tracker_install_transition::native_clone;
    plan.lineage_uuid = source_lineage;
    plan.target_instance = target.runtime_instance();
    plan.global_generation = source.global_generation;
    plan.units = source.units;

    auto foreign = one_cell_placement();
    foreign.cells[0].physical_cell = 6;
    vbr_tracker_import_image image;
    CHECK(!target.prepare_import_image(plan, source, 1, { foreign }, image));
    CHECK(!image.ready());
    CHECK(target.extent_store().live_entries() == 0);

    plan.global_generation++;
    CHECK(!target.prepare_import_image(
        plan, source, 1, { one_cell_placement() }, image));
    CHECK(!image.ready());
    CHECK(target.extent_store().live_entries() == 0);
}

static void test_closed_vocabularies() {
    for (uint8_t i = 0; i < uint8_t(vbr_adopt_phase::_count); ++i) {
        CHECK(std::string(vbr_adopt_phase_name(vbr_adopt_phase(i))) !=
              "invalid");
    }
    for (uint8_t i = 0; i < uint8_t(vbr_adopt_status::_count); ++i) {
        CHECK(std::string(vbr_adopt_status_name(vbr_adopt_status(i))) !=
              "invalid");
    }
    for (uint8_t i = 0;
         i < uint8_t(vbr_adopt_recovery_outcome::_count); ++i) {
        CHECK(std::string(vbr_adopt_recovery_outcome_name(
                  vbr_adopt_recovery_outcome(i))) != "invalid");
    }
    for (uint8_t i = 0;
         i < uint8_t(vbr_downward_reserve_status::_count); ++i) {
        CHECK(std::string(vbr_downward_reserve_status_name(
                  vbr_downward_reserve_status(i))) != "invalid");
    }
    vbr_adopt_stage_result native_stage_default;
    CHECK(native_stage_default.transform_status ==
          vbr_downward_reserve_status::not_attempted);
    CHECK(std::string(vbr_adopt_phase_name(vbr_adopt_phase::_count)) ==
          "invalid");
    CHECK(std::string(vbr_adopt_status_name(vbr_adopt_status::_count)) ==
          "invalid");
}

static bool fake_target_empty(const void * context) noexcept {
    return context && *static_cast<const bool *>(context);
}

static void test_complete_tree_barrier_fail_closed() {
    auto * attention0 = reinterpret_cast<llama_kv_cache *>(uintptr_t(0x1010));
    auto * attention1 = reinterpret_cast<llama_kv_cache *>(uintptr_t(0x2020));
    auto * recurrent = reinterpret_cast<llama_memory_recurrent *>(uintptr_t(0x3030));
    const std::vector<vbr_adopt_expected_attention> expected = {
        { 0, attention0 }, { 1, attention1 },
    };
    std::vector<llama_memory_tree_child> live = {
        { 0, attention0, nullptr,
          checkpoint_child_dependency_mode::payload_complete },
    };
    CHECK(vbr_adopt_check_complete_tree(expected, live, {}) ==
          vbr_adopt_status::target_drift);

    live.push_back({ 1, attention1, recurrent,
                     checkpoint_child_dependency_mode::payload_complete });
    CHECK(vbr_adopt_check_complete_tree(expected, live, {}) ==
          vbr_adopt_status::required_companion_unavailable);

    bool empty = false;
    vbr_companion_adoption_provider provider;
    provider.kind = vbr_artifact_companion_kind::recurrent;
    provider.target_cookie = recurrent;
    provider.context = &empty;
    provider.target_empty = fake_target_empty;
    CHECK(vbr_adopt_check_complete_tree(expected, live, { provider }) ==
          vbr_adopt_status::target_drift);

    empty = true;
    CHECK(vbr_adopt_check_complete_tree(expected, live, { provider }) ==
          vbr_adopt_status::adopted);
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, provider }) ==
          vbr_adopt_status::required_companion_unavailable);

    auto * qsa = reinterpret_cast<llama_memory_hybrid_idx *>(
        uintptr_t(0x4040));
    live[0].qsa_index_owner = qsa;
    CHECK(vbr_adopt_check_complete_tree(expected, live, { provider }) ==
          vbr_adopt_status::required_companion_unavailable);
    vbr_companion_adoption_provider qsa_provider;
    qsa_provider.kind = vbr_artifact_companion_kind::qsa_index;
    qsa_provider.target_cookie = qsa;
    bool qsa_empty_after_prepare = false;
    qsa_provider.context = &qsa_empty_after_prepare;
    qsa_provider.target_empty = fake_target_empty;
    qsa_provider.attention_child_id = 0;
    // QSA prepares reversibly in-place, so it is expected to be non-empty at
    // the complete-tree barrier.  Its later recheck authenticates that state.
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, qsa_provider }) ==
          vbr_adopt_status::adopted);
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, qsa_provider }, true) ==
          vbr_adopt_status::adopted);
    CHECK(vbr_adopt_check_complete_tree(
              expected, live, { provider, qsa_provider, qsa_provider }) ==
          vbr_adopt_status::required_companion_unavailable);
}

namespace adoption_fixture {

struct byte_source {
    std::vector<uint8_t> bytes;

    static bool read(const void * context, uint64_t offset,
                     uint8_t * out, size_t size) noexcept {
        const auto * self = static_cast<const byte_source *>(context);
        if (!self || offset > self->bytes.size() ||
            size > self->bytes.size()-size_t(offset)) {
            return false;
        }
        std::memcpy(out, self->bytes.data()+offset, size);
        return true;
    }

    vbr_artifact_byte_source source() const noexcept {
        return { bytes.size(), this, read };
    }
};

static std::array<uint8_t, 32> marker(uint8_t value) {
    std::array<uint8_t, 32> out;
    out.fill(value);
    return out;
}

static vbr_generation_page_ref page(uint64_t mask) {
    vbr_generation_page_ref out;
    out.page_index = 0;
    out.captured_page_gen = 7;
    out.covered_mask[0] = mask;
    return out;
}

struct storage {
    std::array<byte_source, 4> payload {
        byte_source { { 0x10, 0x11, 0x12, 0x13, 0x14 } },
        byte_source { { 0x20, 0x21, 0x22, 0x23, 0x24 } },
        byte_source { { 0x30, 0x31, 0x32, 0x33, 0x34 } },
        byte_source { { 0x40, 0x41, 0x42, 0x43, 0x44 } },
    };
    std::array<byte_source, 4> stash {
        byte_source { { 0x51, 0x52, 0x53, 0x54, 0x55,
                        0x56, 0x57, 0x58, 0x59, 0x5a } },
        byte_source { { 0x61, 0x62, 0x63, 0x64, 0x65,
                        0x66, 0x67, 0x68, 0x69, 0x6a } },
        byte_source { { 0x71, 0x72, 0x73, 0x74, 0x75,
                        0x76, 0x77, 0x78, 0x79, 0x7a } },
        byte_source { { 0x81, 0x82, 0x83, 0x84, 0x85,
                        0x86, 0x87, 0x88, 0x89, 0x8a } },
    };
    byte_source companion { { 0x90, 0x91, 0x92 } };
};

static vbr_artifact_portable_topology topology() {
    llama_cache_acct_shard_topology out;
    const std::vector<std::string> devices = {
        "g2-device-a", "g2-device-b",
    };
    const float weights[] = { 0.5f, 0.5f };
    CHECK(llama_cache_acct_build_shard_topology(
        devices, LLAMA_SPLIT_MODE_TENSOR, 0, weights, out));
    return out;
}

static vbr_artifact_shard_descriptor shard(
        uint32_t index, const byte_source & source,
        uint64_t row_bytes = 1) {
    vbr_artifact_shard_descriptor out;
    out.shard_index = index;
    out.topology_index = 0;
    out.device_ordinal = uint16_t(index);
    out.logical_offset = 0;
    out.row_count = 5;
    out.column_count = 1;
    out.row_bytes = row_bytes;
    out.payload_bytes = 5*row_bytes;
    out.payload = source.source();
    return out;
}

static vbr_artifact_package package(
        storage & bytes, bool companion,
        ggml_type source_type = GGML_TYPE_TURBO8_0,
        uint8_t promote_hops = 0,
        vbr_artifact_clean_stash_state stash_state =
            vbr_artifact_clean_stash_state::absent_at_source,
        bool partial_stash = false,
        uint32_t child_count = 2,
        uint64_t row_bytes = 1,
        vbr_artifact_companion_kind companion_kind =
            vbr_artifact_companion_kind::recurrent) {
    for (auto & payload : bytes.payload) {
        const size_t old = payload.bytes.size();
        payload.bytes.resize(size_t(5*row_bytes));
        for (size_t i = old; i < payload.bytes.size(); ++i) {
            payload.bytes[i] = uint8_t(0x30 + (i & 0x3f));
        }
    }
    vbr_artifact_package out;
    out.topologies.push_back(topology());
    auto & manifest = out.manifest;
    manifest.identity_policy_order_digest = marker(0x71);
    manifest.identity.execution_identity = "g2:exec";
    manifest.identity.adapter_config_identity = "g2:adapter";
    manifest.identity.media_content_identity = "g2:media";
    manifest.identity.sequence_epoch = 1;
    manifest.identity.token_count = 5;
    manifest.identity.next_position = 5;
    manifest.token_block.tokens = { 1, 2, 3, 4, 5 };
    manifest.generation.version = 1;
    manifest.generation.status = vbr_checkpoint_generation_status::complete;
    manifest.generation.identity_policy_order_digest =
        manifest.identity_policy_order_digest;
    manifest.consistency.kind =
        vbr_artifact_consistency_kind::capture_exact;

    for (uint32_t child_id = 0; child_id < child_count; ++child_id) {
        vbr_artifact_unit_blob blob;
        auto & descriptor = blob.descriptor;
        descriptor.child_id = child_id;
        descriptor.logical_unit_id = 0;
        descriptor.lineage_uuid = {
            UINT64_C(0x1010101010101010)+child_id,
            UINT64_C(0x2020202020202020)+child_id,
        };
        descriptor.repr_gen = 17+child_id;
        descriptor.current_type = source_type;
        descriptor.last_source_type = source_type;
        descriptor.promote_hops = promote_hops;
        descriptor.last_transition = vbr_repr_transition::initial;
        descriptor.representation.kind =
            vbr_artifact_representation_kind::approximate;
        descriptor.representation.codec_id = 0x5438;
        descriptor.representation.codec_version = 1;
        descriptor.representation.reference_digest = marker(0x50+child_id);
        descriptor.side = vbr_artifact_side::key;
        descriptor.n_stream = 1;
        descriptor.unified = true;
        descriptor.wm_cells = 5;
        descriptor.rank = 2;
        descriptor.dimensions = { 5, 2, 0, 0 };
        descriptor.row_alignment = 1;
        descriptor.row_codec_version = 1;
        descriptor.codebook_digest = marker(0x60+child_id);
        descriptor.rotation_digest = marker(0x62+child_id);
        descriptor.meansub_digest = marker(0x64+child_id);
        descriptor.meansub_model_id = 7;
        descriptor.meansub_layer = int32_t(child_id);
        descriptor.meansub_baked = true;
        descriptor.shards = {
            shard(0, bytes.payload[child_id*2], row_bytes),
            shard(1, bytes.payload[child_id*2+1], row_bytes),
        };
        descriptor.clean_stash_state = stash_state;
        if (stash_state == vbr_artifact_clean_stash_state::present) {
            descriptor.clean_stash.valid_rows = 5;
            descriptor.clean_stash.domain = vbr_repr_domain::tapped;
            descriptor.clean_stash.layout = vbr_artifact_layout::row_major;
            descriptor.clean_stash.row_count = 5;
            descriptor.clean_stash.column_count = 2;
            descriptor.clean_stash.row_bytes = 4;
            descriptor.clean_stash.shards = {
                shard(0, bytes.stash[child_id*2]),
                shard(1, bytes.stash[child_id*2+1]),
            };
            for (auto & stash_shard : descriptor.clean_stash.shards) {
                stash_shard.row_bytes = 2;
                stash_shard.payload_bytes = 10;
            }
            descriptor.clean_stash.shards[1].logical_offset = 1;
        }
        out.unit_blobs.push_back(blob);

        vbr_checkpoint_generation_controller controller;
        controller.child_id = child_id;
        controller.dependency_mode =
            checkpoint_child_dependency_mode::live_guarded;
        controller.lineage_uuid = descriptor.lineage_uuid;
        controller.global_generation = 5+child_id;
        controller.units.push_back({
            descriptor.repr_gen, descriptor.current_type,
            descriptor.last_source_type,
            vbr_downward_tier_domain(source_type),
            descriptor.promote_hops, descriptor.last_transition,
        });
        vbr_checkpoint_generation_stream stream;
        stream.stream_index = 0;
        stream.dependency_seq_id = llama_seq_id(child_id);
        stream.computation_frontier = 5;
        stream.captured_dependency_count = 5;
        stream.pages.push_back(page(0x1f));
        controller.streams.push_back(stream);
        manifest.generation.controllers.push_back(controller);

        vbr_artifact_controller_policy policy;
        policy.child_id = child_id;
        policy.dependency_mode = controller.dependency_mode;
        policy.degrade_order_digest = marker(0x72);
        policy.policy_digest = marker(0x73);
        policy.floor_type = GGML_TYPE_TURBO8_0;
        policy.n_stream = 1;
        policy.unified = true;
        policy.wm_cells = 5;
        policy.current_type_vector_digest = marker(0x74);
        policy.completed_wave = true;
        manifest.controller_policy.push_back(policy);

        vbr_artifact_stream_placement placement;
        placement.child_id = child_id;
        placement.stream_index = 0;
        placement.source_sequence = llama_seq_id(child_id);
        placement.computation_frontier = 5;
        for (uint32_t cell = 0; cell < 5; ++cell) {
            placement.cells.push_back({
                cell, llama_pos(cell), uint16_t(10+cell),
                uint16_t(20+cell),
            });
        }
        manifest.stream_placements.push_back(placement);

        vbr_artifact_unit_reference reference;
        reference.lineage_uuid = descriptor.lineage_uuid;
        reference.logical_unit_id = 0;
        reference.repr_gen = descriptor.repr_gen;
        reference.authorized_stream_refs = { 0 };
        if (stash_state == vbr_artifact_clean_stash_state::present) {
            reference.has_stash_reference = true;
            reference.stash_reference.valid_rows = 5;
            reference.stash_reference.domain = vbr_repr_domain::tapped;
            reference.stash_reference.row_count = 5;
            reference.stash_reference.column_count = 2;
            reference.stash_reference.row_bytes = 4;
            reference.stash_reference.captured_sink_count =
                partial_stash ? 4 : 5;
            reference.stash_reference.covered_sink_pages = {
                page(partial_stash ? 0x0f : 0x1f),
            };
        }
        manifest.unit_references.push_back(reference);
    }

    const vbr_artifact_portable_domain device0 {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology, 0, 0,
    };
    const vbr_artifact_portable_domain device1 {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology, 0, 1,
    };
    const vbr_artifact_portable_domain host {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX, UINT16_MAX,
    };
    manifest.accounting = {
        { vbr_artifact_accounting_role::unit_payload,
          device0, 5*row_bytes*child_count, 5*row_bytes*child_count,
          llama_cache_acct_attr_kind::artifact },
        { vbr_artifact_accounting_role::unit_payload,
          device1, 5*row_bytes*child_count, 5*row_bytes*child_count,
          llama_cache_acct_attr_kind::artifact },
        { vbr_artifact_accounting_role::descriptor_metadata,
          host, 512, 512, llama_cache_acct_attr_kind::artifact },
        { vbr_artifact_accounting_role::reference_metadata,
          host, 256, 256, llama_cache_acct_attr_kind::artifact },
    };
    if (stash_state == vbr_artifact_clean_stash_state::present) {
        manifest.accounting.push_back({
            vbr_artifact_accounting_role::clean_stash_payload,
            device0, 10*child_count, 10*child_count,
            llama_cache_acct_attr_kind::artifact,
        });
        manifest.accounting.push_back({
            vbr_artifact_accounting_role::clean_stash_payload,
            device1, 10*child_count, 10*child_count,
            llama_cache_acct_attr_kind::artifact,
        });
    }
    if (companion) {
        vbr_artifact_companion_payload row;
        row.kind = companion_kind;
        row.format_version = 1;
        row.build_identity_digest = marker(0xa1);
        row.domain = host;
        row.payload_bytes = bytes.companion.bytes.size();
        row.payload = bytes.companion.source();
        out.companions.push_back(row);
        manifest.accounting.push_back({
            companion_kind == vbr_artifact_companion_kind::recurrent
                ? vbr_artifact_accounting_role::recurrent_payload
                : vbr_artifact_accounting_role::typed_accelerator_payload,
            host, row.payload_bytes, row.payload_bytes,
            llama_cache_acct_attr_kind::artifact,
        });
    }
    return out;
}

class parsed_companion final : public vbr_parsed_companion_image {
  public:
    explicit parsed_companion(vbr_artifact_companion_kind kind)
        : kind_(kind) {}

    vbr_artifact_companion_kind kind() const noexcept override {
        return kind_;
    }
    uint32_t format_version() const noexcept override { return 1; }

  private:
    vbr_artifact_companion_kind kind_;
};

struct child_state {
    uint32_t child_id = UINT32_MAX;
    vbr_controller_instance_id instance;
    bool armed = false;
    bool image_ready = false;
    bool published = false;
    bool receipts = false;
    std::shared_ptr<void> receipt_owner;
    uint32_t mapped = 0;
    uint32_t unmapped = 0;
    std::vector<uint32_t> mapped_ranges;
    std::vector<uint32_t> unmapped_ranges;
    uint64_t h2d_bytes = 0;
    std::set<uint32_t> completed_units;
    std::array<uint8_t, 24> destination = {};
};

struct recurrent_state {
    vbr_artifact_companion_kind kind =
        vbr_artifact_companion_kind::recurrent;
    bool empty = true;
    bool prepared = false;
    bool published = false;
    bool fail_prepare = false;
    bool report_nonempty = false;
    uint32_t rollbacks = 0;
    uint32_t replacement_prepares = 0;
    uint8_t live_image = 0;
};

class fake_memory final : public llama_memory_i {
  public:
    llama_memory_context_ptr init_batch(
        llama_batch_allocr &, uint32_t, bool) override { return {}; }
    llama_memory_context_ptr init_full() override { return {}; }
    llama_memory_context_ptr init_update(llama_context *, bool) override {
        return {};
    }
    bool get_can_shift() const override { return false; }
    void clear(bool) override {}
    bool seq_rm(llama_seq_id, llama_pos, llama_pos) override { return false; }
    void seq_cp(llama_seq_id, llama_seq_id, llama_pos, llama_pos) override {}
    void seq_keep(llama_seq_id) override {}
    void seq_add(llama_seq_id, llama_pos, llama_pos, llama_pos) override {}
    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override {}
    llama_pos seq_pos_min(llama_seq_id) const override { return -1; }
    llama_pos seq_pos_max(llama_seq_id) const override { return -1; }
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override {
        return {};
    }
    void state_write(llama_io_write_i &, llama_seq_id,
                     llama_state_seq_flags) const override {}
    void state_read(llama_io_read_i &, llama_seq_id,
                    llama_state_seq_flags) override {}
};

class prepared_companion final : public vbr_prepared_companion_image {};

struct seam final : vbr_adopt_test_seam {
    enum class upward_event : uint8_t {
        unit_h2d,
        stash_h2d,
        transform,
        synchronize,
    };
    fake_memory target;
    std::array<child_state, 2> children;
    recurrent_state recurrent;
    std::vector<llama_memory_tree_child> live;
    vbr_adopt_phase inject_phase = vbr_adopt_phase::_count;
    bool inject_after = false;
    uint32_t fail_map_child = UINT32_MAX;
    uint32_t fail_map_after = UINT32_MAX;
    uint32_t fail_transfer_child = UINT32_MAX;
    uint32_t fail_transfer_shard = UINT32_MAX;
    bool drift_recurrent_at_barrier = false;
    bool drop_attention_at_barrier = false;
    bool add_recurrent_at_barrier = false;
    bool drift_serial_at_phase3 = false;
    bool drift_serial_at_phase10 = false;
    bool drift_downward_at_phase10 = false;
    uint64_t * observed_serial = nullptr;
    std::array<uint8_t, 32> * observed_downward_tree = nullptr;
    bool operation_opened = false;
    bool recovery_reserved = false;
    uint32_t events = 0;
    uint32_t publish_calls = 0;
    uint32_t map_calls = 0;
    uint32_t transfer_calls = 0;
    uint32_t mark_calls = 0;
    uint32_t downward_zero_inits = 0;
    uint32_t downward_edges = 0;
    uint32_t downward_stashes = 0;
    uint32_t downward_syncs = 0;
    bool downward_synced = false;
    uint32_t upward_zero_inits = 0;
    uint32_t upward_transforms = 0;
    uint32_t upward_stash_transforms = 0;
    uint32_t upward_null_stash_transforms = 0;
    uint64_t upward_stash_rows = 0;
    uint32_t upward_syncs = 0;
    bool occupied_mode = false;
    bool recycle_mode = false;
    bool transformed_recycle_mode = false;
    bool operation_quarantined = false;
    bool fail_recovery_sync = false;
    uint32_t occupied_rechecks = 0;
    uint32_t relocated_prepares = 0;
    uint32_t relocated_images = 0;
    uint32_t incoming_transfer_calls = 0;
    uint32_t recovery_transfer_calls = 0;
    uint32_t recovery_syncs = 0;
    uint32_t fail_incoming_transfer_at = UINT32_MAX;
    uint32_t fail_incoming_after_bytes = UINT32_MAX;
    uint32_t fail_recovery_transfer_at = UINT32_MAX;
    std::vector<std::pair<uint64_t, uint64_t>> relocated_offsets;
    uint64_t incumbent_sentinel = UINT64_C(0xabcddcba11223344);
    std::array<uint8_t, 24> incumbent_image = {};
    std::vector<vbr_staged_read_kind> transfer_order;
    std::vector<upward_event> upward_events;

    seam() {
        for (uint32_t i = 0; i < children.size(); ++i) {
            children[i].child_id = i;
            children[i].instance = { UINT64_C(0x9000)+i,
                                     UINT64_C(0xa000)+i };
            live.push_back({
                i, reinterpret_cast<llama_kv_cache *>(&children[i]), nullptr,
                checkpoint_child_dependency_mode::live_guarded,
            });
        }
    }

    bool phase_boundary(vbr_adopt_phase phase, bool after) noexcept override {
        if (!after && phase == vbr_adopt_phase::target_recheck &&
            drift_serial_at_phase3 && observed_serial) {
            ++*observed_serial;
        }
        if (!after && phase == vbr_adopt_phase::complete_tree_barrier) {
            if (drift_serial_at_phase10 && observed_serial) {
                ++*observed_serial;
            }
            if (drift_downward_at_phase10 && observed_downward_tree) {
                (*observed_downward_tree)[0] ^= 0xff;
            }
            if (drift_recurrent_at_barrier) {
                recurrent.report_nonempty = true;
            }
            if (drop_attention_at_barrier && live.size() > 1) {
                live.erase(live.begin()+1);
            }
            if (add_recurrent_at_barrier) {
                live.push_back({
                    2, nullptr,
                    reinterpret_cast<llama_memory_recurrent *>(&recurrent),
                    checkpoint_child_dependency_mode::payload_complete,
                });
            }
        }
        return phase == inject_phase && after == inject_after;
    }

    bool collect_tree(llama_memory_i & memory,
                      std::vector<llama_memory_tree_child> & output) noexcept override {
        if (&memory != &target) {
            return false;
        }
        output = live;
        return true;
    }

    vbr_adopt_status operation_open(
            const vbr_validated_manifest & manifest, llama_seq_id,
            const std::vector<llama_memory_tree_child> &,
            std::vector<vbr_controller_instance_id> & instances,
            vbr_operation_id & operation) noexcept override {
        instances.clear();
        for (const auto & child : manifest.tracker_install().children) {
            instances.push_back(child.target_instance);
        }
        if (instances.empty()) {
            return vbr_adopt_status::operation_unavailable;
        }
        operation = { 0xf42a2001 };
        operation_opened = true;
        recovery_reserved = true;
        return vbr_adopt_status::adopted;
    }

    void operation_finish(
            vbr_operation_id, bool, bool quarantine) noexcept override {
        operation_quarantined = quarantine;
        operation_opened = false;
        recovery_reserved = false;
    }

    bool operation_quiescent(
            const std::vector<vbr_controller_instance_id> &,
            vbr_operation_id) const noexcept override {
        return operation_opened && recovery_reserved;
    }

    bool session_recheck(uint32_t child_id,
            const vbr_child_empty_fingerprint & expected,
            bool journal_armed) const noexcept override {
        if (child_id >= children.size()) {
            return false;
        }
        const auto & child = children[child_id];
        return expected.child_id == child_id &&
               expected.memory_cookie == &child &&
               expected.instance_id == child.instance &&
               child.armed == journal_armed && !child.published &&
               (journal_armed ||
                (child.mapped == 0 && child.h2d_bytes == 0));
    }

    bool session_recheck_occupied(
            uint32_t child_id,
            const vbr_occupied_replacement_guard & guard,
            bool) const noexcept override {
        if (!occupied_mode || child_id != 0 || !guard.ready() ||
            guard.destination() != 0) {
            return false;
        }
        ++const_cast<seam *>(this)->occupied_rechecks;
        return true;
    }

    bool session_arm(uint32_t child_id, vbr_operation_id) noexcept override {
        if (child_id >= children.size() || children[child_id].armed) {
            return false;
        }
        children[child_id].armed = true;
        return true;
    }

    bool session_prepare_backing(uint32_t child_id,
            const std::vector<const vbr_validated_child_plan *> & plans) noexcept override {
        if (child_id >= children.size() || plans.empty() ||
            !children[child_id].armed) {
            return false;
        }
        auto & child = children[child_id];
        for (uint32_t range = 0; range < 5; ++range) {
            ++child.mapped;
            ++map_calls;
            child.mapped_ranges.push_back(range);
            if (child_id == fail_map_child &&
                child.mapped == fail_map_after) {
                return false;
            }
        }
        return true;
    }

    bool session_prepare_relocated_backing(
            uint32_t child_id,
            const std::vector<const vbr_validated_child_plan *> & plans,
            const std::vector<vbr_occupied_replacement_relocation_run> & runs) noexcept override {
        if (!occupied_mode || child_id != 0 || plans.empty() || runs.empty() ||
            !children[child_id].armed) {
            return false;
        }
        auto & child = children[child_id];
        child.mapped = uint32_t(runs.size());
        ++relocated_prepares;
        return true;
    }

    bool session_transfer(uint32_t child_id,
            const vbr_staged_read_descriptor & read,
            uint64_t fail_completion, vbr_h2d_stats & stats) noexcept override {
        if (child_id >= children.size() || !read.source ||
            fail_completion == 0 ||
            (child_id == fail_transfer_child &&
             read.shard_index == fail_transfer_shard)) {
            return false;
        }
        std::vector<uint8_t> bytes;
        bytes.reserve(size_t(read.size));
        if (read.projection_ranges.empty()) {
            bytes.resize(size_t(read.size));
            if (!read.source->read(
                    read.source_offset, bytes.data(), bytes.size())) {
                return false;
            }
        } else {
            for (const auto & range : read.projection_ranges) {
                const size_t start = bytes.size();
                bytes.resize(start + size_t(range.size));
                if (!read.source->read(
                        range.source_offset, bytes.data() + start,
                        size_t(range.size))) {
                    return false;
                }
            }
            if (bytes.size() != read.size) {
                return false;
            }
        }
        auto & child = children[child_id];
        const bool incoming =
            read.kind == vbr_staged_read_kind::unit_payload;
        const bool recovery =
            read.kind == vbr_staged_read_kind::recovery_unit_payload;
        const uint32_t transfer_ordinal = incoming
            ? incoming_transfer_calls++
            : recovery ? recovery_transfer_calls++ : UINT32_MAX;
        transfer_order.push_back(read.kind);
        if (occupied_mode && read.kind == vbr_staged_read_kind::unit_payload) {
            if (!child.image_ready || read.projection_ranges.size() != 1) {
                return false;
            }
            relocated_offsets.push_back({
                read.projection_ranges.front().source_offset,
                read.destination_offset,
            });
        }
        size_t written = bytes.size();
        bool fail_after_write = false;
        if (incoming && transfer_ordinal == fail_incoming_transfer_at) {
            written = std::min<size_t>(bytes.size(), fail_incoming_after_bytes);
            fail_after_write = true;
        } else if (recovery &&
                   transfer_ordinal == fail_recovery_transfer_at) {
            fail_after_write = true;
        }
        if (occupied_mode) {
            const size_t base = size_t(read.shard_index)*8;
            const size_t offset = size_t(read.destination_offset % 8);
            if (base + offset > child.destination.size() ||
                written > child.destination.size() - base - offset) {
                return false;
            }
            std::copy_n(bytes.data(), written,
                        child.destination.data() + base + offset);
        } else {
            for (size_t i = 0; i < written; ++i) {
                child.destination[i % child.destination.size()] ^= bytes[i];
            }
        }
        child.h2d_bytes += written;
        ++transfer_calls;
        if (read.kind == vbr_staged_read_kind::unit_payload) {
            upward_events.push_back(upward_event::unit_h2d);
        } else if (read.kind == vbr_staged_read_kind::clean_stash) {
            upward_events.push_back(upward_event::stash_h2d);
        }
        stats.bytes = written;
        stats.chunks = 1;
        ++events;
        return !fail_after_write;
    }

    bool session_synchronize_recovery(uint32_t child_id) noexcept override {
        if (child_id >= children.size()) {
            return false;
        }
        ++recovery_syncs;
        return !fail_recovery_sync;
    }

    bool session_mark_complete(uint32_t child_id,
                               uint32_t unit) noexcept override {
        if (child_id >= children.size() ||
            !children[child_id].completed_units.insert(unit).second) {
            return false;
        }
        ++mark_calls;
        return true;
    }

    vbr_downward_transform_status session_transform_downward(
            uint32_t child_id, const vbr_validated_child_plan & plan,
            bool stashless, uint32_t fail_edge, bool fail_stash,
            uint32_t & stash_valid, uint32_t & edge_reached) noexcept override {
        stash_valid = 0;
        edge_reached = UINT32_MAX;
        if (child_id >= children.size() || !children[child_id].armed ||
            plan.transform_kind != vbr_import_transform_kind::downward ||
            plan.transcode_recipe.n_edges == 0) {
            return vbr_downward_transform_status::invalid_recipe;
        }
        struct driver_state {
            seam * owner = nullptr;
            uint32_t edge = 0;
            uint32_t fail_edge = UINT32_MAX;
            bool fail_stash = false;
            bool stashless = false;
            bool captured = false;
        } state { this, 0, fail_edge, fail_stash, stashless, false };
        vbr_downward_edge_driver driver;
        driver.context = &state;
        driver.stash_available = [](void * opaque) noexcept {
            const auto & value = *static_cast<driver_state *>(opaque);
            return value.stashless || value.captured;
        };
        driver.capture_stash = [](void * opaque,
                                  const vbr_downward_edge &) noexcept {
            auto & value = *static_cast<driver_state *>(opaque);
            if (value.fail_stash) {
                return false;
            }
            value.captured = true;
            ++value.owner->downward_stashes;
            return true;
        };
        driver.transcode = [](void * opaque,
                              const vbr_downward_edge &) noexcept {
            auto & value = *static_cast<driver_state *>(opaque);
            const uint32_t edge = value.edge++;
            if (edge == value.fail_edge) {
                return false;
            }
            ++value.owner->downward_edges;
            return true;
        };
        bool regenerated = false;
        const auto status = vbr_downward_execute_edges(
            plan.transcode_recipe, driver, regenerated, &edge_reached);
        if (status == vbr_downward_transform_status::transformed &&
            regenerated) {
            stash_valid = plan.descriptor.wm_cells;
        }
        return status;
    }

    bool session_initialize_downward_backing(
            uint32_t child_id,
            const vbr_validated_child_plan & plan) noexcept override {
        if (child_id >= children.size() || !children[child_id].armed ||
            plan.transform_kind != vbr_import_transform_kind::downward ||
            plan.descriptor.wm_cells == 0) {
            return false;
        }
        if (recycle_mode && transformed_recycle_mode) {
            // Model the production alias initialization that clears the
            // wider incoming source representation over compact incumbent
            // backing. Recovery must already be armed before this mutation.
            children[child_id].destination.fill(0);
        }
        ++downward_zero_inits;
        return true;
    }

    bool session_initialize_upward_backing(
            uint32_t child_id,
            const vbr_validated_child_plan & plan) noexcept override {
        if (child_id >= children.size() || !children[child_id].armed ||
            (plan.transform_kind !=
                 vbr_import_transform_kind::upward_same_domain &&
             plan.transform_kind !=
                 vbr_import_transform_kind::upward_cross_domain) ||
            plan.descriptor.wm_cells == 0) {
            return false;
        }
        ++upward_zero_inits;
        return true;
    }

    bool session_transform_upward(
            uint32_t child_id,
            const vbr_validated_child_plan & plan,
            uint32_t stash_rows) noexcept override {
        if (child_id >= children.size() || !children[child_id].armed ||
            (plan.transform_kind !=
                 vbr_import_transform_kind::upward_same_domain &&
             plan.transform_kind !=
                 vbr_import_transform_kind::upward_cross_domain) ||
            plan.upward_recipe.n_edges != 1) {
            return false;
        }
        const bool expected_stash =
            plan.source_domain == vbr_repr_domain::tapped &&
            (plan.stash_action == vbr_validated_stash_action::restore_exact ||
             plan.stash_action ==
                 vbr_validated_stash_action::consume_exact_then_drop);
        if ((expected_stash &&
             stash_rows != plan.descriptor.clean_stash.valid_rows) ||
            (!expected_stash && stash_rows != 0)) {
            return false;
        }
        if (stash_rows != 0) {
            ++upward_stash_transforms;
            upward_stash_rows += stash_rows;
        } else {
            ++upward_null_stash_transforms;
        }
        upward_events.push_back(upward_event::transform);
        ++upward_transforms;
        return true;
    }

    bool session_synchronize_downward(
            uint32_t child_id,
            const std::vector<const vbr_validated_child_plan *> & plans) noexcept override {
        if (child_id >= children.size() || plans.empty()) {
            return false;
        }
        ++downward_syncs;
        downward_synced = true;
        return true;
    }

    bool session_synchronize_upward(
            uint32_t child_id,
            const std::vector<const vbr_validated_child_plan *> & plans)
            noexcept override {
        if (child_id >= children.size() || plans.empty() ||
            std::none_of(plans.begin(), plans.end(), [](const auto * plan) {
                return plan &&
                    (plan->transform_kind ==
                         vbr_import_transform_kind::upward_same_domain ||
                     plan->transform_kind ==
                         vbr_import_transform_kind::upward_cross_domain);
            })) {
            return false;
        }
        upward_events.push_back(upward_event::synchronize);
        ++upward_syncs;
        return true;
    }

    bool session_trim_downward(
            uint32_t child_id, const vbr_validated_child_plan & plan,
            uint32_t) noexcept override {
        return downward_synced && child_id < children.size() &&
            plan.transform_kind == vbr_import_transform_kind::downward;
    }

    bool session_build_live_image(uint32_t child_id,
            const std::vector<const vbr_validated_child_plan *> & plans,
            const vbr_tracker_install_child &,
            const vbr_checkpoint_generation_controller &) noexcept override {
        if (child_id >= children.size() || plans.empty() ||
            children[child_id].completed_units.size() != plans.size()) {
            return false;
        }
        children[child_id].image_ready = true;
        return true;
    }

    bool session_build_relocated_live_image(
            uint32_t child_id,
            const std::vector<const vbr_validated_child_plan *> & plans,
            const vbr_tracker_install_child &,
            const vbr_checkpoint_generation_controller &,
            const vbr_occupied_replacement_guard & guard) noexcept override {
        if (!occupied_mode || child_id != 0 || plans.empty() || !guard.ready() ||
            !children[child_id].completed_units.empty()) {
            return false;
        }
        children[child_id].image_ready = true;
        ++relocated_images;
        return true;
    }

    bool session_prepare_receipts(
            uint32_t child_id,
            std::shared_ptr<void> receipt_owner) noexcept override {
        if (child_id >= children.size()) {
            return false;
        }
        if (!receipt_owner || children[child_id].receipt_owner) {
            return false;
        }
        children[child_id].receipt_owner = std::move(receipt_owner);
        children[child_id].receipts = true;
        return true;
    }

    bool session_mapped_prefixes_complete(
            uint32_t child_id) const noexcept override {
        return child_id < children.size() &&
            children[child_id].mapped == (occupied_mode ? 1u : 5u);
    }

    bool session_barrier(uint32_t child_id, uint64_t serial,
                         const vbr_validated_manifest &) const noexcept override {
        return child_id < children.size() && serial != 0 &&
               children[child_id].armed && children[child_id].image_ready;
    }

    void session_publish_metadata(uint32_t child_id) noexcept override {
        auto & child = children[child_id];
        child.published = true;
        ++publish_calls;
    }

    void session_publish_receipts(uint32_t child_id) noexcept override {
        children[child_id].receipts = true;
    }

    void session_finish_publish(uint32_t child_id) noexcept override {
        children[child_id].armed = false;
    }

    bool session_rollback(uint32_t child_id, bool inject_failure) noexcept override {
        if (child_id >= children.size()) {
            return false;
        }
        auto & child = children[child_id];
        child.unmapped += child.mapped;
        child.unmapped_ranges.insert(
            child.unmapped_ranges.end(),
            child.mapped_ranges.rbegin(), child.mapped_ranges.rend());
        child.mapped = 0;
        child.mapped_ranges.clear();
        child.h2d_bytes = 0;
        if (!recycle_mode) {
            child.destination.fill(0);
        }
        child.completed_units.clear();
        child.image_ready = false;
        child.receipts = false;
        child.receipt_owner.reset();
        child.armed = false;
        events = 0;
        downward_zero_inits = 0;
        downward_edges = 0;
        downward_stashes = 0;
        downward_syncs = 0;
        downward_synced = false;
        return !inject_failure;
    }

    bool construction_empty() const noexcept {
        return !operation_opened && !recovery_reserved && events == 0 &&
               downward_zero_inits == 0 && downward_edges == 0 &&
               downward_stashes == 0 &&
               downward_syncs == 0 && !downward_synced &&
               publish_calls == 0 &&
               std::all_of(children.begin(), children.end(),
                   [](const child_state & child) {
                       return !child.armed && !child.image_ready &&
                              !child.published && !child.receipts &&
                              !child.receipt_owner &&
                              child.mapped == 0 && child.h2d_bytes == 0 &&
                              child.mapped_ranges.empty() &&
                              child.completed_units.empty() &&
                              std::all_of(child.destination.begin(),
                                  child.destination.end(),
                                  [](uint8_t value) { return value == 0; });
                   }) &&
               recurrent.empty && !recurrent.prepared &&
               !recurrent.published && !recurrent.report_nonempty;
    }

    void erase_imported() noexcept {
        for (auto & child : children) {
            child.armed = false;
            child.image_ready = false;
            child.published = false;
            child.receipts = false;
            child.mapped = 0;
            child.mapped_ranges.clear();
            child.h2d_bytes = 0;
            child.completed_units.clear();
            child.destination.fill(0);
            // All children hold the same opaque production receipt group. The
            // final reset exercises its real last-owner ledger release.
            child.receipt_owner.reset();
        }
        operation_opened = false;
        recovery_reserved = false;
        events = 0;
        publish_calls = 0;
        map_calls = 0;
        transfer_calls = 0;
        incoming_transfer_calls = 0;
        recovery_transfer_calls = 0;
        recovery_syncs = 0;
        transfer_order.clear();
        mark_calls = 0;
        downward_zero_inits = 0;
        downward_edges = 0;
        downward_stashes = 0;
        downward_syncs = 0;
        downward_synced = false;
        recurrent.empty = true;
        recurrent.prepared = false;
        recurrent.published = false;
    }
};

struct validation_context {
    seam * target = nullptr;
    llama_cache_acct_ledger * ledger = nullptr;
    uint64_t serial_bias = 0;
    uint64_t policy_epoch = 0;
    std::array<uint8_t, 32> downward_tree_digest = {};

    static uint64_t read_accounting(const void * context) noexcept {
        const auto * self = static_cast<const validation_context *>(context);
        return self && self->ledger
            ? self->ledger->snapshot().serial + self->serial_bias
            : 0;
    }

    static uint64_t read_policy(const void * context) noexcept {
        const auto * self = static_cast<const validation_context *>(context);
        return self ? self->policy_epoch : 0;
    }

    static bool read_downward_tree(
            const void * context,
            std::array<uint8_t, 32> & output) noexcept {
        const auto * self = static_cast<const validation_context *>(context);
        if (!self || !vbr_digest_nonzero(self->downward_tree_digest)) {
            return false;
        }
        output = self->downward_tree_digest;
        return true;
    }

    static bool recheck(const void * context,
                        const vbr_target_empty_fingerprint & expected) noexcept {
        const auto * self = static_cast<const validation_context *>(context);
        if (!self || !self->target ||
            expected.accounting_serial != read_accounting(context) ||
            expected.policy_epoch != self->policy_epoch ||
            expected.memory_instance_cookie !=
                uint64_t(reinterpret_cast<uintptr_t>(&self->target->target)) ||
            expected.children.size() != self->target->children.size()) {
            return false;
        }
        for (const auto & child : expected.children) {
            if (child.child_id >= self->target->children.size()) {
                return false;
            }
            const auto & live = self->target->children[child.child_id];
            if (child.memory_cookie != &live ||
                child.instance_id != live.instance || live.published) {
                return false;
            }
        }
        return true;
    }

    static bool parse_companion(
            const void *, const vbr_artifact_companion_payload & descriptor,
            const artifact_segment_chain & source,
            const vbr_target_companion_snapshot & target,
            std::unique_ptr<vbr_parsed_companion_image> & output) noexcept {
        if ((descriptor.kind != vbr_artifact_companion_kind::recurrent &&
             descriptor.kind !=
                 vbr_artifact_companion_kind::required_spec_payload) ||
            descriptor.format_version != 1 || !target.available ||
            target.kind != descriptor.kind ||
            target.format_version != descriptor.format_version ||
            target.build_identity_digest != descriptor.build_identity_digest ||
            source.size() != descriptor.payload_bytes) {
            return false;
        }
        output = std::make_unique<parsed_companion>(descriptor.kind);
        return true;
    }
};

static bool owner_token_valid(
        const void * context, const void * token,
        const llama_memory_i * target) noexcept {
    return context && token == context && target == context;
}

static bool companion_prepare(
        const void * context,
        std::unique_ptr<vbr_parsed_companion_image> parsed,
        llama_seq_id,
        std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
    auto * state = const_cast<recurrent_state *>(
        static_cast<const recurrent_state *>(context));
    if (!state || !state->empty || state->prepared || state->fail_prepare ||
        !parsed || parsed->kind() != state->kind) {
        return false;
    }
    state->prepared = true;
    output = std::make_unique<prepared_companion>();
    return true;
}

static bool companion_prepare_replacement(
        const void * context,
        std::unique_ptr<vbr_parsed_companion_image> incoming,
        std::unique_ptr<vbr_parsed_companion_image> recovery,
        llama_seq_id,
        std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
    auto * state = const_cast<recurrent_state *>(
        static_cast<const recurrent_state *>(context));
    if (!state || state->empty || state->prepared || state->live_image != 1 ||
        !incoming || !recovery || incoming->kind() != state->kind ||
        recovery->kind() != state->kind) {
        return false;
    }
    // Publish the recovery owner before the first destructive mutation.
    output = std::make_unique<prepared_companion>();
    state->prepared = true;
    state->live_image = 2;
    ++state->replacement_prepares;
    return true;
}

static bool companion_empty(const void * context) noexcept {
    const auto * state = static_cast<const recurrent_state *>(context);
    return state && state->empty && !state->published &&
           !state->report_nonempty;
}

static bool companion_recheck(
        const void * context,
        const vbr_prepared_companion_image &) noexcept {
    const auto * state = static_cast<const recurrent_state *>(context);
    return state && state->prepared &&
        !state->empty && state->live_image == 2;
}

static void companion_publish(
        const void * context, vbr_prepared_companion_image &) noexcept {
    auto * state = const_cast<recurrent_state *>(
        static_cast<const recurrent_state *>(context));
    state->prepared = false;
    state->published = true;
    state->empty = false;
    state->live_image = 2;
}

static bool companion_rollback(
        const void * context, vbr_prepared_companion_image &) noexcept {
    auto * state = const_cast<recurrent_state *>(
        static_cast<const recurrent_state *>(context));
    state->prepared = false;
    state->report_nonempty = false;
    if (state->live_image != 0) {
        state->empty = false;
        state->published = true;
        state->live_image = 1;
    }
    ++state->rollbacks;
    return true;
}

static vbr_verified_segment segment(
        uint32_t unit, uint32_t shard_index,
        const byte_source & bytes) {
    auto chain = std::make_shared<artifact_segment_chain>();
    CHECK(chain->append(bytes.bytes.data(), bytes.bytes.size()));
    vbr_verified_segment out;
    out.unit_index = unit;
    out.shard_index = shard_index;
    out.bytes = std::move(chain);
    out.streaming_digest = vbr_capture_stream_digest(*out.bytes);
    return out;
}

static bool measure_upward_destination(
        void *, const std::vector<std::vector<ggml_type>> &,
        const llama_vbr_policy::selection *,
        vbr_import_destination_evidence & evidence) noexcept {
    evidence.active = true;
    evidence.fits = true;
    evidence.pools = 2;
    return true;
}

struct fixture {
    storage bytes;
    vbr_artifact_package source;
    llama_cache_acct_ledger ledger;
    llama_vbr_artifact_catalog catalog;
    std::vector<llama_vbr_artifact_domain_binding> bindings;
    llama_cache_budget_config budget;
    llama_cache_acct_resource_domain host =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pageable_host);
    llama_cache_acct_resource_domain pinned =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    llama_cache_acct_artifact_id reference;
    vbr_artifact_package_view view;
    llama_cache_acct_artifact_id recovery_reference;
    vbr_artifact_package_view recovery_view;
    seam target;
    validation_context validation;
    vbr_target_validation_snapshot snapshot;
    vbr_target_validation_snapshot occupied_live_snapshot;
    vbr_adopt_policy policy;
    uint64_t catalog_live_ops = 0;
    bool with_companion = false;
    bool downward = false;
    bool upward = false;
    ggml_type upward_source_type = GGML_TYPE_TURBO8_0;
    ggml_type upward_target_type = GGML_TYPE_F16;
    bool mixed_exact_upward = false;
    bool occupied = false;
    bool occupied_spec_companion = false;
    bool recycle = false;
    bool transformed_recycle = false;
    vbr_occupied_replacement_guard occupied_guard;
    vbr_downward_policy_projection downward_projection;
    vbr_import_destination_projection upward_destination;
    vbr_import_schedule_quote schedule_quote;
    llama_cache_budget_plan downward_plan;
    vbr_downward_reserve_status downward_reserve_status =
        vbr_downward_reserve_status::reserved;
    uint32_t transform_reserve_plans = 0;
    uint32_t transform_reserve_stash_only = 0;

    explicit fixture(bool companion = false, bool downward_import = false,
                     bool upward_import = false,
                     ggml_type upward_source = GGML_TYPE_TURBO8_0,
                     ggml_type upward_target = GGML_TYPE_F16,
                     uint8_t source_promote_hops = 0,
                     vbr_artifact_clean_stash_state stash_state =
                         vbr_artifact_clean_stash_state::absent_at_source,
                     bool mixed_exact = false,
                     bool partial_stash = false,
                     bool occupied_import = false,
                     bool recycle_import = false,
                     bool transformed_recycle_import = false,
                     bool occupied_spec_companion_import = false)
        : source(package(
              bytes, companion,
              upward_import ? upward_source : GGML_TYPE_TURBO8_0,
              upward_import ? source_promote_hops : 0,
              upward_import ? stash_state :
                  vbr_artifact_clean_stash_state::absent_at_source,
              upward_import && partial_stash,
              occupied_import ? 1 : 2,
              transformed_recycle_import ? 2 : 1,
              occupied_spec_companion_import
                  ? vbr_artifact_companion_kind::required_spec_payload
                  : vbr_artifact_companion_kind::recurrent)),
          catalog(ledger),
          with_companion(companion), downward(downward_import),
          upward(upward_import), upward_source_type(upward_source),
          upward_target_type(upward_target),
          mixed_exact_upward(mixed_exact), occupied(occupied_import),
          occupied_spec_companion(occupied_spec_companion_import),
          recycle(recycle_import),
          transformed_recycle(transformed_recycle_import) {
        CHECK(!(downward && upward));
        CHECK(!recycle || occupied);
        if (occupied) {
            target.live.resize(1);
            target.occupied_mode = true;
            target.recycle_mode = recycle;
            target.transformed_recycle_mode = transformed_recycle;
        }
        if (with_companion && occupied) {
            CHECK(with_companion && occupied);
            target.recurrent.kind = occupied_spec_companion
                ? vbr_artifact_companion_kind::required_spec_payload
                : vbr_artifact_companion_kind::recurrent;
            target.recurrent.empty = false;
            target.recurrent.published = true;
            target.recurrent.live_image = 1;
        }
        if (downward) {
            for (auto & controller : source.manifest.controller_policy) {
                controller.floor_type = GGML_TYPE_TURBO1_TCQ;
            }
        }
        const auto prepared = vbr_artifact_prepare(source);
        if (prepared != vbr_artifact_status::ok) {
            std::fprintf(stderr, "VBR adoption prepare status=%s source=%s hops=%u stash=%u\n",
                         vbr_artifact_status_name(prepared),
                         ggml_type_name(upward_source),
                         unsigned(source_promote_hops), unsigned(stash_state));
        }
        CHECK(prepared == vbr_artifact_status::ok);
        CHECK(catalog.bind_topologies(source.topologies, bindings));
        CHECK(bindings.size() == 2);
        std::vector<llama_cache_acct_completeness_requirement> required = {
            { host, llama_cache_acct_producer::retention_sidecar },
            { pinned, llama_cache_acct_producer::retention_sidecar },
        };
        for (const auto & binding : bindings) {
            required.push_back({
                binding.domain, llama_cache_acct_producer::live_memory,
            });
        }
        CHECK(ledger.configure_required_producers(
            required.data(), required.size()));
        CHECK(catalog.configure_accounting(source));
        const auto initialize_domain = [&](const auto & domain) {
            for (uint8_t raw = 0;
                 raw < uint8_t(llama_cache_acct_category::_count); ++raw) {
                const auto category = llama_cache_acct_category(raw);
                const auto classification =
                    llama_cache_budget_classify(category);
                if (classification.participation !=
                        llama_cache_budget_capacity_participation::participating) {
                    continue;
                }
                for (const auto measure : {
                        llama_cache_acct_measure::logical_payload,
                        llama_cache_acct_measure::resident_allocated,
                        llama_cache_acct_measure::reserved }) {
                    ledger.gauge_set(category, domain, measure, 0);
                }
            }
        };
        initialize_domain(host);
        initialize_domain(pinned);
        for (const auto & binding : bindings) {
            initialize_domain(binding.domain);
        }
        CHECK(ledger.certify_complete(
            host, llama_cache_acct_producer::retention_sidecar));
        CHECK(ledger.certify_complete(
            pinned, llama_cache_acct_producer::retention_sidecar));
        for (size_t i = 0; i < bindings.size(); ++i) {
            CHECK(ledger.certify_complete(
                bindings[i].domain, llama_cache_acct_producer::live_memory));
            llama_cache_budget_device_input input;
            input.backend_device = reinterpret_cast<const void *>(i+1);
            input.domain = bindings[i].domain;
            input.physical_total = 1ull << 30;
            input.physical_free = (1ull << 30) - (1ull << 20);
            input.phys_state = llama_cache_budget_capacity_state::known;
            input.current_compute_allocated = 0;
            input.configured_compute_reserve = 0;
            input.compute_state = llama_cache_budget_capacity_state::known;
            input.cache_cap_state =
                llama_cache_budget_capacity_state::unbounded;
            budget.devices.push_back(input);
        }
        budget.host.pageable_state =
            llama_cache_budget_capacity_state::unbounded;
        budget.host.pinned_state =
            llama_cache_budget_capacity_state::unbounded;

        vbr_capture_stream_status stream_status;
        auto build = catalog.begin_capture(source, budget, {}, stream_status);
        CHECK(build && stream_status == vbr_capture_stream_status::ok);
        if (!build) {
            return;
        }
        for (uint32_t unit_index = 0;
             unit_index < source.unit_blobs.size(); ++unit_index) {
            auto unit = build->begin_unit(unit_index, stream_status);
            CHECK(unit && stream_status == vbr_capture_stream_status::ok);
            if (!unit) {
                return;
            }
            for (uint32_t shard_index = 0; shard_index < 2; ++shard_index) {
                const auto verified = segment(
                    unit_index, shard_index,
                    bytes.payload[unit_index*2+shard_index]);
                CHECK(unit->accept_verified_segment(verified) ==
                      vbr_capture_stream_status::ok);
                if (source.unit_blobs[unit_index].descriptor.clean_stash_state ==
                        vbr_artifact_clean_stash_state::present) {
                    auto verified_stash = segment(
                        unit_index, shard_index,
                        bytes.stash[unit_index*2+shard_index]);
                    verified_stash.clean_stash = true;
                    CHECK(unit->accept_verified_segment(verified_stash) ==
                          vbr_capture_stream_status::ok);
                }
            }
            CHECK(unit->seal_unit() == vbr_capture_stream_status::ok);
        }
        if (companion) {
            auto chain = std::make_shared<artifact_segment_chain>();
            CHECK(chain->append(
                bytes.companion.bytes.data(), bytes.companion.bytes.size()));
            vbr_verified_companion verified;
            verified.companion_index = 0;
            verified.bytes = std::move(chain);
            verified.streaming_digest =
                vbr_capture_stream_digest(*verified.bytes);
            CHECK(build->accept_verified_companion(verified) ==
                  vbr_capture_stream_status::ok);
        }
        const auto published = build->publish_reference();
        if (published.status != vbr_capture_stream_status::ok) {
            std::fprintf(stderr, "VBR adoption publish status=%s\n",
                         vbr_capture_stream_status_name(published.status));
        }
        CHECK(published.status == vbr_capture_stream_status::ok);
        reference = published.reference_artifact;
        CHECK(reference.v != 0);
        build.reset();
        CHECK(catalog.resolve_reference(reference, view) ==
              vbr_artifact_resolve_status::ok);
        CHECK(view.validate() == vbr_artifact_status::ok);
        if (occupied) {
            storage recovery_bytes;
            for (auto & payload : recovery_bytes.payload) {
                for (auto & value : payload.bytes) {
                    value ^= 0x5a;
                }
            }
            const ggml_type recovery_type =
                transformed_recycle && downward
                    ? GGML_TYPE_TURBO3_TCQ
                    : ggml_type(source.unit_blobs.front().descriptor.
                        current_type);
            auto recovery_source = package(
                recovery_bytes, with_companion,
                recovery_type,
                source.unit_blobs.front().descriptor.promote_hops,
                vbr_artifact_clean_stash_state::absent_at_source,
                false, 1, 1,
                occupied_spec_companion
                    ? vbr_artifact_companion_kind::required_spec_payload
                    : vbr_artifact_companion_kind::recurrent);
            for (size_t i = 0;
                 i < recovery_source.manifest.controller_policy.size() &&
                 i < source.manifest.controller_policy.size(); ++i) {
                recovery_source.manifest.controller_policy[i].floor_type =
                    source.manifest.controller_policy[i].floor_type;
            }
            CHECK(vbr_artifact_prepare(recovery_source) ==
                  vbr_artifact_status::ok);
            CHECK(catalog.configure_accounting(recovery_source));
            auto recovery_build = catalog.begin_capture(
                recovery_source, budget, {}, stream_status);
            CHECK(recovery_build &&
                  stream_status == vbr_capture_stream_status::ok);
            if (!recovery_build) {
                return;
            }
            for (uint32_t unit_index = 0;
                 unit_index < recovery_source.unit_blobs.size(); ++unit_index) {
                auto unit = recovery_build->begin_unit(unit_index, stream_status);
                CHECK(unit && stream_status == vbr_capture_stream_status::ok);
                if (!unit) {
                    return;
                }
                for (uint32_t shard_index = 0; shard_index < 2; ++shard_index) {
                    CHECK(unit->accept_verified_segment(segment(
                        unit_index, shard_index,
                              recovery_bytes.payload[
                                  unit_index*2+shard_index])) ==
                          vbr_capture_stream_status::ok);
                }
                CHECK(unit->seal_unit() == vbr_capture_stream_status::ok);
            }
            if (with_companion) {
                auto chain = std::make_shared<artifact_segment_chain>();
                CHECK(chain->append(
                    recovery_bytes.companion.bytes.data(),
                    recovery_bytes.companion.bytes.size()));
                vbr_verified_companion verified;
                verified.companion_index = 0;
                verified.bytes = std::move(chain);
                verified.streaming_digest =
                    vbr_capture_stream_digest(*verified.bytes);
                CHECK(recovery_build->accept_verified_companion(verified) ==
                      vbr_capture_stream_status::ok);
            }
            const auto recovery_published =
                recovery_build->publish_reference();
            CHECK(recovery_published.status ==
                  vbr_capture_stream_status::ok);
            recovery_reference = recovery_published.reference_artifact;
            CHECK(recovery_reference.v != 0);
            CHECK(recovery_reference != reference);
            recovery_build.reset();
            CHECK(catalog.resolve_reference(
                      recovery_reference, recovery_view) ==
                  vbr_artifact_resolve_status::ok);
            CHECK(recovery_view.validate() == vbr_artifact_status::ok);
            for (uint32_t shard_index = 0; shard_index < 2; ++shard_index) {
                const auto & payload = recovery_bytes.payload[shard_index].bytes;
                std::copy(payload.begin(), payload.end(),
                          target.incumbent_image.begin() + shard_index*8);
            }
            if (recycle) {
                target.children[0].destination = target.incumbent_image;
            }
        }
        catalog_live_ops = ledger.snapshot().live_ops;

        validation.target = &target;
        validation.ledger = &ledger;
        validation.policy_epoch = 91;
        target.observed_serial = &validation.serial_bias;
        target.observed_downward_tree =
            &validation.downward_tree_digest;
        fill_snapshot();
        fill_policy();
        if (occupied) {
            refresh();
            std::array<vbr_occupied_replacement_cell, 5> cells;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                cells[i] = { 0, i, llama_pos(i), llama_pos(10+i),
                    llama_pos(20+i), 0, 1 };
            }
            const auto & live_view = transformed_recycle
                ? recovery_view : view;
            const auto & descriptor = live_view.units().front().descriptor;
            const auto & captured = live_view.manifest().generation.
                controllers.front().units.front();
            const vbr_occupied_replacement_unit_currency unit {
                0, descriptor.logical_unit_id,
                { captured.repr_gen, 0, captured.current_type,
                  captured.last_source_type, captured.domain,
                  captured.promote_hops, captured.last_transition },
            };
            const vbr_occupied_replacement_observation observation {
                0, view.manifest().identity.sequence_epoch,
                view.manifest().generation.controllers.front().global_generation,
                snapshot.children.front().state_serial, recycle ? 5u : 10u,
                cells.data(), cells.size(), &unit, 1,
            };
            const auto occupied_status =
                vbr_prepare_occupied_replacement_guard(
                    transformed_recycle
                        ? occupied_live_snapshot : snapshot,
                    snapshot,
                    view, recovery_view, observation, occupied_guard,
                    transformed_recycle ? &schedule_quote : nullptr,
                    nullptr);
            if (occupied_status !=
                    vbr_occupied_replacement_guard_status::ready) {
                std::fprintf(stderr, "VBR occupied guard status=%s\n",
                    vbr_occupied_replacement_guard_status_name(
                        occupied_status));
            }
            CHECK(occupied_status ==
                  vbr_occupied_replacement_guard_status::ready);
            CHECK(occupied_guard.incoming_artifact() == reference);
            CHECK(occupied_guard.recovery_artifact() == recovery_reference);
            CHECK(occupied_guard.incoming_artifact() !=
                  occupied_guard.recovery_artifact());
            CHECK(occupied_guard.strategy() ==
                  (recycle
                       ? vbr_occupied_replacement_strategy::recycle_incumbent_cells
                       : vbr_occupied_replacement_strategy::provisional_free_cells));
            policy.occupied_replacement = &occupied_guard;
            policy.occupied_representation_context = this;
            policy.occupied_representation_identity = [](
                    const void *, int32_t, bool, int32_t,
                    vbr_explicit_representation_identity &) noexcept {
                return false;
            };
        }
    }

    ~fixture() {
        target.erase_imported();
        recovery_view.reset();
        view.reset();
        if (recovery_reference.v != 0) {
            CHECK(catalog.retire(recovery_reference) ==
                  vbr_artifact_retire_status::retired);
        }
        if (reference.v != 0) {
            CHECK(catalog.retire(reference) ==
                  vbr_artifact_retire_status::retired);
        }
        CHECK(ledger.snapshot().live_ops == 0);
    }

    void fill_snapshot() {
        snapshot.memory_instance_cookie =
            uint64_t(reinterpret_cast<uintptr_t>(&target.target));
        snapshot.target_state_serial = 31;
        snapshot.accounting_serial = ledger.snapshot().serial;
        snapshot.tree_shape_digest = 0x4242;
        snapshot.policy_epoch = validation.policy_epoch;
        snapshot.scheduler_idle = true;
        snapshot.destination_sequence_absent = true;
        for (uint32_t child_id = 0; child_id < view.units().size(); ++child_id) {
            const auto & target_view = transformed_recycle
                ? recovery_view : view;
            const auto & descriptor =
                target_view.units()[child_id].descriptor;
            vbr_target_child_snapshot child;
            child.child_id = child_id;
            child.dependency_mode =
                checkpoint_child_dependency_mode::live_guarded;
            child.memory_cookie = &target.children[child_id];
            child.empty = !occupied;
            child.dedicated = true;
            child.armed = true;
            child.lineage_uuid = occupied
                ? recovery_view.manifest().generation.controllers[child_id].
                    lineage_uuid
                : vbr_lineage_uuid { 0x7000+child_id, 0x8000+child_id };
            child.instance_id = target.children[child_id].instance;
            child.state_serial = snapshot.target_state_serial;
            child.previously_observed = occupied;
            child.policy_epoch = snapshot.policy_epoch;
            child.controller_policy =
                target_view.manifest().controller_policy[child_id];
            vbr_target_unit_snapshot unit;
            unit.child_id = child_id;
            unit.logical_unit_id = descriptor.logical_unit_id;
            unit.current_type = descriptor.current_type;
            unit.last_source_type = descriptor.last_source_type;
            unit.promote_hops = descriptor.promote_hops;
            unit.last_transition = descriptor.last_transition;
            unit.representation_kind = descriptor.representation.kind;
            unit.codec_id = descriptor.representation.codec_id;
            unit.codec_version = descriptor.representation.codec_version;
            unit.representation_reference_digest =
                descriptor.representation.reference_digest;
            unit.source_loss_history =
                descriptor.representation.source_loss_history;
            unit.checkpoint_codec_hops =
                descriptor.representation.checkpoint_codec_hops;
            unit.recoverability = descriptor.recoverability;
            unit.side = descriptor.side;
            unit.layout = descriptor.layout;
            unit.row_codec_version = descriptor.row_codec_version;
            unit.current_domain =
                target_view.manifest().generation.controllers[child_id].
                    units[0].domain;
            unit.codebook_digest = descriptor.codebook_digest;
            unit.rotation_digest = descriptor.rotation_digest;
            unit.meansub_digest = descriptor.meansub_digest;
            unit.meansub_model_id = descriptor.meansub_model_id;
            unit.meansub_layer = descriptor.meansub_layer;
            unit.meansub_baked = descriptor.meansub_baked;
            unit.n_stream = descriptor.n_stream;
            unit.unified = descriptor.unified;
            unit.wm_cells = descriptor.wm_cells;
            unit.rank = descriptor.rank;
            unit.dimensions = descriptor.dimensions;
            unit.row_alignment = descriptor.row_alignment;
            for (uint32_t shard_index = 0;
                 shard_index < descriptor.shards.size(); ++shard_index) {
                const auto & source_shard = descriptor.shards[shard_index];
                unit.shards.push_back({
                    shard_index,
                    reinterpret_cast<const void *>(uintptr_t(
                        0x5000+child_id*16+shard_index)),
                    bindings[source_shard.device_ordinal].domain,
                    source_shard.topology_index,
                    source_shard.device_ordinal,
                    source.topologies[source_shard.topology_index].digest,
                    source_shard.logical_offset,
                    source_shard.row_count,
                    source_shard.row_bytes,
                    source_shard.payload_bytes,
                });
                if (occupied) {
                    unit.shards.back().mapped_bytes *= 2;
                }
            }
            child.units.push_back(std::move(unit));
            snapshot.children.push_back(std::move(child));
        }
        snapshot.destination_sequence_absent = !occupied;
        occupied_live_snapshot = snapshot;
        if (downward) {
            std::vector<vbr_downward_policy_child> projected;
            projected.reserve(snapshot.children.size());
            for (auto & child : snapshot.children) {
                CHECK(child.units.size() == 1);
                auto & unit = child.units[0];
                const auto source_type = static_cast<ggml_type>(
                    view.units()[child.child_id].descriptor.current_type);
                const auto target_type = GGML_TYPE_TURBO3_TCQ;
                unit.current_type = target_type;
                unit.current_domain = vbr_repr_domain::tapped;
                unit.downward_supported = true;
                unit.downward_movable = true;
                unit.controller_floor_type = GGML_TYPE_TURBO1_TCQ;
                unit.downward_type = target_type;
                unit.downward_domain = vbr_repr_domain::tapped;
                unit.downward_recipe_id = 1;
                unit.downward_recipe_version = VBR_DOWNWARD_RECIPE_VERSION;
                unit.downward_meansub_model_id = 7;
                CHECK(vbr_downward_resolve_recipe(
                    source_type, target_type, GGML_TYPE_TURBO1_TCQ, true,
                    unit.downward_recipe) ==
                    vbr_downward_recipe_status::resolved);
                unit.downward_row_bytes = transformed_recycle ? 1 :
                    unit.shards[0].row_bytes;
                unit.downward_mapped_bytes = uint64_t(unit.wm_cells)*
                    unit.downward_row_bytes;
                unit.downward_transfer_bytes =
                    view.units()[child.child_id].descriptor.shards[0].payload_bytes;
                unit.downward_codec_workspace_bytes = 64;

                vbr_downward_policy_child policy_child;
                policy_child.initial_types = { source_type };
                policy_child.target_types = { target_type };
                policy_child.initial_cursor =
                    view.manifest().controller_policy[child.child_id].cursor;
                for (size_t edge = 0;
                     edge < unit.downward_recipe.n_edges; ++edge) {
                    const auto & item = unit.downward_recipe.edges[edge];
                    policy_child.policy.steps.push_back({
                        edge, 0, int32_t(item.source_type),
                        int32_t(item.target_type), 1,
                    });
                }
                policy_child.policy.terminal_progress =
                    int64_t(policy_child.policy.steps.size());
                projected.push_back(std::move(policy_child));
            }
            downward_projection =
                vbr_downward_project_policy_prefix(projected);
            CHECK(downward_projection.status ==
                  vbr_downward_policy_status::coherent);
            for (auto & child : snapshot.children) {
                child.controller_policy.current_type_vector_digest =
                    downward_projection.child_type_digests[child.child_id];
                child.controller_policy.cursor =
                    downward_projection.final_cursors[child.child_id];
                auto & unit = child.units[0];
                unit.downward_build_identity_digest =
                    vbr_downward_build_identity(
                        unit.downward_recipe,
                        unit.downward_meansub_model_id,
                        unit.meansub_digest,
                        downward_projection.child_type_digests[child.child_id],
                        downward_projection.tree_digest);
            }
            validation.downward_tree_digest = downward_projection.tree_digest;
            CHECK(vbr_quote_import_schedule(snapshot, view, schedule_quote));
            CHECK(schedule_quote.status() ==
                  vbr_import_schedule_status::downward);
        }
        if (upward) {
            std::vector<vbr_import_destination_child> destination_children;
            destination_children.reserve(snapshot.children.size());
            for (auto & child : snapshot.children) {
                CHECK(child.units.size() == 1);
                auto & unit = child.units[0];
                const auto source_type = static_cast<ggml_type>(
                    view.units()[child.child_id].descriptor.current_type);
                const bool exact = mixed_exact_upward && child.child_id != 0;
                const auto target_type = exact
                    ? source_type : upward_target_type;
                CHECK(source_type == upward_source_type);
                unit.current_type = target_type;
                unit.current_domain = vbr_downward_tier_domain(target_type);
                if (!exact) {
                    unit.upward_supported = true;
                    unit.upward_type = target_type;
                    unit.upward_domain = unit.current_domain;
                    unit.upward_recipe_id = VBR_UPWARD_RECIPE_ID;
                    unit.upward_recipe_version = VBR_UPWARD_RECIPE_VERSION;
                    unit.upward_meansub_model_id = 7;
                    const auto & source_descriptor =
                        view.units()[child.child_id].descriptor;
                    unit.upward_source_identity = {
                        source_descriptor.codebook_digest,
                        source_descriptor.rotation_digest,
                        source_descriptor.meansub_digest,
                        source_descriptor.meansub_model_id,
                        source_descriptor.meansub_layer,
                        source_descriptor.meansub_baked,
                        source_descriptor.representation.codec_id,
                        source_descriptor.representation.codec_version,
                        source_descriptor.representation.reference_digest,
                    };
                    unit.upward_target_identity = unit.upward_source_identity;
                    unit.upward_meansub_model_id =
                        unit.upward_source_identity.meansub_model_id;
                    CHECK(vbr_upward_resolve_recipe(
                        source_type, target_type, unit.upward_recipe) ==
                        vbr_upward_recipe_status::resolved);
                }
                uint64_t mapped = 0;
                uint64_t transfer = 0;
                for (auto & shard : unit.shards) {
                    uint64_t row = ggml_row_size(
                        target_type, int64_t(unit.dimensions[0]));
                    // The model-free fixture uses a narrow synthetic row,
                    // below the real TCQ block width. Preserve only the
                    // expansion relation; the permanent backend oracle covers
                    // real row geometry.
                    if (row == 0 && target_type != source_type) {
                        row = shard.row_bytes + 1;
                    }
                    CHECK(row != 0 &&
                          unit.wm_cells <= UINT64_MAX/row);
                    shard.row_bytes = row;
                    shard.mapped_bytes = unit.wm_cells*row;
                    mapped += shard.mapped_bytes;
                }
                for (const auto & shard :
                        view.units()[child.child_id].descriptor.shards) {
                    transfer += shard.payload_bytes;
                }
                if (!exact) {
                    unit.upward_row_bytes = unit.shards.front().row_bytes;
                    unit.upward_mapped_bytes = mapped;
                    unit.upward_transfer_bytes = transfer;
                    unit.upward_codec_workspace_bytes = 64;
                }

                vbr_import_destination_child destination;
                destination.initial_types = { target_type };
                destination.initial_cursor =
                    view.manifest().controller_policy[child.child_id].cursor;
                destination.watermark_cells = uint32_t(unit.wm_cells);
                destination_children.push_back(std::move(destination));
            }
            upward_destination = vbr_select_import_destination(
                destination_children, nullptr, measure_upward_destination);
            CHECK(upward_destination.status ==
                  vbr_import_destination_status::feasible_current);
            CHECK(vbr_digest_nonzero(upward_destination.tree_digest));
            for (auto & child : snapshot.children) {
                child.controller_policy.current_type_vector_digest =
                    upward_destination.child_type_digests[child.child_id];
                child.controller_policy.cursor =
                    upward_destination.final_cursors[child.child_id];
                auto & unit = child.units[0];
                if (!(mixed_exact_upward && child.child_id != 0)) {
                    unit.upward_build_identity_digest =
                        vbr_upward_build_identity(
                            unit.upward_recipe,
                            unit.upward_source_identity,
                            unit.upward_target_identity,
                            upward_destination.child_type_digests[child.child_id],
                            upward_destination.tree_digest);
                }
            }
            CHECK(vbr_quote_import_schedule(snapshot, view, schedule_quote));
            const auto expected_schedule =
                vbr_downward_tier_domain(upward_source_type) ==
                        vbr_downward_tier_domain(upward_target_type)
                    ? vbr_import_schedule_status::upward_same_domain
                    : vbr_import_schedule_status::upward_cross_domain;
            CHECK(schedule_quote.status() ==
                  expected_schedule);
            CHECK(vbr_rebind_import_schedule_quote(
                snapshot, view, upward_destination, schedule_quote));
            CHECK(schedule_quote.status() ==
                  expected_schedule);
            validation.downward_tree_digest = upward_destination.tree_digest;
        }
        if (with_companion) {
            const auto & descriptor = view.companions()[0].descriptor;
            snapshot.companions.push_back({
                descriptor.kind, descriptor.format_version,
                descriptor.build_identity_digest, true,
                &target.recurrent,
            });
            if (descriptor.kind ==
                    vbr_artifact_companion_kind::recurrent) {
                target.live.push_back({
                    2, nullptr,
                    reinterpret_cast<llama_memory_recurrent *>(
                        &target.recurrent),
                    checkpoint_child_dependency_mode::payload_complete,
                });
            }
        }
    }

    void fill_policy() {
        policy.authorized = true;
        policy.identity.execution_identity =
            view.manifest().identity.execution_identity;
        policy.identity.adapter_config_identity =
            view.manifest().identity.adapter_config_identity;
        policy.identity.media_content_identity =
            view.manifest().identity.media_content_identity;
        policy.identity.sequence_epoch =
            view.manifest().identity.sequence_epoch;
        policy.identity.requested_frontier =
            view.manifest().identity.next_position;
        policy.identity.tokens = &view.manifest().token_block.tokens;
        policy.destination_sequence = 0;
        policy.domain_bindings = bindings;
        policy.domain_bindings.push_back({ UINT32_MAX, UINT16_MAX, host });
        policy.accounting_snapshot = &snapshot_accounting;
        policy.budget_config = &budget;
        policy.context = &validation;
        policy.adoption_nonce = 0xf42a2001;
        policy.parse_companion = validation_context::parse_companion;
        policy.recheck_target_empty = validation_context::recheck;
        policy.read_accounting_serial = validation_context::read_accounting;
        policy.read_policy_epoch = validation_context::read_policy;
        if (downward) {
            downward_plan.accounting_serial = snapshot_accounting.serial;
            policy.transform_budget_plan = &downward_plan;
            policy.downward_projection = &downward_projection;
            policy.read_transform_tree_digest =
                validation_context::read_downward_tree;
        } else if (upward) {
            downward_plan.accounting_serial = snapshot_accounting.serial;
            policy.transform_budget_plan = &downward_plan;
            policy.schedule_quote = &schedule_quote;
            policy.read_transform_tree_digest =
                validation_context::read_downward_tree;
        }
    }

    llama_cache_acct_snapshot snapshot_accounting;

    void refresh() {
        snapshot_accounting = ledger.snapshot();
        snapshot.accounting_serial = snapshot_accounting.serial;
        occupied_live_snapshot.accounting_serial =
            snapshot_accounting.serial;
        policy.accounting_snapshot = &snapshot_accounting;
        if (downward || upward) {
            downward_plan.accounting_serial = snapshot_accounting.serial;
        }
    }

    vbr_adopt_stage_result stage(
            const vbr_adopt_stage_fault & fault = {},
            const llama_cache_budget_config * budget_override = nullptr) {
        refresh();
        auto validated = vbr_validate_unit_manifest_snapshot(
            snapshot, view, policy);
        CHECK(validated.status ==
              vbr_manifest_validation_status::validated);
        CHECK(validated.proof);
        vbr_adopt_stage_policy stage_policy;
        stage_policy.ledger = &ledger;
        stage_policy.budget = budget_override ? budget_override : &budget;
        stage_policy.pinned_domain = pinned;
        stage_policy.pinned_ring_bytes = 32;
        stage_policy.chunk_bytes = 8;
        for (const auto & binding : bindings) {
            stage_policy.lanes.push_back({
                binding.domain, nullptr, nullptr, false,
            });
        }
        if (downward || upward) {
            stage_policy.transform_context = this;
            stage_policy.reserve_transform = [](
                    void * context,
                    const std::vector<vbr_validated_child_plan> & plans,
                    llama_cache_acct_ledger &,
                    const llama_cache_budget_config &,
                    vbr_downward_stage_reservation & output) noexcept {
                auto & self = *static_cast<fixture *>(context);
                const auto expected = self.downward
                    ? vbr_import_transform_kind::downward
                    : vbr_downward_tier_domain(self.upward_source_type) ==
                              vbr_downward_tier_domain(self.upward_target_type)
                        ? vbr_import_transform_kind::upward_same_domain
                        : vbr_import_transform_kind::upward_cross_domain;
                if (plans.empty() || std::any_of(
                        plans.begin(), plans.end(),
                        [&](const vbr_validated_child_plan & plan) {
                            if (plan.transform_kind ==
                                    vbr_import_transform_kind::none) {
                                return false;
                            }
                            return plan.transform_kind != expected ||
                                (expected ==
                                     vbr_import_transform_kind::downward
                                    ? plan.transcode_recipe.n_edges == 0
                                    : plan.upward_recipe.n_edges != 1);
                        })) {
                    return false;
                }
                self.transform_reserve_plans = uint32_t(plans.size());
                self.transform_reserve_stash_only = uint32_t(std::count_if(
                    plans.begin(), plans.end(),
                    [](const vbr_validated_child_plan & plan) {
                        return plan.transform_kind ==
                                   vbr_import_transform_kind::none &&
                            plan.stash_action ==
                                vbr_validated_stash_action::restore_exact;
                    }));
                output.status = self.downward_reserve_status;
                if (self.downward && output.status ==
                        vbr_downward_reserve_status::reserved_stashless) {
                    output.stashless_units.push_back(
                        vbr_downward_unit_key(
                            plans.front().child_id,
                            plans.front().logical_unit_id));
                }
                return true;
            };
        }
        stage_policy.fault = fault;
        return vbr_stage_validated_manifest(
            std::move(validated.proof), stage_policy);
    }

    vbr_companion_adoption_provider companion_provider() {
        vbr_companion_adoption_provider out;
        out.kind = target.recurrent.kind;
        out.target_cookie = &target.recurrent;
        out.context = &target.recurrent;
        out.prepare = companion_prepare;
        out.prepare_replacement = companion_prepare_replacement;
        out.target_empty = companion_empty;
        out.recheck = companion_recheck;
        out.publish_swap = companion_publish;
        out.rollback = companion_rollback;
        return out;
    }
};

static vbr_adopt_result adopt(
        fixture & f, vbr_adopt_fault * fault = nullptr,
        bool install_companion = false) {
    auto staged = f.stage();
    CHECK(staged.status == vbr_adopt_stage_status::staged);
    CHECK(staged.manifest && staged.staged);
    if (!staged.manifest || !staged.staged) {
        return {};
    }
    f.validation.serial_bias = 0;
    vbr_composite_publish_hooks hooks;
    hooks.context = &f.target.target;
    hooks.owner_token = &f.target.target;
    hooks.validate_owner_token = owner_token_valid;
    vbr_adopt_test_control test;
    if (fault) {
        test.fault = *fault;
    }
    test.target = &f.target;
    hooks.test = &test;
    if (install_companion) {
        hooks.companions.push_back(f.companion_provider());
    }
    return vbr_adopt_empty_manifest(
        f.target.target, 0, std::move(*staged.manifest),
        std::move(*staged.staged), f.ledger, hooks);
}

static void test_real_driver_smoke() {
    fixture f;
    const auto result = adopt(f);
    if (result.status != vbr_adopt_status::adopted) {
        std::fprintf(stderr, "VBR adopt status=%s phase=%s accounting=%d\n",
                     vbr_adopt_status_name(result.status),
                     vbr_adopt_phase_name(result.phase),
                     int(result.accounting_status));
    }
    CHECK(result.status == vbr_adopt_status::adopted);
    CHECK(result.phase == vbr_adopt_phase::close);
    CHECK(result.children == 2);
    CHECK(result.units == 2);
    CHECK(f.target.transfer_calls == 4);
    CHECK(f.target.mark_calls == 2);
    CHECK(f.target.publish_calls == 2);
    if (f.ledger.snapshot().live_ops <= f.catalog_live_ops) {
        std::fprintf(stderr, "VBR live_ops=%llu baseline=%llu\n",
                     (unsigned long long) f.ledger.snapshot().live_ops,
                     (unsigned long long) f.catalog_live_ops);
    }
    CHECK(f.ledger.snapshot().live_ops > f.catalog_live_ops);
    f.target.erase_imported();
    CHECK(f.ledger.snapshot().live_ops == f.catalog_live_ops);
}

static vbr_adopt_result adopt_occupied(
        fixture & f, vbr_adopt_phase fail_phase = vbr_adopt_phase::_count,
        bool fail_after = false) {
    auto staged = f.stage();
    CHECK(staged.status == vbr_adopt_stage_status::staged);
    CHECK(staged.manifest && staged.staged);
    if (!staged.manifest || !staged.staged) {
        return {};
    }
    CHECK(staged.staged->read_count() ==
          (f.recycle ? 4u : 2u) + (f.with_companion ? 1u : 0u));
    vbr_composite_publish_hooks hooks;
    hooks.context = &f.target.target;
    hooks.owner_token = &f.target.target;
    hooks.validate_owner_token = owner_token_valid;
    if (f.with_companion) {
        hooks.companions.push_back(f.companion_provider());
    }
    vbr_adopt_test_control test;
    test.target = &f.target;
    if (fail_after) {
        test.fault.fail_after = fail_phase;
    } else {
        test.fault.fail_before = fail_phase;
    }
    hooks.test = &test;
    return vbr_adopt_empty_manifest(
        f.target.target, 0, std::move(*staged.manifest),
        std::move(*staged.staged), f.ledger, hooks);
}

static void test_occupied_spec_companion_replacement_and_rollback() {
    fixture success(
        true, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16, 0,
        vbr_artifact_clean_stash_state::absent_at_source,
        false, false, true, true, false, true);
    const auto adopted = adopt_occupied(success);
    CHECK(adopted.status == vbr_adopt_status::adopted);
    CHECK(adopted.companions == 1);
    CHECK(success.target.recurrent.replacement_prepares == 1);
    CHECK(success.target.recurrent.rollbacks == 0);
    CHECK(success.target.recurrent.live_image == 2);
    CHECK(success.target.recurrent.published);

    fixture late(
        true, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16, 0,
        vbr_artifact_clean_stash_state::absent_at_source,
        false, false, true, true, false, true);
    const auto refused = adopt_occupied(
        late, vbr_adopt_phase::complete_tree_barrier, false);
    CHECK(refused.status != vbr_adopt_status::adopted);
    CHECK(refused.recovery == vbr_adopt_recovery_outcome::replayed);
    CHECK(late.target.recurrent.replacement_prepares == 1);
    CHECK(late.target.recurrent.rollbacks == 1);
    CHECK(late.target.recurrent.live_image == 1);
    CHECK(late.target.recurrent.published);
    CHECK(!late.target.operation_quarantined);

    fixture recurrent(
        true, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16, 0,
        vbr_artifact_clean_stash_state::absent_at_source,
        false, false, true);
    const auto recurrent_adopted = adopt_occupied(recurrent);
    CHECK(recurrent_adopted.status == vbr_adopt_status::adopted);
    CHECK(recurrent_adopted.companions == 1);
    CHECK(recurrent.target.recurrent.kind ==
          vbr_artifact_companion_kind::recurrent);
    CHECK(recurrent.target.recurrent.replacement_prepares == 1);
    CHECK(recurrent.target.recurrent.live_image == 2);
}

static uint64_t device_transfer_staging_reserved(
        const llama_cache_acct_snapshot & snapshot) {
    uint64_t total = 0;
    for (const auto & row : snapshot.cells) {
        if (row.category != llama_cache_acct_category::transfer_staging ||
            row.domain.residency != llama_cache_acct_residency::device) {
            continue;
        }
        const auto & value = row.cell.measures[
            size_t(llama_cache_acct_measure::reserved)];
        if (value.state == llama_cache_acct_known::known) {
            total += value.value;
        }
    }
    return total;
}

static void test_occupied_replacement_free_cell_adoption() {
    fixture f(false, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
              0, vbr_artifact_clean_stash_state::absent_at_source,
              false, false, true);
    const auto result = adopt_occupied(f);
    CHECK(result.status == vbr_adopt_status::adopted);
    CHECK(result.h2d_bytes == 10);
    CHECK(f.target.occupied_rechecks == 2);
    CHECK(f.target.relocated_prepares == 1);
    CHECK(f.target.relocated_images == 1);
    CHECK(f.target.children[0].published);
    CHECK(f.target.children[0].receipts);
    CHECK(f.target.relocated_offsets.size() == 2);
    for (const auto & offset : f.target.relocated_offsets) {
        CHECK(offset.first == 0);
        CHECK(offset.second == 5);
    }
}

static void test_occupied_replacement_fault_preserves_incumbent() {
    fixture f(false, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
              0, vbr_artifact_clean_stash_state::absent_at_source,
              false, false, true);
    const auto sentinel = f.target.incumbent_sentinel;
    const auto result = adopt_occupied(
        f, vbr_adopt_phase::unit_h2d, true);
    CHECK(result.status == vbr_adopt_status::transfer_failed);
    CHECK(f.target.publish_calls == 0);
    CHECK(f.target.transfer_calls == 2);
    CHECK(f.target.children[0].mapped == 0);
    CHECK(!f.target.children[0].published);
    CHECK(!f.target.children[0].image_ready);
    CHECK(f.target.incumbent_sentinel == sentinel);
    CHECK(std::all_of(
        f.target.children[0].destination.begin(),
        f.target.children[0].destination.end(),
        [](uint8_t value) { return value == 0; }));
}

static void check_recycle_restored(
        fixture & f, const vbr_adopt_result & result,
        uint32_t expected_incoming) {
    CHECK(result.status != vbr_adopt_status::adopted);
    CHECK(result.recovery == vbr_adopt_recovery_outcome::replayed);
    CHECK(result.recovery_h2d_bytes == 10);
    CHECK(result.recovery_h2d_chunks == 2);
    CHECK(f.target.incoming_transfer_calls == expected_incoming);
    CHECK(f.target.recovery_transfer_calls == 2);
    CHECK(f.target.recovery_syncs == 1);
    CHECK(!f.target.operation_quarantined);
    CHECK(f.target.publish_calls == 0);
    CHECK(!f.target.children[0].published);
    CHECK(f.target.children[0].destination == f.target.incumbent_image);
    CHECK(f.ledger.snapshot().live_ops == f.catalog_live_ops);
    CHECK(f.target.transfer_order.size() == size_t(expected_incoming + 2));
    for (uint32_t i = 0; i < expected_incoming; ++i) {
        CHECK(f.target.transfer_order[i] ==
              vbr_staged_read_kind::unit_payload);
    }
    for (size_t i = expected_incoming;
         i < f.target.transfer_order.size(); ++i) {
        CHECK(f.target.transfer_order[i] ==
              vbr_staged_read_kind::recovery_unit_payload);
    }
}

static void test_occupied_recycle_success_and_zero_growth() {
    fixture f(false, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
              0, vbr_artifact_clean_stash_state::absent_at_source,
              false, false, true, true);
    auto staged = f.stage();
    CHECK(staged.status == vbr_adopt_stage_status::staged);
    CHECK(staged.manifest && staged.staged);
    if (!staged.manifest || !staged.staged) {
        return;
    }
    CHECK(staged.staged->read_count() == 4);
    CHECK(device_transfer_staging_reserved(f.ledger.snapshot()) == 0);

    vbr_composite_publish_hooks hooks;
    hooks.context = &f.target.target;
    hooks.owner_token = &f.target.target;
    hooks.validate_owner_token = owner_token_valid;
    vbr_adopt_test_control test;
    test.target = &f.target;
    hooks.test = &test;
    const auto result = vbr_adopt_empty_manifest(
        f.target.target, 0, std::move(*staged.manifest),
        std::move(*staged.staged), f.ledger, hooks);
    CHECK(result.status == vbr_adopt_status::adopted);
    CHECK(result.recovery == vbr_adopt_recovery_outcome::not_needed);
    CHECK(result.recovery_h2d_bytes == 0);
    CHECK(result.recovery_h2d_chunks == 0);
    CHECK(f.target.incoming_transfer_calls == 2);
    CHECK(f.target.recovery_transfer_calls == 0);
    CHECK(f.target.recovery_syncs == 0);
    CHECK(f.target.transfer_order ==
          std::vector<vbr_staged_read_kind>({
              vbr_staged_read_kind::unit_payload,
              vbr_staged_read_kind::unit_payload,
          }));
    CHECK(f.target.children[0].destination != f.target.incumbent_image);
    CHECK(f.target.children[0].published);
}

static void test_occupied_recycle_partial_write_and_late_fault_matrix() {
    for (const uint32_t bytes_before_failure : { 1u, 3u, 5u }) {
        fixture f(false, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
                  0, vbr_artifact_clean_stash_state::absent_at_source,
                  false, false, true, true);
        f.target.fail_incoming_transfer_at = 0;
        f.target.fail_incoming_after_bytes = bytes_before_failure;
        const auto result = adopt_occupied(f);
        CHECK(result.status == vbr_adopt_status::transfer_failed);
        check_recycle_restored(f, result, 1);
    }

    fixture late(false, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
                 0, vbr_artifact_clean_stash_state::absent_at_source,
                 false, false, true, true);
    const auto late_result = adopt_occupied(
        late, vbr_adopt_phase::complete_tree_barrier, false);
    CHECK(late_result.status != vbr_adopt_status::adopted);
    check_recycle_restored(late, late_result, 2);
}

static void test_occupied_recycle_replay_failure_quarantines() {
    fixture f(false, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
              0, vbr_artifact_clean_stash_state::absent_at_source,
              false, false, true, true);
    f.target.fail_incoming_transfer_at = 0;
    f.target.fail_incoming_after_bytes = 3;
    f.target.fail_recovery_transfer_at = 1;
    const auto result = adopt_occupied(f);
    CHECK(result.status == vbr_adopt_status::quarantined);
    CHECK(result.phase == vbr_adopt_phase::rollback);
    CHECK(result.recovery == vbr_adopt_recovery_outcome::quarantined);
    CHECK(result.recovery_h2d_bytes == 5);
    CHECK(result.recovery_h2d_chunks == 1);
    CHECK(f.target.recovery_transfer_calls == 2);
    CHECK(f.target.recovery_syncs == 0);
    CHECK(f.target.operation_quarantined);
    CHECK(f.target.publish_calls == 0);
}

static void test_occupied_transformed_recycle_uses_recovery_geometry() {
    fixture f(false, true, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
              0, vbr_artifact_clean_stash_state::absent_at_source,
              false, false, true, true, true);
    auto staged = f.stage();
    CHECK(staged.status == vbr_adopt_stage_status::staged);
    CHECK(staged.manifest && staged.staged);
    if (!staged.manifest || !staged.staged) {
        return;
    }
    uint32_t incoming = 0;
    uint32_t recovery = 0;
    for (const auto & read : staged.staged->reads()) {
        if (read.kind == vbr_staged_read_kind::unit_payload) {
            ++incoming;
            CHECK(read.size == 10);
        } else if (read.kind ==
                       vbr_staged_read_kind::recovery_unit_payload) {
            ++recovery;
            CHECK(read.size == 5);
            CHECK(read.destination_type == GGML_TYPE_TURBO3_TCQ);
        }
    }
    CHECK(incoming == 2);
    CHECK(recovery == 2);

    f.target.fail_incoming_transfer_at = 0;
    f.target.fail_incoming_after_bytes = 3;
    vbr_composite_publish_hooks hooks;
    hooks.context = &f.target.target;
    hooks.owner_token = &f.target.target;
    hooks.validate_owner_token = owner_token_valid;
    vbr_adopt_test_control test;
    test.target = &f.target;
    hooks.test = &test;
    const auto result = vbr_adopt_empty_manifest(
        f.target.target, 0, std::move(*staged.manifest),
        std::move(*staged.staged), f.ledger, hooks);
    CHECK(result.status == vbr_adopt_status::transfer_failed);
    CHECK(result.recovery == vbr_adopt_recovery_outcome::replayed);
    CHECK(result.recovery_h2d_bytes == 10);
    CHECK(f.target.recovery_transfer_calls == 2);
    // Only the compact incumbent row geometry is authoritative after replay;
    // the wider source alias tail is scratch and may remain cleared.
    for (const size_t base : { size_t(0), size_t(8) }) {
        CHECK(std::equal(
            f.target.children[0].destination.begin() + base,
            f.target.children[0].destination.begin() + base + 5,
            f.target.incumbent_image.begin() + base));
    }
    CHECK(!f.target.children[0].published);
    CHECK(!f.target.operation_quarantined);
}

static void test_occupied_replacement_tracker_consumes_canonical_map() {
    fixture f(false, false, false, GGML_TYPE_TURBO8_0, GGML_TYPE_F16,
              0, vbr_artifact_clean_stash_state::absent_at_source,
              false, false, true);
    auto staged = f.stage();
    CHECK(staged.status == vbr_adopt_stage_status::staged);
    CHECK(staged.manifest && staged.manifest->is_occupied_replacement());
    if (!staged.manifest || !staged.manifest->is_occupied_replacement()) {
        return;
    }
    const auto * guard = staged.manifest->occupied_replacement();
    const auto * source = staged.manifest->source_controller(0);
    CHECK(guard && source);
    CHECK(staged.manifest->tracker_install().children.size() == 1);
    if (!guard || !source ||
        staged.manifest->tracker_install().children.size() != 1) {
        return;
    }

    auto plan = staged.manifest->tracker_install().children.front();
    vbr_generation_tracker tracker(
        1, 10, uint32_t(plan.units.size()),
        vbr_lineage_uuid { 0x81, 0x82 });
    CHECK(tracker.active());
    plan.target_instance = tracker.runtime_instance();
    plan.lineage_uuid = tracker.lineage_identity();
    plan.transition = vbr_tracker_install_transition::whole_import;
    plan.global_generation = 1;
    for (auto & unit : plan.units) {
        unit.repr_gen = 1;
        unit.last_transition = vbr_repr_transition::whole_import;
    }
    vbr_tracker_import_image image;
    CHECK(tracker.prepare_relocated_import_image(
        plan, *source, 0, *guard, image));
    CHECK(image.ready());
    CHECK(image.stable());
    auto binding = import_binding(tracker.runtime_instance());
    vbr_scoped_operation operation(binding);
    CHECK(bool(operation));
    CHECK(tracker.import_image_installable(image, operation.id()));
    tracker.install_import_image_swap(image);
    for (const auto & mapping : guard->cell_mapping()) {
        CHECK(tracker.dependency_generation(
                  0, mapping.destination_physical_cell) == 1);
        CHECK(tracker.membership_generation(
                  0, mapping.destination_physical_cell) == 1);
        CHECK(tracker.last_membership_seq(
                  0, mapping.destination_physical_cell) == 0);
    }
    CHECK(operation.close(vbr_operation_outcome::committed));
}

static void test_erase_releases_receipt_for_second_adopt() {
    fixture f;
    const auto first = adopt(f);
    CHECK(first.status == vbr_adopt_status::adopted);
    CHECK(f.ledger.snapshot().live_ops > f.catalog_live_ops);

    // Model the production seq_rm empty transition. This is the same receipt
    // group and real accounting ledger used by the phase driver, not a second
    // accounting algorithm.
    f.target.erase_imported();
    CHECK(f.target.construction_empty());
    CHECK(f.ledger.snapshot().live_ops == f.catalog_live_ops);

    const auto second = adopt(f);
    CHECK(second.status == vbr_adopt_status::adopted);
    CHECK(second.phase == vbr_adopt_phase::close);
    CHECK(f.ledger.snapshot().live_ops > f.catalog_live_ops);
    f.target.erase_imported();
    CHECK(f.ledger.snapshot().live_ops == f.catalog_live_ops);
}

static void check_failed_transaction(
        fixture & f, const vbr_adopt_result & result) {
    CHECK(result.status != vbr_adopt_status::adopted);
    CHECK(f.target.construction_empty());
    CHECK(f.ledger.snapshot().live_ops == f.catalog_live_ops);
    CHECK(f.view.validate() == vbr_artifact_status::ok);
}

static void test_phase_fault_matrix() {
    // Every phase through tracker preparation is fallible on both sides of
    // its work. Composite publication has only the pre-boundary seam; its
    // publication and close are the deliberately seam-free no-fail region.
    for (uint8_t raw = uint8_t(vbr_adopt_phase::consume_capabilities);
         raw <= uint8_t(vbr_adopt_phase::tracker_prepare); ++raw) {
        for (const bool after : { false, true }) {
            fixture f;
            f.target.inject_phase = vbr_adopt_phase(raw);
            f.target.inject_after = after;
            const auto result = adopt(f);
            check_failed_transaction(f, result);
            if (raw < uint8_t(vbr_adopt_phase::private_backing) ||
                (raw == uint8_t(vbr_adopt_phase::private_backing) &&
                 !after)) {
                CHECK(f.target.map_calls == 0);
            }
            if (raw < uint8_t(vbr_adopt_phase::unit_h2d) ||
                (raw == uint8_t(vbr_adopt_phase::unit_h2d) && !after)) {
                CHECK(f.target.transfer_calls == 0);
            }
        }
    }
    {
        fixture f;
        f.target.inject_phase = vbr_adopt_phase::composite_publish;
        const auto result = adopt(f);
        check_failed_transaction(f, result);
    }

    // There is intentionally no post-publication or pre-close injection
    // point: attempts to arm those non-boundaries are inert and must commit.
    for (const auto & attempt : {
            std::make_pair(vbr_adopt_phase::composite_publish, true),
            std::make_pair(vbr_adopt_phase::close, false),
            std::make_pair(vbr_adopt_phase::close, true) }) {
        fixture f;
        f.target.inject_phase = attempt.first;
        f.target.inject_after = attempt.second;
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::adopted);
        CHECK(result.phase == vbr_adopt_phase::close);
        f.target.erase_imported();
        CHECK(f.ledger.snapshot().live_ops == f.catalog_live_ops);
    }
}

static void test_shard_child_and_partial_map_matrix() {
    for (uint32_t child = 0; child < 2; ++child) {
        for (uint32_t shard_index = 0; shard_index < 2; ++shard_index) {
            fixture f;
            f.target.fail_transfer_child = child;
            f.target.fail_transfer_shard = shard_index;
            const auto result = adopt(f);
            CHECK(result.status == vbr_adopt_status::transfer_failed);
            CHECK(result.phase == vbr_adopt_phase::unit_h2d);
            CHECK(result.units == 0);
            CHECK(f.target.mark_calls == 0);
            CHECK(f.target.publish_calls == 0);
            check_failed_transaction(f, result);
        }
    }
    {
        fixture f;
        f.target.fail_map_child = 0;
        f.target.fail_map_after = 3;
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::private_backing_failed);
        CHECK(f.target.map_calls == 3);
        CHECK(f.target.children[0].unmapped == 3);
        CHECK(f.target.children[0].unmapped_ranges ==
              std::vector<uint32_t>({ 2, 1, 0 }));
        CHECK(f.target.children[1].mapped == 0);
        check_failed_transaction(f, result);
    }
    {
        fixture f;
        vbr_adopt_fault fault;
        fault.fail_h2d_completion = 0;
        const auto result = adopt(f, &fault);
        CHECK(result.status == vbr_adopt_status::transfer_failed);
        CHECK(result.h2d_bytes == 0);
        CHECK(f.target.transfer_calls == 0);
        check_failed_transaction(f, result);
    }
}

static void test_complete_tree_barrier_matrix() {
    {
        fixture f;
        f.target.drop_attention_at_barrier = true;
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::target_drift);
        CHECK(result.phase == vbr_adopt_phase::complete_tree_barrier);
        check_failed_transaction(f, result);
    }
    {
        fixture f;
        f.target.add_recurrent_at_barrier = true;
        const auto result = adopt(f);
        CHECK(result.status ==
              vbr_adopt_status::required_companion_unavailable);
        CHECK(result.phase == vbr_adopt_phase::complete_tree_barrier);
        check_failed_transaction(f, result);
    }
    {
        fixture f(true);
        f.target.drift_recurrent_at_barrier = true;
        const auto result = adopt(f, nullptr, true);
        CHECK(result.status == vbr_adopt_status::target_drift);
        CHECK(result.phase == vbr_adopt_phase::complete_tree_barrier);
        CHECK(f.target.recurrent.rollbacks == 1);
        check_failed_transaction(f, result);
    }
    {
        fixture f(true);
        f.target.recurrent.fail_prepare = true;
        const auto result = adopt(f, nullptr, true);
        CHECK(result.status == vbr_adopt_status::companion_failed);
        CHECK(result.phase == vbr_adopt_phase::stash_and_companions);
        // Both attention children reached private backing/H2D first, then
        // the recurrent failure rolled both journals back to empty.
        CHECK(f.target.children[0].unmapped == 5);
        CHECK(f.target.children[1].unmapped == 5);
        CHECK(f.target.transfer_calls == 4);
        CHECK(f.target.mark_calls == 2);
        check_failed_transaction(f, result);
    }
}

static void test_serial_and_admission_faults() {
    {
        fixture f;
        f.target.drift_serial_at_phase3 = true;
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::target_drift);
        CHECK(result.phase == vbr_adopt_phase::target_recheck);
        check_failed_transaction(f, result);
    }
    {
        fixture f;
        f.target.drift_serial_at_phase10 = true;
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::barrier_failed);
        CHECK(result.phase == vbr_adopt_phase::complete_tree_barrier);
        check_failed_transaction(f, result);
    }
    {
        fixture f;
        const uint64_t baseline = f.ledger.snapshot().live_ops;
        vbr_adopt_stage_fault fault;
        fault.fail_before_prepare = true;
        auto staged = f.stage(fault);
        CHECK(staged.status == vbr_adopt_stage_status::internal_error);
        CHECK(!staged.staged);
        CHECK(f.target.construction_empty());
        CHECK(f.ledger.snapshot().live_ops == baseline);
    }
    {
        fixture f;
        const uint64_t baseline = f.ledger.snapshot().live_ops;
        auto refused = f.budget;
        for (auto & device : refused.devices) {
            device.configured_cache_cap = 0;
            device.cache_cap_state =
                llama_cache_budget_capacity_state::known;
        }
        auto staged = f.stage({}, &refused);
        CHECK(staged.status == vbr_adopt_stage_status::admission_refused);
        CHECK(!staged.staged);
        CHECK(f.target.construction_empty());
        CHECK(f.ledger.snapshot().live_ops == baseline);
    }
}

static void test_downward_subphase_matrix() {
    {
        fixture f(false, true);
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::adopted);
        CHECK(result.decision == vbr_import_decision::downward_rebase);
        CHECK(result.consistency ==
              vbr_artifact_consistency_kind::live_rebased);
        CHECK(result.downward_subphase ==
              vbr_downward_adopt_subphase::edge_completion);
        CHECK(f.target.downward_zero_inits == f.view.units().size());
        CHECK(f.target.downward_edges != 0);
        CHECK(f.target.downward_syncs == 2);
        // The fake reserve makes only child 0 stashless. Child 1 proves that
        // the first outgoing tapped edge regenerates a clean stash and that
        // stashless evidence is keyed by (child,unit), not a colliding unit id.
        CHECK(f.target.downward_stashes ==
              f.view.units()[1].descriptor.shards.size());
    }

    vbr_downward_recipe recipe;
    CHECK(vbr_downward_resolve_recipe(
        GGML_TYPE_TURBO8_0, GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO1_TCQ, true, recipe) ==
        vbr_downward_recipe_status::resolved);
    for (uint32_t edge = 0; edge < recipe.n_edges; ++edge) {
        fixture f(false, true);
        vbr_adopt_fault fault;
        fault.fail_downward_edge = edge;
        const auto result = adopt(f, &fault);
        CHECK(result.status == vbr_adopt_status::downward_transform_failed);
        CHECK(result.phase == vbr_adopt_phase::unit_h2d);
        CHECK(result.downward_subphase ==
              vbr_downward_adopt_subphase::edge_transcode);
        CHECK(result.downward_edge == edge);
        check_failed_transaction(f, result);
    }
    {
        fixture f(false, true);
        f.target.drift_downward_at_phase10 = true;
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::barrier_failed);
        CHECK(result.phase == vbr_adopt_phase::complete_tree_barrier);
        check_failed_transaction(f, result);
    }
    {
        fixture f(false, true);
        f.target.fail_transfer_child = 0;
        f.target.fail_transfer_shard = 0;
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::transfer_failed);
        CHECK(result.downward_subphase ==
              vbr_downward_adopt_subphase::source_h2d);
        check_failed_transaction(f, result);
    }
    {
        fixture f(false, true);
        vbr_adopt_fault fault;
        fault.fail_downward_stash = true;
        const auto result = adopt(f, &fault);
        CHECK(result.status == vbr_adopt_status::downward_stash_unavailable);
        CHECK(result.phase == vbr_adopt_phase::unit_h2d);
        CHECK(result.downward_subphase ==
              vbr_downward_adopt_subphase::edge_stash_capture);
        check_failed_transaction(f, result);
    }
    {
        fixture f(false, true);
        vbr_adopt_fault fault;
        fault.fail_downward_completion = true;
        const auto result = adopt(f, &fault);
        CHECK(result.status == vbr_adopt_status::downward_transform_failed);
        CHECK(result.phase == vbr_adopt_phase::unit_h2d);
        CHECK(result.downward_subphase ==
              vbr_downward_adopt_subphase::edge_completion);
        check_failed_transaction(f, result);
    }
    for (const auto status : {
            vbr_downward_reserve_status::not_attempted,
            vbr_downward_reserve_status::projection_unavailable,
            vbr_downward_reserve_status::accounting_refused,
            vbr_downward_reserve_status::workspace_reserve_failed,
            vbr_downward_reserve_status::required_stash_reserve_failed,
            vbr_downward_reserve_status::internal_error }) {
        fixture f(false, true);
        const uint64_t baseline = f.ledger.snapshot().live_ops;
        f.downward_reserve_status = status;
        auto staged = f.stage();
        CHECK(staged.status ==
              vbr_adopt_stage_status::transform_reserve_failed);
        CHECK(staged.transform_status == status);
        CHECK(!staged.staged);
        CHECK(f.target.construction_empty());
        CHECK(f.ledger.snapshot().live_ops == baseline);
    }
}

static void test_upward_reconstruction() {
    {
        fixture f(false, false, true);
        uint64_t compact_bytes = 0;
        for (const auto & unit : f.view.units()) {
            for (const auto & shard : unit.descriptor.shards) {
                compact_bytes += shard.payload_bytes;
            }
        }
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::adopted);
        CHECK(result.decision == vbr_import_decision::upward_reconstruct);
        CHECK(result.consistency ==
              vbr_artifact_consistency_kind::live_rebased);
        CHECK(result.h2d_bytes == compact_bytes);
        CHECK(f.target.upward_zero_inits == f.view.units().size());
        CHECK(f.target.upward_transforms == f.view.units().size());
        CHECK(f.target.upward_stash_transforms == 0);
        CHECK(f.target.upward_null_stash_transforms ==
              f.view.units().size());
        CHECK(f.target.upward_syncs == 2);
        CHECK(f.target.downward_edges == 0);
        CHECK(f.target.downward_stashes == 0);
        CHECK(f.target.downward_syncs == 0);
        CHECK(result.downward_subphase ==
              vbr_downward_adopt_subphase::none);
        CHECK(result.downward_edge == UINT32_MAX);
    }
    {
        fixture f(false, false, true);
        vbr_adopt_fault fault;
        fault.fail_after = vbr_adopt_phase::unit_h2d;
        const auto result = adopt(f, &fault);
        CHECK(result.status == vbr_adopt_status::transfer_failed);
        CHECK(result.phase == vbr_adopt_phase::unit_h2d);
        CHECK(f.target.upward_transforms == f.view.units().size());
        CHECK(f.target.upward_syncs == 2);
        check_failed_transaction(f, result);
    }
    // A tapped-domain reconstruction is one direct transform over the
    // source-sized upload.  Its authenticated clean stash survives and the
    // quality history advances exactly once.
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 0,
            vbr_artifact_clean_stash_state::present);
        f.refresh();
        auto validation = vbr_validate_unit_manifest_snapshot(
            f.snapshot, f.view, f.policy);
        CHECK(validation.status ==
              vbr_manifest_validation_status::validated);
        CHECK(validation.proof);
        for (const auto & plan : validation.proof->children()) {
            CHECK(plan.transform_kind ==
                  vbr_import_transform_kind::upward_same_domain);
            CHECK(plan.source_domain == vbr_repr_domain::tapped);
            CHECK(plan.selected_target_domain == vbr_repr_domain::tapped);
            CHECK(plan.target_last_source_type == GGML_TYPE_TURBO2_TCQ);
            CHECK(plan.target_promote_hops == 1);
            CHECK(plan.stash_action ==
                  vbr_validated_stash_action::restore_exact);
            CHECK(plan.descriptor.clean_stash.valid_rows == 5);
        }
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::adopted);
        CHECK(result.h2d_bytes == 60); // 20 source + 40 f16 clean-stash bytes
        CHECK(f.target.upward_transforms == 2);
        CHECK(f.target.upward_stash_transforms == 2);
        CHECK(f.target.upward_null_stash_transforms == 0);
        CHECK(f.target.upward_stash_rows == 10);
        CHECK(f.target.upward_syncs == 2);
        CHECK(std::is_sorted(
            f.target.upward_events.begin(), f.target.upward_events.end()));
        for (const auto event : {
                seam::upward_event::unit_h2d,
                seam::upward_event::stash_h2d,
                seam::upward_event::transform,
                seam::upward_event::synchronize }) {
            CHECK(std::find(
                f.target.upward_events.begin(),
                f.target.upward_events.end(), event) !=
                f.target.upward_events.end());
        }
    }
    // Cross-domain reconstruction consumes the authenticated centered stash,
    // adds the baked source mean, and publishes full-domain bytes without
    // retaining the now-inapplicable tapped stash.
    for (const ggml_type target : {
            GGML_TYPE_TURBO8_0, GGML_TYPE_F16 }) {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, target, 0,
            vbr_artifact_clean_stash_state::present);
        f.refresh();
        auto validation = vbr_validate_unit_manifest_snapshot(
            f.snapshot, f.view, f.policy);
        CHECK(validation.status ==
              vbr_manifest_validation_status::validated);
        CHECK(validation.proof);
        for (const auto & plan : validation.proof->children()) {
            CHECK(plan.transform_kind ==
                  vbr_import_transform_kind::upward_cross_domain);
            CHECK(plan.upward_recipe.edges[0].mean_action ==
                  vbr_upward_mean_action::add_baked_source_mean);
            CHECK(plan.transcode_source_identity.meansub_baked);
            CHECK(plan.transcode_source_identity.meansub_model_id ==
                  plan.transcode_target_identity.meansub_model_id);
            CHECK(plan.transcode_source_identity.meansub_layer ==
                  plan.transcode_target_identity.meansub_layer);
            CHECK(plan.target_last_source_type == GGML_TYPE_TURBO2_TCQ);
            CHECK(plan.target_promote_hops == 1);
            CHECK(plan.stash_action ==
                  vbr_validated_stash_action::consume_exact_then_drop);
        }
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::adopted);
        CHECK(result.h2d_bytes == 60);
        CHECK(f.target.upward_transforms == 2);
        CHECK(f.target.upward_stash_transforms == 2);
        CHECK(f.target.upward_stash_rows == 10);
        CHECK(f.target.upward_syncs == 2);
    }
    // A tapped exact-stash import cannot use downward's stashless fallback.
    // Its mandatory physical endpoint is rejected while staging, before any
    // source or stash byte reaches the target.
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 0,
            vbr_artifact_clean_stash_state::present);
        f.downward_reserve_status =
            vbr_downward_reserve_status::required_stash_reserve_failed;
        auto staged = f.stage();
        CHECK(staged.status ==
              vbr_adopt_stage_status::transform_reserve_failed);
        CHECK(staged.transform_status ==
              vbr_downward_reserve_status::required_stash_reserve_failed);
        CHECK(!staged.staged);
        CHECK(f.target.transfer_calls == 0);
        CHECK(std::all_of(
            f.target.children.begin(), f.target.children.end(),
            [](const child_state & child) { return child.h2d_bytes == 0; }));
        CHECK(f.target.construction_empty());
    }
    // A second tapped promotion is allowed, while a third is rejected before
    // any stage/adoption work is created.
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 1,
            vbr_artifact_clean_stash_state::present, false, true);
        f.refresh();
        auto validation = vbr_validate_unit_manifest_snapshot(
            f.snapshot, f.view, f.policy);
        CHECK(validation.status ==
              vbr_manifest_validation_status::validated);
        CHECK(validation.proof);
        for (const auto & plan : validation.proof->children()) {
            CHECK(plan.target_promote_hops == 2);
            CHECK(plan.stash_action ==
                  vbr_validated_stash_action::omit_live_rebased);
        }
        CHECK(adopt(f).status == vbr_adopt_status::adopted);
        CHECK(f.target.upward_stash_transforms == 0);
        CHECK(f.target.upward_null_stash_transforms == 2);
    }
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 2);
        f.refresh();
        auto validation = vbr_validate_unit_manifest_snapshot(
            f.snapshot, f.view, f.policy);
        CHECK(validation.status ==
              vbr_manifest_validation_status::representation_mismatch);
        CHECK(!validation.proof);
        CHECK(f.target.construction_empty());
    }
    // Exact siblings stay byte-identical and do not acquire a transform; the
    // transformed child still uses one source upload and one backend sync.
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 0,
            vbr_artifact_clean_stash_state::present, true, true);
        f.refresh();
        auto validation = vbr_validate_unit_manifest_snapshot(
            f.snapshot, f.view, f.policy);
        CHECK(validation.status ==
              vbr_manifest_validation_status::validated);
        CHECK(validation.proof);
        CHECK(validation.proof->children().size() == 2);
        CHECK(validation.proof->children()[0].transform_kind ==
              vbr_import_transform_kind::upward_same_domain);
        CHECK(validation.proof->children()[1].transform_kind ==
              vbr_import_transform_kind::none);
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::adopted);
        CHECK(result.h2d_bytes == 20);
        CHECK(f.target.upward_zero_inits == 1);
        CHECK(f.target.upward_transforms == 1);
        CHECK(f.target.upward_stash_transforms == 0);
        CHECK(f.target.upward_null_stash_transforms == 1);
        CHECK(f.target.upward_syncs == 1);
    }
    // A full-stash exact sibling in a distinct child participates in the same
    // pre-transfer resource wave even though it has no codec workspace. Both
    // authenticated stash reads precede the one transformed child's work.
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 0,
            vbr_artifact_clean_stash_state::present, true, false);
        const auto result = adopt(f);
        CHECK(result.status == vbr_adopt_status::adopted);
        CHECK(f.transform_reserve_plans == 2);
        CHECK(f.transform_reserve_stash_only == 1);
        CHECK(f.target.upward_transforms == 1);
        CHECK(f.target.upward_stash_transforms == 1);
        CHECK(f.target.upward_syncs == 1);
    }
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 0,
            vbr_artifact_clean_stash_state::present, true, false);
        f.downward_reserve_status =
            vbr_downward_reserve_status::required_stash_reserve_failed;
        auto staged = f.stage();
        CHECK(staged.status ==
              vbr_adopt_stage_status::transform_reserve_failed);
        CHECK(!staged.staged);
        CHECK(f.transform_reserve_plans == 2);
        CHECK(f.transform_reserve_stash_only == 1);
        CHECK(f.target.transfer_calls == 0);
        CHECK(f.target.construction_empty());
    }
    // A post-transform fault synchronizes the submitted work, then returns to
    // the construction-empty state without publishing partial metadata.
    {
        fixture f(
            false, false, true,
            GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO4_0, 0,
            vbr_artifact_clean_stash_state::present);
        vbr_adopt_fault fault;
        fault.fail_after = vbr_adopt_phase::unit_h2d;
        const auto result = adopt(f, &fault);
        CHECK(result.status == vbr_adopt_status::transfer_failed);
        CHECK(f.target.upward_transforms == 2);
        CHECK(f.target.upward_stash_transforms == 2);
        CHECK(f.target.upward_null_stash_transforms == 0);
        CHECK(f.target.upward_stash_rows == 10);
        CHECK(f.target.upward_syncs == 2);
        check_failed_transaction(f, result);
    }
}

} // namespace adoption_fixture

namespace {

struct packed_h2d_source {
    std::vector<uint8_t> bytes;
    uint64_t bytes_read = 0;
    uint64_t read_calls = 0;

    static bool read(const void * opaque, uint64_t offset,
                     uint8_t * output, size_t size) noexcept {
        auto & source = *const_cast<packed_h2d_source *>(
            static_cast<const packed_h2d_source *>(opaque));
        if (!output || offset > source.bytes.size() ||
            size > source.bytes.size() - size_t(offset)) {
            return false;
        }
        std::copy_n(source.bytes.data() + size_t(offset), size, output);
        source.bytes_read += size;
        ++source.read_calls;
        return true;
    }
};

struct packed_h2d_destination {
    std::vector<uint8_t> bytes;

    static bool issue(void * opaque, uint64_t, uint64_t offset,
                      const uint8_t * data, size_t size,
                      bool asynchronous) noexcept {
        auto & destination =
            *static_cast<packed_h2d_destination *>(opaque);
        if (!data || asynchronous || offset > destination.bytes.size() ||
            size > destination.bytes.size() - size_t(offset)) {
            return false;
        }
        std::copy_n(data, size, destination.bytes.data() + size_t(offset));
        return true;
    }

    static bool complete(void *, uint64_t) noexcept { return true; }
};

static void test_packed_h2d_projection_stream() {
    packed_h2d_source source;
    source.bytes.resize(64);
    for (size_t i = 0; i < source.bytes.size(); ++i) {
        source.bytes[i] = uint8_t(i + 1);
    }
    const std::array<vbr_h2d_source_range, 3> ranges {{
        { 40, 7 }, { 3, 5 }, { 20, 9 },
    }};
    std::vector<uint8_t> expected(4, 0xa5);
    for (const auto & range : ranges) {
        expected.insert(expected.end(),
            source.bytes.begin() + range.source_offset,
            source.bytes.begin() + range.source_offset + range.size);
    }
    expected.resize(32, 0xa5);
    packed_h2d_destination destination { std::vector<uint8_t>(32, 0xa5) };

    vbr_h2d_status status;
    auto ring = vbr_h2d_chunk_ring::create({ {} }, 16, 8, status);
    CHECK(ring && status == vbr_h2d_status::ok);
    if (!ring) {
        return;
    }
    auto operation = ring->try_begin_operation();
    CHECK(bool(operation));
    vbr_h2d_packed_transfer transfer;
    transfer.source = { source.bytes.size(), &source, packed_h2d_source::read };
    transfer.ranges = ranges.data();
    transfer.range_count = ranges.size();
    transfer.size = 21;
    transfer.destination_offset = 4;
    transfer.fake = {
        &destination, packed_h2d_destination::issue,
        packed_h2d_destination::complete, false,
    };
    vbr_h2d_stats stats;
    CHECK(ring->stream_packed_reserved(operation, transfer, stats) ==
          vbr_h2d_status::ok);
    CHECK(destination.bytes == expected);
    CHECK(source.bytes_read == transfer.size);
    CHECK(stats.bytes == transfer.size);
    CHECK(stats.chunks == 3);

    // The reservation spans the complete projected unit set rather than
    // permitting the other direction to splice work between shards.
    vbr_h2d_transfer competing;
    competing.source = transfer.source;
    competing.size = 1;
    competing.fake = transfer.fake;
    CHECK(ring->stream(competing, stats) ==
          vbr_h2d_status::ring_unavailable);
    operation = {};
    CHECK(ring->stream(competing, stats) == vbr_h2d_status::ok);
}

static void test_packed_h2d_projection_max_ranges() {
    static constexpr size_t n = 1024*1024;
    packed_h2d_source source;
    source.bytes.resize(n);
    std::vector<vbr_h2d_source_range> ranges(n);
    for (size_t i = 0; i < n; ++i) {
        source.bytes[i] = uint8_t(i & 0xff);
        ranges[i] = { n - 1 - i, 1 };
    }
    packed_h2d_destination destination { std::vector<uint8_t>(n) };
    vbr_h2d_status status;
    auto ring = vbr_h2d_chunk_ring::create(
        { {} }, 128*1024, 64*1024, status);
    CHECK(ring && status == vbr_h2d_status::ok);
    if (!ring) {
        return;
    }
    auto operation = ring->try_begin_operation();
    vbr_h2d_packed_transfer transfer;
    transfer.source = { source.bytes.size(), &source, packed_h2d_source::read };
    transfer.ranges = ranges.data();
    transfer.range_count = ranges.size();
    transfer.size = n;
    transfer.fake = {
        &destination, packed_h2d_destination::issue,
        packed_h2d_destination::complete, false,
    };
    vbr_h2d_stats stats;
    CHECK(ring->stream_packed_reserved(operation, transfer, stats) ==
          vbr_h2d_status::ok);
    CHECK(stats.bytes == n);
    CHECK(stats.chunks == 16);
    CHECK(source.bytes_read == n);
    CHECK(source.read_calls == n);
    CHECK(destination.bytes.front() == source.bytes.back());
    CHECK(destination.bytes.back() == source.bytes.front());
}

} // namespace

static void test_cuda_h2d_adapter() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * candidate = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(candidate) ==
                GGML_BACKEND_DEVICE_TYPE_GPU) {
            device = candidate;
            break;
        }
    }
    CHECK(device != nullptr);
    if (!device) {
        return;
    }
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    CHECK(backend != nullptr);
    if (!backend) {
        return;
    }
    constexpr size_t n = 3*1024*1024 + 29;
    std::vector<uint8_t> expected(n);
    for (size_t i = 0; i < n; ++i) {
        expected[i] = uint8_t((i*41 + 7) & 0xff);
    }
    auto chain = std::make_shared<artifact_segment_chain>();
    CHECK(chain->append(expected.data(), expected.size()));
    ggml_init_params params = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context * context = ggml_init(params);
    CHECK(context != nullptr);
    ggml_tensor * tensor = context
        ? ggml_new_tensor_1d(context, GGML_TYPE_I8, n) : nullptr;
    ggml_backend_buffer_t buffer = tensor
        ? ggml_backend_alloc_ctx_tensors(context, backend) : nullptr;
    CHECK(tensor != nullptr && buffer != nullptr);
    if (tensor && buffer) {
        vbr_h2d_status ring_status;
        vbr_h2d_lane_binding lane;
        lane.device = device;
        lane.backend = backend;
        auto ring = vbr_h2d_chunk_ring::create(
            { lane }, 2*1024*1024,
            1024*1024, ring_status);
        CHECK(ring != nullptr);
        CHECK(ring_status == vbr_h2d_status::ok);
        if (ring) {
            vbr_h2d_transfer transfer;
            transfer.lane = 0;
            transfer.source = chain->source();
            transfer.size = chain->size();
            transfer.backend = backend;
            transfer.device = device;
            transfer.destination = tensor;
            vbr_h2d_stats stats;
            CHECK(ring->stream(transfer, stats) == vbr_h2d_status::ok);
            CHECK(stats.bytes == n);
            CHECK(stats.chunks >= 4);
            ggml_backend_synchronize(backend);
            std::vector<uint8_t> actual(n);
            ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
            CHECK(actual == expected);
        }
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (context) {
        ggml_free(context);
    }
    ggml_backend_free(backend);
}

static bool model_backed_adoption(
        const char * model_path, bool downward_mode,
        ggml_type source_type = GGML_TYPE_F16,
        ggml_type target_type = GGML_TYPE_F16,
        uint64_t live_budget_mib = 0,
        size_t live_token_count = 0,
        bool require_straddled = false,
        uint64_t destination_budget_mib = 0,
        ggml_type expected_destination_type = GGML_TYPE_COUNT) {
    const bool live_matrix = live_token_count != 0;
    const bool require_destination_degraded =
        expected_destination_type != GGML_TYPE_COUNT;
    vbr_upward_recipe destination_upward_recipe;
    const bool destination_upward = require_destination_degraded &&
        vbr_upward_resolve_recipe(
            source_type, expected_destination_type,
            destination_upward_recipe) ==
                vbr_upward_recipe_status::resolved;
    const bool needs_transform = downward_mode || destination_upward;
    const auto expected_schedule_status = downward_mode
        ? vbr_import_schedule_status::downward
        : destination_upward
            ? (vbr_downward_tier_domain(source_type) ==
                       vbr_downward_tier_domain(expected_destination_type)
                   ? vbr_import_schedule_status::upward_same_domain
                   : vbr_import_schedule_status::upward_cross_domain)
            : vbr_import_schedule_status::exact;
    ggml_backend_load_all();
    if (live_matrix) {
        // The live-schedule gate validates the shipped, model-matched order.
        // The older homogeneous
        // tier oracle intentionally keeps the generic test order.
        unsetenv("VBR_FORCE_GENERIC");
    } else {
        setenv("VBR_FORCE_GENERIC", "1", 1);
    }
    setenv("VBR_PROMOTE", "0", 1);
    setenv("VBR_STASH_ROWS", (downward_mode || live_matrix) ? "64" : "0", 1);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model_ptr model(llama_model_load_from_file(model_path, model_params));
    CHECK(model != nullptr);
    if (!model) {
        return false;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = live_matrix
        ? uint32_t(std::max<size_t>(256, live_token_count + 128)) : 256;
    context_params.n_batch = live_matrix ? 512 : 64;
    context_params.n_ubatch = live_matrix ? 512 : 64;
    context_params.n_seq_max = 1;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.type_k = GGML_TYPE_F16;
    context_params.type_v = GGML_TYPE_F16;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.vbr_dynamic = true;
    context_params.vbr_budget_explicit = true;
    context_params.vbr_vram_budget_bytes = live_matrix
        ? live_budget_mib*1024*1024 : 4ull*1024*1024*1024;
    if (downward_mode) {
        context_params.vbr_min_bits = 1.0;
        context_params.vbr_min_bits_explicit = true;
    }

    auto make_context = [&](ggml_type entry_type, uint64_t budget_mib = 0) {
        auto params = context_params;
        params.type_k = entry_type;
        params.type_v = entry_type;
        if (budget_mib != 0) {
            params.vbr_vram_budget_bytes = budget_mib*1024*1024;
        }
        return llama_context_ptr(llama_init_from_model(model.get(), params));
    };
    auto source = make_context(source_type);
    CHECK(source != nullptr);
    if (!source) {
        return false;
    }
    std::string prompt = "Atomic artifact restore parity fixture.";
    if (live_matrix) {
        prompt.clear();
        prompt.reserve(live_token_count*8);
        for (size_t i = 0; i < live_token_count; ++i) {
            prompt += " VBR ledger row ";
            prompt += std::to_string(i);
            prompt += " preserves deterministic controller pressure.";
        }
    }
    auto tokens = common_tokenize(source.get(), prompt, true, false);
    if (tokens.size() < 4) {
        return false;
    }
    if (live_matrix && tokens.size() < live_token_count) {
        return false;
    }
    if (live_matrix) {
        tokens.resize(live_token_count);
    } else if (tokens.size() > 32) {
        tokens.resize(32);
    }
    const llama_token continuation = tokens.back();
    const int32_t vocab_size = llama_vocab_n_tokens(
        llama_model_get_vocab(model.get()));
    std::vector<float> live_source_logits;
    if (live_matrix) {
        // Dynamic VBR deliberately permits one controller context per process.
        // Build the uninterrupted oracle first, destroy it, then reproduce the
        // frozen schedule in the context that will be captured. This proves
        // schedule reproducibility without mutating the sealed controller.
        CHECK(model_adoption_decode(source.get(), tokens, 0));
        CHECK(model_adoption_decode(
            source.get(), { continuation }, llama_pos(tokens.size())));
        float * logits = llama_get_logits_ith(source.get(), -1);
        CHECK(logits != nullptr && vocab_size > 0);
        if (!logits || vocab_size <= 0) {
            return false;
        }
        live_source_logits.assign(logits, logits+vocab_size);
        llama_synchronize(source.get());
        source.reset();
        source = make_context(GGML_TYPE_F16);
        CHECK(source != nullptr);
        if (!source) {
            return false;
        }
    }
    CHECK(model_adoption_decode(source.get(), tokens, 0));
    if (failures) {
        return false;
    }
    // Explicit capture is an idle decode-thread transaction. Long live-schedule prefills
    // span several scheduler submissions, so close the final submitted VBR
    // operation before asking the capture door to prove quiescence.
    llama_synchronize(source.get());

    llama_memory_i * source_memory = llama_get_memory(source.get());
    std::vector<vbr_explicit_capture_runtime_pool> source_pools;
    uint32_t source_children = 0;
    CHECK(source_memory && vbr_explicit_capture_runtime_pools(
        *source_memory, source_pools, source_children));
    CHECK(source_children == 1 && !source_pools.empty());
    if (!source_memory || source_children != 1 || source_pools.empty()) {
        return false;
    }
    ggml_backend_dev_t device = source_pools.front().backend_device;
    ggml_backend_t source_backend = source_pools.front().backend;
    if (!source_backend) {
        return false;
    }

    vbr_artifact_portable_topology topology;
    const std::vector<std::string> identities = {
        std::string(ggml_backend_dev_name(device)) + "\n" +
        ggml_backend_dev_description(device),
    };
    const float split[] = { 1.0f };
    CHECK(llama_cache_acct_build_shard_topology(
        identities, LLAMA_SPLIT_MODE_LAYER, 0, split, topology));

    llama_cache_acct_ledger ledger;
    llama_vbr_artifact_catalog catalog(ledger);
    std::vector<llama_vbr_artifact_domain_binding> bindings;
    CHECK(catalog.bind_topologies({ topology }, bindings));
    CHECK(bindings.size() == 1);
    const auto host = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const auto pinned = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pinned_host);
    CHECK(model_adoption_initialize_accounting(
        ledger, { bindings[0].domain, host, pinned }));
    auto budget = model_adoption_budget(device, bindings[0].domain);

    std::vector<vbr_capture_lane> capture_lanes = {
        { device, source_backend, false },
    };
    vbr_capture_stream_status ring_status;
    auto capture_ring = vbr_pinned_chunk_ring::create(
        capture_lanes, 64ull*1024*1024, 8ull*1024*1024,
        ring_status);
    CHECK(capture_ring && ring_status == vbr_capture_stream_status::ok);
    if (!capture_ring) {
        return false;
    }

    vbr_explicit_capture_request request;
    request.sequence = 0;
    request.identity = {
        "vbr:model", "vbr:no-adapter", "vbr:text-only",
        1, int64_t(tokens.size()), llama_pos(tokens.size()),
    };
    request.token_block = tokens;
    request.frontier.execution_identity =
        request.identity.execution_identity.data();
    request.frontier.execution_identity_len =
        request.identity.execution_identity.size();
    request.frontier.adapter_config_identity =
        request.identity.adapter_config_identity.data();
    request.frontier.adapter_config_identity_len =
        request.identity.adapter_config_identity.size();
    request.frontier.media_content_identity =
        request.identity.media_content_identity.data();
    request.frontier.media_content_identity_len =
        request.identity.media_content_identity.size();
    request.frontier.sequence_epoch = request.identity.sequence_epoch;
    request.frontier.token_count = request.identity.token_count;
    request.frontier.next_position = request.identity.next_position;
    request.idle_decode_thread = true;
    request.ring = capture_ring.get();
    request.topologies = { topology };
    for (const auto & pool : source_pools) {
        request.pool_bindings.push_back({
            pool.instance_id, pool.device, 0, 0, 0,
        });
    }
    const vbr_explicit_representation_policy representation_policy {
        "vbr-model-harness", sizeof("vbr-model-harness")-1,
    };
    request.representation_context = &representation_policy;
    request.representation_identity =
        vbr_explicit_capture_representation_identity;

    vbr_explicit_capture_accounting capture_accounting;
    capture_accounting.budget = &budget;
    capture_accounting.context = &catalog;
    capture_accounting.prepare = [](
            void * context,
            const vbr_artifact_package & package) noexcept {
        return static_cast<llama_vbr_artifact_catalog *>(context)
            ->prepare_capture_package(package);
    };

    // CR-1: the exact/stateful route must quote and refuse before begin_capture
    // or D2H, rather than discovering its size by copying the artifact. The
    // cancellation probe is likewise before accounting/ring work.
    struct exact_pretransfer_probe {
        uint32_t admission_calls = 0;
        uint32_t continuation_calls = 0;
        bool continue_allowed = true;
    } exact_probe;
    auto refused_request = request;
    refused_request.max_packed_bytes = 1;
    refused_request.pretransfer_context = &exact_probe;
    refused_request.pretransfer_admit = [](
            void * opaque,
            const vbr_explicit_capture_pretransfer_quote &) noexcept {
        auto * probe = static_cast<exact_pretransfer_probe *>(opaque);
        ++probe->admission_calls;
        return true;
    };
    const auto refused_capture = vbr_capture_explicit_manifest(
        *source_memory, refused_request, catalog, capture_accounting);
    CHECK(refused_capture.status ==
          vbr_explicit_capture_status::admission_refused);
    CHECK(refused_capture.phase ==
          vbr_explicit_capture_phase::reservation_preparation);
    CHECK(refused_capture.pretransfer.planned_packed_bytes > 1);
    CHECK(refused_capture.chunks == 0);
    CHECK(refused_capture.payload_bytes == 0);
    CHECK(refused_capture.stash_bytes == 0);
    CHECK(exact_probe.admission_calls == 0);

    auto cancelled_request = request;
    exact_probe = {};
    exact_probe.continue_allowed = false;
    cancelled_request.max_packed_bytes = UINT64_MAX;
    cancelled_request.pretransfer_context = &exact_probe;
    cancelled_request.pretransfer_admit = refused_request.pretransfer_admit;
    cancelled_request.continue_context = &exact_probe;
    cancelled_request.continue_transfer = [](void * opaque) noexcept {
        auto * probe = static_cast<exact_pretransfer_probe *>(opaque);
        ++probe->continuation_calls;
        return probe->continue_allowed;
    };
    const auto cancelled_capture = vbr_capture_explicit_manifest(
        *source_memory, cancelled_request, catalog, capture_accounting);
    CHECK(cancelled_capture.status == vbr_explicit_capture_status::cancelled);
    CHECK(cancelled_capture.phase ==
          vbr_explicit_capture_phase::reservation_preparation);
    CHECK(cancelled_capture.chunks == 0);
    CHECK(cancelled_capture.payload_bytes == 0);
    CHECK(cancelled_capture.stash_bytes == 0);
    CHECK(exact_probe.admission_calls == 1);
    CHECK(exact_probe.continuation_calls == 1);

    auto cancelled_during_admit = request;
    exact_probe = {};
    cancelled_during_admit.max_packed_bytes = UINT64_MAX;
    cancelled_during_admit.pretransfer_context = &exact_probe;
    cancelled_during_admit.pretransfer_admit = [](
            void * opaque,
            const vbr_explicit_capture_pretransfer_quote &) noexcept {
        auto * probe = static_cast<exact_pretransfer_probe *>(opaque);
        ++probe->admission_calls;
        probe->continue_allowed = false;
        return false;
    };
    cancelled_during_admit.continue_context = &exact_probe;
    cancelled_during_admit.continue_transfer =
        cancelled_request.continue_transfer;
    const auto cancelled_from_admit = vbr_capture_explicit_manifest(
        *source_memory, cancelled_during_admit, catalog,
        capture_accounting);
    CHECK(cancelled_from_admit.status ==
          vbr_explicit_capture_status::cancelled);
    CHECK(cancelled_from_admit.phase ==
          vbr_explicit_capture_phase::reservation_preparation);
    CHECK(cancelled_from_admit.chunks == 0);
    CHECK(exact_probe.admission_calls == 1);
    CHECK(exact_probe.continuation_calls == 1);

    // A stateful provider writes directly into the bounded chain writer. A
    // task arrival after the first 1 MiB quantum must terminate the private
    // operation without publishing or reaching attention D2H.
    struct bounded_companion_probe {
        uint32_t completed_writes = 0;
        std::array<uint8_t, 1024*1024> bytes = {};
    } bounded_probe;
    auto bounded_request = request;
    vbr_explicit_companion_provider bounded;
    bounded.kind = vbr_artifact_companion_kind::typed_accelerator;
    bounded.build_identity_digest.fill(7);
    bounded.domain = {
        llama_cache_acct_residency::pageable_host,
        llama_cache_acct_domain_kind::not_applicable,
        UINT32_MAX, UINT16_MAX,
    };
    bounded.context = &bounded_probe;
    bounded.size = [](
            const void *, llama_seq_id, uint64_t & output) noexcept {
        output = 3ull*1024ull*1024ull;
        return true;
    };
    bounded.capture_stream = [](
            const void * opaque,
            llama_seq_id,
            llama_io_write_i & output) {
        auto * probe = const_cast<bounded_companion_probe *>(
            static_cast<const bounded_companion_probe *>(opaque));
        for (uint32_t i = 0; i < 3; ++i) {
            output.write(probe->bytes.data(), probe->bytes.size());
            ++probe->completed_writes;
        }
        return true;
    };
    bounded_request.companions.push_back(bounded);
    bounded_request.max_packed_bytes = UINT64_MAX;
    bounded_request.continue_context = &bounded_probe;
    bounded_request.continue_transfer = [](void * opaque) noexcept {
        const auto * probe =
            static_cast<const bounded_companion_probe *>(opaque);
        return probe && probe->completed_writes == 0;
    };
    const auto catalog_before_bounded = catalog.snapshot();
    vbr_explicit_capture_operation bounded_operation;
    const auto bounded_prepared = vbr_prepare_explicit_manifest(
        *source_memory, bounded_request, catalog, capture_accounting,
        bounded_operation);
    CHECK(bounded_prepared.status == vbr_explicit_capture_status::ok);
    CHECK(bounded_operation.ready_for_transfer());
    const auto bounded_cancelled =
        vbr_transfer_explicit_manifest(bounded_operation);
    CHECK(bounded_cancelled.status ==
          vbr_explicit_capture_status::cancelled);
    CHECK(bounded_cancelled.phase ==
          vbr_explicit_capture_phase::companion_capture);
    CHECK(bounded_probe.completed_writes == 1);
    CHECK(bounded_cancelled.chunks == 0);
    CHECK(!bounded_operation.ready_for_publication());
    bounded_operation.reset();
    const auto catalog_after_bounded = catalog.snapshot();
    CHECK(catalog_after_bounded.references ==
          catalog_before_bounded.references);
    CHECK(catalog_after_bounded.published ==
          catalog_before_bounded.published);

    bounded_probe.completed_writes = 0;
    bounded_request.continue_context = nullptr;
    bounded_request.continue_transfer = nullptr;
    const auto bounded_complete = vbr_capture_explicit_manifest(
        *source_memory, bounded_request, catalog, capture_accounting);
    CHECK(bounded_complete.status == vbr_explicit_capture_status::ok);
    CHECK(bounded_complete.companions == 1);
    CHECK(bounded_probe.completed_writes == 3);
    CHECK(bounded_complete.sink.reference_artifact.v != 0);
    if (bounded_complete.sink.reference_artifact.v != 0) {
        CHECK(catalog.discard_unowned_reference(
            bounded_complete.sink.reference_artifact) ==
            vbr_artifact_retire_status::retired);
    }

    const auto captured = vbr_capture_explicit_manifest(
        *source_memory, request, catalog, capture_accounting);
    CHECK(captured.status == vbr_explicit_capture_status::ok);
    CHECK(captured.sink.reference_artifact.v != 0);
    if (captured.status != vbr_explicit_capture_status::ok ||
        captured.sink.reference_artifact.v == 0) {
        std::fprintf(stderr,
            "capture failed status=%s phase=%s inner=%s\n",
            vbr_explicit_capture_status_name(captured.status),
            vbr_explicit_capture_phase_name(captured.phase),
            vbr_capture_stream_status_name(captured.inner_stream_status));
        return false;
    }
    const auto reference = captured.sink.reference_artifact;
    vbr_artifact_package_view package;
    CHECK(catalog.resolve_reference(reference, package) ==
          vbr_artifact_resolve_status::ok);
    CHECK(package && package.validate() == vbr_artifact_status::ok);
    if (live_matrix) {
        std::set<int32_t> types;
        std::map<uint32_t, int32_t> by_unit;
        std::array<size_t, size_t(vbr_artifact_clean_stash_state::_count)>
            stash_states = {};
        for (const auto & unit : package.units()) {
            types.insert(unit.descriptor.current_type);
            by_unit[unit.descriptor.logical_unit_id] =
                unit.descriptor.current_type;
            if (unit.descriptor.clean_stash_state <
                    vbr_artifact_clean_stash_state::_count) {
                stash_states[size_t(
                    unit.descriptor.clean_stash_state)]++;
            }
            if (unit.descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::absent_at_source) {
                const auto & controller =
                    package.manifest().generation.controllers[
                        unit.descriptor.child_id];
                if (unit.descriptor.logical_unit_id <
                        controller.units.size() &&
                    controller.units[unit.descriptor.logical_unit_id].domain !=
                        vbr_repr_domain::full) {
                    std::fprintf(stderr,
                        "VBR tapped-without-stash unit=%u type=%d domain=%u\n",
                        unit.descriptor.logical_unit_id,
                        unit.descriptor.current_type,
                        unsigned(controller.units[
                            unit.descriptor.logical_unit_id].domain));
                }
            }
            if (unit.descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present) {
                const auto reference = std::find_if(
                    package.manifest().unit_references.begin(),
                    package.manifest().unit_references.end(),
                    [&](const vbr_artifact_unit_reference & value) {
                        return value.logical_unit_id ==
                                   unit.descriptor.logical_unit_id &&
                               value.lineage_uuid ==
                                   unit.descriptor.lineage_uuid;
                    });
                uint64_t covered = 0;
                if (reference !=
                        package.manifest().unit_references.end()) {
                    for (const auto & page :
                            reference->stash_reference.covered_sink_pages) {
                        for (uint64_t word : page.covered_mask) {
                            covered += llama_popcount_u64(word);
                        }
                    }
                    std::fprintf(stderr,
                        "VBR stash unit=%u rows=%" PRIu64
                        " captured=%u covered=%" PRIu64 " pages=%zu\n",
                        unit.descriptor.logical_unit_id,
                        reference->stash_reference.valid_rows,
                        reference->stash_reference.captured_sink_count,
                        covered,
                        reference->stash_reference.covered_sink_pages.size());
                }
            }
        }
        const auto tier_rank = [](int32_t type) {
            switch (ggml_type(type)) {
                case GGML_TYPE_F16: return 6;
                case GGML_TYPE_TURBO8_0: return 5;
                case GGML_TYPE_TURBO4_0: return 4;
                case GGML_TYPE_TURBO3_TCQ: return 3;
                case GGML_TYPE_TURBO2_TCQ: return 2;
                case GGML_TYPE_TURBO1_TCQ: return 1;
                default: return -1;
            }
        };
        bool straddled = false;
        for (const auto & [unit, type] : by_unit) {
            if ((unit & 1u) != 0) {
                continue;
            }
            const auto peer = by_unit.find(unit + 1);
            if (peer != by_unit.end() && tier_rank(type) >= 0 &&
                tier_rank(peer->second) >= 0 &&
                std::abs(tier_rank(type) - tier_rank(peer->second)) >= 2) {
                straddled = true;
            }
        }
        std::fprintf(stderr,
            "VBR source_state tokens=%zu budget_mib=%" PRIu64
            " types=%zu straddled=%d stash_absent=%zu stash_present=%zu "
            "stash_omitted=%zu\n",
            tokens.size(), live_budget_mib, types.size(), straddled,
            stash_states[size_t(
                vbr_artifact_clean_stash_state::absent_at_source)],
            stash_states[size_t(
                vbr_artifact_clean_stash_state::present)],
            stash_states[size_t(
                vbr_artifact_clean_stash_state::omitted_source_present)]);
        std::fprintf(stderr, "VBR manifest_consistency=%u\n",
            unsigned(package.manifest().consistency.kind));
        CHECK(types.size() > 1);
        CHECK(!require_straddled || straddled);
        if (types.size() <= 1 || (require_straddled && !straddled)) {
            return false;
        }
    }
    if (downward_mode) {
        CHECK(std::all_of(
            package.units().begin(), package.units().end(),
            [&](const vbr_artifact_unit_view & unit) {
                return unit.descriptor.current_type == int32_t(source_type);
            }));
    }
    const auto source_instance = source_pools.front().instance_id;

    std::vector<float> source_logits = std::move(live_source_logits);
    if (!live_matrix) {
        CHECK(model_adoption_decode(
            source.get(), { continuation }, llama_pos(tokens.size())));
        float * source_logits_ptr = llama_get_logits_ith(source.get(), -1);
        CHECK(source_logits_ptr != nullptr && vocab_size > 0);
        if (!source_logits_ptr || vocab_size <= 0) {
            return false;
        }
        source_logits.assign(
            source_logits_ptr, source_logits_ptr+vocab_size);
        // Context destruction is a terminal scheduler boundary. Keep the
        // retained hardware harness explicit about the production settle.
        llama_synchronize(source.get());
    }
    capture_ring.reset();
    if (!live_matrix) {
        source.reset();
    }

    // Two-standard oracle: the ordinary native adoption cell remains byte-exact to
    // its F16 source. Downward compares encoded rows against the shipped live
    // degrade path and continuation against a separate context that encoded
    // the same tokens natively at the declared final tier.
    std::vector<float> expected_logits = source_logits;
    llama_context_ptr live_degrade_reference;
    const llama_kv_cache * live_degrade_cache = nullptr;
    if (downward_mode) {
        const auto final_oracle_type = require_destination_degraded
            ? expected_destination_type : target_type;
        live_degrade_reference = make_context(source_type);
        CHECK(live_degrade_reference != nullptr);
        std::vector<llama_memory_tree_child> degrade_tree;
        if (!live_degrade_reference ||
            !model_adoption_decode(live_degrade_reference.get(), tokens, 0)) {
            return false;
        }
        llama_synchronize(live_degrade_reference.get());
        if (!llama_memory_tree_collect(
                llama_get_memory(live_degrade_reference.get()),
                degrade_tree) ||
            degrade_tree.size() != 1 || !degrade_tree[0].attention ||
            !llama_kv_cache_vbr_epoch_test::degrade_to(
                degrade_tree[0].attention, final_oracle_type)) {
            return false;
        }
        live_degrade_cache = degrade_tree[0].attention;

        auto same_tier = make_context(final_oracle_type);
        CHECK(same_tier != nullptr);
        if (!same_tier || !model_adoption_decode(same_tier.get(), tokens, 0) ||
            !model_adoption_decode(same_tier.get(), { continuation },
                         llama_pos(tokens.size()))) {
            return false;
        }
        float * same_tier_logits =
            llama_get_logits_ith(same_tier.get(), -1);
        CHECK(same_tier_logits != nullptr);
        if (!same_tier_logits) {
            return false;
        }
        expected_logits.assign(
            same_tier_logits, same_tier_logits+vocab_size);
        llama_synchronize(same_tier.get());
    }

    const auto run_import = [&](bool previously_observed,
                                vbr_import_decision expected_decision,
                                vbr_adopt_phase fail_before_phase =
                                    vbr_adopt_phase::_count) {
        const auto find_attention_child = [](auto & tree) {
            return std::find_if(
                tree.begin(), tree.end(),
                [](const llama_memory_tree_child & child) {
                    return child.attention != nullptr;
                });
        };
        const uint64_t baseline_live_ops = ledger.snapshot().live_ops;
        llama_context_ptr target_context = live_matrix
            ? std::move(source)
            : make_context(
                require_destination_degraded || downward_mode
                    ? target_type : GGML_TYPE_F16,
                destination_budget_mib);
        CHECK(target_context != nullptr);
        if (!target_context) {
            return false;
        }
        llama_memory_i * target_memory = llama_get_memory(target_context.get());
        if (live_matrix) {
            // Consume the checkpoint back into the same live controller after
            // the reference continuation. This is the production slot shape:
            // the naturally-created mixed tier/generation history is retained,
            // while the sequence itself is removed before native import.
            CHECK(target_memory && target_memory->seq_rm(0, -1, -1));
            std::vector<llama_memory_tree_child> empty_tree;
            CHECK(target_memory &&
                  llama_memory_tree_collect(target_memory, empty_tree));
            const auto attention = find_attention_child(empty_tree);
            CHECK(attention != empty_tree.end() &&
                  llama_kv_cache_vbr_epoch_test::
                      make_construction_empty_preserve_tiers(
                          attention->attention));
            CHECK(target_memory && target_memory->seq_pos_min(0) < 0 &&
                  target_memory->seq_pos_max(0) < 0);
            if (!target_memory || target_memory->seq_pos_max(0) >= 0) {
                return false;
            }
        }
        std::vector<vbr_explicit_capture_runtime_pool> target_pools;
        uint32_t target_children = 0;
        CHECK(target_memory && vbr_explicit_capture_runtime_pools(
            *target_memory, target_pools, target_children));
        if (!target_memory || target_children != 1 || target_pools.empty()) {
            return false;
        }
        CHECK(live_matrix
            ? target_pools.front().instance_id == source_instance
            : target_pools.front().instance_id != source_instance);

        const auto accounting_snapshot = ledger.snapshot();
        llama_cache_budget_coordinator baseline_budget;
        CHECK(baseline_budget.reset(accounting_snapshot, budget));
        llama_cache_budget_plan baseline_plan;
        baseline_plan.accounting_serial = accounting_snapshot.serial;
        const auto baseline_fit = baseline_budget.fits(baseline_plan);
        CHECK(baseline_fit.state == llama_cache_budget_fit_state::fits);
        if (baseline_fit.state != llama_cache_budget_fit_state::fits) {
            std::fprintf(stderr,
                "VBR baseline budget state=%u cells=%zu completeness=%zu\n",
                unsigned(baseline_fit.state), accounting_snapshot.cells.size(),
                accounting_snapshot.completeness.size());
            for (const auto & row : accounting_snapshot.cells) {
                const auto resident = row.cell.measures[size_t(
                    llama_cache_acct_measure::resident_allocated)];
                const auto reserved = row.cell.measures[size_t(
                    llama_cache_acct_measure::reserved)];
                if (row.certification != llama_cache_acct_known::known ||
                    resident.state != llama_cache_acct_known::known ||
                    reserved.state != llama_cache_acct_known::known) {
                    std::fprintf(stderr,
                        "VBR accounting category=%u residency=%u cert=%u "
                        "resident=%u reserved=%u\n",
                        unsigned(row.category), unsigned(row.domain.residency),
                        unsigned(row.certification), unsigned(resident.state),
                        unsigned(reserved.state));
                }
            }
        }
        vbr_target_validation_snapshot target_snapshot;
        vbr_downward_policy_projection downward_projection;
        bool detected_downward = false;
        vbr_import_schedule_quote schedule_quote;
        if (destination_upward) {
            for (auto drift_kind : {
                    model_adoption_representation_drift_context::kind::codec_version,
                    model_adoption_representation_drift_context::kind::codebook_digest,
                    model_adoption_representation_drift_context::kind::rotation_digest,
                 }) {
                model_adoption_representation_drift_context drift_context {
                    drift_kind, &representation_policy,
                    int32_t(source_type),
                };
                vbr_target_validation_snapshot drifted_target;
                vbr_downward_policy_projection drifted_projection;
                bool drifted_downward = false;
                vbr_import_schedule_quote drifted_quote;
                CHECK(vbr_explicit_import_target_schedule_snapshot(
                    *target_memory, 0, package, bindings,
                    previously_observed, accounting_snapshot.serial,
                    &drift_context,
                    model_adoption_representation_identity_with_source_drift,
                    drifted_target, drifted_projection,
                    drifted_downward, drifted_quote) ==
                    vbr_import_target_snapshot_status::unavailable);
                CHECK(drifted_target.children.empty());
                CHECK(drifted_projection.final_types.empty());
                CHECK(drifted_quote.status() ==
                      vbr_import_schedule_status::_count);
            }
        }
        CHECK(vbr_explicit_import_target_schedule_snapshot(
            *target_memory, 0, package, bindings, previously_observed,
            accounting_snapshot.serial,
            &representation_policy,
            vbr_explicit_capture_representation_identity,
            target_snapshot,
            downward_projection, detected_downward, schedule_quote) ==
            vbr_import_target_snapshot_status::actionable);
        CHECK(detected_downward == downward_mode);
        CHECK(schedule_quote.status() == expected_schedule_status);
        if (require_destination_degraded) {
            CHECK(schedule_quote.destination().status ==
                vbr_import_destination_status::feasible_degraded);
            CHECK(!schedule_quote.destination().prefix.empty());
            CHECK(std::all_of(
                schedule_quote.destination().final_types.begin(),
                schedule_quote.destination().final_types.end(),
                [&](const auto & types) {
                    return std::all_of(
                        types.begin(), types.end(), [&](ggml_type type) {
                            return type == expected_destination_type;
                        });
                }));
        }
        CHECK(vbr_import_schedule_quote_matches(
            schedule_quote, target_snapshot, package));
        CHECK(downward_mode
            ? downward_projection.status ==
                vbr_downward_policy_status::coherent
            : downward_projection.final_types.empty());
        target_snapshot.scheduler_idle = true;
        if (target_snapshot.children.empty()) {
            return false;
        }
        const auto target_lineage = target_snapshot.children[0].lineage_uuid;
        const auto target_instance = target_snapshot.children[0].instance_id;

        model_adoption_harness_target target_owner;
        target_owner.memory = target_memory;
        target_owner.ledger = &ledger;
        target_owner.package = &package;
        target_owner.bindings = &bindings;
        target_owner.schedule_quote = &schedule_quote;
        target_owner.representation_context = &representation_policy;
        target_owner.representation_identity =
            vbr_explicit_capture_representation_identity;
        if (needs_transform) {
            target_owner.downward_tree_digest =
                downward_mode
                    ? downward_projection.tree_digest
                    : schedule_quote.destination().tree_digest;
        }
        vbr_adopt_policy policy;
        policy.authorized = true;
        policy.identity = {
            package.manifest().identity.execution_identity,
            package.manifest().identity.adapter_config_identity,
            package.manifest().identity.media_content_identity,
            package.manifest().identity.sequence_epoch,
            package.manifest().identity.next_position,
            &package.manifest().token_block.tokens,
        };
        policy.destination_sequence = 0;
        policy.adoption_nonce = previously_observed ? 0xf42a02 : 0xf42a01;
        policy.domain_bindings = bindings;
        policy.domain_bindings.push_back({ UINT32_MAX, UINT16_MAX, host });
        policy.domain_bindings.push_back({ UINT32_MAX, UINT16_MAX, pinned });
        policy.accounting_snapshot = &accounting_snapshot;
        policy.budget_config = &budget;
        policy.schedule_quote = &schedule_quote;
        policy.context = &target_owner;
        policy.recheck_target_empty = model_adoption_harness_target::recheck;
        policy.read_accounting_serial =
            model_adoption_harness_target::accounting_serial;
        policy.read_policy_epoch = model_adoption_harness_target::policy_serial;
        policy.parse_companion = vbr_parse_recurrent_companion;
        llama_cache_budget_plan transform_budget_plan;
        if (needs_transform) {
            transform_budget_plan.accounting_serial =
                accounting_snapshot.serial;
            policy.transform_budget_plan = &transform_budget_plan;
            policy.downward_projection = downward_mode
                ? &downward_projection : nullptr;
            policy.read_transform_tree_digest =
                model_adoption_harness_target::transform_tree;
        }

        auto validated = vbr_validate_unit_manifest_snapshot(
            target_snapshot, package, policy);
        if (validated.status != vbr_manifest_validation_status::validated ||
            validated.decision != expected_decision || !validated.proof) {
            std::fprintf(stderr,
                "VBR validation failed status=%s decision=%s expected=%s "
                "observed=%d downward=%d\n",
                vbr_manifest_validation_status_name(validated.status),
                vbr_import_decision_name(validated.decision),
                vbr_import_decision_name(expected_decision),
                previously_observed, detected_downward);
            for (const auto & unit : package.units()) {
                if (unit.descriptor.child_id >= target_snapshot.children.size()) {
                    continue;
                }
                const auto & target_units =
                    target_snapshot.children[unit.descriptor.child_id].units;
                const auto found = std::find_if(
                    target_units.begin(), target_units.end(),
                    [&](const vbr_target_unit_snapshot & value) {
                        return value.logical_unit_id ==
                            unit.descriptor.logical_unit_id;
                    });
                if (found == target_units.end()) {
                    continue;
                }
                const auto & source_unit = unit.descriptor;
                if (source_unit.current_type != found->current_type ||
                    source_unit.last_source_type != found->last_source_type ||
                    source_unit.promote_hops != found->promote_hops ||
                    source_unit.last_transition != found->last_transition ||
                    source_unit.representation.source_loss_history !=
                        found->source_loss_history ||
                    source_unit.representation.checkpoint_codec_hops !=
                        found->checkpoint_codec_hops) {
                    std::fprintf(stderr,
                        "VBR representation unit=%u type=%d/%d source=%d/%d "
                        "hops=%u/%u transition=%u/%u loss=%u/%u codec=%u/%u\n",
                        source_unit.logical_unit_id,
                        source_unit.current_type, found->current_type,
                        source_unit.last_source_type, found->last_source_type,
                        unsigned(source_unit.promote_hops),
                        unsigned(found->promote_hops),
                        unsigned(source_unit.last_transition),
                        unsigned(found->last_transition),
                        source_unit.representation.source_loss_history,
                        found->source_loss_history,
                        source_unit.representation.checkpoint_codec_hops,
                        found->checkpoint_codec_hops);
                    break;
                }
            }
        }
        CHECK(validated.status == vbr_manifest_validation_status::validated);
        CHECK(validated.decision == expected_decision);
        CHECK(validated.proof);
        if (!validated.proof || validated.decision != expected_decision) {
            return false;
        }

        vbr_adopt_stage_policy stage_policy;
        stage_policy.ledger = &ledger;
        stage_policy.budget = &budget;
        stage_policy.pinned_domain = pinned;
        stage_policy.pinned_ring_bytes = 64ull*1024*1024;
        stage_policy.chunk_bytes = 8ull*1024*1024;
        stage_policy.lanes.push_back({
            bindings[0].domain,
            target_pools.front().backend_device,
            target_pools.front().backend,
            false,
        });
        if (needs_transform) {
            stage_policy.transform_context = target_memory;
            stage_policy.reserve_transform = model_adoption_reserve_transform;
        }
        auto staged = vbr_stage_validated_manifest(
            std::move(validated.proof), stage_policy);
        CHECK(staged.status == vbr_adopt_stage_status::staged);
        CHECK(staged.manifest && staged.staged);
        if (!staged.manifest || !staged.staged) {
            return false;
        }

        vbr_composite_publish_hooks hooks;
        hooks.context = target_memory;
        hooks.owner_token = target_memory;
        hooks.validate_owner_token = model_adoption_owner_token;
        std::vector<llama_memory_tree_child> adoption_tree;
        CHECK(llama_memory_tree_collect(target_memory, adoption_tree));
        for (const auto & child : adoption_tree) {
            if (child.recurrent) {
                hooks.companions.push_back(
                    vbr_recurrent_companion_adoption_provider(
                        *child.recurrent));
            }
        }
        vbr_adopt_test_control test;
        test.fault.fail_before = fail_before_phase;
        hooks.test = fail_before_phase == vbr_adopt_phase::_count
            ? nullptr : &test;
        auto adopted = vbr_adopt_empty_manifest(
            *target_memory, 0, std::move(*staged.manifest),
            std::move(*staged.staged), ledger, hooks);
        if (fail_before_phase != vbr_adopt_phase::_count) {
            CHECK(adopted.status != vbr_adopt_status::adopted);
            vbr_target_validation_snapshot rolled_back;
            vbr_downward_policy_projection rolled_back_projection;
            bool rolled_back_downward = false;
            vbr_import_schedule_quote rolled_back_quote;
            CHECK(vbr_explicit_import_target_schedule_snapshot(
                *target_memory, 0, package, bindings, previously_observed,
                ledger.snapshot().serial,
                &representation_policy,
                vbr_explicit_capture_representation_identity,
                rolled_back,
                rolled_back_projection, rolled_back_downward,
                rolled_back_quote) ==
                vbr_import_target_snapshot_status::actionable);
            CHECK(std::all_of(
                rolled_back.children.begin(), rolled_back.children.end(),
                [](const vbr_target_child_snapshot & child) {
                    return child.empty;
                }));
            CHECK(!vbr_recovery_pending_for(target_instance));
            CHECK(vbr_operation_registry_quiescent_for(&target_instance, 1));
            target_context.reset();
            CHECK(ledger.snapshot().live_ops == baseline_live_ops);
            return adopted.status != vbr_adopt_status::adopted;
        }
        CHECK(adopted.status == vbr_adopt_status::adopted);
        CHECK(adopted.decision == expected_decision);
        if (adopted.status != vbr_adopt_status::adopted) {
            std::fprintf(stderr, "adopt failed status=%s phase=%s accounting=%s\n",
                vbr_adopt_status_name(adopted.status),
                vbr_adopt_phase_name(adopted.phase),
                llama_cache_transaction_status_name(adopted.accounting_status));
            return false;
        }

        std::vector<llama_memory_tree_child> adopted_tree;
        CHECK(llama_memory_tree_collect(target_memory, adopted_tree));
        const auto adopted_attention = find_attention_child(adopted_tree);
        CHECK(adopted_attention != adopted_tree.end());
        if (adopted_attention == adopted_tree.end()) {
            return false;
        }
        auto * adopted_cache = adopted_attention->attention;
        vbr_controller_instance_id adopted_instance;
        vbr_lineage_uuid adopted_lineage;
        uint64_t adopted_generation = 0;
        vbr_repr_transition adopted_transition = vbr_repr_transition::initial;
        CHECK(llama_kv_cache_vbr_epoch_test::tracker_identity(
            adopted_cache, adopted_instance, adopted_lineage,
            adopted_generation, adopted_transition));
        CHECK(adopted_instance == target_instance);
        if (expected_decision == vbr_import_decision::native_import) {
            CHECK(llama_kv_cache_vbr_epoch_test::adopted_matches(
                adopted_cache, package, package.manifest(), 0));
        } else {
            CHECK(adopted_lineage == target_lineage);
            CHECK(adopted_generation == 1);
            CHECK(adopted_transition == vbr_repr_transition::whole_import);
        }
        if (downward_mode) {
            CHECK(expected_decision ==
                  vbr_import_decision::downward_rebase);
            CHECK(adopted.consistency ==
                  vbr_artifact_consistency_kind::live_rebased);
            CHECK(!downward_projection.final_types.empty());
            CHECK(llama_kv_cache_vbr_epoch_test::representation_types(
                adopted_cache, downward_projection.final_types[0]));
            if (!live_degrade_cache ||
                !llama_kv_cache_vbr_epoch_test::representation_bytes_equal(
                    adopted_cache, live_degrade_cache,
                    package.manifest())) {
                CHECK(false &&
                      "downward rows differ from shipped live-degrade oracle");
                return false;
            }
        }
        if (require_destination_degraded) {
            CHECK(llama_kv_cache_vbr_epoch_test::representation_types(
                adopted_cache,
                schedule_quote.destination().final_types[0]));
        }

        CHECK(model_adoption_decode(target_context.get(),
                          { continuation }, llama_pos(tokens.size())));
        float * target_logits = llama_get_logits_ith(target_context.get(), -1);
        CHECK(target_logits != nullptr);
        if (!target_logits) {
            return false;
        }
        if (!downward_mode &&
            expected_decision == vbr_import_decision::native_import) {
            if (std::memcmp(
                    expected_logits.data(), target_logits,
                    expected_logits.size()*sizeof(float)) != 0) {
                CHECK(false && "capture/adopt continuation logits differ");
                return false;
            }
        } else {
            float max_abs = 0.0f;
            for (size_t i = 0; i < expected_logits.size(); ++i) {
                max_abs = std::max(
                    max_abs, std::abs(expected_logits[i]-target_logits[i]));
            }
            std::fprintf(stderr,
                "VBR representation-level continuation max_abs=%g\n",
                double(max_abs));
            // The target representation is deterministic on pinned kernels;
            // the retained 27B gate records this metric and applies the
            // predeclared representation-level tolerance only here.
            CHECK(max_abs <= 1.0e-4f);
            if (max_abs > 1.0e-4f) {
                return false;
            }
        }
        CHECK(!vbr_recovery_pending_for(target_instance));
        llama_synchronize(target_context.get());
        CHECK(vbr_operation_registry_quiescent_for(&target_instance, 1));
        CHECK(llama_kv_cache_vbr_epoch_test::tracker_teardown_state(
                  adopted_cache) == vbr_generation_teardown_state::clean);
        target_context.reset();
        return true;
    };

    // Every fallible boundary from operation-open through the final
    // pre-publication checkpoint is driven on a real target. A fail-before
    // at a boundary also proves rollback after the preceding boundary. Composite
    // publication and close deliberately have no post-boundary fault seam.
    if (live_matrix) {
        CHECK(run_import(false, require_straddled
            ? vbr_import_decision::live_rebased
            : vbr_import_decision::native_import));
    } else if (!needs_transform) {
        for (uint8_t phase = uint8_t(vbr_adopt_phase::operation_open);
             phase <= uint8_t(vbr_adopt_phase::composite_publish); ++phase) {
            CHECK(run_import(false, vbr_import_decision::native_import,
                             vbr_adopt_phase(phase)));
        }
        CHECK(run_import(false, vbr_import_decision::native_import));
        CHECK(run_import(true, vbr_import_decision::live_rebased));
    } else if (downward_mode && !require_destination_degraded) {
        CHECK(run_import(false, vbr_import_decision::downward_rebase));
    } else {
        const auto transform_decision = destination_upward
            ? vbr_import_decision::upward_reconstruct
            : vbr_import_decision::downward_rebase;
        CHECK(run_import(false,
            transform_decision,
            vbr_adopt_phase::composite_publish));
        CHECK(run_import(false, transform_decision));
    }
    package.reset();
    CHECK(catalog.retire(reference) == vbr_artifact_retire_status::retired);
    CHECK(ledger.snapshot().live_ops == 0);
    return failures == 0;
}

static bool model_backed_occupied_store(
        const char * model_path, ggml_type entry_type) {
    ggml_backend_load_all();
    setenv("VBR_FORCE_GENERIC", "1", 1);
    setenv("VBR_PROMOTE", "0", 1);
    setenv("VBR_STASH_ROWS", "0", 1);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model_ptr model(llama_model_load_from_file(model_path, model_params));
    CHECK(model != nullptr);
    if (!model) {
        return false;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 256;
    context_params.n_batch = 64;
    context_params.n_ubatch = 64;
    context_params.n_seq_max = 1;
    context_params.n_threads = 2;
    context_params.n_threads_batch = 2;
    context_params.type_k = entry_type;
    context_params.type_v = entry_type;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.vbr_dynamic = true;
    context_params.vbr_budget_explicit = true;
    context_params.vbr_vram_budget_bytes = 4ull*1024*1024*1024;
    // Imported KV receipts retain their accounting owner until the context
    // releases the adopted cache image. Keep the ledger alive across context
    // destruction, just as the server's cache-accounting owner does.
    llama_cache_acct_ledger ledger;
    auto context = llama_context_ptr(
        llama_init_from_model(model.get(), context_params));
    CHECK(context != nullptr);
    if (!context) {
        return false;
    }

    auto incoming_tokens = common_tokenize(
        context.get(), "Occupied restore production composition fixture.",
        true, false);
    CHECK(incoming_tokens.size() >= 4);
    if (incoming_tokens.size() < 4) {
        return false;
    }
    const int32_t vocab_size = llama_vocab_n_tokens(
        llama_model_get_vocab(model.get()));
    CHECK(vocab_size > 1);
    if (vocab_size <= 1) {
        return false;
    }
    const llama_token continuation = incoming_tokens.back();
    const uint32_t context_cells = llama_n_ctx(context.get());
    CHECK(context_cells >= 8);
    if (context_cells < 8 || incoming_tokens.size() >= context_cells) {
        return false;
    }
    // Leave exactly one cache cell for the continuation. An equal-length
    // occupied replacement cannot fit in the remaining free space and must
    // therefore exercise authenticated incumbent-cell recycling.
    incoming_tokens.resize(
        size_t(context_cells)-1,
        incoming_tokens.size() > 1 ? incoming_tokens[1]
                                   : incoming_tokens.front());
    auto incumbent_tokens = incoming_tokens;
    incumbent_tokens.back() = llama_token(
        (uint32_t(incumbent_tokens.back()) + 1u) % uint32_t(vocab_size));
    if (incumbent_tokens.back() == incoming_tokens.back()) {
        incumbent_tokens.back() = llama_token(
            (uint32_t(incumbent_tokens.back()) + 1u) % uint32_t(vocab_size));
    }

    const size_t prefix_count = std::min<size_t>(8, incoming_tokens.size()-1);
    const llama_token prefix_continuation = incoming_tokens[prefix_count];
    llama_tokens prefix_tokens(
        incoming_tokens.begin(), incoming_tokens.begin()+prefix_count);
    std::vector<uint8_t> prefix_expected_rows;
    std::vector<float> prefix_expected_logits;

    CHECK(model_adoption_decode(context.get(), incoming_tokens, 0));
    llama_synchronize(context.get());
    llama_memory_i * memory = llama_get_memory(context.get());
    std::vector<vbr_explicit_capture_runtime_pool> pools;
    uint32_t attention_children = 0;
    CHECK(memory && vbr_explicit_capture_runtime_pools(
        *memory, pools, attention_children));
    CHECK(attention_children == 1 && !pools.empty());
    if (!memory || attention_children != 1 || pools.empty()) {
        return false;
    }

    const auto & first_pool = pools.front();
    CHECK(first_pool.backend_device != nullptr);
    if (!first_pool.backend_device) {
        return false;
    }
    ggml_backend_t ring_backend = first_pool.backend;
    if (ring_backend == nullptr) {
        ring_backend = context->backend_for_device(first_pool.backend_device);
    }
    CHECK(ring_backend != nullptr);
    if (!ring_backend) {
        return false;
    }
    vbr_artifact_portable_topology topology;
    const std::vector<std::string> identities = {
        std::string(ggml_backend_dev_name(first_pool.backend_device)) + "\n" +
        ggml_backend_dev_description(first_pool.backend_device),
    };
    const float split[] = { 1.0f };
    CHECK(llama_cache_acct_build_shard_topology(
        identities, LLAMA_SPLIT_MODE_LAYER, 0, split, topology));

    llama_cache_acct_resource_domain device_domain;
    CHECK(ledger.make_device_domain(
        topology, llama_cache_acct_device_ordinal { 0 }, device_domain));
    const auto host_domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);
    const auto pinned_domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pinned_host);
    CHECK(model_adoption_initialize_accounting(
        ledger, { device_domain, host_domain, pinned_domain }));
    occupied_budget_source budget_source {
        model_adoption_budget(first_pool.backend_device, device_domain),
    };
    server_vbr_artifact_store_config store_config;
    store_config.ledger = &ledger;
    store_config.pinned_domain = pinned_domain;
    store_config.topologies = { topology };
    for (const auto & pool : pools) {
        store_config.pool_bindings.push_back({
            pool.instance_id, pool.device, 0, 0, 0,
        });
    }
    store_config.lanes = {
        { first_pool.backend_device, ring_backend, false },
    };
    store_config.attention_children = attention_children;
    store_config.ring_bytes = 64ull*1024*1024;
    store_config.chunk_bytes = 8ull*1024*1024;
    store_config.budget_context = &budget_source;
    store_config.sample_budget = occupied_sample_budget;
    server_vbr_artifact_capture_status store_status;
    auto store = server_vbr_artifact_store::create(
        store_config, store_status);
    CHECK(store && store_status == server_vbr_artifact_capture_status::ok);
    if (!store) {
        return false;
    }

    static constexpr char execution_identity[] = "vbr:occupied:model";
    static constexpr char adapter_identity[] = "vbr:occupied:no-adapter";
    const auto make_host_capture_request = [&] (const llama_tokens & tokens) {
        server_prompt prompt;
        prompt.tokens = server_tokens(tokens, false);
        prompt.sequence_epoch = 1;
        std::string media_identity;
        CHECK(prompt.tokens.media_content_identity(
            prompt.n_tokens(), media_identity));
        vbr_explicit_capture_request request;
        request.sequence = 0;
        request.identity = {
            execution_identity, adapter_identity, media_identity,
            prompt.sequence_epoch, int64_t(tokens.size()),
            llama_pos(tokens.size()),
        };
        request.token_block = tokens;
        request.frontier.execution_identity =
            request.identity.execution_identity.data();
        request.frontier.execution_identity_len =
            request.identity.execution_identity.size();
        request.frontier.adapter_config_identity =
            request.identity.adapter_config_identity.data();
        request.frontier.adapter_config_identity_len =
            request.identity.adapter_config_identity.size();
        request.frontier.media_content_identity =
            request.identity.media_content_identity.data();
        request.frontier.media_content_identity_len =
            request.identity.media_content_identity.size();
        request.frontier.sequence_epoch = request.identity.sequence_epoch;
        request.frontier.token_count = request.identity.token_count;
        request.frontier.next_position = request.identity.next_position;
        request.idle_decode_thread = true;
        return request;
    };
    const auto capture_owner = [&] (
            const llama_tokens & tokens,
            std::shared_ptr<const server_prompt_cache_vbr_payload> & owner) {
        server_vbr_explicit_host_capture operation;
        const auto prepared = store->prepare_host_payload(
            *memory, make_host_capture_request(tokens), operation);
        const auto transferred = prepared.status ==
                    server_vbr_artifact_capture_status::ok &&
                operation.ready_for_transfer()
            ? store->transfer_host_payload(operation)
            : prepared;
        const auto captured = transferred.status ==
                    server_vbr_artifact_capture_status::ok &&
                operation.ready_for_publication()
            ? store->publish_host_payload(operation, owner)
            : transferred;
        const bool ok =
            captured.status == server_vbr_artifact_capture_status::ok &&
            owner && owner->package();
        if (!ok) {
            std::fprintf(stderr,
                "VBR occupied capture failed: store=%s library=%s phase=%s "
                "generation=%s size=%s owner=%d package=%d\n",
                server_vbr_artifact_capture_status_name(captured.status),
                vbr_explicit_capture_status_name(captured.library_status),
                vbr_explicit_capture_phase_name(captured.phase),
                vbr_explicit_generation_failure_name(captured.generation_failure),
                vbr_explicit_size_failure_name(captured.size_failure),
                bool(owner), bool(owner && owner->package()));
        }
        return ok;
    };

    // Prefix derivation is intentionally parent-bound to a projected-sealed
    // artifact (it retains the authenticated range tree that an exact whole
    // capture does not need). Capture a short parent from the still-live full
    // sequence so this hardware arm exercises the same production owner used
    // by automatic projected host publication.
    const size_t projected_parent_count = incoming_tokens.size();
    vbr_projected_capture_manifest_request projected_manifest;
    projected_manifest.manifest_id = 1;
    projected_manifest.sequence = 0;
    projected_manifest.token_block.assign(
        incoming_tokens.begin(),
        incoming_tokens.begin() + projected_parent_count);
    projected_manifest.identity =
        make_host_capture_request(projected_manifest.token_block).identity;
    projected_manifest.text_only = true;
    std::vector<server_vbr_projected_host_publish_result> projected_results;
    server_vbr_projected_host_capture_diagnostics projected_diagnostics;
    CHECK(store->capture_projected_host_batch(
        *memory, { std::move(projected_manifest) },
        256ull*1024*1024, projected_results, nullptr,
        &projected_diagnostics));
    CHECK(projected_results.size() == 1);
    std::shared_ptr<const server_prompt_cache_vbr_payload> projected_owner =
        projected_results.size() == 1 ? projected_results.front().payload
                                      : nullptr;
    CHECK(projected_owner && projected_owner->package());

    // The scheduler may abandon an admitted operation after preparation. The
    // private catalog build and its ledger claims must disappear without a
    // host-visible publication or a successful-capture counter increment.
    const auto ledger_before_abandon = ledger.snapshot();
    const auto counters_before_abandon = store->counters();
    {
        server_vbr_explicit_host_capture abandoned;
        const auto prepared = store->prepare_host_payload(
            *memory, make_host_capture_request(incoming_tokens), abandoned);
        CHECK(prepared.status == server_vbr_artifact_capture_status::ok);
        CHECK(abandoned.ready_for_transfer());
        abandoned.reset();
        CHECK(!abandoned.ready_for_transfer());
        CHECK(!abandoned.ready_for_publication());
    }
    const auto ledger_after_abandon = ledger.snapshot();
    CHECK(ledger_after_abandon.live_ops == ledger_before_abandon.live_ops);
    CHECK(ledger_after_abandon.allocations.size() ==
          ledger_before_abandon.allocations.size());
    CHECK(store->counters().exact_published ==
          counters_before_abandon.exact_published);

    std::shared_ptr<const server_prompt_cache_vbr_payload> incoming_owner;
    CHECK(capture_owner(incoming_tokens, incoming_owner));
    if (!incoming_owner) {
        return false;
    }
    CHECK(model_adoption_decode(
        context.get(), { continuation }, llama_pos(incoming_tokens.size())));
    llama_synchronize(context.get());
    float * expected_logits_ptr = llama_get_logits_ith(context.get(), -1);
    CHECK(expected_logits_ptr != nullptr);
    if (!expected_logits_ptr) {
        return false;
    }
    std::vector<float> expected_logits(
        expected_logits_ptr, expected_logits_ptr + vocab_size);
    // Build the prefix oracle from the exact rows used by the projected parent.
    // A separately decoded short prompt may take a different GEMM shape and is
    // not a byte-exact cache oracle even though it is semantically equivalent.
    CHECK(llama_memory_seq_rm(
        memory, 0, llama_pos(prefix_count), -1));
    llama_synchronize(context.get());
    CHECK(llama_kv_cache_vbr_epoch_test::snapshot_sequence_rows(
        static_cast<llama_kv_cache *>(memory), 0, uint32_t(prefix_count),
        prefix_expected_rows));
    CHECK(model_adoption_decode(
        context.get(), { prefix_continuation }, llama_pos(prefix_count)));
    llama_synchronize(context.get());
    float * prefix_expected_ptr = llama_get_logits_ith(context.get(), -1);
    CHECK(prefix_expected_ptr != nullptr);
    if (prefix_expected_ptr) {
        prefix_expected_logits.assign(
            prefix_expected_ptr, prefix_expected_ptr+vocab_size);
    }
    llama_memory_clear(memory, true);
    CHECK(model_adoption_decode(context.get(), incumbent_tokens, 0));
    llama_synchronize(context.get());
    CHECK(memory->seq_pos_min(0) == 0);
    CHECK(memory->seq_pos_max(0) ==
          llama_pos(incumbent_tokens.size()-1));

    std::shared_ptr<const server_prompt_cache_vbr_payload> recovery_owner;
    CHECK(capture_owner(incumbent_tokens, recovery_owner));
    CHECK(recovery_owner);
    CHECK(incoming_owner && recovery_owner &&
          incoming_owner->reference_artifact() !=
              recovery_owner->reference_artifact());
    if (!recovery_owner ||
        incoming_owner->reference_artifact() ==
            recovery_owner->reference_artifact()) {
        return false;
    }

    server_cache_authority authority;
    server_retention_sidecar_store retention;
    retention.configure(&ledger, host_domain, &authority.leases);
    CHECK(retention.enable_prefix_tracking());
    server_prompt_cache cache(0, 0);
    cache.acct = &ledger;
    cache.retention_obs = &retention;
    cache.lease_obs = &authority.leases;
    const std::string execution_key = execution_identity;
    const std::string adapter_key = adapter_identity;
    cache.lease_execution_identity = &execution_key;

    server_prompt incoming_prompt;
    incoming_prompt.tokens = server_tokens(incoming_tokens, false);
    incoming_prompt.sequence_epoch = 1;
    server_prompt incumbent_prompt;
    incumbent_prompt.tokens = server_tokens(incumbent_tokens, false);
    incumbent_prompt.sequence_epoch = 1;
    const common_cache_family_binding incumbent_family_initial {
        common_cache_family_id { 41 }, common_cache_family_role::branch,
    };
    const common_cache_family_binding incoming_family {
        common_cache_family_id { 42 }, common_cache_family_role::background,
    };
    common_cache_family_binding incumbent_family = incumbent_family_initial;
    constexpr int32_t incoming_source_slot = 4;
    constexpr int32_t destination_slot = 0;
    const auto publish_live_source = [&] (
            int32_t slot, const server_prompt & prompt) {
        common_chat_msg_spans spans;
        spans.add(COMMON_CHAT_ROLE_USER, 0, prompt.n_tokens());
        const auto key = server_retention_instance_key::for_slot(slot);
        return retention.publish(
                   key, common_retention_pool::attention, spans, true,
                   prompt.n_tokens(), prompt.n_tokens(), true) &&
            server_prompt_retention_publish_exact_prefix(
                retention, key, prompt, adapter_key, prompt.n_tokens());
    };
    CHECK(publish_live_source(incoming_source_slot, incoming_prompt));
    CHECK(publish_live_source(destination_slot, incumbent_prompt));

    server_cache_lease_identity incumbent_lease_identity;
    CHECK(server_cache_lease_build_identity(
        execution_key, adapter_key, incumbent_prompt.tokens,
        incumbent_prompt.n_tokens(), incumbent_lease_identity));
    const server_cache_lease_subject incumbent_subject {
        retention.artifact_id(
            server_retention_instance_key::for_slot(destination_slot)),
        common_retention_artifact_kind::live_slot, destination_slot,
    };
    auto incumbent_lease = authority.leases.grant_soft(
        incumbent_subject,
        server_cache_lease_scope::from(authority.leases.new_context_scope()),
        incumbent_lease_identity, UINT64_MAX/2);
    CHECK(incumbent_lease);

    const auto publish_host = [&] (
            int32_t source_slot, const server_prompt & prompt,
            const std::shared_ptr<const server_prompt_cache_vbr_payload> & owner,
            const common_cache_family_binding & family) {
        server_prompt_cache_vbr_publication_metadata metadata;
        if (!cache.prepare_vbr_publication_metadata(
                prompt, execution_key, adapter_key, source_slot, metadata)) {
            return false;
        }
        auto payload = server_prompt_cache_payload::from_vbr(owner);
        return cache.publish_vbr(metadata, payload, family, false);
    };
    CHECK(publish_host(
        incoming_source_slot, incoming_prompt, incoming_owner,
        incoming_family));
    CHECK(publish_host(
        destination_slot, incumbent_prompt, recovery_owner,
        incumbent_family));
    CHECK(cache.states.size() == 2);

    std::vector<llama_memory_tree_child> occupied_tree;
    CHECK(llama_memory_tree_collect(memory, occupied_tree));
    const auto occupied_attention = std::find_if(
        occupied_tree.begin(), occupied_tree.end(),
        [](const llama_memory_tree_child & child) {
            return child.attention != nullptr;
        });
    CHECK(occupied_attention != occupied_tree.end());
    if (occupied_attention == occupied_tree.end()) {
        return false;
    }
    auto * occupied_cache = occupied_attention->attention;

    llama_tokens request_tokens = incoming_tokens;
    request_tokens.push_back(continuation);
    struct publish_state {
        server_prompt_cache * cache = nullptr;
        server_prompt_cache_vbr_replacement_ticket * ticket = nullptr;
        bool published = false;
    };
    const auto import_request = [&] (publish_state & state) {
        server_vbr_artifact_import_target request;
        request.memory = memory;
        request.destination = destination_slot;
        request.execution_identity = execution_key;
        request.adapter_config_identity = adapter_key;
        request.previously_observed = true;
        request.publish_context = &state;
        request.prepare_publish = [] (
                void * opaque, const std::vector<llama_token> & tokens,
                uint64_t sequence_epoch) noexcept {
            auto * current = static_cast<publish_state *>(opaque);
            return current && current->cache && current->ticket &&
                current->ticket->ready() &&
                tokens.size() == current->ticket->incoming_prefix_tokens() &&
                sequence_epoch ==
                    current->ticket->replacement_prompt().sequence_epoch &&
                current->cache->prepare_vbr_occupied_replacement_publish(
                    *current->ticket);
        };
        request.publish = [] (void * opaque) noexcept {
            auto * current = static_cast<publish_state *>(opaque);
            GGML_ASSERT(current && current->cache && current->ticket);
            current->cache->publish_vbr_occupied_replacement(
                *current->ticket);
            current->published = true;
        };
        return request;
    };
    const auto prepare_ticket = [&] (
            server_prompt_cache_vbr_replacement_ticket & ticket) {
        server_prompt_cache_vbr_restore_candidate candidate;
        return cache.prepare_vbr_restore(
                   server_tokens(request_tokens, false), execution_key,
                   adapter_key, candidate, false) &&
            cache.prepare_vbr_occupied_replacement(
                std::move(candidate), incumbent_prompt, incumbent_family,
                incoming_family, destination_slot, execution_key,
                adapter_key, ticket);
    };

    const auto incumbent_artifact = retention.artifact_id(
        server_retention_instance_key::for_slot(destination_slot));
    server_prompt_cache_vbr_replacement_ticket refused_ticket;
    CHECK(prepare_ticket(refused_ticket));
    publish_state refused_state { &cache, &refused_ticket, false };
    incumbent_family = {
        common_cache_family_id { 77 }, common_cache_family_role::main,
    };
    const auto refused = store->import_host_occupied_replacement(
        import_request(refused_state), refused_ticket.incoming_payload(),
        refused_ticket.recovery_payload());
    CHECK(refused.status == server_vbr_artifact_import_status::unavailable);
    CHECK(!refused.adopt_attempted && !refused_state.published);
    CHECK(incumbent_prompt.tokens.retention_token_ids() == incumbent_tokens);
    CHECK(retention.artifact_id(
              server_retention_instance_key::for_slot(destination_slot)) ==
          incumbent_artifact);
    CHECK(llama_kv_cache_vbr_epoch_test::adopted_matches(
        occupied_cache, recovery_owner->package(),
        recovery_owner->package().manifest(), destination_slot));
    refused_ticket = {};
    incumbent_family = incumbent_family_initial;

    server_prompt_cache_vbr_replacement_ticket ticket;
    CHECK(prepare_ticket(ticket));
    publish_state state { &cache, &ticket, false };
    const auto imported = store->import_host_occupied_replacement(
        import_request(state), ticket.incoming_payload(),
        ticket.recovery_payload());
    if (imported.status != server_vbr_artifact_import_status::ok ||
        imported.adopt_status != vbr_adopt_status::adopted ||
        !state.published) {
        std::fprintf(stderr,
            "VBR occupied import failed: store=%s validation=%s stage=%s "
            "reserve=%s adopt=%s recovery=%s phase=%s schedule=%s "
            "destination=%s decision=%s attempted=%d published=%d\n",
            server_vbr_artifact_import_status_name(imported.status),
            vbr_manifest_validation_status_name(imported.validation_status),
            vbr_adopt_stage_status_name(imported.stage_status),
            vbr_downward_reserve_status_name(imported.downward_reserve_status),
            vbr_adopt_status_name(imported.adopt_status),
            vbr_adopt_recovery_outcome_name(imported.recovery),
            vbr_adopt_phase_name(imported.phase),
            vbr_import_schedule_status_name(imported.schedule_status),
            vbr_import_destination_status_name(imported.destination_status),
            vbr_import_decision_name(imported.decision),
            imported.adopt_attempted, state.published);
    }
    CHECK(imported.status == server_vbr_artifact_import_status::ok);
    CHECK(imported.adopt_status == vbr_adopt_status::adopted);
    CHECK(state.published);
    CHECK(incumbent_prompt.tokens.retention_token_ids() == incoming_tokens);
    CHECK(incumbent_family == incoming_family);
    CHECK(retention.prepared_for_launch(
        server_retention_instance_key::for_slot(destination_slot)));
    CHECK(llama_kv_cache_vbr_epoch_test::adopted_matches(
        occupied_cache, incoming_owner->package(),
        incoming_owner->package().manifest(), destination_slot));
    if (imported.status != server_vbr_artifact_import_status::ok ||
        imported.adopt_status != vbr_adopt_status::adopted ||
        !state.published) {
        return false;
    }
    cache.commit_vbr_occupied_replacement(
        ticket, incumbent_prompt, incumbent_family, destination_slot);
    CHECK(!ticket.ready());

    CHECK(model_adoption_decode(
        context.get(), { continuation }, llama_pos(incoming_tokens.size())));
    llama_synchronize(context.get());
    float * actual_logits = llama_get_logits_ith(context.get(), -1);
    CHECK(actual_logits != nullptr);
    if (actual_logits) {
        CHECK(std::memcmp(
            expected_logits.data(), actual_logits,
            expected_logits.size()*sizeof(float)) == 0);
    }
    // The no-fail sidecar swap retired the displaced live artifact and its
    // soft lease. Drop the now-stale local handle without a second terminal.
    incumbent_lease = {};

    // PT production composition: derive a true shorter/divergent prefix from
    // the same store-owned parent, import it through the real empty-target
    // store door, and prove that no suffix row was needed for continuation
    // parity.
    llama_memory_clear(memory, true);
    CHECK(llama_kv_cache_vbr_epoch_test::
        make_construction_empty_preserve_tiers(occupied_cache));
    llama_tokens divergent_request = prefix_tokens;
    llama_token divergent = llama_token(
        (uint32_t(prefix_continuation)+1u)%uint32_t(vocab_size));
    if (divergent == prefix_continuation) {
        divergent = llama_token(
            (uint32_t(divergent)+1u)%uint32_t(vocab_size));
    }
    divergent_request.push_back(divergent);
    vbr_artifact_attention_prefix_projection prefix_projection;
    CHECK(store->prepare_host_prefix_projection(
        projected_owner, divergent_request, prefix_count,
        prefix_projection) ==
        vbr_artifact_prefix_projection_status::projected);
    struct prefix_publish_state {
        const llama_tokens * expected = nullptr;
        bool published = false;
    } prefix_state { &prefix_tokens, false };
    server_vbr_artifact_import_target prefix_request;
    prefix_request.memory = memory;
    prefix_request.destination = destination_slot;
    prefix_request.execution_identity = execution_key;
    prefix_request.adapter_config_identity = adapter_key;
    prefix_request.previously_observed = true;
    prefix_request.publish_context = &prefix_state;
    prefix_request.prepare_publish = [](
            void * opaque,
            const std::vector<llama_token> & tokens,
            uint64_t sequence_epoch) noexcept {
        const auto * state = static_cast<const prefix_publish_state *>(opaque);
        return state && state->expected && sequence_epoch == 1 &&
            tokens == *state->expected;
    };
    prefix_request.publish = [](void * opaque) noexcept {
        static_cast<prefix_publish_state *>(opaque)->published = true;
    };
    const auto prefix_import = store->import_host_prefix_payload(
        std::move(prefix_request), projected_owner,
        std::move(prefix_projection));
    if (prefix_import.status != server_vbr_artifact_import_status::ok ||
        prefix_import.adopt_status != vbr_adopt_status::adopted ||
        !prefix_state.published) {
        std::fprintf(stderr,
            "VBR prefix import failed: store=%s validation=%s stage=%s "
            "reserve=%s adopt=%s phase=%s schedule=%s destination=%s "
            "decision=%s attempted=%d published=%d\n",
            server_vbr_artifact_import_status_name(prefix_import.status),
            vbr_manifest_validation_status_name(
                prefix_import.validation_status),
            vbr_adopt_stage_status_name(prefix_import.stage_status),
            vbr_downward_reserve_status_name(
                prefix_import.downward_reserve_status),
            vbr_adopt_status_name(prefix_import.adopt_status),
            vbr_adopt_phase_name(prefix_import.phase),
            vbr_import_schedule_status_name(prefix_import.schedule_status),
            vbr_import_destination_status_name(
                prefix_import.destination_status),
            vbr_import_decision_name(prefix_import.decision),
            prefix_import.adopt_attempted, prefix_state.published);
    }
    CHECK(prefix_import.status == server_vbr_artifact_import_status::ok);
    CHECK(prefix_import.adopt_status == vbr_adopt_status::adopted);
    CHECK(prefix_state.published);
    CHECK(memory->seq_pos_min(destination_slot) == 0);
    CHECK(memory->seq_pos_max(destination_slot) ==
          llama_pos(prefix_count-1));
    std::vector<uint8_t> prefix_actual_rows;
    CHECK(llama_kv_cache_vbr_epoch_test::snapshot_sequence_rows(
        occupied_cache, destination_slot, uint32_t(prefix_count),
        prefix_actual_rows));
    if (prefix_actual_rows != prefix_expected_rows) {
        size_t mismatch = 0;
        while (mismatch < prefix_actual_rows.size() &&
               mismatch < prefix_expected_rows.size() &&
               prefix_actual_rows[mismatch] == prefix_expected_rows[mismatch]) {
            ++mismatch;
        }
        std::fprintf(stderr,
            "VBR prefix KV mismatch byte=%zu expected_size=%zu actual_size=%zu\n",
            mismatch, prefix_expected_rows.size(), prefix_actual_rows.size());
    }
    CHECK(prefix_actual_rows == prefix_expected_rows);
    CHECK(model_adoption_decode(
        context.get(), { prefix_continuation }, llama_pos(prefix_count)));
    llama_synchronize(context.get());
    float * prefix_actual = llama_get_logits_ith(context.get(), -1);
    CHECK(prefix_actual != nullptr);
    if (prefix_actual && !prefix_expected_logits.empty()) {
        const bool equal = std::memcmp(
            prefix_expected_logits.data(), prefix_actual,
            prefix_expected_logits.size()*sizeof(float)) == 0;
        if (!equal) {
            float max_abs = 0.0f;
            size_t max_index = 0;
            for (size_t i = 0; i < prefix_expected_logits.size(); ++i) {
                const float delta = std::fabs(
                    prefix_expected_logits[i] - prefix_actual[i]);
                if (delta > max_abs) {
                    max_abs = delta;
                    max_index = i;
                }
            }
            std::fprintf(stderr,
                "VBR prefix parity mismatch max_abs=%g index=%zu expected=%g actual=%g\n",
                max_abs, max_index, prefix_expected_logits[max_index],
                prefix_actual[max_index]);
        }
        CHECK(equal);
    }
    return failures == 0;
}

static void test_final_recheck_excludes_only_own_reservation() {
    vbr_generation_tracker tracker(1, 256, 1,
                                  vbr_lineage_uuid { 0xe1, 0xf2 });
    CHECK(tracker.active());
    auto binding = import_binding(tracker.runtime_instance());
    vbr_scoped_operation operation(binding);
    CHECK(bool(operation));
    const int32_t own = vbr_recovery_reserve(
        operation.id(), tracker.runtime_instance());
    CHECK(own >= 0);
    CHECK(vbr_recovery_pending_for(tracker.runtime_instance()));
    CHECK(!vbr_recovery_pending_for_except(
        tracker.runtime_instance(), operation.id()));

    auto second_binding = import_binding(tracker.runtime_instance());
    vbr_scoped_operation second(second_binding);
    CHECK(bool(second));
    const int32_t foreign = vbr_recovery_reserve(
        second.id(), tracker.runtime_instance());
    CHECK(foreign >= 0);
    CHECK(vbr_recovery_pending_for_except(
        tracker.runtime_instance(), operation.id()));
    CHECK(vbr_recovery_release_unused(foreign, second.id()));
    second.close(vbr_operation_outcome::aborted);
    CHECK(!vbr_recovery_pending_for_except(
        tracker.runtime_instance(), operation.id()));
    CHECK(vbr_recovery_release_unused(own, operation.id()));
    operation.close(vbr_operation_outcome::aborted);
}

static void test_checkpoint_recurrent_frontier_header() {
    std::array<uint8_t,
        2*sizeof(uint32_t)+sizeof(llama_seq_id)+sizeof(llama_pos)> bytes = {};
    const uint32_t magic = 0xaf143cd8;
    const llama_seq_id sequence = 3;
    const uint32_t cell_count = 1;
    const llama_pos terminal = 37;
    size_t cursor = 0;
    std::memcpy(bytes.data()+cursor, &magic, sizeof(magic));
    cursor += sizeof(magic);
    std::memcpy(bytes.data()+cursor, &sequence, sizeof(sequence));
    cursor += sizeof(sequence);
    std::memcpy(bytes.data()+cursor, &cell_count, sizeof(cell_count));
    cursor += sizeof(cell_count);
    std::memcpy(bytes.data()+cursor, &terminal,
                sizeof(terminal));
    llama_pos parsed = -1;
    CHECK(vbr_explicit_recurrent_companion_terminal(
        bytes.data(), bytes.size(), parsed));
    CHECK(parsed == terminal);

    const llama_pos swapped_frontier = 19;
    std::memcpy(bytes.data()+cursor, &swapped_frontier,
                sizeof(swapped_frontier));
    CHECK(vbr_explicit_recurrent_companion_terminal(
        bytes.data(), bytes.size(), parsed));
    CHECK(parsed != terminal);
    const uint32_t multiple_cells = 2;
    std::memcpy(bytes.data()+cursor-sizeof(cell_count), &multiple_cells,
                sizeof(multiple_cells));
    CHECK(!vbr_explicit_recurrent_companion_terminal(
        bytes.data(), bytes.size(), parsed));
}

static void test_dflash_ring_frontier_header() {
    std::array<uint8_t, 6*sizeof(int32_t)> bytes = {};
    auto * header = reinterpret_cast<int32_t *>(bytes.data());
    header[0] = 1;
    header[1] = 1;
    header[2] = 38;
    header[3] = 1;
    header[4] = 1;
    header[5] = 1;
    llama_pos terminal = -1;
    CHECK(common_speculative_ring_state_serialized_terminal(
        bytes.data(), bytes.size(), terminal));
    CHECK(terminal == 37);
    header[2] = 19;
    CHECK(common_speculative_ring_state_serialized_terminal(
        bytes.data(), bytes.size(), terminal));
    CHECK(terminal == 18);
    header[2] = -1;
    CHECK(!common_speculative_ring_state_serialized_terminal(
        bytes.data(), bytes.size(), terminal));
    header[2] = INT32_MAX;
    CHECK(common_speculative_ring_state_serialized_terminal(
        bytes.data(), bytes.size(), terminal));
    CHECK(terminal == llama_pos(INT32_MAX)-1);
}

static bool parse_vbr_type(const std::string & name, ggml_type & output) {
    if (name == "f16") {
        output = GGML_TYPE_F16;
    } else if (name == "t8") {
        output = GGML_TYPE_TURBO8_0;
    } else if (name == "t4") {
        output = GGML_TYPE_TURBO4_0;
    } else if (name == "t3") {
        output = GGML_TYPE_TURBO3_TCQ;
    } else if (name == "t2") {
        output = GGML_TYPE_TURBO2_TCQ;
    } else if (name == "t1") {
        output = GGML_TYPE_TURBO1_TCQ;
    } else {
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    adoption_fixture::test_occupied_replacement_free_cell_adoption();
    adoption_fixture::test_occupied_replacement_fault_preserves_incumbent();
    adoption_fixture::test_occupied_spec_companion_replacement_and_rollback();
    adoption_fixture::test_occupied_recycle_success_and_zero_growth();
    adoption_fixture::test_occupied_recycle_partial_write_and_late_fault_matrix();
    adoption_fixture::test_occupied_recycle_replay_failure_quarantines();
    adoption_fixture::test_occupied_transformed_recycle_uses_recovery_geometry();
    adoption_fixture::test_occupied_replacement_tracker_consumes_canonical_map();
    test_checkpoint_recurrent_frontier_header();
    test_dflash_ring_frontier_header();
    test_epoch_capacity_preflight();
    test_closed_vocabularies();
    test_complete_tree_barrier_fail_closed();
    adoption_fixture::test_real_driver_smoke();
    adoption_fixture::test_erase_releases_receipt_for_second_adopt();
    adoption_fixture::test_phase_fault_matrix();
    adoption_fixture::test_shard_child_and_partial_map_matrix();
    adoption_fixture::test_complete_tree_barrier_matrix();
    adoption_fixture::test_serial_and_admission_faults();
    adoption_fixture::test_downward_subphase_matrix();
    adoption_fixture::test_upward_reconstruction();
    test_final_recheck_excludes_only_own_reservation();
    test_native_tracker_image_preserves_lineage_and_runtime();
    test_real_tracker_import_settle_and_teardown();
    test_live_rebased_tracker_image_is_fresh();
    test_native_tracker_rejects_uncovered_cell_and_tuple_splice();
    if (argc >= 2 &&
        (std::string(argv[1]) == "--vbr-adopt-cuda" ||
         std::string(argv[1]) == "--vbr-transform-cuda" ||
         std::string(argv[1]) == "--vbr-live-schedule-cuda" ||
         std::string(argv[1]) == "--vbr-transformed-destination-cuda" ||
         std::string(argv[1]) == "--vbr-occupied-restore-cuda")) {
        const bool downward = std::string(argv[1]) == "--vbr-transform-cuda";
        const bool live_schedule =
            std::string(argv[1]) == "--vbr-live-schedule-cuda";
        const bool transformed_destination =
            std::string(argv[1]) == "--vbr-transformed-destination-cuda";
        const bool occupied_restore =
            std::string(argv[1]) == "--vbr-occupied-restore-cuda";
        if ((!downward && !live_schedule && !transformed_destination && !occupied_restore &&
                 argc != 3) ||
            (live_schedule && argc != 6) ||
            (downward && argc != 3 && argc != 5) ||
            (transformed_destination && argc != 7) ||
            (occupied_restore && argc != 4)) {
            std::fprintf(stderr,
                "usage: %s --vbr-adopt-cuda MODEL\n"
                "       %s --vbr-transform-cuda MODEL [SOURCE TARGET]\n"
                "       %s --vbr-transformed-destination-cuda MODEL SOURCE ENTRY "
                "EXPECTED BUDGET_MIB\n"
                "       %s --vbr-occupied-restore-cuda MODEL ENTRY\n"
                "       %s --vbr-live-schedule-cuda MODEL BUDGET_MIB TOKENS "
                "mixed|straddled\n"
                "tiers: f16 t8 t4 t3 t2 t1\n",
                argv[0], argv[0], argv[0], argv[0], argv[0]);
            return 2;
        }
        ggml_type source_type = GGML_TYPE_F16;
        ggml_type target_type = downward ?
            GGML_TYPE_TURBO3_TCQ : GGML_TYPE_F16;
        if (downward && argc == 5 &&
            (!parse_vbr_type(argv[3], source_type) ||
             !parse_vbr_type(argv[4], target_type))) {
            std::fprintf(stderr, "invalid VBR transform tier pair: %s -> %s\n",
                argv[3], argv[4]);
            return 2;
        }
        vbr_downward_recipe requested_recipe;
        ggml_type expected_destination_type = GGML_TYPE_COUNT;
        uint64_t destination_budget_mib = 0;
        bool destination_downward = downward;
        if (occupied_restore && !parse_vbr_type(argv[3], target_type)) {
            std::fprintf(stderr, "invalid occupied-restore entry tier: %s\n", argv[3]);
            return 2;
        }
        if (transformed_destination) {
            char * budget_end = nullptr;
            if (!parse_vbr_type(argv[3], source_type) ||
                !parse_vbr_type(argv[4], target_type) ||
                !parse_vbr_type(argv[5], expected_destination_type)) {
                std::fprintf(stderr, "invalid transformed-destination tier\n");
                return 2;
            }
            destination_budget_mib = std::strtoull(
                argv[6], &budget_end, 10);
            vbr_downward_recipe destination_recipe;
            const auto relation = vbr_downward_resolve_recipe(
                source_type, expected_destination_type,
                GGML_TYPE_TURBO1_TCQ, true, destination_recipe);
            destination_downward =
                relation == vbr_downward_recipe_status::resolved;
            vbr_upward_recipe destination_upward_recipe;
            const bool destination_upward =
                vbr_upward_resolve_recipe(
                    source_type, expected_destination_type,
                    destination_upward_recipe) ==
                    vbr_upward_recipe_status::resolved;
            if (!destination_budget_mib ||
                destination_budget_mib > UINT64_MAX/(1024*1024) ||
                *budget_end ||
                (relation != vbr_downward_recipe_status::resolved &&
                 relation != vbr_downward_recipe_status::equal_tier &&
                 !destination_upward)) {
                std::fprintf(stderr, "invalid transformed-destination schedule\n");
                return 2;
            }
        }
        if (downward && vbr_downward_resolve_recipe(
                source_type, target_type, GGML_TYPE_TURBO1_TCQ, true,
                requested_recipe) != vbr_downward_recipe_status::resolved) {
            std::fprintf(stderr, "unsupported VBR transform tier pair: %s -> %s\n",
                argc == 5 ? argv[3] : "f16",
                argc == 5 ? argv[4] : "t3");
            return 2;
        }
        test_packed_h2d_projection_stream();
        test_packed_h2d_projection_max_ranges();
        test_cuda_h2d_adapter();
        if (failures == 0) {
            if (occupied_restore) {
                model_backed_occupied_store(argv[2], target_type);
            } else if (transformed_destination) {
                model_backed_adoption(
                    argv[2], destination_downward,
                    source_type, target_type, 0, 0, false,
                    destination_budget_mib, expected_destination_type);
            } else if (live_schedule) {
                char * budget_end = nullptr;
                char * tokens_end = nullptr;
                const uint64_t budget = std::strtoull(
                    argv[3], &budget_end, 10);
                const uint64_t n_tokens = std::strtoull(
                    argv[4], &tokens_end, 10);
                const bool straddled = std::string(argv[5]) == "straddled";
                if (!budget || !n_tokens || *budget_end || *tokens_end ||
                    (!straddled && std::string(argv[5]) != "mixed")) {
                    std::fprintf(stderr, "invalid VBR live-schedule arguments\n");
                    return 2;
                }
                model_backed_adoption(
                    argv[2], false, GGML_TYPE_F16, GGML_TYPE_F16,
                    budget, size_t(n_tokens), straddled);
            } else {
                model_backed_adoption(
                    argv[2], downward, source_type, target_type);
            }
        }
    }
    if (failures != 0) {
        std::fprintf(stderr, "%d VBR adoption test(s) failed\n", failures);
        return 1;
    }
    std::printf("VBR artifact adoption: PASS\n");
    return 0;
}
