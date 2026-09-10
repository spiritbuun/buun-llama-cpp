#include "llama-vbr-generation.h"

#include "llama-vbr-explicit-capture.h"
#include "llama-vbr-artifact-validate.h"

#include "ggml.h"
#include "llama-cparams.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <new>
#include <set>

namespace {

static_assert(LLAMA_MAX_SEQ <= std::numeric_limits<int16_t>::max(),
              "generation tracker packed membership provenance must widen with LLAMA_MAX_SEQ");

enum class generation_dispatch_effect : uint8_t {
    dependency,
    membership,
    global,
    unit,
    delegated_transaction,
};

// VBR_GENERATION_MUTATION_DISPATCH_EXHAUSTIVE
constexpr std::array<generation_dispatch_effect,
                     static_cast<size_t>(vbr_mutation_registrant::count)>
    VBR_GENERATION_DISPATCH = {
        {
         generation_dispatch_effect::dependency,             // apply_ubatch_append
            generation_dispatch_effect::dependency,             // apply_ubatch_occupied_reuse
            generation_dispatch_effect::membership,             // seq_rm
            generation_dispatch_effect::membership,             // seq_cp
            generation_dispatch_effect::membership,             // seq_keep
            generation_dispatch_effect::dependency,             // seq_add
            generation_dispatch_effect::dependency,             // seq_div
            generation_dispatch_effect::delegated_transaction,  // state_read_meta
            generation_dispatch_effect::delegated_transaction,  // state_read_data
            generation_dispatch_effect::global,                 // state_read_install
            generation_dispatch_effect::delegated_transaction,  // state_read_cleanup
            generation_dispatch_effect::global,                 // whole_import
            generation_dispatch_effect::global,                 // explicit_restore_adopt
            generation_dispatch_effect::global,                 // clear
            generation_dispatch_effect::global,                 // full_reset
            generation_dispatch_effect::unit,                   // degrade_next
            generation_dispatch_effect::unit,                   // promote_next
            generation_dispatch_effect::delegated_transaction,  // execute_shed -> degrade_next
            generation_dispatch_effect::global,                 // authenticated_recovery
        }
};
static_assert(VBR_GENERATION_DISPATCH.size() == static_cast<size_t>(vbr_mutation_registrant::count),
              "every closed operation registry mutation registrant must have an generation tracker generation effect");

uint16_t pack_provenance(vbr_mutation_family family, vbr_operation_class operation_class) {
    return uint16_t(static_cast<uint8_t>(family)) | uint16_t(uint16_t(static_cast<uint8_t>(operation_class)) << 8);
}

const vbr_mutation_registration * registration_for(vbr_mutation_registrant registrant) {
    const size_t index = static_cast<size_t>(registrant);
    if (index >= VBR_MUTATION_REGISTRY.size()) {
        return nullptr;
    }
    const auto & registration = VBR_MUTATION_REGISTRY[index];
    return registration.registrant == registrant ? &registration : nullptr;
}

bool class_allowed(const vbr_mutation_registration & registration, vbr_operation_class operation_class) {
    const size_t index = static_cast<size_t>(operation_class);
    return index < static_cast<size_t>(vbr_operation_class::count) &&
           (registration.allowed_classes & (uint16_t(1u) << index)) != 0;
}

uint32_t page_count(uint32_t cells) {
    return (cells + VBR_GENERATION_PAGE_CELLS - 1) / VBR_GENERATION_PAGE_CELLS;
}

bool mask_test(const std::array<uint64_t, VBR_GENERATION_MASK_WORDS> & mask, uint32_t offset) {
    return (mask[offset / 64] & (uint64_t(1) << (offset % 64))) != 0;
}

void mask_set(std::array<uint64_t, VBR_GENERATION_MASK_WORDS> & mask, uint32_t offset) {
    mask[offset / 64] |= uint64_t(1) << (offset % 64);
}

bool unit_equal(const vbr_checkpoint_unit_generation & captured, const vbr_unit_generation & current) {
    return captured.repr_gen == current.repr_gen && captured.current_type == current.current_type &&
           captured.last_source_type == current.last_source_type && captured.domain == current.domain &&
           captured.promote_hops == current.promote_hops && captured.last_transition == current.last_transition;
}

}  // namespace

std::vector<uint32_t> vbr_generation_production_covered_set(
        const vbr_checkpoint_generation_stream & production_record) {
    std::vector<uint32_t> result;
    result.reserve(production_record.captured_dependency_count);
    for (const auto & page : production_record.pages) {
        const uint64_t base =
            uint64_t(page.page_index) * VBR_GENERATION_PAGE_CELLS;
        for (uint32_t offset = 0;
             offset < VBR_GENERATION_PAGE_CELLS; ++offset) {
            if (mask_test(page.covered_mask, offset)) {
                const uint64_t cell = base + offset;
                if (cell > UINT32_MAX) {
                    return {};
                }
                result.push_back(uint32_t(cell));
            }
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

vbr_generation_event::~vbr_generation_event() {
    if (owner_ != nullptr && !finish()) {
        std::abort();
    }
}

vbr_generation_event::vbr_generation_event(vbr_generation_event && other) noexcept :
    serial(other.serial),
    stream(other.stream),
    stamp_kind(other.stamp_kind),
    family(other.family),
    operation_class(other.operation_class),
    destructive(other.destructive),
    imported(other.imported),
    operation_id(other.operation_id),
    manifest(other.manifest),
    registrant_bit(other.registrant_bit),
    poisoned(other.poisoned),
    extent_fn(other.extent_fn),
    extent_ctx(other.extent_ctx),
    owner_(other.owner_) {
    other.serial       = 0;
    other.operation_id = {};
    other.extent_fn    = nullptr;
    other.extent_ctx   = nullptr;
    other.owner_       = nullptr;
}

bool vbr_generation_event::finish() {
    if (owner_ == nullptr) {
        return false;
    }
    if (!owner_->finish_event(serial)) {
        return false;
    }
    owner_ = nullptr;
    serial = 0;
    return true;
}

struct vbr_generation_stream_state {
    std::vector<uint32_t> page_event_gen;
    std::vector<uint32_t> page_last_destructive_gen;
    std::vector<uint32_t> page_last_import_gen;
    std::vector<uint64_t> page_event_serial;

    std::vector<uint32_t> cell_last_dependency_gen;
    std::vector<uint32_t> cell_last_membership_gen;
    std::vector<uint16_t> cell_dependency_provenance;
    std::vector<uint16_t> cell_membership_provenance;
    std::vector<int16_t>  cell_last_membership_seq;

    // Durable committed-extent references. One per stamp kind: a cell
    // can retain two different events (latest dependency + latest membership).
    std::vector<vbr_extent_ref> cell_dependency_extent;
    std::vector<vbr_extent_ref> cell_membership_extent;
    // Stamp-time range-proof bits: the position was inside the authenticated target range.
    std::vector<uint64_t> cell_dependency_in_range;
    std::vector<uint64_t> cell_membership_in_range;
};

struct vbr_tracker_import_image::impl {
    vbr_lineage_uuid lineage;
    vbr_controller_instance_id instance;
    uint64_t global_generation = 0;
    uint64_t mutation_serial = 0;
    uint32_t active_event_depth = 0;
    std::vector<vbr_generation_stream_state> streams;
    std::vector<vbr_unit_generation> units;
    vbr_extent_store * extent_store = nullptr;
    std::vector<vbr_extent_handle> extent_handles;
    std::vector<vbr_extent_ref> extent_guard_refs;
    bool ready = false;

    ~impl() {
        if (extent_store == nullptr) {
            return;
        }
        for (auto & stream : streams) {
            for (const auto ref : stream.cell_dependency_extent) {
                extent_store->release_ref(ref);
            }
            for (const auto ref : stream.cell_membership_extent) {
                extent_store->release_ref(ref);
            }
        }
        for (const auto ref : extent_guard_refs) {
            extent_store->release_ref(ref);
        }
        for (const auto handle : extent_handles) {
            extent_store->fail(handle);
        }
    }
};

static void resize_stream_state(
        vbr_generation_stream_state & stream,
        uint32_t n_pages, uint32_t n_cells) {
    stream.page_event_gen.resize(n_pages);
    stream.page_last_destructive_gen.resize(n_pages);
    stream.page_last_import_gen.resize(n_pages);
    stream.page_event_serial.resize(n_pages);
    stream.cell_last_dependency_gen.resize(n_cells);
    stream.cell_last_membership_gen.resize(n_cells);
    stream.cell_dependency_provenance.resize(n_cells);
    stream.cell_membership_provenance.resize(n_cells);
    stream.cell_last_membership_seq.resize(n_cells, -1);
    stream.cell_dependency_extent.resize(n_cells);
    stream.cell_membership_extent.resize(n_cells);
    stream.cell_dependency_in_range.resize((n_cells + 63)/64);
    stream.cell_membership_in_range.resize((n_cells + 63)/64);
}

vbr_tracker_import_image::vbr_tracker_import_image()
    : impl_(new impl) {}
vbr_tracker_import_image::~vbr_tracker_import_image() = default;
vbr_tracker_import_image::vbr_tracker_import_image(
        vbr_tracker_import_image &&) noexcept = default;
vbr_tracker_import_image & vbr_tracker_import_image::operator=(
        vbr_tracker_import_image &&) noexcept = default;
bool vbr_tracker_import_image::ready() const noexcept {
    return impl_ && impl_->ready;
}
bool vbr_tracker_import_image::stable() const noexcept {
    return impl_ && impl_->ready && impl_->active_event_depth == 0 &&
           (impl_->mutation_serial & 1u) == 0 &&
           std::all_of(impl_->units.begin(), impl_->units.end(),
               [](const vbr_unit_generation & unit) {
                   return (unit.publish_seq & 1u) == 0;
               });
}

static void set_range_bit(std::vector<uint64_t> & bits, uint32_t cell, bool value) {
    if (value) {
        bits[cell / 64] |= uint64_t(1) << (cell % 64);
    } else {
        bits[cell / 64] &= ~(uint64_t(1) << (cell % 64));
    }
}

static bool get_range_bit(const std::vector<uint64_t> & bits, uint32_t cell) {
    return (bits[cell / 64] & (uint64_t(1) << (cell % 64))) != 0;
}

const char * vbr_generation_teardown_state_name(
        vbr_generation_teardown_state state) noexcept {
    switch (state) {
        case vbr_generation_teardown_state::clean: return "clean";
        case vbr_generation_teardown_state::active_event: return "active_event";
        case vbr_generation_teardown_state::operation_live: return "operation_live";
        case vbr_generation_teardown_state::recovery_owned: return "recovery_owned";
        case vbr_generation_teardown_state::instance_owner_mismatch: return "instance_owner_mismatch";
        case vbr_generation_teardown_state::_count: break;
    }
    return "invalid";
}

vbr_generation_teardown_state
vbr_generation_tracker::teardown_state() const noexcept {
    if (active_event_depth_ != 0 || (mutation_serial_ & 1u) != 0) {
        return vbr_generation_teardown_state::active_event;
    }
    if (!instance_enrolled_) {
        return vbr_generation_teardown_state::clean;
    }
    if (!vbr_operation_registry_quiescent_for(&instance_id_, 1)) {
        return vbr_generation_teardown_state::operation_live;
    }
    if (vbr_recovery_owned_by(instance_id_)) {
        return vbr_generation_teardown_state::recovery_owned;
    }
    if (!vbr_controller_instance_owned_by(instance_id_, this)) {
        return vbr_generation_teardown_state::instance_owner_mismatch;
    }
    return vbr_generation_teardown_state::clean;
}

vbr_generation_tracker::~vbr_generation_tracker() {
    if (teardown_state() != vbr_generation_teardown_state::clean) {
        std::abort();
    }
    if (instance_enrolled_) {
        if (!vbr_controller_instance_release(instance_id_, this)) {
            std::abort();
        }
        instance_enrolled_ = false;
    }
}

vbr_generation_tracker::vbr_generation_tracker(uint32_t n_stream, uint32_t n_cells, uint32_t n_units,
                                               vbr_lineage_uuid lineage) :
    n_cells_(n_cells),
    streams_(n_stream),
    units_(n_units) {
    const uint32_t n_pages = page_count(n_cells);
    for (auto & stream : streams_) {
        resize_stream_state(stream, n_pages, n_cells);
    }

    lineage_uuid_ = vbr_lineage_uuid_is_set(lineage)
                        ? lineage
                        : vbr_lineage_uuid_allocate();
    if (vbr_lineage_uuid_is_set(lineage_uuid_)) {
        instance_id_ = vbr_controller_instance_id_allocate();
        instance_enrolled_ = vbr_controller_instance_check_and_claim(instance_id_, this);
    }
}

bool vbr_generation_tracker::active() const {
    return vbr_lineage_uuid_is_set(lineage_uuid_) &&
           vbr_controller_instance_id_is_set(instance_id_) && instance_enrolled_ &&
           !streams_.empty() && n_cells_ != 0;
}

uint32_t vbr_generation_tracker::stream_count() const {
    return static_cast<uint32_t>(streams_.size());
}

uint32_t vbr_generation_tracker::cell_count() const {
    return n_cells_;
}

uint32_t vbr_generation_tracker::unit_count() const {
    return static_cast<uint32_t>(units_.size());
}

bool vbr_generation_tracker::stable() const {
    if ((mutation_serial_ & 1u) != 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(units_mutex_);
    for (const auto & unit : units_) {
        if ((unit.publish_seq & 1u) != 0) {
            return false;
        }
    }
    return true;
}

vbr_lineage_uuid vbr_generation_tracker::lineage_identity() const {
    return lineage_uuid_;
}

vbr_controller_instance_id vbr_generation_tracker::runtime_instance() const {
    return instance_id_;
}

uint64_t vbr_generation_tracker::controller_generation() const {
    return global_generation_;
}

vbr_generation_event vbr_generation_tracker::begin_event(vbr_mutation_registrant   registrant,
                                                         vbr_operation_class       operation_class,
                                                         uint32_t                  stream,
                                                         vbr_generation_stamp_kind stamp_kind,
                                                         vbr_operation_id          operation_id,
                                                         vbr_event_extent_fn       extent_fn,
                                                         void *                    extent_ctx,
                                                         bool                      destructive,
                                                         bool                      imported) {
    // One named return object for every path (NRVO): the ~656-byte manifest is copied exactly
    // once, registry slot -> event, and refused paths return it with serial 0 (falsy).
    vbr_generation_event result;
    const auto * registration = registration_for(registrant);
    const size_t registrant_index = static_cast<size_t>(registrant);
    const generation_dispatch_effect expected_effect =
            stamp_kind == vbr_generation_stamp_kind::dependency
                    ? generation_dispatch_effect::dependency
                    : generation_dispatch_effect::membership;
    if (!active() || shadow_unavailable_ ||
        registration == nullptr || !class_allowed(*registration, operation_class) ||
        registrant_index >= VBR_GENERATION_DISPATCH.size() ||
        VBR_GENERATION_DISPATCH[registrant_index] != expected_effect ||
        stream >= streams_.size() || event_serial_ == std::numeric_limits<uint64_t>::max() ||
        active_event_depth_ == MAX_EVENT_DEPTH ||
        (active_event_depth_ == 0 && mutation_serial_ == std::numeric_limits<uint64_t>::max())) {
        return result;
    }

    // Full manifest authentication. The event must cite a live
    // operation whose manifest (a) lists this registrant in its closed mask, (b) declares
    // this exact operation class, (c) is in the mutate phase, and (d) carries a target
    // covering this tracker's runtime instance and the event's stream.
    if (!operation_id || !vbr_operation_registry_binding(operation_id, result.manifest)) {
        return result;
    }
    // Begin still refuses when no target could ever cover this
    // instance/stream/class/registrant; the per-(seq, position) selection happens at EACH STAMP
    // against the event's manifest copy, so multi-target manifests authenticate
    // multi-sequence ubatches exactly instead of citing target zero.
    if (result.manifest.find_covering_target(instance_id_, stream, operation_class,
                                             vbr_registrant_bit(registrant)) == nullptr) {
        return result;
    }

    if (active_event_depth_ == 0) {
        ++mutation_serial_;
    }
    ++event_serial_;
    active_event_stack_[active_event_depth_] = event_serial_;
    ++active_event_depth_;
    result.serial          = event_serial_;
    result.stream          = stream;
    result.stamp_kind      = stamp_kind;
    result.family          = registration->family;
    result.operation_class = operation_class;
    result.destructive     = destructive;
    result.imported        = imported;
    result.operation_id    = operation_id;
    result.registrant_bit  = vbr_registrant_bit(registrant);
    result.extent_fn       = extent_fn;
    result.extent_ctx      = extent_ctx;
    result.owner_          = this;
    return result;
}

bool vbr_generation_tracker::finish_event(uint64_t serial) {
    if (serial == 0 || active_event_depth_ == 0 || active_event_stack_[active_event_depth_ - 1] != serial ||
        mutation_serial_ == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    --active_event_depth_;
    active_event_stack_[active_event_depth_] = 0;
    if (active_event_depth_ == 0) {
        ++mutation_serial_;
    }
    return true;
}

bool vbr_generation_tracker::stamp_cell(vbr_generation_event & event,
                                        uint32_t               cell,
                                        llama_seq_id           membership_seq,
                                        llama_pos              pre_mutation_pos) {
    return stamp_cell(event, cell, &membership_seq, 1, pre_mutation_pos);
}

bool vbr_generation_tracker::stamp_cell(vbr_generation_event & event,
                                        uint32_t               cell,
                                        const llama_seq_id *   seqs,
                                        int32_t                n_seqs,
                                        llama_pos              pre_mutation_pos) {
    // A poisoned event stays inert: no further metadata moves under it.
    if (event.poisoned) {
        return false;
    }
    bool event_is_live = false;
    for (uint32_t depth = 0; depth < active_event_depth_; ++depth) {
        event_is_live = event_is_live || active_event_stack_[depth] == event.serial;
    }
    // Wiring-bug refusals (wrong owner, dead event, out-of-bounds): plain false, no poison —
    // these are not authorization failures against this event's manifest.
    const bool binds_evidence = event.destructive || event.imported;
    if (!event || event.owner_ != this || !event_is_live ||
        event.stream >= streams_.size() || cell >= n_cells_ || seqs == nullptr || n_seqs < 1 ||
        (event.stamp_kind == vbr_generation_stamp_kind::membership && n_seqs != 1)) {
        return false;
    }
    // Destructive/import evidence binds to exactly ONE selected target; a shared multi-member
    // cell has no single exact citation, so it goes unavailable instead.
    if (binds_evidence && n_seqs != 1) {
        event.poisoned = true;
        set_shadow_unavailable();
        return false;
    }
    // Per-stamp covering-target selection over the event's authenticated manifest,
    // keyed by (instance, stream, class, registrant, seq, pre-mutation position). EVERY member of
    // a shared cell's sequence set needs a covering target (target-set proof); the first
    // member's selection supplies the durable evidence binding. NO cover => the stamp
    // refuses, POISONS the event, and latches shadow-unavailable IMMEDIATELY — metadata may
    // already have moved under an unauthenticated claim, so no strict accept can be allowed
    // to form until a sanctioned transition provably follows.
    uint8_t selected_index = 0;
    bool    all_real_range = true;
    for (int32_t i = 0; i < n_seqs; ++i) {
        if (seqs[i] < -1 || seqs[i] > std::numeric_limits<int16_t>::max()) {
            return false;  // wiring-bug refusal: out-of-domain value, not an auth failure
        }
        uint8_t      index    = 0;
        const auto * covering = event.manifest.find_covering_target_at(
                instance_id_, static_cast<uint16_t>(event.stream),
                event.operation_class, event.registrant_bit, seqs[i], pre_mutation_pos, &index);
        if (covering == nullptr) {
            event.poisoned = true;
            set_shadow_unavailable();
            return false;
        }
        all_real_range = all_real_range && covering->range.p0 >= 0;
        if (i == 0) {
            selected_index = index;
        }
    }
    vbr_extent_handle extent = {};
    if (binds_evidence && event.extent_fn != nullptr) {
        extent = event.extent_fn(event.extent_ctx, selected_index);
        if (!extent) {
            // Per-target reservation failed — the owning scope already took the availability
            // path; poison so the rest of this event is inert and the operation reports
            // failed instead of committing partial evidence.
            event.poisoned = true;
            set_shadow_unavailable();
            return false;
        }
    }
    // Record the range proof (`in_range`) so tombstone rows can prove
    // membership from committed evidence. True only when the position is known AND every
    // member's selected target carries a real (non-wildcard) range containing it.
    const bool in_authorized_range = pre_mutation_pos >= 0 && all_real_range;
    const llama_seq_id membership_seq = seqs[0];

    auto &         stream = streams_[event.stream];
    const uint32_t page   = cell / VBR_GENERATION_PAGE_CELLS;
    if (stream.page_event_serial[page] != event.serial) {
        if (stream.page_event_gen[page] == std::numeric_limits<uint32_t>::max()) {
            if (!reset_page_generations_before_wrap()) {
                return false;
            }
        }
        stream.page_event_serial[page] = event.serial;
        ++stream.page_event_gen[page];
        if (event.destructive) {
            stream.page_last_destructive_gen[page] = stream.page_event_gen[page];
        }
        if (event.imported) {
            stream.page_last_import_gen[page] = stream.page_event_gen[page];
        }
    }

    const uint32_t generation = stream.page_event_gen[page];
    const uint16_t provenance = pack_provenance(event.family, event.operation_class);
    if (event.stamp_kind == vbr_generation_stamp_kind::dependency) {
        stream.cell_last_dependency_gen[cell]   = generation;
        stream.cell_dependency_provenance[cell] = provenance;
        extents_.release_ref(stream.cell_dependency_extent[cell]);
        stream.cell_dependency_extent[cell] = extent ? extents_.add_ref(extent) : vbr_extent_ref{};
        set_range_bit(stream.cell_dependency_in_range, cell, in_authorized_range);
    } else {
        stream.cell_last_membership_gen[cell]   = generation;
        stream.cell_membership_provenance[cell] = provenance;
        stream.cell_last_membership_seq[cell]   = static_cast<int16_t>(membership_seq);
        extents_.release_ref(stream.cell_membership_extent[cell]);
        stream.cell_membership_extent[cell] = extent ? extents_.add_ref(extent) : vbr_extent_ref{};
        set_range_bit(stream.cell_membership_in_range, cell, in_authorized_range);
    }
    return true;
}

vbr_extent_ref vbr_generation_tracker::dependency_extent(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_dependency_extent.at(cell);
}

vbr_extent_ref vbr_generation_tracker::membership_extent(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_membership_extent.at(cell);
}

bool vbr_generation_tracker::dependency_in_range(uint32_t stream, uint32_t cell) const {
    return get_range_bit(streams_.at(stream).cell_dependency_in_range, cell);
}

bool vbr_generation_tracker::membership_in_range(uint32_t stream, uint32_t cell) const {
    return get_range_bit(streams_.at(stream).cell_membership_in_range, cell);
}

bool vbr_generation_tracker::global_invalidate_and_reset_extents(vbr_mutation_registrant registrant,
                                                                 vbr_operation_class     operation_class,
                                                                 vbr_operation_id        operation_id) {
    if (!global_transition(registrant, operation_class, operation_id)) {
        return false;
    }
    // Every stored extent reference is obsolete after the global invalidation; drop them so
    // reset_all() reclaims a coherent slab.
    for (auto & stream : streams_) {
        std::fill(stream.cell_dependency_extent.begin(), stream.cell_dependency_extent.end(), vbr_extent_ref{});
        std::fill(stream.cell_membership_extent.begin(), stream.cell_membership_extent.end(), vbr_extent_ref{});
    }
    extents_.reset_all();
    return true;
}

bool vbr_generation_tracker::reset_page_generations_before_wrap() {
    if (global_generation_ == std::numeric_limits<uint64_t>::max() ||
        (active_event_depth_ == 0 &&
         mutation_serial_ > std::numeric_limits<uint64_t>::max() - 2)) {
        return false;
    }
    const bool owns_stability_barrier = active_event_depth_ == 0;
    if (owns_stability_barrier) {
        ++mutation_serial_;
    } else if ((mutation_serial_ & 1u) == 0) {
        return false;
    }
    ++global_generation_;
    for (auto & stream : streams_) {
        std::fill(stream.page_event_gen.begin(), stream.page_event_gen.end(), 0);
        std::fill(stream.page_last_destructive_gen.begin(), stream.page_last_destructive_gen.end(), 0);
        std::fill(stream.page_last_import_gen.begin(), stream.page_last_import_gen.end(), 0);
        std::fill(stream.page_event_serial.begin(), stream.page_event_serial.end(), 0);
        std::fill(stream.cell_last_dependency_gen.begin(), stream.cell_last_dependency_gen.end(), 0);
        std::fill(stream.cell_last_membership_gen.begin(), stream.cell_last_membership_gen.end(), 0);
        std::fill(stream.cell_dependency_provenance.begin(), stream.cell_dependency_provenance.end(), 0);
        std::fill(stream.cell_membership_provenance.begin(), stream.cell_membership_provenance.end(), 0);
        std::fill(stream.cell_last_membership_seq.begin(), stream.cell_last_membership_seq.end(), -1);
        // Pre-wrap reset is a global invalidation: every stored extent reference is obsolete.
        std::fill(stream.cell_dependency_extent.begin(), stream.cell_dependency_extent.end(), vbr_extent_ref{});
        std::fill(stream.cell_membership_extent.begin(), stream.cell_membership_extent.end(), vbr_extent_ref{});
    }
    extents_.reset_all();
    if (owns_stability_barrier) {
        ++mutation_serial_;
    }
    return true;
}

bool vbr_generation_tracker::try_rearm() {
    if (!shadow_unavailable_) {
        return true;
    }
    if (vbr_recovery_pending_for(instance_id_) ||
        !vbr_operation_registry_has_capacity()) {
        return false;
    }
    if (!global_invalidate_and_reset_extents(vbr_mutation_registrant::authenticated_recovery,
                                             vbr_operation_class::controller)) {
        return false;
    }
    return try_clear_shadow_unavailable();
}

bool vbr_generation_tracker::global_transition(vbr_mutation_registrant registrant,
                                               vbr_operation_class     operation_class,
                                               vbr_operation_id        operation_id) {
    // Cited operations validate at manifest depth: the cited
    // binding must carry a covering target for THIS instance authorizing this registrant + class.
    // The recovery drain and registry-refusal fallback run OUTSIDE any operation (empty id) —
    // a NAMED exemption: they are the paths that CREATE availability.
    if (operation_id) {
        vbr_operation_binding cited;
        if (!vbr_operation_registry_binding(operation_id, cited) ||
            cited.find_covering_target(instance_id_, 0, operation_class,
                                       vbr_registrant_bit(registrant)) == nullptr) {
            return false;
        }
    }
    const auto * registration = registration_for(registrant);
    const size_t registrant_index = static_cast<size_t>(registrant);
    if (!active() || registration == nullptr || !class_allowed(*registration, operation_class) ||
        registrant_index >= VBR_GENERATION_DISPATCH.size() ||
        VBR_GENERATION_DISPATCH[registrant_index] != generation_dispatch_effect::global ||
        active_event_depth_ != 0 || (mutation_serial_ & 1u) != 0 ||
        global_generation_ == std::numeric_limits<uint64_t>::max() ||
        mutation_serial_ > std::numeric_limits<uint64_t>::max() - 2) {
        return false;
    }
    ++mutation_serial_;
    ++global_generation_;
    ++mutation_serial_;
    // The unavailable state does not auto-clear here: the cause (registry or
    // slab exhaustion) may persist. try_clear_shadow_unavailable() probes the cause.
    return true;
}

bool vbr_generation_tracker::initialize_unit(uint32_t unit, int32_t type, vbr_repr_domain domain) {
    std::lock_guard<std::mutex> lock(units_mutex_);
    if (unit >= units_.size()) {
        return false;
    }
    auto & state           = units_[unit];
    state.repr_gen         = 1;
    state.current_type     = type;
    state.last_source_type = type;
    state.domain           = domain;
    state.last_transition  = vbr_repr_transition::initial;
    return true;
}

bool vbr_generation_tracker::publish_unit(uint32_t                unit,
                                          int32_t                 source_type,
                                          int32_t                 target_type,
                                          vbr_repr_domain         domain,
                                          uint8_t                 promote_hops,
                                          vbr_repr_transition     transition,
                                          vbr_mutation_registrant registrant,
                                          vbr_operation_id        operation_id) {
    if (operation_id) {
        vbr_operation_binding cited;
        // The citation authenticates at manifest depth with the exact
        // registrant driving this publication — never an OR of plausible ones.
        if (!vbr_operation_registry_binding(operation_id, cited) ||
            cited.find_covering_target(instance_id_, 0,
                                       vbr_operation_class::controller,
                                       vbr_registrant_bit(registrant)) == nullptr) {
            return false;
        }
    }
    std::lock_guard<std::mutex> lock(units_mutex_);
    if (unit >= units_.size() || active_event_depth_ != 0 || (mutation_serial_ & 1u) != 0 ||
        units_[unit].current_type != source_type || (units_[unit].publish_seq & 1u) != 0) {
        return false;
    }
    if ((units_[unit].repr_gen == std::numeric_limits<uint64_t>::max() ||
         units_[unit].publish_seq > std::numeric_limits<uint64_t>::max() - 2) &&
        !reset_unit_generations_before_wrap()) {
        return false;
    }
    auto & state = units_[unit];
    ++state.publish_seq;
    ++state.repr_gen;
    state.last_source_type = source_type;
    state.current_type     = target_type;
    state.domain           = domain;
    state.promote_hops     = promote_hops;
    state.last_transition  = transition;
    ++state.publish_seq;
    return true;
}

// VBR_GENERATION_IMPORT_REGION_BEGIN
// 's validator-authenticated import is the fourth narrow authority allowed
// to consume captured page generations. It constructs an off-side image only;
// the artifact adoption validator remains the live admission authority.
bool vbr_generation_tracker::prepare_import_image(
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        llama_seq_id destination,
        const std::vector<vbr_artifact_stream_placement> & placements,
        vbr_tracker_import_image & output) noexcept {
    return prepare_import_image_impl(
        plan, source, destination, &placements, nullptr, output);
}

bool vbr_generation_tracker::prepare_relocated_import_image(
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        llama_seq_id destination,
        const vbr_occupied_replacement_guard & replacement,
        vbr_tracker_import_image & output) noexcept {
    return prepare_import_image_impl(
        plan, source, destination, nullptr, &replacement, output);
}

bool vbr_generation_tracker::prepare_import_image_impl(
        const vbr_tracker_install_child & plan,
        const vbr_checkpoint_generation_controller & source,
        llama_seq_id destination,
        const std::vector<vbr_artifact_stream_placement> * placements,
        const vbr_occupied_replacement_guard * replacement,
        vbr_tracker_import_image & output) noexcept {
    try {
        if (!active() || !stable() || !output.impl_ ||
            ((placements == nullptr) == (replacement == nullptr)) ||
            plan.child_id != source.child_id ||
            plan.target_instance != instance_id_ ||
            destination < 0 || destination >= LLAMA_MAX_SEQ ||
            plan.units.size() != units_.size() ||
            source.units.size() != units_.size() ||
            source.streams.size() != streams_.size() ||
            (plan.transition != vbr_tracker_install_transition::native_clone &&
             plan.transition != vbr_tracker_install_transition::whole_import)) {
            return false;
        }
        if (replacement &&
            (!replacement->ready() || replacement->destination() != destination ||
             plan.transition != vbr_tracker_install_transition::whole_import ||
             source.streams.size() != 1 ||
             replacement->cell_mapping().empty() ||
             replacement->cell_mapping().size() > n_cells_)) {
            return false;
        }
        if (plan.transition == vbr_tracker_install_transition::native_clone) {
            if (plan.lineage_uuid != source.lineage_uuid ||
                plan.global_generation != source.global_generation ||
                plan.units != source.units) {
                return false;
            }
        } else {
            if (plan.lineage_uuid != lineage_uuid_ ||
                plan.global_generation != 1 ||
                std::any_of(plan.units.begin(), plan.units.end(),
                    [](const vbr_checkpoint_unit_generation & unit) {
                        return unit.repr_gen != 1 ||
                               unit.last_transition !=
                                   vbr_repr_transition::whole_import;
                    })) {
                return false;
            }
        }
        auto next = std::make_unique<vbr_tracker_import_image::impl>();
        next->instance = instance_id_;
        next->lineage = plan.transition ==
                vbr_tracker_install_transition::native_clone
            ? plan.lineage_uuid : lineage_uuid_;
        // An occupied replacement changes one logical sequence while the
        // guarded map retains the other live sequences.  Their durable host
        // images remain valid only if the controller representation lineage
        // is preserved; the empty-cache whole-import generation belongs to
        // the ordinary import path.
        next->global_generation = replacement
            ? global_generation_ : plan.global_generation;
        if (!vbr_lineage_uuid_is_set(next->lineage) ||
            next->global_generation == 0) {
            return false;
        }
        next->streams.resize(streams_.size());
        const uint32_t n_pages = page_count(n_cells_);
        for (auto & stream : next->streams) {
            resize_stream_state(stream, n_pages, n_cells_);
        }
        next->extent_store = &extents_;
        std::vector<vbr_extent_handle> import_extents(streams_.size());
        for (const auto & source_stream : source.streams) {
            if (source_stream.stream_index >= streams_.size() ||
                source_stream.computation_frontier <= 0 ||
                import_extents[source_stream.stream_index]) {
                return false;
            }
            const auto handle = extents_.reserve(
                vbr_mutation_family::import,
                vbr_operation_class::state_api,
                uint16_t(source_stream.stream_index), destination,
                0, source_stream.computation_frontier);
            if (!handle) {
                return false;
            }
            const auto guard = extents_.add_ref(handle);
            if (!guard) {
                extents_.fail(handle);
                return false;
            }
            next->extent_guard_refs.push_back(guard);
            if (!extents_.commit(handle)) {
                return false;
            }
            import_extents[source_stream.stream_index] = handle;
            next->extent_handles.push_back(handle);
        }

        if (replacement) {
            // Rebuild one coherent tracker image for the shared physical
            // stream. The guard authenticates the destination mapping and all
            // single-owner cells retained for the other logical sequences.
            uint32_t previous_physical = 0;
            bool first = true;
            auto & stream = next->streams.front();
            std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> ranges;
            for (const auto & cell : replacement->preserved_cells()) {
                if (cell.stream_index != 0 ||
                    cell.physical_cell >= n_cells_ ||
                    cell.logical_position < 0 ||
                    cell.reference_count != 1 ||
                    cell.owner_sequence < 0 ||
                    cell.owner_sequence == destination ||
                    cell.owns_destination) {
                    return false;
                }
                auto inserted = ranges.emplace(
                    cell.owner_sequence,
                    std::make_pair(
                        cell.logical_position, cell.logical_position));
                inserted.first->second.first = std::min(
                    inserted.first->second.first, cell.logical_position);
                inserted.first->second.second = std::max(
                    inserted.first->second.second, cell.logical_position);
            }
            std::map<llama_seq_id, vbr_extent_handle> retained_extents;
            for (const auto & entry : ranges) {
                if (entry.second.second ==
                        std::numeric_limits<llama_pos>::max()) {
                    return false;
                }
                const auto handle = extents_.reserve(
                    vbr_mutation_family::import,
                    vbr_operation_class::state_api,
                    0, entry.first, entry.second.first,
                    entry.second.second + 1);
                if (!handle) {
                    return false;
                }
                const auto guard = extents_.add_ref(handle);
                if (!guard) {
                    extents_.fail(handle);
                    return false;
                }
                next->extent_guard_refs.push_back(guard);
                if (!extents_.commit(handle)) {
                    return false;
                }
                next->extent_handles.push_back(handle);
                retained_extents.emplace(entry.first, handle);
            }
            const auto stamp = [&](uint32_t physical, llama_seq_id seq,
                                   vbr_extent_handle handle) {
                if (physical >= n_cells_ || seq < 0 || !handle ||
                    stream.cell_last_dependency_gen[physical] != 0 ||
                    stream.cell_last_membership_gen[physical] != 0) {
                    return false;
                }
                const uint32_t page =
                    physical / VBR_GENERATION_PAGE_CELLS;
                constexpr uint32_t generation = 1;
                stream.page_event_gen[page] = std::max(
                    stream.page_event_gen[page], generation);
                stream.page_last_import_gen[page] = std::max(
                    stream.page_last_import_gen[page], generation);
                stream.cell_last_dependency_gen[physical] = generation;
                stream.cell_last_membership_gen[physical] = generation;
                const uint16_t provenance = pack_provenance(
                    vbr_mutation_family::import,
                    vbr_operation_class::state_api);
                stream.cell_dependency_provenance[physical] = provenance;
                stream.cell_membership_provenance[physical] = provenance;
                stream.cell_last_membership_seq[physical] =
                    static_cast<int16_t>(seq);
                stream.cell_dependency_extent[physical] =
                    extents_.add_ref(handle);
                stream.cell_membership_extent[physical] =
                    extents_.add_ref(handle);
                if (!stream.cell_dependency_extent[physical] ||
                    !stream.cell_membership_extent[physical]) {
                    return false;
                }
                set_range_bit(
                    stream.cell_dependency_in_range, physical, true);
                set_range_bit(
                    stream.cell_membership_in_range, physical, true);
                return true;
            };
            for (const auto & cell : replacement->preserved_cells()) {
                const auto handle = retained_extents.find(
                    cell.owner_sequence);
                if (handle == retained_extents.end() ||
                    !stamp(cell.physical_cell, cell.owner_sequence,
                           handle->second)) {
                    return false;
                }
            }
            const auto destination_handle = import_extents.front();
            for (const auto & cell : replacement->cell_mapping()) {
                if (cell.source_stream != 0 ||
                    cell.destination_physical_cell >= n_cells_ ||
                    cell.logical_position < 0 ||
                    (!first && cell.destination_physical_cell <= previous_physical)) {
                    return false;
                }
                first = false;
                previous_physical = cell.destination_physical_cell;
                const uint32_t physical = cell.destination_physical_cell;
                if (!stamp(physical, destination, destination_handle)) {
                    return false;
                }
            }
        } else {
            std::set<std::pair<uint32_t, uint32_t>> installed_cells;
            for (const auto & placement : *placements) {
                if (placement.child_id != plan.child_id ||
                    placement.stream_index >= next->streams.size()) {
                    return false;
                }
                const auto source_stream = std::find_if(
                    source.streams.begin(), source.streams.end(),
                    [&](const vbr_checkpoint_generation_stream & value) {
                        return value.stream_index == placement.stream_index;
                    });
                if (source_stream == source.streams.end()) {
                    return false;
                }
                auto & stream = next->streams[placement.stream_index];
                for (const auto & cell : placement.cells) {
                    if (cell.physical_cell >= n_cells_) {
                        return false;
                    }
                    const uint32_t page =
                        cell.physical_cell / VBR_GENERATION_PAGE_CELLS;
                    uint32_t generation = 1;
                    if (plan.transition ==
                            vbr_tracker_install_transition::native_clone) {
                        const auto page_ref = std::find_if(
                            source_stream->pages.begin(), source_stream->pages.end(),
                            [&](const vbr_generation_page_ref & value) {
                                return value.page_index == page;
                            });
                        if (page_ref == source_stream->pages.end() ||
                            page_ref->captured_page_gen == 0 ||
                            (page_ref->covered_mask[
                                (cell.physical_cell % VBR_GENERATION_PAGE_CELLS)/64] &
                             (uint64_t(1) <<
                                (cell.physical_cell % 64))) == 0) {
                            return false;
                        }
                        generation = page_ref->captured_page_gen;
                    }
                    const auto installed = std::make_pair(
                        placement.stream_index, cell.physical_cell);
                    if (!installed_cells.insert(installed).second) {
                        if (stream.cell_last_dependency_gen[cell.physical_cell] !=
                                generation ||
                            stream.cell_last_membership_seq[cell.physical_cell] !=
                                destination) {
                            return false;
                        }
                        continue;
                    }
                    stream.page_event_gen[page] =
                        std::max(stream.page_event_gen[page], generation);
                    stream.page_last_import_gen[page] =
                        std::max(stream.page_last_import_gen[page], generation);
                    stream.cell_last_dependency_gen[cell.physical_cell] = generation;
                    stream.cell_last_membership_gen[cell.physical_cell] = generation;
                    const uint16_t provenance = pack_provenance(
                        vbr_mutation_family::import,
                        vbr_operation_class::state_api);
                    stream.cell_dependency_provenance[cell.physical_cell] = provenance;
                    stream.cell_membership_provenance[cell.physical_cell] = provenance;
                    stream.cell_last_membership_seq[cell.physical_cell] =
                        static_cast<int16_t>(destination);
                    const auto handle = import_extents[placement.stream_index];
                    stream.cell_dependency_extent[cell.physical_cell] =
                        extents_.add_ref(handle);
                    stream.cell_membership_extent[cell.physical_cell] =
                        extents_.add_ref(handle);
                    if (!stream.cell_dependency_extent[cell.physical_cell] ||
                        !stream.cell_membership_extent[cell.physical_cell]) {
                        return false;
                    }
                    set_range_bit(stream.cell_dependency_in_range,
                                  cell.physical_cell, true);
                    set_range_bit(stream.cell_membership_in_range,
                                  cell.physical_cell, true);
                }
            }
        }

        for (const auto guard : next->extent_guard_refs) {
            extents_.release_ref(guard);
        }
        next->extent_guard_refs.clear();

        if (replacement) {
            std::lock_guard<std::mutex> lock(units_mutex_);
            if (std::any_of(units_.begin(), units_.end(),
                    [](const vbr_unit_generation & unit) {
                        return unit.repr_gen == 0 ||
                               (unit.publish_seq & 1u) != 0;
                    })) {
                return false;
            }
            next->units = units_;
        } else {
            next->units.reserve(plan.units.size());
            for (const auto & unit : plan.units) {
                if (unit.repr_gen == 0) {
                    return false;
                }
                vbr_unit_generation installed;
                installed.repr_gen = unit.repr_gen;
                installed.publish_seq = 0;
                installed.current_type = unit.current_type;
                installed.last_source_type = unit.last_source_type;
                installed.domain = unit.domain;
                installed.promote_hops = unit.promote_hops;
                installed.last_transition = unit.last_transition;
                next->units.push_back(installed);
            }
        }
        next->mutation_serial = 0;
        next->ready = true;
        output.impl_ = std::move(next);
        return true;
    } catch (...) {
        output.impl_.reset(new (std::nothrow) vbr_tracker_import_image::impl);
        return false;
    }
}

bool vbr_generation_tracker::import_image_installable(
        const vbr_tracker_import_image & image,
        vbr_operation_id operation_id) const noexcept {
    if (!image.impl_ || !image.stable() || !active() || !stable() ||
        image.impl_->instance != instance_id_ ||
        image.impl_->streams.size() != streams_.size() ||
        image.impl_->units.size() != units_.size() ||
        !vbr_controller_instance_owned_by(instance_id_, this)) {
        return false;
    }
    vbr_operation_binding binding;
    return operation_id &&
           vbr_operation_registry_binding(operation_id, binding) &&
           binding.find_covering_target(
               instance_id_, VBR_STREAM_ANY,
               vbr_operation_class::state_api,
               vbr_registrant_bit(vbr_mutation_registrant::whole_import)) != nullptr;
}

// BEGIN VBR_IMPORT_TRACKER_METADATA_SWAP
void vbr_generation_tracker::install_import_image_swap(
        vbr_tracker_import_image & image) noexcept {
    GGML_ASSERT(image.impl_ && image.stable());
    GGML_ASSERT(image.impl_->instance == instance_id_);
    lineage_uuid_ = image.impl_->lineage;
    global_generation_ = image.impl_->global_generation;
    mutation_serial_ = image.impl_->mutation_serial;
    event_serial_ = 0;
    active_event_depth_ = 0;
    generation_at_latch_ = 0;
    shadow_unavailable_ = false;
    streams_.swap(image.impl_->streams);
    units_.swap(image.impl_->units);
    // Extent references are now owned by the live stream vectors.  Detach the
    // off-side image's rollback owner without touching the imported handles.
    image.impl_->extent_store = nullptr;
    image.impl_->ready = false;
}
// END VBR_IMPORT_TRACKER_METADATA_SWAP
// VBR_GENERATION_IMPORT_REGION_END

bool vbr_generation_tracker::reset_unit_generations_before_wrap() {
    if (global_generation_ == std::numeric_limits<uint64_t>::max() ||
        mutation_serial_ > std::numeric_limits<uint64_t>::max() - 2) {
        return false;
    }
    for (const auto & unit : units_) {
        if ((unit.publish_seq & 1u) != 0) {
            return false;
        }
    }
    ++mutation_serial_;
    ++global_generation_;
    for (auto & unit : units_) {
        unit.repr_gen    = 1;
        unit.publish_seq = 0;
        unit.flags       = 0;
    }
    ++mutation_serial_;
    return true;
}

uint32_t vbr_generation_tracker::page_generation(uint32_t stream, uint32_t page) const {
    return streams_.at(stream).page_event_gen.at(page);
}

uint32_t vbr_generation_tracker::page_destructive_generation(uint32_t stream, uint32_t page) const {
    return streams_.at(stream).page_last_destructive_gen.at(page);
}

uint32_t vbr_generation_tracker::page_import_generation(uint32_t stream, uint32_t page) const {
    return streams_.at(stream).page_last_import_gen.at(page);
}

uint32_t vbr_generation_tracker::dependency_generation(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_last_dependency_gen.at(cell);
}

uint32_t vbr_generation_tracker::membership_generation(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_last_membership_gen.at(cell);
}

uint16_t vbr_generation_tracker::dependency_provenance(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_dependency_provenance.at(cell);
}

uint16_t vbr_generation_tracker::membership_provenance(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_membership_provenance.at(cell);
}

llama_seq_id vbr_generation_tracker::last_membership_seq(uint32_t stream, uint32_t cell) const {
    return streams_.at(stream).cell_last_membership_seq.at(cell);
}

vbr_unit_generation vbr_generation_tracker::unit_generation(uint32_t unit) const {
    std::lock_guard<std::mutex> lock(units_mutex_);
    return units_.at(unit);
}

// VBR_GENERATION_CAPTURE_REGION_BEGIN
bool vbr_generation_capture_stream(const vbr_generation_tracker &     tracker,
                                   uint32_t                           stream,
                                   llama_seq_id                       dependency_seq_id,
                                   llama_pos                          computation_frontier,
                                   const std::vector<uint32_t> &      canonical_dependency_cells,
                                   vbr_checkpoint_generation_stream & output) {
    if (!tracker.active() || !tracker.stable() || stream >= tracker.stream_count() || dependency_seq_id < 0 ||
        computation_frontier < 0 ||
        canonical_dependency_cells.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }

    output                           = {};
    output.stream_index              = stream;
    output.dependency_seq_id         = dependency_seq_id;
    output.computation_frontier      = computation_frontier;
    output.captured_dependency_count = static_cast<uint32_t>(canonical_dependency_cells.size());

    uint32_t previous = std::numeric_limits<uint32_t>::max();
    for (uint32_t cell : canonical_dependency_cells) {
        if (cell >= tracker.cell_count() || (previous != std::numeric_limits<uint32_t>::max() && cell <= previous)) {
            output = {};
            return false;
        }
        previous = cell;

        const uint32_t page = cell / VBR_GENERATION_PAGE_CELLS;
        if (output.pages.empty() || output.pages.back().page_index != page) {
            vbr_generation_page_ref ref;
            ref.page_index        = page;
            ref.captured_page_gen = tracker.page_generation(stream, page);
            output.pages.push_back(ref);
        }
        mask_set(output.pages.back().covered_mask, cell % VBR_GENERATION_PAGE_CELLS);
    }
    return true;
}

bool vbr_generation_capture_controller(const vbr_generation_tracker &                        tracker,
                                       uint32_t                                              child_id,
                                       checkpoint_child_dependency_mode                      dependency_mode,
                                       const std::vector<vbr_checkpoint_generation_stream> & streams,
                                       vbr_checkpoint_generation_controller &                output) {
    if (!tracker.active() || tracker.shadow_unavailable() || !tracker.stable() ||
        dependency_mode != checkpoint_child_dependency_mode::live_guarded) {
        return false;
    }

    output                   = {};
    output.child_id          = child_id;
    output.dependency_mode   = dependency_mode;
    output.lineage_uuid      = tracker.lineage_identity();
    output.global_generation = tracker.controller_generation();
    output.streams           = streams;
    uint32_t previous_stream  = std::numeric_limits<uint32_t>::max();
    for (const auto & stream : output.streams) {
        if (stream.stream_index >= tracker.stream_count() ||
            (previous_stream != std::numeric_limits<uint32_t>::max() &&
             stream.stream_index <= previous_stream)) {
            output = {};
            return false;
        }
        previous_stream = stream.stream_index;
        for (const auto & page : stream.pages) {
            if (page.page_index >= page_count(tracker.cell_count()) ||
                tracker.page_generation(stream.stream_index, page.page_index) !=
                        page.captured_page_gen) {
                output = {};
                return false;
            }
        }
    }
    output.units.reserve(tracker.unit_count());
    for (uint32_t unit = 0; unit < tracker.unit_count(); ++unit) {
        const auto live_unit = tracker.unit_generation(unit);
        if ((live_unit.publish_seq & 1u) != 0) {
            output = {};
            return false;
        }
        output.units.push_back({
            live_unit.repr_gen,
            live_unit.current_type,
            live_unit.last_source_type,
            live_unit.domain,
            live_unit.promote_hops,
            live_unit.last_transition,
        });
    }

    if (!tracker.stable() || tracker.lineage_identity() != output.lineage_uuid ||
        tracker.controller_generation() != output.global_generation) {
        output = {};
        return false;
    }
    for (uint32_t unit = 0; unit < tracker.unit_count(); ++unit) {
        if (!unit_equal(output.units[unit], tracker.unit_generation(unit))) {
            output = {};
            return false;
        }
    }
    for (const auto & stream : output.streams) {
        for (const auto & page : stream.pages) {
            if (tracker.page_generation(stream.stream_index, page.page_index) !=
                    page.captured_page_gen) {
                output = {};
                return false;
            }
        }
    }
    return true;
}
// VBR_GENERATION_CAPTURE_REGION_END
