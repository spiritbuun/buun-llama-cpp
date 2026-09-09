#include "llama-vbr-artifact-stage.h"

#include <algorithm>
#include <future>
#include <limits>
#include <new>
#include <thread>
#include <utility>

const char * vbr_h2d_status_name(vbr_h2d_status status) noexcept {
    switch (status) {
        case vbr_h2d_status::ok: return "ok";
        case vbr_h2d_status::invalid_argument: return "invalid_argument";
        case vbr_h2d_status::ring_unavailable: return "ring_unavailable";
        case vbr_h2d_status::source_read_failed: return "source_read_failed";
        case vbr_h2d_status::transfer_failed: return "transfer_failed";
        case vbr_h2d_status::event_failed: return "event_failed";
        case vbr_h2d_status::internal_error: return "internal_error";
        case vbr_h2d_status::_count: break;
    }
    return "invalid";
}

const char * vbr_adopt_stage_status_name(
        vbr_adopt_stage_status status) noexcept {
    switch (status) {
        case vbr_adopt_stage_status::staged: return "staged";
        case vbr_adopt_stage_status::invalid_proof: return "invalid_proof";
        case vbr_adopt_stage_status::unsupported_decision: return "unsupported_decision";
        case vbr_adopt_stage_status::source_unavailable: return "source_unavailable";
        case vbr_adopt_stage_status::source_hash_mismatch: return "source_hash_mismatch";
        case vbr_adopt_stage_status::accounting_unavailable: return "accounting_unavailable";
        case vbr_adopt_stage_status::admission_refused: return "admission_refused";
        case vbr_adopt_stage_status::ring_unavailable: return "ring_unavailable";
        case vbr_adopt_stage_status::transform_projection_unavailable: return "transform_projection_unavailable";
        case vbr_adopt_stage_status::transform_reserve_failed: return "transform_reserve_failed";
        case vbr_adopt_stage_status::internal_error: return "internal_error";
        case vbr_adopt_stage_status::_count: break;
    }
    return "invalid";
}

struct vbr_h2d_chunk_ring::impl {
    std::shared_ptr<vbr_bounded_pinned_ring_core> core;
    std::vector<vbr_h2d_lane_binding> lanes;
};

vbr_h2d_ring_operation::vbr_h2d_ring_operation(
        vbr_h2d_ring_operation && other) noexcept
    : keepalive_(std::move(other.keepalive_)),
      operation_(std::move(other.operation_)) {}

vbr_h2d_ring_operation & vbr_h2d_ring_operation::operator=(
        vbr_h2d_ring_operation && other) noexcept {
    if (this != &other) {
        // Release the old mutex while its core is still alive.
        operation_ = {};
        keepalive_.reset();
        keepalive_ = std::move(other.keepalive_);
        operation_ = std::move(other.operation_);
    }
    return *this;
}

vbr_h2d_ring_operation::~vbr_h2d_ring_operation() = default;

vbr_h2d_chunk_ring::vbr_h2d_chunk_ring(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}

vbr_h2d_chunk_ring::~vbr_h2d_chunk_ring() = default;

std::unique_ptr<vbr_h2d_chunk_ring> vbr_h2d_chunk_ring::create(
        const std::vector<vbr_h2d_lane_binding> & lanes,
        uint64_t total_bytes,
        size_t chunk_bytes,
        vbr_h2d_status & status,
        vbr_pinned_ring_create_failure * failure) noexcept {
    status = vbr_h2d_status::ring_unavailable;
    vbr_pinned_ring_create_failure reason =
        vbr_pinned_ring_create_failure::none;
    try {
        std::vector<vbr_pinned_ring_lane> bindings;
        bindings.reserve(lanes.size());
        for (const auto & lane : lanes) {
            bindings.push_back({
                lane.device, lane.backend, lane.force_synchronous,
            });
        }
        std::unique_ptr<impl> state(new impl);
        state->core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
            vbr_bounded_pinned_ring_core::create(
                bindings, total_bytes, chunk_bytes, nullptr, reason));
        if (!state->core) {
            status = reason == vbr_pinned_ring_create_failure::internal_error
                ? vbr_h2d_status::internal_error
                : vbr_h2d_status::ring_unavailable;
            if (failure) {
                *failure = reason;
            }
            return nullptr;
        }
        state->lanes = lanes;
        if (failure) {
            *failure = reason;
        }
        status = vbr_h2d_status::ok;
        return std::unique_ptr<vbr_h2d_chunk_ring>(
            new vbr_h2d_chunk_ring(std::move(state)));
    } catch (...) {
        status = vbr_h2d_status::internal_error;
        if (failure) {
            *failure = vbr_pinned_ring_create_failure::internal_error;
        }
        return nullptr;
    }
}

std::shared_ptr<vbr_h2d_chunk_ring> vbr_h2d_chunk_ring::attach(
        std::shared_ptr<vbr_bounded_pinned_ring_core> core,
        const std::vector<vbr_h2d_lane_binding> & lanes) noexcept {
    try {
        if (!core || core->lane_count() != lanes.size()) {
            return nullptr;
        }
        for (size_t i = 0; i < lanes.size(); ++i) {
            const auto * binding = core->lane_binding(uint32_t(i));
            if (!binding || binding->device != lanes[i].device ||
                binding->backend != lanes[i].backend ||
                binding->force_synchronous != lanes[i].force_synchronous) {
                return nullptr;
            }
        }
        std::unique_ptr<impl> state(new impl);
        state->core = std::move(core);
        state->lanes = lanes;
        return std::shared_ptr<vbr_h2d_chunk_ring>(
            new vbr_h2d_chunk_ring(std::move(state)));
    } catch (...) {
        return nullptr;
    }
}

uint64_t vbr_h2d_chunk_ring::capacity_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->capacity_bytes() : 0;
}

size_t vbr_h2d_chunk_ring::chunk_bytes() const noexcept {
    return impl_ && impl_->core ? impl_->core->chunk_bytes() : 0;
}

size_t vbr_h2d_chunk_ring::lane_count() const noexcept {
    return impl_ && impl_->core ? impl_->core->lane_count() : 0;
}

bool vbr_h2d_chunk_ring::compatible_with(
        const llama_cache_acct_ledger * ledger,
        const llama_cache_acct_snapshot & snapshot,
        const llama_cache_acct_resource_domain & domain,
        uint64_t capacity_bytes,
        size_t chunk_bytes,
        const std::vector<vbr_h2d_lane_binding> & lanes) const noexcept {
    if (!impl_ || !impl_->core || impl_->lanes.size() != lanes.size() ||
        impl_->core->capacity_bytes() != capacity_bytes ||
        impl_->core->chunk_bytes() != chunk_bytes ||
        !impl_->core->accounted_to(
            ledger, snapshot, domain,
            llama_cache_acct_category::pinned_preimage_ring)) {
        return false;
    }
    for (size_t i = 0; i < lanes.size(); ++i) {
        if (impl_->lanes[i].domain != lanes[i].domain ||
            impl_->lanes[i].device != lanes[i].device ||
            impl_->lanes[i].backend != lanes[i].backend ||
            impl_->lanes[i].force_synchronous !=
                lanes[i].force_synchronous) {
            return false;
        }
    }
    return true;
}

vbr_h2d_status vbr_h2d_chunk_ring::stream(
        const vbr_h2d_transfer & transfer,
        vbr_h2d_stats & stats) noexcept {
    stats = {};
    if (!impl_ || !impl_->core || transfer.size == 0 ||
        transfer.lane >= impl_->core->lane_count() ||
        !transfer.source.valid() ||
        transfer.source_offset > transfer.source.size ||
        transfer.size > transfer.source.size - transfer.source_offset ||
        transfer.destination_offset >
            std::numeric_limits<uint64_t>::max() - transfer.size) {
        return vbr_h2d_status::invalid_argument;
    }
    const auto * lane = impl_->core->lane_binding(transfer.lane);
    const bool tensor_destination = transfer.destination != nullptr;
    const bool fake_destination = transfer.fake.issue != nullptr;
    if (tensor_destination == fake_destination) {
        return vbr_h2d_status::invalid_argument;
    }
    if (tensor_destination) {
        if (!lane || !transfer.backend || !transfer.device ||
            transfer.device != lane->device ||
            ggml_backend_get_device(transfer.backend) != lane->device ||
            transfer.destination_offset > ggml_nbytes(transfer.destination) ||
            transfer.size > ggml_nbytes(transfer.destination) -
                transfer.destination_offset) {
            return vbr_h2d_status::invalid_argument;
        }
    } else if (!transfer.fake.complete) {
        return vbr_h2d_status::invalid_argument;
    }
    struct h2d_pump_context {
        const vbr_h2d_transfer * transfer = nullptr;
        uint64_t offset = 0;
        uint64_t next_ticket = 1;
        bool tensor_destination = false;
    } pump_context { &transfer, 0, 1, tensor_destination };
    vbr_bounded_pinned_ring_core::pump_callbacks callbacks;
    callbacks.context = &pump_context;
    callbacks.ok = uint32_t(vbr_h2d_status::ok);
    callbacks.ring_unavailable = uint32_t(vbr_h2d_status::ring_unavailable);
    callbacks.submit_failed = uint32_t(vbr_h2d_status::internal_error);
    callbacks.wait_failed = uint32_t(vbr_h2d_status::event_failed);
    callbacks.internal_error = uint32_t(vbr_h2d_status::internal_error);
    callbacks.more = [](void * opaque) noexcept {
        const auto & context = *static_cast<h2d_pump_context *>(opaque);
        return context.offset < context.transfer->size;
    };
    callbacks.fill = [](void * opaque, uint8_t * output,
                        size_t capacity,
                        vbr_bounded_pinned_ring_core::pump_step & step)
            noexcept -> uint32_t {
        auto & context = *static_cast<h2d_pump_context *>(opaque);
        const auto & transfer = *context.transfer;
        try {
            const size_t count = size_t(std::min<uint64_t>(
                capacity, transfer.size - context.offset));
            if (!transfer.source.read(
                    transfer.source.context,
                    transfer.source_offset + context.offset,
                    output, count)) {
                return uint32_t(vbr_h2d_status::source_read_failed);
            }
            const uint64_t destination_offset =
                transfer.destination_offset + context.offset;
            const uint64_t ticket = context.next_ticket++;
            bool fake_async = false;
            if (context.tensor_destination) {
                ggml_backend_tensor_set_async(
                    transfer.backend, transfer.destination,
                    output, size_t(destination_offset), count);
            } else {
                fake_async = transfer.fake.supports_events;
                if (!transfer.fake.issue(
                        transfer.fake.context, ticket,
                        destination_offset, output, count,
                        fake_async)) {
                    return uint32_t(vbr_h2d_status::transfer_failed);
                }
                if (!fake_async) {
                    if (!transfer.fake.complete(
                            transfer.fake.context, ticket)) {
                        return uint32_t(vbr_h2d_status::event_failed);
                    }
                    step.adapter_synchronous_fallback = true;
                }
            }
            step.valid = count;
            step.backend = context.tensor_destination
                ? transfer.backend : nullptr;
            step.tag = ticket;
            step.adapter_async = fake_async;
            context.offset += count;
            return uint32_t(vbr_h2d_status::ok);
        } catch (...) {
            return uint32_t(vbr_h2d_status::internal_error);
        }
    };
    callbacks.consume = [](void * opaque, const uint8_t *, size_t,
                           uint64_t ticket, bool fake_async,
                           uint64_t ordinal, bool & event_completion)
            noexcept -> uint32_t {
        const auto & context = *static_cast<h2d_pump_context *>(opaque);
        const auto & transfer = *context.transfer;
        if (fake_async) {
            if (!transfer.fake.complete(
                    transfer.fake.context, ticket)) {
                return uint32_t(vbr_h2d_status::event_failed);
            }
            event_completion = true;
        }
        return ordinal == transfer.fail_completion_at
            ? uint32_t(vbr_h2d_status::transfer_failed)
            : uint32_t(vbr_h2d_status::ok);
    };
    callbacks.abandon = [](void * opaque, uint64_t ticket,
                           bool fake_async) noexcept {
        const auto & context = *static_cast<h2d_pump_context *>(opaque);
        if (fake_async) {
            (void) context.transfer->fake.complete(
                context.transfer->fake.context, ticket);
        }
    };

    vbr_bounded_pinned_ring_core::pump_stats pump_stats;
    const auto pumped = vbr_h2d_status(
        impl_->core->pump(transfer.lane, callbacks, pump_stats));
    stats.bytes = pump_stats.bytes;
    stats.chunks = pump_stats.chunks;
    stats.backpressure_waits = pump_stats.backpressure_waits;
    stats.event_completions = pump_stats.event_completions;
    stats.synchronous_fallbacks = pump_stats.synchronous_fallbacks;
    stats.peak_pinned_bytes = pump_stats.peak_pinned_bytes;
    if (pumped != vbr_h2d_status::ok) {
        return pumped;
    }
    return stats.bytes == transfer.size
        ? vbr_h2d_status::ok
        : vbr_h2d_status::transfer_failed;
}

vbr_h2d_ring_operation
vbr_h2d_chunk_ring::try_begin_operation() noexcept {
    vbr_h2d_ring_operation result;
    if (impl_ && impl_->core) {
        result.keepalive_ = impl_->core;
        result.operation_ = impl_->core->try_begin_operation();
        if (!result.operation_) {
            result.keepalive_.reset();
        }
    }
    return result;
}

vbr_h2d_status vbr_h2d_chunk_ring::stream_packed_reserved(
        const vbr_h2d_ring_operation & operation,
        const vbr_h2d_packed_transfer & transfer,
        vbr_h2d_stats & stats) noexcept {
    static constexpr size_t max_ranges = 1024*1024;
    stats = {};
    if (!impl_ || !impl_->core || !operation || transfer.size == 0 ||
        transfer.lane >= impl_->core->lane_count() ||
        !transfer.source.valid() || !transfer.ranges ||
        transfer.range_count == 0 || transfer.range_count > max_ranges ||
        transfer.destination_offset >
            std::numeric_limits<uint64_t>::max() - transfer.size) {
        return vbr_h2d_status::invalid_argument;
    }
    uint64_t selected = 0;
    for (size_t i = 0; i < transfer.range_count; ++i) {
        const auto & range = transfer.ranges[i];
        if (range.size == 0 || range.source_offset > transfer.source.size ||
            range.size > transfer.source.size - range.source_offset ||
            range.size > std::numeric_limits<uint64_t>::max() - selected) {
            return vbr_h2d_status::invalid_argument;
        }
        selected += range.size;
    }
    if (selected != transfer.size) {
        return vbr_h2d_status::invalid_argument;
    }
    const auto * lane = impl_->core->lane_binding(transfer.lane);
    const bool tensor_destination = transfer.destination != nullptr;
    const bool fake_destination = transfer.fake.issue != nullptr;
    if (tensor_destination == fake_destination) {
        return vbr_h2d_status::invalid_argument;
    }
    if (tensor_destination) {
        if (!lane || !transfer.backend || !transfer.device ||
            transfer.device != lane->device ||
            ggml_backend_get_device(transfer.backend) != lane->device ||
            transfer.destination_offset > ggml_nbytes(transfer.destination) ||
            transfer.size > ggml_nbytes(transfer.destination) -
                transfer.destination_offset) {
            return vbr_h2d_status::invalid_argument;
        }
    } else if (!transfer.fake.complete) {
        return vbr_h2d_status::invalid_argument;
    }

    struct packed_pump_context {
        const vbr_h2d_packed_transfer * transfer = nullptr;
        size_t range_index = 0;
        uint64_t range_offset = 0;
        uint64_t transferred = 0;
        uint64_t next_ticket = 1;
        bool tensor_destination = false;
    } pump_context { &transfer, 0, 0, 0, 1, tensor_destination };
    vbr_bounded_pinned_ring_core::pump_callbacks callbacks;
    callbacks.context = &pump_context;
    callbacks.ok = uint32_t(vbr_h2d_status::ok);
    callbacks.ring_unavailable = uint32_t(vbr_h2d_status::ring_unavailable);
    callbacks.submit_failed = uint32_t(vbr_h2d_status::internal_error);
    callbacks.wait_failed = uint32_t(vbr_h2d_status::event_failed);
    callbacks.internal_error = uint32_t(vbr_h2d_status::internal_error);
    callbacks.more = [](void * opaque) noexcept {
        const auto & context = *static_cast<packed_pump_context *>(opaque);
        return context.transferred < context.transfer->size;
    };
    callbacks.fill = [](void * opaque, uint8_t * output,
                        size_t capacity,
                        vbr_bounded_pinned_ring_core::pump_step & step)
            noexcept -> uint32_t {
        auto & context = *static_cast<packed_pump_context *>(opaque);
        const auto & transfer = *context.transfer;
        try {
            size_t filled = 0;
            while (filled < capacity &&
                   context.range_index < transfer.range_count) {
                const auto & range =
                    transfer.ranges[context.range_index];
                const size_t count = size_t(std::min<uint64_t>(
                    capacity - filled, range.size - context.range_offset));
                if (!transfer.source.read(
                        transfer.source.context,
                        range.source_offset + context.range_offset,
                        output + filled, count)) {
                    return uint32_t(vbr_h2d_status::source_read_failed);
                }
                filled += count;
                context.range_offset += count;
                if (context.range_offset == range.size) {
                    ++context.range_index;
                    context.range_offset = 0;
                }
            }
            if (filled == 0 ||
                filled > transfer.size - context.transferred) {
                return uint32_t(vbr_h2d_status::internal_error);
            }
            const uint64_t destination_offset =
                transfer.destination_offset + context.transferred;
            const uint64_t ticket = context.next_ticket++;
            bool fake_async = false;
            if (context.tensor_destination) {
                ggml_backend_tensor_set_async(
                    transfer.backend, transfer.destination,
                    output, size_t(destination_offset), filled);
            } else {
                fake_async = transfer.fake.supports_events;
                if (!transfer.fake.issue(
                        transfer.fake.context, ticket,
                        destination_offset, output, filled, fake_async)) {
                    return uint32_t(vbr_h2d_status::transfer_failed);
                }
                if (!fake_async) {
                    if (!transfer.fake.complete(
                            transfer.fake.context, ticket)) {
                        return uint32_t(vbr_h2d_status::event_failed);
                    }
                    step.adapter_synchronous_fallback = true;
                }
            }
            step.valid = filled;
            step.backend = context.tensor_destination
                ? transfer.backend : nullptr;
            step.tag = ticket;
            step.adapter_async = fake_async;
            context.transferred += filled;
            return uint32_t(vbr_h2d_status::ok);
        } catch (...) {
            return uint32_t(vbr_h2d_status::internal_error);
        }
    };
    callbacks.consume = [](void * opaque, const uint8_t *, size_t,
                           uint64_t ticket, bool fake_async,
                           uint64_t ordinal, bool & event_completion)
            noexcept -> uint32_t {
        const auto & context = *static_cast<packed_pump_context *>(opaque);
        const auto & transfer = *context.transfer;
        if (fake_async) {
            if (!transfer.fake.complete(transfer.fake.context, ticket)) {
                return uint32_t(vbr_h2d_status::event_failed);
            }
            event_completion = true;
        }
        return ordinal == transfer.fail_completion_at
            ? uint32_t(vbr_h2d_status::transfer_failed)
            : uint32_t(vbr_h2d_status::ok);
    };
    callbacks.abandon = [](void * opaque, uint64_t ticket,
                           bool fake_async) noexcept {
        const auto & context = *static_cast<packed_pump_context *>(opaque);
        if (fake_async) {
            (void) context.transfer->fake.complete(
                context.transfer->fake.context, ticket);
        }
    };

    vbr_bounded_pinned_ring_core::pump_stats pump_stats;
    const auto pumped = vbr_h2d_status(impl_->core->pump_reserved(
        operation.operation_, transfer.lane, callbacks, pump_stats));
    stats.bytes = pump_stats.bytes;
    stats.chunks = pump_stats.chunks;
    stats.backpressure_waits = pump_stats.backpressure_waits;
    stats.event_completions = pump_stats.event_completions;
    stats.synchronous_fallbacks = pump_stats.synchronous_fallbacks;
    stats.peak_pinned_bytes = pump_stats.peak_pinned_bytes;
    if (pumped != vbr_h2d_status::ok) {
        return pumped;
    }
    return stats.bytes == transfer.size &&
            pump_context.range_index == transfer.range_count
        ? vbr_h2d_status::ok
        : vbr_h2d_status::transfer_failed;
}

struct vbr_staged_payloads::impl {
    uint64_t nonce = 0;
    uint64_t post_prepare_serial = 0;
    vbr_manifest_digest manifest;
    vbr_target_empty_fingerprint target;
    vbr_import_decision import_decision = vbr_import_decision::reject;
    std::vector<vbr_staged_read_descriptor> read_plan;
    std::vector<llama_cache_transaction_leaf> leaves;
    std::vector<llama_cache_acct_op_id> committed_ops;
    std::vector<llama_cache_acct_alloc_id> allocations;
    llama_cache_prepared_claim_group prepared;
    std::shared_ptr<vbr_h2d_chunk_ring> ring;
    std::vector<uint64_t> transform_stashless;
    bool transform_resources = false;
};

vbr_staged_payloads::vbr_staged_payloads(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {}
vbr_staged_payloads::~vbr_staged_payloads() = default;
vbr_staged_payloads::vbr_staged_payloads(
        vbr_staged_payloads &&) noexcept = default;
vbr_staged_payloads & vbr_staged_payloads::operator=(
        vbr_staged_payloads &&) noexcept = default;

uint64_t vbr_staged_payloads::adoption_nonce() const noexcept {
    return impl_ ? impl_->nonce : 0;
}
uint64_t vbr_staged_payloads::validation_accounting_serial() const noexcept {
    return impl_ ? impl_->target.accounting_serial : 0;
}
uint64_t vbr_staged_payloads::accounting_serial_after_prepare() const noexcept {
    return impl_ ? impl_->post_prepare_serial : 0;
}
const vbr_manifest_digest & vbr_staged_payloads::manifest_digest() const noexcept {
    static const vbr_manifest_digest empty;
    return impl_ ? impl_->manifest : empty;
}
const vbr_target_empty_fingerprint &
vbr_staged_payloads::target_fingerprint() const noexcept {
    static const vbr_target_empty_fingerprint empty;
    return impl_ ? impl_->target : empty;
}
vbr_import_decision vbr_staged_payloads::decision() const noexcept {
    return impl_ ? impl_->import_decision : vbr_import_decision::reject;
}
size_t vbr_staged_payloads::read_count() const noexcept {
    return impl_ ? impl_->read_plan.size() : 0;
}
const std::vector<vbr_staged_read_descriptor> &
vbr_staged_payloads::reads() const noexcept {
    static const std::vector<vbr_staged_read_descriptor> empty;
    return impl_ ? impl_->read_plan : empty;
}
uint64_t vbr_staged_payloads::ring_capacity_bytes() const noexcept {
    return impl_ && impl_->ring ? impl_->ring->capacity_bytes() : 0;
}
bool vbr_staged_payloads::claims_ready() const noexcept {
    return impl_ && impl_->prepared.ready();
}
const std::vector<uint64_t> &
vbr_staged_payloads::transform_stashless_units() const noexcept {
    static const std::vector<uint64_t> empty;
    return impl_ ? impl_->transform_stashless : empty;
}
bool vbr_staged_payloads::transform_resources_ready() const noexcept {
    return impl_ && impl_->transform_resources;
}

llama_cache_transaction_result
vbr_staged_payloads::adoption_materialize_claims() noexcept {
    if (!impl_) {
        return {};
    }
    return impl_->prepared.materialize_and_commit(impl_->leaves);
}

vbr_h2d_chunk_ring * vbr_staged_payloads::adoption_ring() noexcept {
    return impl_ ? impl_->ring.get() : nullptr;
}

const std::vector<llama_cache_acct_op_id> &
vbr_staged_payloads::adoption_committed_ops() const noexcept {
    static const std::vector<llama_cache_acct_op_id> empty;
    return impl_ ? impl_->committed_ops : empty;
}

namespace {

bool add_checked(uint64_t a, uint64_t b, uint64_t & out) noexcept {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

uint32_t find_lane(
        const std::vector<vbr_h2d_lane_binding> & lanes,
        const llama_cache_acct_resource_domain & domain) noexcept {
    uint32_t result = UINT32_MAX;
    for (uint32_t i = 0; i < lanes.size(); ++i) {
        if (lanes[i].domain != domain) {
            continue;
        }
        if (result != UINT32_MAX) {
            return UINT32_MAX - 1;
        }
        result = i;
    }
    return result;
}

bool digest_nonzero(const std::array<uint8_t, 32> & digest) noexcept {
    return std::any_of(digest.begin(), digest.end(), [](uint8_t value) {
        return value != 0;
    });
}

template<class AppendRead>
bool stage_child(
        const vbr_validated_child_plan & child,
        const vbr_artifact_package_view & source_package,
        const std::vector<vbr_h2d_lane_binding> & lanes,
        AppendRead && append_read,
        vbr_adopt_stage_status & failure) {
    for (const auto & shard : child.shards) {
        const uint32_t lane = find_lane(lanes, shard.domain);
        if (lane >= lanes.size()) {
            failure = vbr_adopt_stage_status::source_unavailable;
            return false;
        }
        for (const auto & run : child.authorized_runs) {
            if (run.cell_count == 0 ||
                uint64_t(run.first_physical_cell) >
                    std::numeric_limits<uint64_t>::max()/shard.row_bytes ||
                uint64_t(run.cell_count) >
                    std::numeric_limits<uint64_t>::max()/shard.row_bytes) {
                failure = vbr_adopt_stage_status::source_unavailable;
                return false;
            }
            const uint64_t offset =
                uint64_t(run.first_physical_cell)*shard.row_bytes;
            const uint64_t size = uint64_t(run.cell_count)*shard.row_bytes;
            if (offset > shard.payload_bytes ||
                size > shard.payload_bytes - offset ||
                !append_read({
                    vbr_staged_read_kind::unit_payload,
                    child.child_id, child.logical_unit_id,
                    shard.shard_index, lane, offset, size,
                    shard.source, {}, 0, {},
                })) {
                failure = vbr_adopt_stage_status::source_hash_mismatch;
                return false;
            }
        }
    }

    if (child.stash_action !=
            vbr_validated_stash_action::restore_exact &&
        child.stash_action !=
            vbr_validated_stash_action::consume_exact_then_drop) {
        return true;
    }
    const auto unit = std::find_if(
        source_package.units().begin(), source_package.units().end(),
        [&](const vbr_artifact_unit_view & value) {
            return value.descriptor.child_id == child.child_id &&
                   value.descriptor.logical_unit_id == child.logical_unit_id;
        });
    if (unit == source_package.units().end() ||
        unit->stash_shards.size() != child.shards.size()) {
        failure = vbr_adopt_stage_status::source_unavailable;
        return false;
    }
    for (size_t i = 0; i < unit->stash_shards.size(); ++i) {
        const uint32_t lane = find_lane(lanes, child.shards[i].domain);
        if (lane >= lanes.size() ||
            !append_read({
                vbr_staged_read_kind::clean_stash,
                child.child_id, child.logical_unit_id,
                uint32_t(i), lane, 0,
                unit->stash_shards[i]
                    ? unit->stash_shards[i]->size() : 0,
                unit->stash_shards[i], {}, 0, {},
            })) {
            failure = vbr_adopt_stage_status::source_hash_mismatch;
            return false;
        }
    }
    return true;
}

template<class AppendRead>
bool stage_projection_child(
        const vbr_validated_child_plan & child,
        const std::vector<vbr_h2d_lane_binding> & lanes,
        AppendRead && append_read,
        vbr_adopt_stage_status & failure) {
    for (const auto & shard : child.shards) {
        const uint32_t lane = find_lane(lanes, shard.domain);
        if (lane >= lanes.size() || !shard.source ||
            !shard.projection_proof || shard.projection_runs.empty() ||
            shard.projection_runs.front().destination_offset != 0) {
            failure = vbr_adopt_stage_status::source_unavailable;
            return false;
        }
        // The restricted proof is the payload authority for a projection.
        // Authenticate its selected leaves against the still-borrowed parent
        // before any accounting claim or H2D can consume the row mapping.
        // Verification deliberately reads only proof-selected chunks; it
        // never hashes the omitted parent suffix.
        uint64_t proof_verified_bytes = 0;
        if (!vbr_capture_range_verify(
                shard.projection_proof, shard.source->source(),
                &proof_verified_bytes) || proof_verified_bytes == 0) {
            failure = vbr_adopt_stage_status::source_hash_mismatch;
            return false;
        }
        vbr_staged_read_descriptor read;
        read.kind = vbr_staged_read_kind::unit_payload;
        read.child_id = child.child_id;
        read.logical_unit_id = child.logical_unit_id;
        read.shard_index = shard.shard_index;
        read.lane = lane;
        read.destination_offset =
            shard.projection_runs.front().destination_offset;
        read.source = shard.source;
        read.verified_digest = shard.projection_proof.root();
        read.proof_verified_bytes = proof_verified_bytes;
        read.projection_ranges.reserve(shard.projection_runs.size());
        uint64_t next_destination = read.destination_offset;
        for (const auto & run : shard.projection_runs) {
            if (run.size == 0 || run.destination_offset != next_destination ||
                run.source_offset > shard.source->size() ||
                run.size > shard.source->size() - run.source_offset ||
                run.size > std::numeric_limits<uint64_t>::max() - read.size ||
                run.size > std::numeric_limits<uint64_t>::max() -
                    next_destination) {
                failure = vbr_adopt_stage_status::invalid_proof;
                return false;
            }
            read.projection_ranges.push_back({
                run.source_offset, run.size,
            });
            read.size += run.size;
            next_destination += run.size;
        }
        if (read.size == 0 || !digest_nonzero(read.verified_digest) ||
            !append_read(std::move(read))) {
            failure = vbr_adopt_stage_status::source_hash_mismatch;
            return false;
        }
    }
    return true;
}

template<class AppendRead>
bool stage_projection_relocated_child(
        const vbr_validated_child_plan & child,
        const std::vector<vbr_occupied_replacement_relocation_run> & runs,
        const std::vector<vbr_h2d_lane_binding> & lanes,
        AppendRead && append_read,
        vbr_adopt_stage_status & failure) {
    if (runs.empty()) {
        failure = vbr_adopt_stage_status::invalid_proof;
        return false;
    }
    for (const auto & shard : child.shards) {
        const uint32_t lane = find_lane(lanes, shard.domain);
        if (lane >= lanes.size() || !shard.source ||
            !shard.projection_proof || shard.row_bytes == 0) {
            failure = vbr_adopt_stage_status::source_unavailable;
            return false;
        }
        uint64_t proof_verified_bytes = 0;
        if (!vbr_capture_range_verify(
                shard.projection_proof, shard.source->source(),
                &proof_verified_bytes) || proof_verified_bytes == 0) {
            failure = vbr_adopt_stage_status::source_hash_mismatch;
            return false;
        }
        for (const auto & run : runs) {
            if (run.cell_count == 0 ||
                run.first_source_packed_row > UINT64_MAX/shard.row_bytes ||
                uint64_t(run.first_destination_physical_cell) >
                    UINT64_MAX/shard.row_bytes ||
                uint64_t(run.cell_count) > UINT64_MAX/shard.row_bytes) {
                failure = vbr_adopt_stage_status::invalid_proof;
                return false;
            }
            const uint64_t source_offset =
                run.first_source_packed_row*shard.row_bytes;
            const uint64_t destination_offset =
                uint64_t(run.first_destination_physical_cell)*shard.row_bytes;
            const uint64_t size = uint64_t(run.cell_count)*shard.row_bytes;
            if (source_offset > shard.source->size() ||
                size > shard.source->size()-source_offset) {
                failure = vbr_adopt_stage_status::source_unavailable;
                return false;
            }
            vbr_staged_read_descriptor read;
            read.kind = vbr_staged_read_kind::unit_payload;
            read.child_id = child.child_id;
            read.logical_unit_id = child.logical_unit_id;
            read.shard_index = shard.shard_index;
            read.lane = lane;
            read.size = size;
            read.source = shard.source;
            read.verified_digest = shard.projection_proof.root();
            read.destination_offset = destination_offset;
            read.projection_ranges.push_back({ source_offset, size });
            read.proof_verified_bytes = proof_verified_bytes;
            if (!append_read(std::move(read))) {
                failure = vbr_adopt_stage_status::source_hash_mismatch;
                return false;
            }
        }
    }
    return true;
}

template<class AppendRead>
bool stage_relocated_child(
        const vbr_validated_child_plan & child,
        const std::vector<vbr_occupied_replacement_relocation_run> & runs,
        vbr_staged_read_kind kind,
        const vbr_artifact_unit_view * recovery_unit,
        const std::vector<vbr_h2d_lane_binding> & lanes,
        AppendRead && append_read,
        vbr_adopt_stage_status & failure) {
    if (kind != vbr_staged_read_kind::unit_payload &&
        kind != vbr_staged_read_kind::recovery_unit_payload) {
        failure = vbr_adopt_stage_status::invalid_proof;
        return false;
    }
    if (recovery_unit &&
        (kind != vbr_staged_read_kind::recovery_unit_payload ||
         recovery_unit->descriptor.child_id != child.child_id ||
         recovery_unit->descriptor.logical_unit_id != child.logical_unit_id ||
         recovery_unit->descriptor.shards.size() != child.shards.size() ||
         recovery_unit->payload_shards.size() != child.shards.size())) {
        failure = vbr_adopt_stage_status::invalid_proof;
        return false;
    }
    for (const auto & shard : child.shards) {
        const uint32_t lane = find_lane(lanes, shard.domain);
        if (shard.shard_index >= child.shards.size() ||
            (recovery_unit &&
             shard.shard_index >= recovery_unit->payload_shards.size())) {
            failure = vbr_adopt_stage_status::source_unavailable;
            return false;
        }
        const auto source = recovery_unit
            ? recovery_unit->payload_shards[shard.shard_index]
            : shard.source;
        const auto * recovery_shard = recovery_unit
            ? &recovery_unit->descriptor.shards[shard.shard_index]
            : nullptr;
        const uint64_t row_bytes = recovery_shard
            ? recovery_shard->row_bytes : shard.row_bytes;
        const uint64_t source_bytes = recovery_unit && source
            ? source->size() : shard.payload_bytes;
        if (lane >= lanes.size() || !source || row_bytes == 0 ||
            (recovery_shard &&
             (recovery_shard->shard_index != shard.shard_index ||
              recovery_shard->payload_bytes != source_bytes))) {
            failure = vbr_adopt_stage_status::source_unavailable;
            return false;
        }
        auto digest = source->sealed_digest();
        if (!digest_nonzero(digest)) {
            digest = vbr_capture_stream_digest(*source);
        }
        if (!digest_nonzero(digest)) {
            failure = vbr_adopt_stage_status::source_hash_mismatch;
            return false;
        }
        for (const auto & run : runs) {
            if (run.first_source_packed_row > UINT64_MAX/row_bytes ||
                uint64_t(run.first_destination_physical_cell) >
                    UINT64_MAX/row_bytes ||
                uint64_t(run.cell_count) > UINT64_MAX/row_bytes) {
                failure = vbr_adopt_stage_status::source_unavailable;
                return false;
            }
            const uint64_t source_offset =
                run.first_source_packed_row*row_bytes;
            const uint64_t destination_offset =
                uint64_t(run.first_destination_physical_cell)*row_bytes;
            const uint64_t size = uint64_t(run.cell_count)*row_bytes;
            if (run.cell_count == 0 || source_offset > source_bytes ||
                size > source_bytes-source_offset) {
                failure = vbr_adopt_stage_status::source_unavailable;
                return false;
            }
            vbr_staged_read_descriptor read;
            read.kind = kind;
            read.child_id = child.child_id;
            read.logical_unit_id = child.logical_unit_id;
            read.shard_index = shard.shard_index;
            read.lane = lane;
            read.size = size;
            read.source = source;
            read.verified_digest = digest;
            read.destination_offset = destination_offset;
            read.projection_ranges.push_back({ source_offset, size });
            if (recovery_unit) {
                read.destination_type = recovery_unit->descriptor.current_type;
            }
            if (!append_read(std::move(read))) {
                failure = vbr_adopt_stage_status::source_hash_mismatch;
                return false;
            }
        }
    }
    return true;
}

} // namespace

vbr_adopt_stage_result vbr_stage_validated_manifest(
        std::unique_ptr<vbr_validated_manifest> proof,
        const vbr_adopt_stage_policy & policy) noexcept {
    vbr_adopt_stage_result out;
    out.manifest = std::move(proof);
    try {
        if (!out.manifest || !policy.ledger || !policy.budget ||
            policy.lanes.empty() || policy.pinned_ring_bytes == 0 ||
            policy.chunk_bytes == 0 ||
            policy.pinned_domain.residency !=
                llama_cache_acct_residency::pinned_host ||
            out.manifest->adoption_nonce() == 0 ||
            !out.manifest->manifest_digest().valid()) {
            out.status = vbr_adopt_stage_status::invalid_proof;
            return out;
        }
        if (out.manifest->decision() != vbr_import_decision::native_import &&
            out.manifest->decision() != vbr_import_decision::live_rebased &&
            out.manifest->decision() != vbr_import_decision::downward_rebase &&
            out.manifest->decision() !=
                vbr_import_decision::upward_reconstruct) {
            out.status = vbr_adopt_stage_status::unsupported_decision;
            return out;
        }
        for (size_t i = 0; i < policy.lanes.size(); ++i) {
            if ((policy.lanes[i].device == nullptr) !=
                    (policy.lanes[i].backend == nullptr)) {
                out.status = vbr_adopt_stage_status::ring_unavailable;
                return out;
            }
            for (size_t j = 0; j < i; ++j) {
                if (policy.lanes[i].domain == policy.lanes[j].domain) {
                    out.status = vbr_adopt_stage_status::source_unavailable;
                    return out;
                }
            }
        }
        const auto preflight_accounting = policy.ledger->snapshot();
        if (policy.persistent_ring &&
            !policy.persistent_ring->compatible_with(
                policy.ledger, preflight_accounting,
                policy.pinned_domain,
                policy.pinned_ring_bytes, policy.chunk_bytes,
                policy.lanes)) {
            out.status = vbr_adopt_stage_status::ring_unavailable;
            return out;
        }
        if (preflight_accounting.serial !=
                out.manifest->target().accounting_serial) {
            out.status = vbr_adopt_stage_status::accounting_unavailable;
            return out;
        }
        const bool prefix_projection = out.manifest->is_prefix_projection();
        const bool occupied_replacement =
            out.manifest->is_occupied_replacement();
        if (prefix_projection) {
            if (!out.manifest->projection_transfer_ready() ||
                !out.manifest->companions().empty()) {
                out.status = vbr_adopt_stage_status::invalid_proof;
                return out;
            }
        } else if (!occupied_replacement &&
                   out.manifest->source_package().validate() !=
                       vbr_artifact_status::ok) {
            out.status = vbr_adopt_stage_status::source_hash_mismatch;
            return out;
        }

        std::unique_ptr<vbr_staged_payloads::impl> state(
            new vbr_staged_payloads::impl);
        state->nonce = out.manifest->adoption_nonce();
        state->manifest = out.manifest->manifest_digest();
        state->target = out.manifest->target();
        state->import_decision = out.manifest->decision();

        if (state->import_decision == vbr_import_decision::downward_rebase ||
            state->import_decision ==
                vbr_import_decision::upward_reconstruct) {
            if (policy.reserve_transform == nullptr) {
                out.status =
                    vbr_adopt_stage_status::transform_projection_unavailable;
                return out;
            }
            vbr_downward_stage_reservation reservation;
            const bool projected = policy.reserve_transform(
                    policy.transform_context, out.manifest->children(),
                    *policy.ledger, *policy.budget, reservation);
            out.transform_status = reservation.status;
            if (!projected) {
                out.status =
                    vbr_adopt_stage_status::transform_projection_unavailable;
                return out;
            }
            if (reservation.status != vbr_downward_reserve_status::reserved &&
                reservation.status !=
                    vbr_downward_reserve_status::reserved_stashless) {
                out.status = vbr_adopt_stage_status::transform_reserve_failed;
                return out;
            }
            state->transform_stashless =
                std::move(reservation.stashless_units);
            std::sort(
                state->transform_stashless.begin(),
                state->transform_stashless.end());
            state->transform_resources = true;
        }

        uint32_t source_index = 0;
        const auto append_read = [&](vbr_staged_read_descriptor read) {
            const bool projected = !read.projection_ranges.empty();
            if (!read.source || read.size == 0 ||
                (!projected &&
                 (read.source_offset > read.source->size() ||
                  read.size > read.source->size() - read.source_offset))) {
                return false;
            }
            if (source_index++ == policy.fault.fail_source_verify_at) {
                return false;
            }
            if (!projected) {
                read.verified_digest = read.source->sealed_digest();
                if (!digest_nonzero(read.verified_digest)) {
                    read.verified_digest =
                        vbr_capture_stream_digest(*read.source);
                }
            }
            if (!digest_nonzero(read.verified_digest)) {
                return false;
            }
            state->read_plan.push_back(std::move(read));
            return true;
        };

        struct child_stage_result {
            bool ok = false;
            vbr_adopt_stage_status status =
                vbr_adopt_stage_status::internal_error;
            std::vector<vbr_staged_read_descriptor> reads;
        };
        const auto stage_children =
                [&](size_t count, bool parallel, auto && stage_one) {
            std::vector<child_stage_result> results(count);
            const uint32_t available = std::thread::hardware_concurrency();
            const size_t worker_count = parallel
                ? std::min<size_t>(
                    16, std::min<size_t>(
                        count, available == 0 ? 1 : available))
                : std::min<size_t>(count, 1);
            const auto worker = [&](size_t worker_index) {
                const size_t begin = count*worker_index/worker_count;
                const size_t end = count*(worker_index + 1)/worker_count;
                for (size_t i = begin; i < end; ++i) {
                    auto & result = results[i];
                    const auto collect = [&](vbr_staged_read_descriptor read) {
                        result.reads.push_back(std::move(read));
                        return true;
                    };
                    result.ok = stage_one(i, collect, result.status);
                }
            };
            if (worker_count == 1) {
                worker(0);
            } else {
                std::vector<std::future<void>> workers;
                workers.reserve(worker_count);
                for (size_t i = 0; i < worker_count; ++i) {
                    workers.push_back(std::async(
                        std::launch::async, worker, i));
                }
                for (auto & pending : workers) {
                    pending.get();
                }
            }
            for (auto & result : results) {
                if (!result.ok) {
                    out.status = result.status;
                    return false;
                }
                for (auto & read : result.reads) {
                    if (!append_read(std::move(read))) {
                        out.status =
                            vbr_adopt_stage_status::source_hash_mismatch;
                        return false;
                    }
                }
            }
            return true;
        };

        const auto & manifest_children = out.manifest->children();
        if (!stage_children(
                manifest_children.size(),
                prefix_projection,
                [&](size_t i, const auto & collect,
                    vbr_adopt_stage_status & status) {
                    const auto & child = manifest_children[i];
                    return prefix_projection
                        ? occupied_replacement
                            ? stage_projection_relocated_child(
                                child, out.manifest->relocation_runs(),
                                policy.lanes, collect, status)
                            : stage_projection_child(
                                child, policy.lanes, collect, status)
                        : occupied_replacement
                            ? stage_relocated_child(
                                child, out.manifest->relocation_runs(),
                                vbr_staged_read_kind::unit_payload, nullptr,
                                policy.lanes, collect, status)
                            : stage_child(
                                child, out.manifest->source_package(),
                                policy.lanes, collect, status);
                })) {
            return out;
        }
        const auto * replacement = out.manifest->occupied_replacement();
        const bool recycle = replacement && replacement->strategy() ==
            vbr_occupied_replacement_strategy::recycle_incumbent_cells;
        if (recycle) {
            const auto & recovery = replacement->recovery_package();
            const auto & runs = replacement->recovery_runs();
            if (!recovery || runs.empty()) {
                out.status = vbr_adopt_stage_status::source_unavailable;
                return out;
            }
            const auto & recovery_units = recovery.units();
            if (!stage_children(
                    manifest_children.size(), false,
                    [&](size_t i, const auto & collect,
                        vbr_adopt_stage_status & status) {
                        const auto & child = manifest_children[i];
                        if (child.logical_unit_id >= recovery_units.size()) {
                            status = vbr_adopt_stage_status::source_unavailable;
                            return false;
                        }
                        return stage_relocated_child(
                                child, runs,
                                vbr_staged_read_kind::recovery_unit_payload,
                                &recovery_units[child.logical_unit_id],
                                policy.lanes, collect, status);
                    })) {
                return out;
            }
        }
        for (uint32_t i = 0; i < out.manifest->companions().size(); ++i) {
            const auto & companion = out.manifest->companions()[i];
            if (!append_read({
                    vbr_staged_read_kind::companion,
                    UINT32_MAX, UINT32_MAX, i, UINT32_MAX, 0,
                    companion.source ? companion.source->size() : 0,
                    companion.source, {}, 0, {},
                })) {
                out.status = vbr_adopt_stage_status::source_hash_mismatch;
                return out;
            }
        }
        if (state->read_plan.empty()) {
            out.status = vbr_adopt_stage_status::source_unavailable;
            return out;
        }

        state->leaves = out.manifest->accounting_leaves();
        std::vector<std::pair<llama_cache_acct_resource_domain, uint64_t>>
            transfer_totals;
        const auto add_transfer = [&](const llama_cache_acct_resource_domain & domain,
                                      uint64_t bytes) {
            auto found = std::find_if(
                transfer_totals.begin(), transfer_totals.end(),
                [&](const auto & value) { return value.first == domain; });
            if (found == transfer_totals.end()) {
                transfer_totals.push_back({ domain, bytes });
                return true;
            }
            uint64_t next;
            if (!add_checked(found->second, bytes, next)) {
                return false;
            }
            found->second = next;
            return true;
        };
        // A persistent ring is already physically charged by its store owner.
        // The standalone path still reserves and charges its per-stage ring.
        if (!policy.persistent_ring &&
            !add_transfer(policy.pinned_domain, policy.pinned_ring_bytes)) {
            out.status = vbr_adopt_stage_status::accounting_unavailable;
            return out;
        }
        // Recycle overwrites already-backed incumbent cells and reuses the
        // store-owned persistent ring for both the incoming write and any
        // recovery replay. Neither byte stream is new device residency.
        if (!recycle) {
            for (const auto & read : state->read_plan) {
                if (read.lane != UINT32_MAX &&
                    !add_transfer(policy.lanes[read.lane].domain, read.size)) {
                    out.status = vbr_adopt_stage_status::accounting_unavailable;
                    return out;
                }
            }
        }
        for (const auto & total : transfer_totals) {
            if (total.second == 0) {
                continue;
            }
            llama_cache_transaction_leaf leaf;
            leaf.category = llama_cache_acct_category::transfer_staging;
            leaf.domain = total.first;
            leaf.attribution = {
                llama_cache_acct_attr_kind::artifact, -1,
                out.manifest->source_artifact(),
            };
            leaf.expected_logical = total.second;
            leaf.reserve_resident = total.second;
            leaf.stage_resident = total.second;
            leaf.artifact = out.manifest->source_artifact();
            state->leaves.push_back(leaf);
        }
        state->committed_ops.resize(state->leaves.size());
        state->allocations.resize(state->leaves.size());
        for (size_t i = 0; i < state->leaves.size(); ++i) {
            state->leaves[i].committed_op = &state->committed_ops[i];
            state->leaves[i].allocation_out = &state->allocations[i];
        }
        if (policy.fault.fail_before_prepare) {
            out.status = vbr_adopt_stage_status::internal_error;
            return out;
        }
        state->prepared = llama_cache_prepare_reservation_transaction(
            *policy.ledger, *policy.budget, state->leaves);
        if (!state->prepared.ready()) {
            out.status = state->prepared.preparation().status ==
                    llama_cache_prepare_status::admission_refused
                ? vbr_adopt_stage_status::admission_refused
                : state->prepared.preparation().status ==
                      llama_cache_prepare_status::invalid_argument
                    ? vbr_adopt_stage_status::accounting_unavailable
                    : vbr_adopt_stage_status::internal_error;
            return out;
        }

        const auto post_prepare = policy.ledger->snapshot();
        if (policy.persistent_ring &&
            !policy.persistent_ring->compatible_with(
                policy.ledger, post_prepare, policy.pinned_domain,
                policy.pinned_ring_bytes, policy.chunk_bytes,
                policy.lanes)) {
            out.status = vbr_adopt_stage_status::accounting_unavailable;
            return out;
        }
        llama_cache_budget_coordinator coordinator;
        llama_cache_budget_plan empty_plan;
        empty_plan.accounting_serial = post_prepare.serial;
        if (!coordinator.reset(post_prepare, *policy.budget) ||
            coordinator.fits(empty_plan).state !=
                llama_cache_budget_fit_state::fits) {
            out.status = vbr_adopt_stage_status::accounting_unavailable;
            return out;
        }
        state->post_prepare_serial = post_prepare.serial;
        if (state->post_prepare_serial ==
                state->target.accounting_serial) {
            out.status = vbr_adopt_stage_status::accounting_unavailable;
            return out;
        }

        if (policy.fault.fail_ring_allocation) {
            out.status = vbr_adopt_stage_status::ring_unavailable;
            return out;
        }
        state->ring = policy.persistent_ring;
        if (!state->ring) {
            vbr_h2d_status ring_status;
            state->ring = vbr_h2d_chunk_ring::create(
                policy.lanes, policy.pinned_ring_bytes,
                policy.chunk_bytes, ring_status);
            if (!state->ring || ring_status != vbr_h2d_status::ok) {
                out.status = vbr_adopt_stage_status::ring_unavailable;
                return out;
            }
        }

        out.staged.reset(new vbr_staged_payloads(std::move(state)));
        out.status = vbr_adopt_stage_status::staged;
        return out;
    } catch (...) {
        out.staged.reset();
        out.status = vbr_adopt_stage_status::internal_error;
        return out;
    }
}
