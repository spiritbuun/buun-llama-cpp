#include "server-vbr-artifact-store.h"

#include "server-prompt-cache-payload.h"
#include "../../common/speculative.h"

#include "build-info.h"
#include "../../src/llama-context.h"
#include "../../src/llama-io.h"
#include "../../src/llama-memory-hybrid-idx.h"
#include "../../src/llama-sha256.h"
#include "../../src/llama-vbr-qsa-index.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>

bool server_vbr_companion_codec_for(
        vbr_artifact_companion_kind kind,
        server_vbr_companion_codec & output) noexcept {
    output = {};
    const char * domain = nullptr;
    switch (kind) {
        case vbr_artifact_companion_kind::required_spec_payload:
            domain = "buun.vbr.draft-state-codec/v1";
            break;
        case vbr_artifact_companion_kind::typed_accelerator:
            domain = "buun.vbr.accelerator-ring-codec/v1";
            break;
        case vbr_artifact_companion_kind::frontier_logits:
            domain = "buun.vbr.frontier-logits-codec/v1";
            break;
        case vbr_artifact_companion_kind::qsa_index:
            output.kind = kind;
            output.format_version =
                vbr_qsa_index_companion_format_version();
            output.build_identity_digest =
                vbr_qsa_index_companion_build_identity();
            return true;
        default:
            return false;
    }
    llama_sha256_writer writer;
    writer.string(domain, std::strlen(domain));
    const char * commit = llama_commit();
    writer.string(commit, std::strlen(commit));
    output.kind = kind;
    output.format_version = 1;
    output.build_identity_digest = writer.finish();
    return true;
}

namespace {

// One prefix keeps the reference builder and the authorizer in lock-step.
constexpr char VBR_REFERENCE_PREFIX[] = "vbrref_";

class server_vbr_parsed_bytes final : public vbr_parsed_companion_image {
public:
    vbr_artifact_companion_kind companion_kind =
        vbr_artifact_companion_kind::_count;
    uint32_t version = 0;
    std::vector<uint8_t> bytes;

    vbr_artifact_companion_kind kind() const noexcept override {
        return companion_kind;
    }
    uint32_t format_version() const noexcept override { return version; }
};

class server_vbr_parsed_chain final : public vbr_parsed_companion_image {
public:
    vbr_artifact_companion_kind companion_kind =
        vbr_artifact_companion_kind::_count;
    uint32_t version = 0;
    const artifact_segment_chain * source = nullptr;
    uint64_t bytes = 0;

    vbr_artifact_companion_kind kind() const noexcept override {
        return companion_kind;
    }
    uint32_t format_version() const noexcept override { return version; }
};

class server_vbr_chain_reader final : public llama_io_read_i {
public:
    explicit server_vbr_chain_reader(const artifact_segment_chain & source)
        : source_(source) {}

    void read(void * destination, size_t size) override {
        if (!destination || offset_ > source_.size() ||
            size > source_.size()-offset_) {
            throw std::runtime_error("VBR companion chain short read");
        }
        constexpr size_t QUANTUM = 1u << 20;
        auto * out = static_cast<uint8_t *>(destination);
        for (size_t done = 0; done < size;) {
            const size_t take = std::min(QUANTUM, size-done);
            if (!source_.read(offset_, out+done, take)) {
                throw std::runtime_error("VBR companion chain short read");
            }
            offset_ += take;
            done += take;
        }
    }

    void read_tensor(
            ggml_tensor * tensor, size_t tensor_offset,
            size_t size) override {
        if (!tensor || offset_ > source_.size() ||
            size > source_.size()-offset_) {
            throw std::runtime_error("VBR companion tensor short read");
        }
        constexpr size_t QUANTUM = 1u << 20;
        if (scratch_.empty()) {
            scratch_.resize(QUANTUM);
        }
        for (size_t done = 0; done < size;) {
            const size_t take = std::min(QUANTUM, size-done);
            if (!source_.read(offset_, scratch_.data(), take)) {
                throw std::runtime_error("VBR companion tensor short read");
            }
            ggml_backend_tensor_set(
                tensor, scratch_.data(), tensor_offset+done, take);
            offset_ += take;
            done += take;
        }
    }

    size_t n_bytes() override {
        return size_t(offset_);
    }

private:
    const artifact_segment_chain & source_;
    std::vector<uint8_t> scratch_;
    uint64_t offset_ = 0;
};

class server_vbr_chain_match_writer final : public llama_io_write_i {
public:
    explicit server_vbr_chain_match_writer(
            const artifact_segment_chain & expected)
        : expected_(expected), scratch_(1u << 20) {}

    void write(const void * source, size_t size) override {
        if ((!source && size != 0) || offset_ > expected_.size() ||
            size > expected_.size()-offset_) {
            matches_ = false;
            return;
        }
        constexpr size_t QUANTUM = 1u << 20;
        const auto * bytes = static_cast<const uint8_t *>(source);
        for (size_t done = 0; done < size;) {
            const size_t take = std::min(QUANTUM, size-done);
            if (!expected_.read(offset_, scratch_.data(), take) ||
                std::memcmp(bytes+done, scratch_.data(), take) != 0) {
                matches_ = false;
                return;
            }
            offset_ += take;
            done += take;
        }
    }

    void write_tensor(ggml_tensor *, size_t, size_t) override {
        matches_ = false;
    }

    size_t n_bytes() override { return size_t(offset_); }

    bool matches() const noexcept {
        return matches_ && offset_ == expected_.size();
    }

private:
    const artifact_segment_chain & expected_;
    std::vector<uint8_t> scratch_;
    uint64_t offset_ = 0;
    bool matches_ = true;
};

struct server_vbr_draft_target {
    llama_context * ctx = nullptr;
    llama_seq_id destination = -1;
    llama_pos expected_terminal = -1;
    llama_pos recovery_terminal = -1;
};

class server_vbr_draft_image final : public vbr_prepared_companion_image {
public:
    server_vbr_draft_target target;
    uint64_t expected_bytes = 0;
    std::vector<uint8_t> recovery;
    llama_pos recovery_live_terminal = -1;

    static bool empty(const void * opaque) noexcept {
        const auto * target = static_cast<const server_vbr_draft_target *>(opaque);
        if (!target || !target->ctx || target->destination < 0) return false;
        auto * memory = llama_get_memory(target->ctx);
        return memory &&
            llama_memory_seq_pos_min(memory, target->destination) < 0 &&
            llama_memory_seq_pos_max(memory, target->destination) < 0;
    }
    static bool prepare(
            const void * opaque,
            std::unique_ptr<vbr_parsed_companion_image> parsed_base,
            llama_seq_id destination,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        try {
            output.reset();
            const auto * target = static_cast<const server_vbr_draft_target *>(opaque);
            auto * parsed = dynamic_cast<server_vbr_parsed_chain *>(parsed_base.get());
            if (!target || !target->ctx || target->destination != destination ||
                !parsed || parsed->kind() !=
                    vbr_artifact_companion_kind::required_spec_payload ||
                !parsed->source || parsed->bytes == 0 || !empty(opaque)) {
                return false;
            }
            auto image = std::make_unique<server_vbr_draft_image>();
            image->target = *target;
            image->expected_bytes = parsed->bytes;
            server_vbr_chain_reader reader(*parsed->source);
            const size_t written = target->ctx->state_seq_read_data_stream(
                reader, destination, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (written != image->expected_bytes ||
                reader.n_bytes() != image->expected_bytes) {
                llama_memory_seq_rm(
                    llama_get_memory(target->ctx), destination, -1, -1);
                return false;
            }
            auto * memory = llama_get_memory(target->ctx);
            if (!memory || target->expected_terminal < 0 ||
                llama_memory_seq_pos_max(memory, destination) !=
                    target->expected_terminal) {
                if (memory) {
                    llama_memory_seq_rm(memory, destination, -1, -1);
                }
                return false;
            }
            output = std::move(image);
            return true;
        } catch (...) {
            output.reset();
            return false;
        }
    }
    static bool load(
            const server_vbr_draft_target & target,
            const artifact_segment_chain & source,
            uint64_t bytes,
            llama_pos terminal) {
        server_vbr_chain_reader reader(source);
        const size_t written = target.ctx->state_seq_read_data_stream(
            reader, target.destination, LLAMA_STATE_SEQ_FLAGS_NONE);
        auto * memory = llama_get_memory(target.ctx);
        return written == bytes && reader.n_bytes() == bytes && memory &&
            llama_memory_seq_pos_max(memory, target.destination) == terminal;
    }
    static bool prepare_replacement(
            const void * opaque,
            std::unique_ptr<vbr_parsed_companion_image> incoming_base,
            std::unique_ptr<vbr_parsed_companion_image> recovery_base,
            llama_seq_id destination,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        try {
            output.reset();
            const auto * target =
                static_cast<const server_vbr_draft_target *>(opaque);
            auto * incoming = dynamic_cast<server_vbr_parsed_chain *>(
                incoming_base.get());
            auto * recovery = dynamic_cast<server_vbr_parsed_chain *>(
                recovery_base.get());
            if (!target || !target->ctx || target->destination != destination ||
                target->expected_terminal < 0 ||
                target->recovery_terminal < 0 || !incoming || !recovery ||
                !incoming->source || !recovery->source ||
                incoming->bytes == 0 || recovery->bytes == 0) {
                return false;
            }
            auto image = std::make_unique<server_vbr_draft_image>();
            image->target = *target;
            image->expected_bytes = incoming->bytes;
            auto * memory = llama_get_memory(target->ctx);
            if (!memory) {
                return false;
            }
            image->recovery_live_terminal =
                llama_memory_seq_pos_max(memory, destination);
            const llama_pos recovery_live_min =
                llama_memory_seq_pos_min(memory, destination);
            const size_t live_bytes = llama_state_seq_get_size_ext(
                target->ctx, destination, LLAMA_STATE_SEQ_FLAGS_NONE);
            static constexpr size_t MAX_DRAFT_RECOVERY_BYTES = 64u << 20;
            if (image->recovery_live_terminal != target->recovery_terminal ||
                recovery_live_min != target->recovery_terminal ||
                live_bytes == 0 || live_bytes > MAX_DRAFT_RECOVERY_BYTES ||
                live_bytes != recovery->bytes ||
                recovery->source->size() != recovery->bytes) {
                return false;
            }
            image->recovery.resize(live_bytes);
            if (llama_state_seq_get_data_ext(
                    target->ctx, image->recovery.data(), live_bytes,
                    destination, LLAMA_STATE_SEQ_FLAGS_NONE) != live_bytes) {
                return false;
            }
            std::vector<uint8_t> recovery_chunk(
                std::min<size_t>(size_t(1) << 20, live_bytes));
            for (size_t offset = 0; offset < live_bytes;) {
                const size_t take = std::min(
                    recovery_chunk.size(), live_bytes-offset);
                if (!recovery->source->read(
                        offset, recovery_chunk.data(), take) ||
                    std::memcmp(
                        recovery_chunk.data(),
                        image->recovery.data()+offset, take) != 0) {
                    return false;
                }
                offset += take;
            }
            auto * raw = image.get();
            output = std::move(image);
            if (!llama_memory_seq_rm(
                    llama_get_memory(target->ctx), destination, -1, -1) ||
                !load(*target, *incoming->source, incoming->bytes,
                      target->expected_terminal)) {
                return false;
            }
            return raw != nullptr;
        } catch (...) {
            return false;
        }
    }
    static bool recheck(
            const void * opaque,
            const vbr_prepared_companion_image & base) noexcept {
        auto & image = const_cast<server_vbr_draft_image &>(
            static_cast<const server_vbr_draft_image &>(base));
        const auto * target = static_cast<const server_vbr_draft_target *>(opaque);
        return target && target->ctx == image.target.ctx &&
            target->destination == image.target.destination &&
            target->expected_terminal == image.target.expected_terminal &&
            target->recovery_terminal == image.target.recovery_terminal &&
            llama_memory_seq_pos_max(
                llama_get_memory(target->ctx), target->destination) ==
                    target->expected_terminal &&
            llama_state_seq_get_size_ext(
                target->ctx, target->destination,
                LLAMA_STATE_SEQ_FLAGS_NONE) == image.expected_bytes;
    }
    static void publish(const void *, vbr_prepared_companion_image &) noexcept {}
    static bool rollback(
            const void *, vbr_prepared_companion_image & base) noexcept {
        auto & image = static_cast<server_vbr_draft_image &>(base);
        if (!image.target.ctx) {
            return false;
        }
        if (!llama_memory_seq_rm(
                llama_get_memory(image.target.ctx),
                image.target.destination, -1, -1)) {
            return false;
        }
        if (image.recovery.empty()) {
            return empty(&image.target);
        }
        try {
            const size_t restored = llama_state_seq_set_data_ext(
                image.target.ctx, image.recovery.data(),
                image.recovery.size(), image.target.destination,
                LLAMA_STATE_SEQ_FLAGS_NONE);
            auto * memory = llama_get_memory(image.target.ctx);
            return restored == image.recovery.size() && memory &&
                llama_memory_seq_pos_max(
                    memory, image.target.destination) ==
                        image.recovery_live_terminal;
        } catch (...) {
            return false;
        }
    }
};

struct server_vbr_accelerator_target {
    common_speculative * spec = nullptr;
    llama_pos expected_terminal = -1;
    llama_pos recovery_terminal = -1;
};

class server_vbr_accelerator_image final : public vbr_prepared_companion_image {
public:
    common_speculative * spec = nullptr;
    common_speculative_ring_state_currency installed;
    const artifact_segment_chain * recovery = nullptr;
    uint64_t recovery_bytes = 0;

    static bool empty(const void * opaque) noexcept {
        const auto * target =
            static_cast<const server_vbr_accelerator_target *>(opaque);
        return target && target->spec &&
            common_speculative_ring_state_empty(target->spec);
    }
    static bool prepare(
            const void * opaque,
            std::unique_ptr<vbr_parsed_companion_image> parsed_base,
            llama_seq_id,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        try {
            output.reset();
            const auto * target =
                static_cast<const server_vbr_accelerator_target *>(opaque);
            auto * parsed = dynamic_cast<server_vbr_parsed_chain *>(
                parsed_base.get());
            if (!target || !target->spec || !parsed ||
                parsed->kind() !=
                    vbr_artifact_companion_kind::typed_accelerator ||
                !parsed->source || parsed->bytes == 0 ||
                target->expected_terminal < 0 ||
                !empty(opaque)) {
                return false;
            }
            std::array<uint8_t, 6*sizeof(int32_t)> header = {};
            llama_pos serialized_terminal = -1;
            if (parsed->bytes < header.size() ||
                !parsed->source->read(0, header.data(), header.size()) ||
                !common_speculative_ring_state_serialized_terminal(
                    header.data(), header.size(), serialized_terminal) ||
                serialized_terminal != target->expected_terminal) {
                return false;
            }
            auto image = std::make_unique<server_vbr_accelerator_image>();
            image->spec = target->spec;
            server_vbr_chain_reader reader(*parsed->source);
            if (!common_speculative_ring_state_read(
                    image->spec, reader, size_t(parsed->bytes)) ||
                reader.n_bytes() != parsed->bytes ||
                !common_speculative_ring_state_get_currency(
                    image->spec, image->installed) ||
                image->installed.serialized_bytes != parsed->bytes ||
                image->installed.terminal != target->expected_terminal) {
                common_speculative_ring_state_reset(image->spec);
                return false;
            }
            output = std::move(image);
            return true;
        } catch (...) {
            output.reset();
            return false;
        }
    }
    static bool live_matches(
            const server_vbr_accelerator_target & target,
            const server_vbr_parsed_chain & expected,
            llama_pos terminal) {
        if (!target.spec || !expected.source || expected.bytes == 0 ||
            terminal < 0 ||
            common_speculative_ring_state_size(target.spec) !=
                expected.bytes) {
            return false;
        }
        llama_pos current_terminal = -1;
        if (!common_speculative_ring_state_terminal(
                target.spec, current_terminal) ||
            current_terminal != terminal) {
            return false;
        }
        server_vbr_chain_match_writer writer(*expected.source);
        return common_speculative_ring_state_write(target.spec, writer) &&
            writer.n_bytes() == expected.bytes &&
            writer.matches();
    }
    static bool prepare_replacement(
            const void * opaque,
            std::unique_ptr<vbr_parsed_companion_image> incoming_base,
            std::unique_ptr<vbr_parsed_companion_image> recovery_base,
            llama_seq_id,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        try {
            output.reset();
            const auto * target = static_cast<
                const server_vbr_accelerator_target *>(opaque);
            auto * incoming = dynamic_cast<server_vbr_parsed_chain *>(
                incoming_base.get());
            auto * recovery = dynamic_cast<server_vbr_parsed_chain *>(
                recovery_base.get());
            if (!target || !target->spec ||
                target->expected_terminal < 0 ||
                target->recovery_terminal < 0 || !incoming || !recovery ||
                incoming->kind() !=
                    vbr_artifact_companion_kind::typed_accelerator ||
                recovery->kind() !=
                    vbr_artifact_companion_kind::typed_accelerator ||
                !incoming->source || !recovery->source ||
                incoming->bytes == 0 || recovery->bytes == 0) {
                return false;
            }
            if (!live_matches(
                    *target, *recovery,
                    target->recovery_terminal)) {
                return false;
            }
            auto image = std::make_unique<server_vbr_accelerator_image>();
            image->spec = target->spec;
            image->recovery = recovery->source;
            image->recovery_bytes = recovery->bytes;
            output = std::move(image);
            auto & prepared = static_cast<server_vbr_accelerator_image &>(
                *output);
            server_vbr_chain_reader reader(*incoming->source);
            if (!common_speculative_ring_state_read(
                    target->spec, reader, size_t(incoming->bytes)) ||
                reader.n_bytes() != incoming->bytes ||
                !common_speculative_ring_state_get_currency(
                    target->spec, prepared.installed) ||
                prepared.installed.serialized_bytes != incoming->bytes ||
                prepared.installed.terminal != target->expected_terminal) {
                return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }
    static bool recheck(
            const void * opaque,
            const vbr_prepared_companion_image & base) noexcept {
        const auto * target =
            static_cast<const server_vbr_accelerator_target *>(opaque);
        const auto & image = static_cast<
            const server_vbr_accelerator_image &>(base);
        if (!target || target->spec != image.spec ||
            target->expected_terminal < 0) {
            return false;
        }
        common_speculative_ring_state_currency current;
        return common_speculative_ring_state_get_currency(
                image.spec, current) &&
            current.serialized_bytes == image.installed.serialized_bytes &&
            current.terminal == image.installed.terminal &&
            current.mutation_epoch == image.installed.mutation_epoch &&
            current.terminal == target->expected_terminal;
    }
    static void publish(const void *, vbr_prepared_companion_image &) noexcept {}
    static bool rollback(
            const void *, vbr_prepared_companion_image & base) noexcept {
        auto & image = static_cast<server_vbr_accelerator_image &>(base);
        common_speculative_ring_state_reset(image.spec);
        if (!image.recovery) {
            return common_speculative_ring_state_empty(image.spec);
        }
        try {
            server_vbr_chain_reader reader(*image.recovery);
            return common_speculative_ring_state_read(
                    image.spec, reader, size_t(image.recovery_bytes)) &&
                reader.n_bytes() == image.recovery_bytes;
        } catch (...) {
            return false;
        }
    }
};

struct server_vbr_frontier_logits_target {
    std::vector<float> * logits = nullptr;
    uint32_t count = 0;
    bool replace = false;
};

class server_vbr_frontier_logits_image final : public vbr_prepared_companion_image {
public:
    server_vbr_frontier_logits_target target;
    std::vector<float> prepared;
    std::vector<float> original;

    static bool empty(const void * opaque) noexcept {
        const auto * target =
            static_cast<const server_vbr_frontier_logits_target *>(opaque);
        return target && target->logits && target->count > 0 &&
            (target->replace || target->logits->empty());
    }
    static bool prepare(
            const void * opaque,
            std::unique_ptr<vbr_parsed_companion_image> parsed_base,
            llama_seq_id,
            std::unique_ptr<vbr_prepared_companion_image> & output) noexcept {
        try {
            output.reset();
            const auto * target =
                static_cast<const server_vbr_frontier_logits_target *>(opaque);
            auto * parsed = dynamic_cast<server_vbr_parsed_bytes *>(parsed_base.get());
            const uint64_t expected_bytes = target
                ? uint64_t(target->count)*sizeof(float) : 0;
            if (!target || !target->logits || target->count == 0 ||
                !parsed || parsed->companion_kind !=
                    vbr_artifact_companion_kind::frontier_logits ||
                parsed->bytes.size() != expected_bytes || !empty(opaque)) {
                return false;
            }
            auto image = std::make_unique<server_vbr_frontier_logits_image>();
            image->target = *target;
            image->original = *target->logits;
            image->prepared.resize(target->count);
            std::memcpy(image->prepared.data(), parsed->bytes.data(),
                        parsed->bytes.size());
            output = std::move(image);
            return true;
        } catch (...) {
            output.reset();
            return false;
        }
    }
    static bool recheck(
            const void * opaque,
            const vbr_prepared_companion_image & base) noexcept {
        const auto * target =
            static_cast<const server_vbr_frontier_logits_target *>(opaque);
        const auto & image =
            static_cast<const server_vbr_frontier_logits_image &>(base);
        return target && target->logits == image.target.logits &&
            target->count == image.target.count &&
            target->replace == image.target.replace &&
            image.prepared.size() == target->count && empty(opaque) &&
            *target->logits == image.original;
    }
    static void publish(
            const void * opaque,
            vbr_prepared_companion_image & base) noexcept {
        const auto * target =
            static_cast<const server_vbr_frontier_logits_target *>(opaque);
        auto & image = static_cast<server_vbr_frontier_logits_image &>(base);
        GGML_ASSERT(target && target->logits);
        target->logits->swap(image.prepared);
    }
    static bool rollback(
            const void * opaque,
            vbr_prepared_companion_image & base) noexcept {
        const auto * target =
            static_cast<const server_vbr_frontier_logits_target *>(opaque);
        const auto & image =
            static_cast<const server_vbr_frontier_logits_image &>(base);
        return target && target->logits &&
            *target->logits == image.original;
    }
};

bool capture_capacity_category_applies(
        llama_cache_acct_category category,
        const llama_cache_acct_resource_domain & domain,
        bool include_live_scope) {
    const auto row = llama_cache_budget_classify(category);
    if (row.participation !=
            llama_cache_budget_capacity_participation::
                participating) {
        return false;
    }
    if (row.scope ==
            llama_cache_budget_residency_scope::by_domain) {
        return domain.residency ==
                   llama_cache_acct_residency::device ||
               domain.residency ==
                   llama_cache_acct_residency::pinned_host ||
               domain.residency ==
                   llama_cache_acct_residency::pageable_host;
    }
    if (row.scope ==
            llama_cache_budget_residency_scope::host) {
        return
            (domain.residency ==
                 llama_cache_acct_residency::pinned_host ||
             domain.residency ==
                 llama_cache_acct_residency::pageable_host);
    }
    return include_live_scope &&
           row.scope ==
               llama_cache_budget_residency_scope::device &&
           domain.residency ==
               llama_cache_acct_residency::device;
}

std::string opaque_reference(
        uint64_t nonce,
        uint64_t ordinal,
        llama_cache_acct_artifact_id artifact,
        const std::string & tenant) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] = "buun.vbr.server-reference/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u64(nonce);
    writer.u64(ordinal);
    writer.u64(artifact.v);
    writer.string(tenant.data(), tenant.size());
    const auto digest = writer.finish();
    static constexpr char HEX[] = "0123456789abcdef";
    std::string out = VBR_REFERENCE_PREFIX;
    out.reserve(out.size() + 32);
    for (size_t i = 0; i < 16; ++i) {
        out.push_back(HEX[digest[i] >> 4]);
        out.push_back(HEX[digest[i] & 0x0f]);
    }
    return out;
}

server_vbr_artifact_capture_status map_status(
        vbr_explicit_capture_status status) {
    switch (status) {
        case vbr_explicit_capture_status::ok:
            return server_vbr_artifact_capture_status::ok;
        case vbr_explicit_capture_status::not_armed:
        case vbr_explicit_capture_status::unsupported_layout:
            return server_vbr_artifact_capture_status::unsupported;
        case vbr_explicit_capture_status::slot_not_idle:
            return server_vbr_artifact_capture_status::slot_processing;
        case vbr_explicit_capture_status::identity_unavailable:
            return server_vbr_artifact_capture_status::identity_unavailable;
        case vbr_explicit_capture_status::required_companion_unavailable:
            return server_vbr_artifact_capture_status::
                required_companion_unavailable;
        case vbr_explicit_capture_status::admission_refused:
            return server_vbr_artifact_capture_status::admission_refused;
        case vbr_explicit_capture_status::cancelled:
            return server_vbr_artifact_capture_status::cancelled;
        case vbr_explicit_capture_status::source_changed:
            return server_vbr_artifact_capture_status::source_changed;
        case vbr_explicit_capture_status::generation_unavailable:
        case vbr_explicit_capture_status::registry_busy:
        case vbr_explicit_capture_status::recovery_pending:
        case vbr_explicit_capture_status::geometry_mismatch:
        case vbr_explicit_capture_status::stash_inconsistent:
        case vbr_explicit_capture_status::size_overflow:
        case vbr_explicit_capture_status::ring_unavailable:
        case vbr_explicit_capture_status::transfer_failed:
        case vbr_explicit_capture_status::short_read:
        case vbr_explicit_capture_status::event_failed:
        case vbr_explicit_capture_status::hash_mismatch:
        case vbr_explicit_capture_status::dedup_validation_failed:
        case vbr_explicit_capture_status::accounting_failed:
        case vbr_explicit_capture_status::publication_failed:
            return server_vbr_artifact_capture_status::unavailable;
        case vbr_explicit_capture_status::internal_error:
        case vbr_explicit_capture_status::_count:
            return server_vbr_artifact_capture_status::internal_error;
    }
    return server_vbr_artifact_capture_status::internal_error;
}

void copy_capture_result(
        const vbr_explicit_capture_result & result,
        server_vbr_artifact_capture_output & output) {
    output.library_status = result.status;
    output.phase = result.phase;
    output.inner_stream_status = result.inner_stream_status;
    output.generation_failure = result.generation_failure;
    output.size_failure = result.size_failure;
    output.begin_diagnostics = result.begin_diagnostics;
    output.pretransfer = result.pretransfer;
    output.status = map_status(result.status);
    output.controllers = result.controllers;
    output.units = result.units;
    output.companions = result.companions;
    output.companion_failure_index = result.companion_failure_index;
    output.companion_failure_kind = result.companion_failure_kind;
    output.payload_bytes = result.payload_bytes;
    output.stash_bytes = result.stash_bytes;
    output.companion_bytes = result.companion_bytes;
    output.chunks = result.chunks;
    output.backpressure_waits = result.backpressure_waits;
    output.event_completions = result.event_completions;
    output.synchronous_fallbacks = result.synchronous_fallbacks;
    output.dedup = result.sink.adopted;
}

struct live_import_context {
    llama_memory_i * memory = nullptr;
    llama_cache_acct_ledger * ledger = nullptr;
    const vbr_artifact_package_view * package = nullptr;
    const std::vector<llama_vbr_artifact_domain_binding> * bindings = nullptr;
    const vbr_import_schedule_quote * schedule_quote = nullptr;
    const void * representation_context = nullptr;
    vbr_explicit_capture_request::representation_identity_fn
        representation_identity = nullptr;
    llama_seq_id destination = -1;
    vbr_target_validation_snapshot snapshot;
};

bool import_inspect_target(
        const void * opaque,
        llama_memory_i & memory,
        const std::vector<llama_memory_tree_child> &,
        vbr_target_validation_snapshot & output) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    if (!context || context->memory != &memory) {
        return false;
    }
    try {
        output = context->snapshot;
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

uint64_t import_accounting_serial(const void * opaque) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    return context && context->ledger
        ? context->ledger->snapshot().serial : 0;
}

uint64_t import_policy_epoch(const void * opaque) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    // This is deliberately a fresh read even though target recheck also reads
    // the policy: the validator compares two independent live observations to
    // close the adopt-time TOCTOU window.
    return context && context->memory
        ? vbr_explicit_import_policy_epoch(*context->memory) : 0;
}

bool import_target_recheck(
        const void * opaque,
        const vbr_target_empty_fingerprint & expected) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    return context && context->memory &&
        vbr_explicit_import_target_recheck(
            *context->memory, context->destination, expected);
}

bool import_transform_digest(
        const void * opaque,
        std::array<uint8_t, 32> & output) noexcept {
    const auto * context = static_cast<const live_import_context *>(opaque);
    if (!context || !context->memory || !context->package ||
        !context->bindings || !context->schedule_quote) {
        return false;
    }
    return vbr_explicit_import_transform_projection_recheck(
            *context->memory, context->destination,
            *context->package, *context->bindings,
            *context->schedule_quote, context->representation_context,
            context->representation_identity, output);
}

bool import_parse_companion(
        const void * opaque,
        const vbr_artifact_companion_payload & descriptor,
        const artifact_segment_chain & source,
        const vbr_target_companion_snapshot & target,
        std::unique_ptr<vbr_parsed_companion_image> & output) noexcept {
    if (descriptor.kind == vbr_artifact_companion_kind::recurrent) {
        return vbr_parse_recurrent_companion(
            opaque, descriptor, source, target, output);
    }
    if (descriptor.kind == vbr_artifact_companion_kind::qsa_index) {
        return vbr_parse_qsa_index_companion(
            opaque, descriptor, source, target, output);
    }
    if (descriptor.kind ==
            vbr_artifact_companion_kind::required_spec_payload ||
        descriptor.kind ==
            vbr_artifact_companion_kind::typed_accelerator) {
        try {
            output.reset();
            if (!target.available || target.target_cookie == nullptr ||
                descriptor.format_version != 1 || source.size() == 0 ||
                descriptor.payload_bytes != source.size() ||
                source.size() > std::numeric_limits<size_t>::max()) {
                return false;
            }
            auto parsed = std::make_unique<server_vbr_parsed_chain>();
            parsed->companion_kind = descriptor.kind;
            parsed->version = descriptor.format_version;
            parsed->source = &source;
            parsed->bytes = source.size();
            output = std::move(parsed);
            return true;
        } catch (...) {
            output.reset();
            return false;
        }
    }
    try {
        output.reset();
        if (!target.available || target.target_cookie == nullptr ||
            descriptor.format_version != 1 ||
            descriptor.payload_bytes != source.size() ||
            source.size() == 0 ||
            source.size() > std::numeric_limits<size_t>::max() ||
            descriptor.kind !=
                 vbr_artifact_companion_kind::frontier_logits) {
            return false;
        }
        auto parsed = std::make_unique<server_vbr_parsed_bytes>();
        parsed->companion_kind = descriptor.kind;
        parsed->version = descriptor.format_version;
        parsed->bytes.resize(size_t(source.size()));
        if (!source.read(0, parsed->bytes.data(), parsed->bytes.size())) {
            return false;
        }
        output = std::move(parsed);
        return true;
    } catch (...) {
        output.reset();
        return false;
    }
}

bool import_reserve_transform(
        void * opaque,
        const std::vector<vbr_validated_child_plan> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept {
    auto * context = static_cast<live_import_context *>(opaque);
    return context && context->memory &&
        vbr_explicit_import_reserve_transform(
            *context->memory, plans, ledger, budget, output);
}

bool add_bytes(uint64_t & total, uint64_t value) noexcept {
    if (value > UINT64_MAX - total) {
        return false;
    }
    total += value;
    return true;
}

bool package_bytes(
        const vbr_artifact_package_view & package,
        uint64_t & payload,
        uint64_t & companions) noexcept {
    payload = 0;
    companions = 0;
    for (const auto & unit : package.units()) {
        for (const auto & shard : unit.descriptor.shards) {
            if (!add_bytes(payload, shard.payload_bytes)) {
                return false;
            }
        }
        for (const auto & stash : unit.descriptor.clean_stash.shards) {
            if (!add_bytes(payload, stash.payload_bytes)) {
                return false;
            }
        }
    }
    for (const auto & companion : package.companions()) {
        if (!add_bytes(companions, companion.descriptor.payload_bytes)) {
            return false;
        }
    }
    return true;
}

} // namespace

server_vbr_artifact_import_status server_vbr_artifact_import_route_precheck(
        bool store_available,
        bool slot_exists,
        bool slot_processing,
        bool target_available,
        bool slot_empty) noexcept {
    if (!store_available) {
        return server_vbr_artifact_import_status::unsupported;
    }
    if (!slot_exists) {
        return server_vbr_artifact_import_status::invalid_slot;
    }
    if (slot_processing) {
        return server_vbr_artifact_import_status::slot_processing;
    }
    if (!target_available) {
        return server_vbr_artifact_import_status::unavailable;
    }
    return slot_empty
        ? server_vbr_artifact_import_status::ok
        : server_vbr_artifact_import_status::slot_not_empty;
}

server_vbr_artifact_import_status
server_vbr_artifact_import_validation_disposition(
        vbr_manifest_validation_status status,
        vbr_import_decision decision) noexcept {
    if (status != vbr_manifest_validation_status::validated) {
        return server_vbr_artifact_import_status::validation_failed;
    }
    switch (decision) {
        case vbr_import_decision::native_import:
        case vbr_import_decision::live_rebased:
        case vbr_import_decision::downward_rebase:
        case vbr_import_decision::upward_reconstruct:
            return server_vbr_artifact_import_status::ok;
        case vbr_import_decision::rebuild:
        case vbr_import_decision::cold:
            return server_vbr_artifact_import_status::report_only;
        case vbr_import_decision::reject:
        case vbr_import_decision::_count:
            return server_vbr_artifact_import_status::validation_failed;
    }
    return server_vbr_artifact_import_status::validation_failed;
}

bool server_vbr_artifact_import_variant_fallback_safe(
        const server_vbr_artifact_import_output & output) noexcept {
    if (output.adopt_attempted || output.h2d_bytes != 0 ||
        output.h2d_chunks != 0) {
        return false;
    }
    switch (output.status) {
        case server_vbr_artifact_import_status::validation_failed:
        case server_vbr_artifact_import_status::report_only:
        case server_vbr_artifact_import_status::stage_failed:
            return true;
        case server_vbr_artifact_import_status::ok:
        case server_vbr_artifact_import_status::unsupported:
        case server_vbr_artifact_import_status::not_found:
        case server_vbr_artifact_import_status::invalid_slot:
        case server_vbr_artifact_import_status::slot_processing:
        case server_vbr_artifact_import_status::slot_not_empty:
        case server_vbr_artifact_import_status::adopt_failed:
        case server_vbr_artifact_import_status::unavailable:
        case server_vbr_artifact_import_status::internal_error:
        case server_vbr_artifact_import_status::_count:
            return false;
    }
    return false;
}

bool server_vbr_artifact_reference_index::publish(
        std::string reference,
        std::string tenant_key,
        llama_cache_acct_artifact_id artifact) noexcept {
    try {
        return reference.rfind(VBR_REFERENCE_PREFIX, 0) == 0 &&
               !tenant_key.empty() && artifact.v != 0 &&
               entries_.emplace(
                   std::move(reference),
                   binding { std::move(tenant_key), artifact }).second;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_reference_index::authorize(
        const std::string & reference,
        const std::string & tenant_key,
        llama_cache_acct_artifact_id & artifact) const noexcept {
    artifact = {};
    try {
        if (reference.rfind(VBR_REFERENCE_PREFIX, 0) != 0 || tenant_key.empty()) {
            return false;
        }
        const auto found = entries_.find(reference);
        if (found == entries_.end() ||
            found->second.tenant_key != tenant_key) {
            return false;
        }
        artifact = found->second.artifact;
        return artifact.v != 0;
    } catch (...) {
        artifact = {};
        return false;
    }
}

struct server_vbr_artifact_store::impl {
    llama_cache_acct_ledger * ledger = nullptr;
    llama_vbr_artifact_catalog catalog;
    std::unique_ptr<vbr_pinned_chunk_ring> ring;
    std::shared_ptr<vbr_h2d_chunk_ring> import_ring;
    std::vector<vbr_artifact_portable_topology> topologies;
    std::vector<vbr_explicit_capture_pool_binding> pool_bindings;
    std::vector<llama_vbr_artifact_domain_binding> domain_bindings;
    std::vector<llama_vbr_artifact_domain_binding> policy_bindings;
    std::vector<vbr_h2d_lane_binding> h2d_lanes;
    llama_cache_acct_resource_domain pinned_domain;
    uint64_t import_ring_bytes = 0;
    size_t import_chunk_bytes = 0;
    void * budget_context = nullptr;
    server_vbr_artifact_store_config::sample_budget_fn sample_budget = nullptr;
    server_vbr_artifact_store_counters counters;
    uint64_t nonce = 0;
    uint64_t next_reference = 1;
    uint32_t n_attention_children = 0;
    server_vbr_artifact_reference_index references;
    bool fail_projected_host_adoption_once = false;

    explicit impl(llama_cache_acct_ledger & ledger)
        : ledger(&ledger), catalog(ledger) {
    }

    bool bind_import_transport(vbr_adopt_stage_policy & policy) const noexcept {
        if (!ledger || !import_ring || h2d_lanes.empty() ||
            import_ring_bytes == 0 || import_chunk_bytes == 0) {
            return false;
        }
        policy.ledger = ledger;
        policy.lanes = h2d_lanes;
        policy.pinned_domain = pinned_domain;
        policy.pinned_ring_bytes = import_ring_bytes;
        policy.chunk_bytes = import_chunk_bytes;
        policy.persistent_ring = import_ring;
        return true;
    }
};

struct server_vbr_explicit_host_capture::impl {
    vbr_explicit_capture_operation capture;
    server_vbr_artifact_capture_output output;
};

server_vbr_explicit_host_capture::server_vbr_explicit_host_capture()
    noexcept = default;
server_vbr_explicit_host_capture::server_vbr_explicit_host_capture(
        server_vbr_explicit_host_capture && other) noexcept = default;
server_vbr_explicit_host_capture &
server_vbr_explicit_host_capture::operator=(
        server_vbr_explicit_host_capture && other) noexcept = default;
server_vbr_explicit_host_capture::~server_vbr_explicit_host_capture() =
    default;

bool server_vbr_explicit_host_capture::ready_for_transfer() const noexcept {
    return impl_ && impl_->capture.ready_for_transfer();
}

bool server_vbr_explicit_host_capture::ready_for_publication() const noexcept {
    return impl_ && impl_->capture.ready_for_publication();
}

void server_vbr_explicit_host_capture::reset() noexcept {
    impl_.reset();
}

server_vbr_artifact_store::server_vbr_artifact_store(
        std::unique_ptr<impl> state) noexcept
    : impl_(std::move(state)) {
}

server_vbr_artifact_store::~server_vbr_artifact_store() = default;

namespace {

class projected_capture_staging_session {
public:
    using quote =
        vbr_projected_capture_batch_request::pretransfer_quote;

    projected_capture_staging_session(
            llama_cache_acct_ledger & ledger,
            const std::vector<llama_vbr_artifact_domain_binding> & bindings)
        : ledger_(ledger), bindings_(bindings) {
    }

    bool admit_initial(
            const llama_cache_budget_config & budget,
            const quote & value) noexcept {
        if (initialized_) {
            return false;
        }
        initialized_ = true;
        if (value.planned_packed_bytes == 0) {
            return value.staging.empty();
        }
        preparation_attempted_ = true;
        try {
            uint64_t total = 0;
            portable_domains_.reserve(value.staging.size());
            leaves_.resize(value.staging.size());
            committed_ops_.resize(value.staging.size());
            current_bytes_.resize(value.staging.size());
            shrink_bytes_.resize(value.staging.size());
            for (size_t i = 0; i < value.staging.size(); ++i) {
                const auto & row = value.staging[i];
                if (row.bytes == 0 || row.bytes > UINT64_MAX - total ||
                    (i != 0 &&
                     !vbr_artifact_portable_domain_less(
                         value.staging[i - 1].domain, row.domain))) {
                    return false;
                }
                total += row.bytes;
                llama_cache_acct_resource_domain domain;
                if (!resolve(row.domain, domain)) {
                    return false;
                }
                portable_domains_.push_back(row.domain);
                current_bytes_[i] = row.bytes;
                auto & leaf = leaves_[i];
                leaf.category =
                    llama_cache_acct_category::transfer_staging;
                leaf.domain = domain;
                leaf.attribution = {
                    llama_cache_acct_attr_kind::server, -1, {},
                };
                leaf.expected_logical = row.bytes;
                leaf.reserve_resident = row.bytes;
                leaf.stage_resident = row.bytes;
                leaf.committed_op = &committed_ops_[i];
            }
            if (total != value.planned_packed_bytes) {
                return false;
            }
            prepared_ = llama_cache_prepare_reservation_transaction(
                ledger_, budget, leaves_);
            preparation_ = prepared_.preparation();
            if (!prepared_.ready()) {
                return false;
            }
            bytes_ = total;
            return true;
        } catch (...) {
            return false;
        }
    }

    bool shrink(const quote & value) noexcept {
        if (!initialized_ || value.planned_packed_bytes > bytes_) {
            return false;
        }
        if (!preparation_attempted_) {
            return value.planned_packed_bytes == 0 && value.staging.empty();
        }
        try {
            std::fill(shrink_bytes_.begin(), shrink_bytes_.end(), 0);
            uint64_t total = 0;
            for (const auto & row : value.staging) {
                if (row.bytes == 0 || row.bytes > UINT64_MAX - total) {
                    return false;
                }
                total += row.bytes;
                const auto found = std::lower_bound(
                    portable_domains_.begin(), portable_domains_.end(),
                    row.domain, vbr_artifact_portable_domain_less);
                if (found == portable_domains_.end() ||
                    *found != row.domain) {
                    return false;
                }
                const size_t index = size_t(found - portable_domains_.begin());
                if (shrink_bytes_[index] != 0 ||
                    row.bytes > current_bytes_[index]) {
                    return false;
                }
                shrink_bytes_[index] = row.bytes;
            }
            if (total != value.planned_packed_bytes ||
                !prepared_.shrink_equal_reservations(shrink_bytes_)) {
                return false;
            }
            current_bytes_ = shrink_bytes_;
            bytes_ = total;
            return true;
        } catch (...) {
            return false;
        }
    }

    uint64_t bytes() const noexcept { return bytes_; }
    bool reserved() const noexcept {
        return preparation_attempted_ && prepared_.ready();
    }
    bool preparation_required() const noexcept {
        return preparation_attempted_;
    }
    const llama_cache_prepare_result & preparation() const noexcept {
        return preparation_;
    }
    void cancel() noexcept {
        prepared_ = {};
        bytes_ = 0;
    }

private:
    bool resolve(
            const vbr_artifact_portable_domain & portable,
            llama_cache_acct_resource_domain & domain) const noexcept {
        if (portable.residency ==
                llama_cache_acct_residency::device) {
            const auto found = std::find_if(
                bindings_.begin(), bindings_.end(),
                [&](const auto & binding) {
                    return binding.topology_index ==
                               portable.topology_index &&
                           binding.device_ordinal ==
                               portable.device_ordinal;
                });
            if (portable.kind !=
                    llama_cache_acct_domain_kind::device_topology ||
                found == bindings_.end()) {
                return false;
            }
            domain = found->domain;
            return true;
        }
        if (portable.kind !=
                llama_cache_acct_domain_kind::not_applicable ||
            portable.topology_index != UINT32_MAX ||
            portable.device_ordinal != UINT16_MAX ||
            portable.residency ==
                llama_cache_acct_residency::not_applicable ||
            portable.residency >= llama_cache_acct_residency::_count) {
            return false;
        }
        domain = llama_cache_acct_resource_domain::non_device(
            portable.residency);
        return true;
    }

    llama_cache_acct_ledger & ledger_;
    const std::vector<llama_vbr_artifact_domain_binding> & bindings_;
    std::vector<vbr_artifact_portable_domain> portable_domains_;
    std::vector<llama_cache_transaction_leaf> leaves_;
    std::vector<llama_cache_acct_op_id> committed_ops_;
    std::vector<uint64_t> current_bytes_;
    std::vector<uint64_t> shrink_bytes_;
    llama_cache_prepared_claim_group prepared_;
    llama_cache_prepare_result preparation_;
    uint64_t bytes_ = 0;
    bool initialized_ = false;
    bool preparation_attempted_ = false;
};

class projected_capture_resource_admission {
public:
    using quote = projected_capture_staging_session::quote;
    using status =
        server_vbr_projected_host_capture_diagnostics::resource_status;

    projected_capture_resource_admission(
            llama_cache_acct_ledger & ledger,
            void * budget_context,
            server_vbr_artifact_store_config::sample_budget_fn sample_budget,
            const std::vector<llama_vbr_artifact_domain_binding> & bindings,
            const server_vbr_projected_capture_admission * scheduler,
            llama_vbr_artifact_catalog * catalog = nullptr)
        : budget_context_(budget_context),
          sample_budget_(sample_budget),
          scheduler_(scheduler),
          catalog_(catalog),
          session_(ledger, bindings) {
    }

    bool admit(const quote & value) noexcept {
        if (!sample_budget_) {
            status_ = status::budget_failed;
            return false;
        }
        if (!scheduler_checked_) {
            if (value.planned_packed_bytes == 0) {
                scheduler_checked_ = true;
                if (value.projected_host_resident_bytes != 0 ||
                    !value.staging.empty() || !value.durable.empty() ||
                    value.manifests != 0 || value.projected_units != 0 ||
                    value.union_cells != 0) {
                    status_ = status::invalid_quote;
                    return false;
                }
                const llama_cache_budget_config unused;
                const bool accepted = session_.admit_initial(unused, value);
                status_ = accepted
                    ? status::zero_work_admitted
                    : status::invalid_quote;
                return accepted;
            }
            scheduler_checked_ = true;
            try {
                llama_cache_budget_config budget;
                if (!sample_budget_(budget_context_, budget)) {
                    status_ = status::budget_failed;
                    return false;
                }
                if (!prepare_durable(budget, value)) {
                    publication_claim_ = {};
                    durable_.clear();
                    status_ = status::durable_preparation_refused;
                    return false;
                }
                const bool accepted = session_.admit_initial(budget, value);
                if (!accepted) {
                    publication_claim_ = {};
                    durable_.clear();
                    status_ = status::preparation_refused;
                    return false;
                }
                if (scheduler_ && scheduler_->admit &&
                    !scheduler_->admit(scheduler_->context, value)) {
                    session_.cancel();
                    publication_claim_ = {};
                    durable_.clear();
                    status_ = status::scheduler_refused;
                    return false;
                }
                status_ = status::prepared;
                return true;
            } catch (...) {
                status_ = status::preparation_refused;
                return false;
            }
        }
        return session_.shrink(value) && shrink_durable(value);
    }

    projected_capture_staging_session & session() noexcept {
        return session_;
    }
    status terminal() const noexcept { return status_; }
    uint32_t claim_count() const noexcept {
        return publication_claim_.manifests();
    }
    bool claims_ready() const noexcept {
        return publication_claim_.ready();
    }
    bool take_claim_for(
            const vbr_capture_manifest_assembly & assembly,
            llama_vbr_projected_publication_batch_claim & output)
            noexcept {
        if (!assembly || output.ready()) {
            return false;
        }
        size_t source = 0;
        for (const auto & manifest : assembly.manifests()) {
            if (manifest.state != vbr_capture_manifest_state::ready) {
                continue;
            }
            while (source < durable_.size() &&
                   durable_[source].manifest_id < manifest.manifest_id) {
                ++source;
            }
            if (source == durable_.size() ||
                durable_[source].manifest_id != manifest.manifest_id ||
                !publication_claim_.ready()) {
                return false;
            }
            ++source;
        }
        if (source != durable_.size() || !catalog_) {
            return false;
        }
        output = std::move(publication_claim_);
        durable_.clear();
        return true;
    }
    std::vector<llama_vbr_projected_publication_claim>
    take_all_claims_for_test() noexcept {
        std::vector<llama_vbr_projected_publication_claim> output;
        if (catalog_) {
            catalog_->partition_projected_publication_claims(
                std::move(publication_claim_), output);
        }
        durable_.clear();
        return output;
    }

private:
    bool prepare_durable(
            const llama_cache_budget_config & budget,
            const quote & value) {
        if (value.durable.empty()) {
            return value.manifests == 0 || catalog_ == nullptr;
        }
        if (!catalog_ || value.durable.size() != value.manifests) {
            return false;
        }
        durable_ = value.durable;
        std::vector<llama_vbr_projected_publication_request> requests;
        requests.reserve(durable_.size());
        uint64_t previous = 0;
        for (const auto & row : durable_) {
            if (row.manifest_id == 0 || row.manifest_id <= previous ||
                row.accounting.empty()) {
                return false;
            }
            previous = row.manifest_id;
            requests.push_back({
                row.manifest_id, row.accounting, row.reserve_accounting, true,
            });
        }
        publication_claim_ = catalog_->prepare_projected_publication_claims(
            requests, budget);
        return publication_claim_.ready();
    }

    static bool accounting_covers(
            const std::vector<vbr_artifact_portable_accounting_row> & initial,
            const std::vector<vbr_artifact_portable_accounting_row> & shrink) {
        if (shrink.empty()) {
            return false;
        }
        for (size_t index = 0; index < shrink.size(); ++index) {
            const auto & row = shrink[index];
            if (std::any_of(
                    shrink.begin(), shrink.begin() + index,
                    [&](const auto & prior) {
                        return prior.role == row.role &&
                               prior.domain == row.domain;
                    })) {
                return false;
            }
            const auto found = std::find_if(
                initial.begin(), initial.end(), [&](const auto & value) {
                    return value.role == row.role &&
                           value.domain == row.domain;
                });
            if (found == initial.end() || row.logical_bytes == 0 ||
                row.logical_bytes != row.resident_bytes ||
                row.logical_bytes > found->logical_bytes ||
                row.attribution != found->attribution) {
                return false;
            }
        }
        return true;
    }

    bool shrink_durable(const quote & value) noexcept {
        try {
            if (value.durable.size() > durable_.size() ||
                value.durable.size() != value.manifests) {
                return false;
            }
            if (value.durable.empty()) {
                return durable_.empty() && !publication_claim_.ready();
            }
            if (!catalog_ || !publication_claim_.ready()) {
                return false;
            }
            size_t source = 0;
            uint64_t previous = 0;
            std::vector<llama_vbr_projected_publication_request> requests;
            requests.reserve(value.durable.size());
            for (const auto & row : value.durable) {
                if (row.manifest_id == 0 || row.manifest_id <= previous) {
                    return false;
                }
                previous = row.manifest_id;
                while (source < durable_.size() &&
                       durable_[source].manifest_id < row.manifest_id) {
                    ++source;
                }
                if (source == durable_.size() ||
                    durable_[source].manifest_id != row.manifest_id ||
                    !accounting_covers(
                        durable_[source].accounting, row.accounting)) {
                    return false;
                }
                requests.push_back({
                    row.manifest_id, row.accounting,
                    row.reserve_accounting, true,
                });
                ++source;
            }
            if (!catalog_->shrink_projected_publication_claims(
                    publication_claim_, requests)) {
                return false;
            }
            durable_ = value.durable;
            return true;
        } catch (...) {
            return false;
        }
    }

    void * budget_context_ = nullptr;
    server_vbr_artifact_store_config::sample_budget_fn sample_budget_ =
        nullptr;
    const server_vbr_projected_capture_admission * scheduler_ = nullptr;
    llama_vbr_artifact_catalog * catalog_ = nullptr;
    projected_capture_staging_session session_;
    std::vector<vbr_projected_capture_batch_request::pretransfer_quote::
        durable_manifest> durable_;
    llama_vbr_projected_publication_batch_claim publication_claim_;
    bool scheduler_checked_ = false;
    status status_ = status::not_called;
};

} // namespace

bool server_vbr_artifact_store_test_door::import_transport_policy(
        const server_vbr_artifact_store & store,
        vbr_adopt_stage_policy & policy) noexcept {
    policy = {};
    return store.impl_ && store.impl_->bind_import_transport(policy);
}

void server_vbr_artifact_store_test_door::
fail_projected_host_adoption_once(
        server_vbr_artifact_store & store) noexcept {
    if (store.impl_) {
        store.impl_->fail_projected_host_adoption_once = true;
    }
}

bool server_vbr_artifact_store_test_door::projected_staging_lifecycle(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        const vbr_projected_capture_batch_request::pretransfer_quote & initial,
        const vbr_projected_capture_batch_request::pretransfer_quote & shrink,
        const vbr_projected_capture_batch_request::pretransfer_quote & growth,
        projected_staging_lifecycle_result & result) noexcept {
    result = {};
    try {
        const uint64_t baseline = ledger.snapshot().live_ops;
        struct fixed_budget_context {
            const llama_cache_budget_config * budget = nullptr;
            uint32_t * samples = nullptr;
        } budget_context { &budget, &result.budget_samples };
        struct scheduler_context {
            uint32_t calls = 0;
            llama_cache_acct_ledger * ledger = nullptr;
            llama_cache_acct_snapshot * observed = nullptr;
        } scheduler_state { 0, &ledger, &result.scheduler };
        server_vbr_projected_capture_admission scheduler;
        scheduler.context = &scheduler_state;
        scheduler.admit = +[](
                void * opaque,
                const vbr_projected_capture_batch_request::
                    pretransfer_quote &) noexcept {
            auto * state = static_cast<scheduler_context *>(opaque);
            if (!state) {
                return false;
            }
            state->calls++;
            if (state->ledger && state->observed) {
                *state->observed = state->ledger->snapshot();
            }
            return true;
        };
        {
            projected_capture_resource_admission admission(
                ledger, &budget_context,
                +[](void * opaque,
                    llama_cache_budget_config & output) noexcept {
                    const auto * state =
                        static_cast<const fixed_budget_context *>(opaque);
                    if (!state || !state->budget) {
                        return false;
                    }
                    (*state->samples)++;
                    output = *state->budget;
                    return true;
                },
                bindings, &scheduler);
            result.initial_admitted = admission.admit(initial);
            result.initial = ledger.snapshot();
            if (!result.initial_admitted) {
                result.preparation = admission.session().preparation();
                result.scheduler_calls = scheduler_state.calls;
                result.resources = admission.terminal();
                return false;
            }
            result.shrink_admitted = admission.admit(shrink);
            result.shrunk = ledger.snapshot();
            result.growth_refused = !admission.admit(growth);
            result.publication = ledger.snapshot();
            result.live_at_publication =
                admission.session().reserved() &&
                result.publication.live_ops > baseline;
            result.preparation = admission.session().preparation();
            result.resources = admission.terminal();
        }
        result.scheduler_calls = scheduler_state.calls;
        result.after = ledger.snapshot();
        return result.initial_admitted && result.shrink_admitted &&
               result.growth_refused && result.live_at_publication &&
               result.scheduler_calls == 1 &&
               result.after.live_ops == baseline;
    } catch (...) {
        result.after = ledger.snapshot();
        return false;
    }
}

bool server_vbr_artifact_store_test_door::projected_staging_initial(
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        const vbr_projected_capture_batch_request::pretransfer_quote & quote,
        bool scheduler_accept,
        projected_staging_lifecycle_result & result) noexcept {
    result = {};
    try {
        const uint64_t baseline = ledger.snapshot().live_ops;
        struct fixed_budget_context {
            const llama_cache_budget_config * budget = nullptr;
            uint32_t * samples = nullptr;
        } budget_context { &budget, &result.budget_samples };
        struct scheduler_context {
            bool accept = false;
            uint32_t calls = 0;
            llama_cache_acct_ledger * ledger = nullptr;
            llama_cache_acct_snapshot * observed = nullptr;
        } scheduler_state {
            scheduler_accept, 0, &ledger, &result.scheduler,
        };
        server_vbr_projected_capture_admission scheduler;
        scheduler.context = &scheduler_state;
        scheduler.admit = +[](
                void * opaque,
                const vbr_projected_capture_batch_request::
                    pretransfer_quote &) noexcept {
            auto * state = static_cast<scheduler_context *>(opaque);
            if (!state) {
                return false;
            }
            state->calls++;
            if (state->ledger && state->observed) {
                *state->observed = state->ledger->snapshot();
            }
            return state->accept;
        };
        bool accepted = false;
        {
            projected_capture_resource_admission admission(
                ledger, &budget_context,
                +[](void * opaque,
                    llama_cache_budget_config & output) noexcept {
                    auto * state =
                        static_cast<fixed_budget_context *>(opaque);
                    if (!state || !state->budget || !state->samples) {
                        return false;
                    }
                    (*state->samples)++;
                    output = *state->budget;
                    return true;
                },
                bindings, &scheduler);
            accepted = admission.admit(quote);
            result.initial_admitted = accepted;
            result.initial = ledger.snapshot();
            result.preparation = admission.session().preparation();
            result.resources = admission.terminal();
        }
        result.scheduler_calls = scheduler_state.calls;
        result.after = ledger.snapshot();
        return accepted && result.after.live_ops == baseline;
    } catch (...) {
        result.after = ledger.snapshot();
        return false;
    }
}

bool server_vbr_artifact_store_test_door::projected_resource_initial(
        server_vbr_artifact_store & store,
        const vbr_projected_capture_batch_request::pretransfer_quote & quote,
        bool scheduler_accept,
        projected_staging_lifecycle_result & result) noexcept {
    result = {};
    if (!store.impl_ || !store.impl_->ledger ||
        !store.impl_->sample_budget) {
        return false;
    }
    auto & state = *store.impl_;
    try {
        const uint64_t baseline = state.ledger->snapshot().live_ops;
        struct budget_context {
            server_vbr_artifact_store::impl * store = nullptr;
            uint32_t * samples = nullptr;
        } budget_state { &state, &result.budget_samples };
        struct scheduler_context {
            bool accept = false;
            uint32_t calls = 0;
            llama_cache_acct_ledger * ledger = nullptr;
            llama_cache_acct_snapshot * observed = nullptr;
        } scheduler_state {
            scheduler_accept, 0, state.ledger, &result.scheduler,
        };
        server_vbr_projected_capture_admission scheduler;
        scheduler.context = &scheduler_state;
        scheduler.admit = +[](
                void * opaque,
                const vbr_projected_capture_batch_request::
                    pretransfer_quote &) noexcept {
            auto * current = static_cast<scheduler_context *>(opaque);
            if (!current) {
                return false;
            }
            ++current->calls;
            *current->observed = current->ledger->snapshot();
            return current->accept;
        };
        bool accepted = false;
        {
            projected_capture_resource_admission admission(
                *state.ledger, &budget_state,
                +[](void * opaque,
                    llama_cache_budget_config & output) noexcept {
                    auto * current = static_cast<budget_context *>(opaque);
                    if (!current || !current->store || !current->samples) {
                        return false;
                    }
                    ++*current->samples;
                    return current->store->sample_budget(
                        current->store->budget_context, output);
                },
                state.domain_bindings, &scheduler, &state.catalog);
            accepted = admission.admit(quote);
            result.initial_admitted = accepted;
            result.initial = state.ledger->snapshot();
            result.preparation = admission.session().preparation();
            result.resources = admission.terminal();
            if (accepted) {
                auto claims = admission.take_all_claims_for_test();
                result.durable_claims = claims.size();
                result.publication = state.ledger->snapshot();
                result.live_at_publication =
                    admission.session().reserved() &&
                    std::all_of(
                        claims.begin(), claims.end(),
                        [](const auto & claim) { return claim.ready(); }) &&
                    result.publication.live_ops > baseline;
            }
        }
        result.scheduler_calls = scheduler_state.calls;
        result.after = state.ledger->snapshot();
        return accepted && result.after.live_ops == baseline;
    } catch (...) {
        result.after = state.ledger->snapshot();
        return false;
    }
}

bool server_vbr_artifact_store_test_door::publish_after_capture_checkpoint(
        server_vbr_artifact_store & store,
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        std::vector<server_vbr_projected_host_publish_result> & output,
        const server_vbr_projected_capture_admission & admission,
        server_vbr_projected_host_capture_diagnostics & diagnostics) noexcept {
    return store.publish_captured_projected_host_batch(
        assembly, std::move(publications), {}, output, &admission,
        diagnostics);
}

bool server_vbr_artifact_store_observe_empty_accounting(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) noexcept {
    try {
        if (!llama_cache_acct_resource_domain_valid(domain) ||
            (domain.residency !=
                 llama_cache_acct_residency::device &&
             domain.residency !=
                 llama_cache_acct_residency::pinned_host &&
             domain.residency !=
                 llama_cache_acct_residency::pageable_host)) {
            return false;
        }

        const auto before = ledger.snapshot();
        if (before.completeness_manifest !=
                llama_cache_acct_known::known) {
            return false;
        }
        if (std::none_of(
                before.completeness.begin(),
                before.completeness.end(),
                [&](const llama_cache_acct_completeness_row & row) {
                    return row.domain == domain &&
                           row.state !=
                               llama_cache_acct_known::
                                   unavailable;
                })) {
            return false;
        }

        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            if (!capture_capacity_category_applies(
                    category, domain, false)) {
                continue;
            }
            const auto cell = std::find_if(
                before.cells.begin(), before.cells.end(),
                [&](const llama_cache_acct_cell_row & row) {
                    return row.category == category &&
                           row.domain == domain;
                });
            if (cell == before.cells.end()) {
                return false;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                const auto value =
                    cell->cell.measures[size_t(measure)];
                if (value.state ==
                        llama_cache_acct_known::unavailable ||
                    (value.state == llama_cache_acct_known::known &&
                     value.value != 0)) {
                    return false;
                }
            }
        }

        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            if (!capture_capacity_category_applies(
                    category, domain, false)) {
                continue;
            }
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated,
                    llama_cache_acct_measure::reserved }) {
                ledger.gauge_set(category, domain, measure, 0);
            }
        }
        const auto after = ledger.snapshot();
        if (after.faults_invalid_transition !=
                before.faults_invalid_transition ||
            after.faults_overflow != before.faults_overflow ||
            after.faults_allocation !=
                before.faults_allocation) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_store_verify_accounting(
        llama_cache_acct_ledger & ledger,
        const std::vector<llama_cache_acct_resource_domain> &
            domains) noexcept {
    try {
        const auto snapshot = ledger.snapshot();
        if (snapshot.completeness_manifest !=
                llama_cache_acct_known::known ||
            domains.empty()) {
            return false;
        }
        for (size_t i = 0; i < domains.size(); ++i) {
            const auto & domain = domains[i];
            if (!llama_cache_acct_resource_domain_valid(domain) ||
                std::find(
                    domains.begin(), domains.begin() + i,
                    domain) != domains.begin() + i) {
                return false;
            }
            bool has_requirement = false;
            for (const auto & row : snapshot.completeness) {
                if (row.domain == domain) {
                    has_requirement = true;
                    if (row.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
            }
            if (!has_requirement) {
                return false;
            }
        }
        for (size_t i = 0;
             i < size_t(llama_cache_acct_category::_count);
             ++i) {
            const auto category =
                llama_cache_acct_category(i);
            const auto classification =
                llama_cache_budget_classify(category);
            for (const auto & domain : domains) {
                if (!capture_capacity_category_applies(
                        category, domain, true)) {
                    continue;
                }
                const auto cell = std::find_if(
                    snapshot.cells.begin(), snapshot.cells.end(),
                    [&](const llama_cache_acct_cell_row & row) {
                        return row.category == category &&
                               row.domain == domain;
                    });
                if (cell == snapshot.cells.end() ||
                    cell->certification !=
                        llama_cache_acct_known::known) {
                    return false;
                }
                const auto resident =
                    cell->cell.measures[size_t(
                        llama_cache_acct_measure::
                            resident_allocated)];
                if (resident.state !=
                        llama_cache_acct_known::known) {
                    return false;
                }
                if (classification.mode ==
                        llama_cache_budget_accounting_mode::
                            transactional) {
                    const auto reserved =
                        cell->cell.measures[size_t(
                            llama_cache_acct_measure::reserved)];
                    if (reserved.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
                if (classification.scope !=
                        llama_cache_budget_residency_scope::
                            device) {
                    const auto logical =
                        cell->cell.measures[size_t(
                            llama_cache_acct_measure::
                                logical_payload)];
                    if (logical.state !=
                            llama_cache_acct_known::known) {
                        return false;
                    }
                }
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool server_vbr_artifact_store_configure_pinned_accounting(
        llama_cache_acct_ledger & ledger,
        const llama_cache_acct_resource_domain & domain) noexcept {
    const auto canonical =
        llama_cache_acct_resource_domain::non_device(
            llama_cache_acct_residency::pinned_host);
    if (domain != canonical ||
        !server_vbr_artifact_store_observe_empty_accounting(
            ledger, domain) ||
        !ledger.certify_complete(
            domain,
            llama_cache_acct_producer::retention_sidecar)) {
        return false;
    }
    return server_vbr_artifact_store_verify_accounting(
        ledger, { domain });
}

std::unique_ptr<server_vbr_artifact_store>
server_vbr_artifact_store::create(
        const server_vbr_artifact_store_config & config,
        server_vbr_artifact_capture_status & status,
        server_vbr_artifact_store_create_diagnostics * diagnostics) noexcept {
    status = server_vbr_artifact_capture_status::unavailable;
    server_vbr_artifact_store_create_diagnostics observed;
    observed.requested_ring_bytes = config.ring_bytes;
    observed.chunk_bytes = config.chunk_bytes;
    observed.lane_count = config.lanes.size();
    observed.attention_children = config.attention_children;
    const auto fail =
        [&](server_vbr_artifact_store_create_failure failure) {
            observed.failure = failure;
            if (diagnostics) {
                *diagnostics = observed;
            }
        };
    try {
        if (config.ledger == nullptr) {
            fail(server_vbr_artifact_store_create_failure::ledger_missing);
            return nullptr;
        }
        if (config.sample_budget == nullptr) {
            fail(server_vbr_artifact_store_create_failure::
                budget_sampler_missing);
            return nullptr;
        }
        if (config.topologies.empty()) {
            fail(server_vbr_artifact_store_create_failure::
                topology_missing);
            return nullptr;
        }
        if (config.pool_bindings.empty()) {
            fail(server_vbr_artifact_store_create_failure::
                pool_binding_missing);
            return nullptr;
        }
        if (config.lanes.empty()) {
            fail(server_vbr_artifact_store_create_failure::lane_missing);
            return nullptr;
        }
        if (config.attention_children == 0) {
            fail(server_vbr_artifact_store_create_failure::
                attention_child_missing);
            return nullptr;
        }
        if (config.ring_bytes == 0 ||
            config.ring_bytes >
                VBR_CAPTURE_PINNED_RING_MAX_BYTES) {
            fail(server_vbr_artifact_store_create_failure::
                ring_size_invalid);
            return nullptr;
        }
        if (config.chunk_bytes == 0 ||
            config.lanes.size() >
                std::numeric_limits<uint64_t>::max()/2 ||
            uint64_t(config.lanes.size()*2) >
                std::numeric_limits<uint64_t>::max() /
                    uint64_t(config.chunk_bytes)) {
            fail(server_vbr_artifact_store_create_failure::
                chunk_size_invalid);
            return nullptr;
        }
        auto state = std::make_unique<impl>(*config.ledger);
        state->topologies = config.topologies;
        state->pool_bindings = config.pool_bindings;
        state->pinned_domain = config.pinned_domain;
        state->import_ring_bytes = config.ring_bytes;
        state->import_chunk_bytes = config.chunk_bytes;
        state->budget_context = config.budget_context;
        state->sample_budget = config.sample_budget;
        state->n_attention_children = config.attention_children;
        if (!state->catalog.bind_topologies(
                config.topologies, state->domain_bindings)) {
            fail(server_vbr_artifact_store_create_failure::topology_missing);
            return nullptr;
        }
        state->policy_bindings = state->domain_bindings;
        state->policy_bindings.push_back({
            UINT32_MAX, UINT16_MAX,
            llama_cache_acct_resource_domain::non_device(
                llama_cache_acct_residency::pageable_host),
        });
        state->policy_bindings.push_back({
            UINT32_MAX, UINT16_MAX, state->pinned_domain,
        });
        state->h2d_lanes.resize(config.lanes.size());
        std::vector<bool> lane_bound(config.lanes.size(), false);
        for (const auto & pool : config.pool_bindings) {
            if (pool.lane >= config.lanes.size()) {
                fail(server_vbr_artifact_store_create_failure::lane_missing);
                return nullptr;
            }
            const auto domain = std::find_if(
                state->domain_bindings.begin(),
                state->domain_bindings.end(),
                [&](const llama_vbr_artifact_domain_binding & binding) {
                    return binding.topology_index == pool.topology_index &&
                           binding.device_ordinal == pool.device_ordinal;
                });
            if (domain == state->domain_bindings.end()) {
                fail(server_vbr_artifact_store_create_failure::pool_binding_missing);
                return nullptr;
            }
            const auto & capture_lane = config.lanes[pool.lane];
            auto & lane = state->h2d_lanes[pool.lane];
            if (lane_bound[pool.lane] &&
                (lane.domain != domain->domain ||
                 lane.device != capture_lane.device ||
                 lane.backend != capture_lane.backend)) {
                fail(server_vbr_artifact_store_create_failure::lane_missing);
                return nullptr;
            }
            lane = {
                domain->domain, capture_lane.device,
                capture_lane.backend, capture_lane.force_synchronous,
            };
            lane_bound[pool.lane] = true;
        }
        if (std::find(lane_bound.begin(), lane_bound.end(), false) !=
                lane_bound.end()) {
            fail(server_vbr_artifact_store_create_failure::lane_missing);
            return nullptr;
        }
        std::random_device random;
        state->nonce = (uint64_t(random()) << 32) ^ random();
        if (state->nonce == 0) {
            state->nonce = 1;
        }

        llama_cache_budget_config budget;
        if (!state->sample_budget(state->budget_context, budget)) {
            fail(server_vbr_artifact_store_create_failure::
                budget_sample_failed);
            return nullptr;
        }
        vbr_capture_ring_accounting accounting {
            config.ledger, config.pinned_domain, &budget,
        };
        const uint64_t minimum_ring_bytes =
            uint64_t(config.lanes.size()*2) *
            uint64_t(config.chunk_bytes);
        uint64_t attempt = config.ring_bytes;
        for (;;) {
            observed.attempted_ring_bytes = attempt;
            auto core = std::shared_ptr<vbr_bounded_pinned_ring_core>(
                vbr_bounded_pinned_ring_core::create(
                    config.lanes, attempt, config.chunk_bytes,
                    &accounting, observed.ring_failure));
            if (core) {
                state->ring = vbr_pinned_chunk_ring::attach(core);
                state->import_ring = vbr_h2d_chunk_ring::attach(
                    std::move(core), state->h2d_lanes);
                observed.ring_status = state->ring && state->import_ring
                    ? vbr_capture_stream_status::ok
                    : vbr_capture_stream_status::internal_error;
                if (!state->ring || !state->import_ring) {
                    observed.ring_failure =
                        vbr_capture_ring_create_failure::internal_error;
                }
            } else {
                observed.ring_status =
                    vbr_capture_ring_failure_status(
                        observed.ring_failure);
            }
            if (state->ring && state->import_ring) {
                break;
            }
            state->ring.reset();
            state->import_ring.reset();
            // Pinned allocation pressure is recoverable without weakening
            // the ring protocol: two chunks per physical lane are sufficient
            // for bounded producer/consumer overlap. Other failures are
            // evidence/configuration failures and remain fail closed.
            if (observed.ring_failure !=
                    vbr_capture_ring_create_failure::
                        host_buffer_allocation_failed ||
                attempt <= minimum_ring_bytes) {
                break;
            }
            uint64_t next =
                (attempt/2/uint64_t(config.chunk_bytes)) *
                uint64_t(config.chunk_bytes);
            next = std::max(next, minimum_ring_bytes);
            if (next >= attempt) {
                break;
            }
            attempt = next;
        }
        if (!state->ring) {
            status = observed.ring_status ==
                        vbr_capture_stream_status::accounting_refused
                ? server_vbr_artifact_capture_status::admission_refused
                : observed.ring_status ==
                        vbr_capture_stream_status::internal_error
                    ? server_vbr_artifact_capture_status::internal_error
                    : server_vbr_artifact_capture_status::unavailable;
            fail(server_vbr_artifact_store_create_failure::
                ring_create_failed);
            return nullptr;
        }
        observed.constructed_ring_bytes =
            state->ring->capacity_bytes();
        state->import_ring_bytes = observed.constructed_ring_bytes;
        state->counters.pinned_bytes =
            observed.constructed_ring_bytes;
        status = server_vbr_artifact_capture_status::ok;
        if (diagnostics) {
            *diagnostics = observed;
        }
        return std::unique_ptr<server_vbr_artifact_store>(
            new server_vbr_artifact_store(std::move(state)));
    } catch (...) {
        status = server_vbr_artifact_capture_status::internal_error;
        fail(server_vbr_artifact_store_create_failure::internal_error);
        return nullptr;
    }
}

server_vbr_artifact_capture_output server_vbr_artifact_store::capture(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        const std::string & tenant_key) noexcept {
    return capture_impl(memory, std::move(request), tenant_key);
}

server_vbr_artifact_capture_output
server_vbr_artifact_store::prepare_host_payload(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        server_vbr_explicit_host_capture & operation) noexcept {
    operation.reset();
    server_vbr_artifact_capture_output output;
    impl_->counters.requested++;
    try {
        llama_cache_budget_config budget;
        if (!impl_->sample_budget(impl_->budget_context, budget)) {
            output.status = server_vbr_artifact_capture_status::unavailable;
            impl_->counters.unavailable++;
            return output;
        }
        request.ring = impl_->ring.get();
        request.topologies = impl_->topologies;
        request.pool_bindings = impl_->pool_bindings;
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
        };
        request.representation_context = &representation_policy;
        request.representation_identity =
            vbr_explicit_capture_representation_identity;

        vbr_explicit_capture_accounting accounting;
        accounting.budget = &budget;
        accounting.context = &impl_->catalog;
        accounting.prepare = [](
                void * context,
                const vbr_artifact_package & package) noexcept {
            return static_cast<llama_vbr_artifact_catalog *>(context)
                ->prepare_capture_package(package);
        };
        auto prepared = std::make_unique<
            server_vbr_explicit_host_capture::impl>();
        const auto result = vbr_prepare_explicit_manifest(
            memory, std::move(request), impl_->catalog, accounting,
            prepared->capture);
        copy_capture_result(result, output);
        if (result.status != vbr_explicit_capture_status::ok ||
            !prepared->capture.ready_for_transfer()) {
            if (result.status != vbr_explicit_capture_status::_count) {
                impl_->counters.capture_outcomes[size_t(result.status)]++;
            }
            if (output.status ==
                    server_vbr_artifact_capture_status::admission_refused) {
                impl_->counters.refused++;
            } else if (output.status ==
                    server_vbr_artifact_capture_status::internal_error) {
                impl_->counters.internal_error++;
            } else {
                impl_->counters.unavailable++;
            }
            return output;
        }
        operation.impl_ = std::move(prepared);
        operation.impl_->output = output;
        return output;
    } catch (...) {
        output.status = server_vbr_artifact_capture_status::internal_error;
        impl_->counters.internal_error++;
        return output;
    }
}

server_vbr_artifact_capture_output
server_vbr_artifact_store::transfer_host_payload(
        server_vbr_explicit_host_capture & operation) noexcept {
    server_vbr_artifact_capture_output output;
    if (!operation.ready_for_transfer()) {
        return output;
    }
    const auto result =
        vbr_transfer_explicit_manifest(operation.impl_->capture);
    copy_capture_result(result, output);
    operation.impl_->output = output;
    return output;
}

server_vbr_artifact_capture_output
server_vbr_artifact_store::publish_host_payload(
        server_vbr_explicit_host_capture & operation,
        std::shared_ptr<const server_prompt_cache_vbr_payload> & payload)
        noexcept {
    payload.reset();
    server_vbr_artifact_capture_output output;
    if (!operation.ready_for_publication()) {
        if (!operation.impl_) {
            return output;
        }
        output = operation.impl_->output;
        if (output.library_status != vbr_explicit_capture_status::_count) {
            impl_->counters.capture_outcomes[
                size_t(output.library_status)]++;
        }
        if (output.status ==
                server_vbr_artifact_capture_status::admission_refused) {
            impl_->counters.refused++;
        } else if (output.status ==
                server_vbr_artifact_capture_status::internal_error) {
            impl_->counters.internal_error++;
        } else {
            impl_->counters.unavailable++;
        }
        operation.reset();
        return output;
    }
    try {
        const auto result =
            vbr_publish_explicit_manifest(operation.impl_->capture);
        copy_capture_result(result, output);
        if (result.status != vbr_explicit_capture_status::_count) {
            impl_->counters.capture_outcomes[size_t(result.status)]++;
        }
        if (result.status != vbr_explicit_capture_status::ok ||
            result.sink.reference_artifact.v == 0) {
            if (result.status == vbr_explicit_capture_status::ok) {
                output.status =
                    server_vbr_artifact_capture_status::internal_error;
            }
            if (output.status ==
                    server_vbr_artifact_capture_status::internal_error) {
                impl_->counters.internal_error++;
            } else {
                impl_->counters.unavailable++;
            }
            operation.reset();
            return output;
        }
        std::vector<llama_cache_acct_artifact_id> references {
            result.sink.reference_artifact,
        };
        std::vector<vbr_artifact_package_view> packages;
        if (!impl_->catalog.claim_fresh_host_batch(
                references, packages) || packages.size() != 1) {
            (void) impl_->catalog.discard_unowned_reference(
                result.sink.reference_artifact);
            output.status = server_vbr_artifact_capture_status::internal_error;
            impl_->counters.internal_error++;
            operation.reset();
            return output;
        }
        payload = server_prompt_cache_vbr_payload::adopt(
            std::move(packages.front()));
        if (!payload) {
            output.status = server_vbr_artifact_capture_status::internal_error;
            impl_->counters.internal_error++;
            operation.reset();
            return output;
        }
        output.consistency = vbr_artifact_consistency_kind::capture_exact;
        impl_->counters.exact_published++;
        impl_->counters.payload_bytes += result.payload_bytes;
        impl_->counters.stash_bytes += result.stash_bytes;
        impl_->counters.companion_bytes += result.companion_bytes;
        impl_->counters.chunks += result.chunks;
        impl_->counters.event_completions += result.event_completions;
        impl_->counters.synchronous_fallbacks +=
            result.synchronous_fallbacks;
        impl_->counters.backpressure_waits += result.backpressure_waits;
        if (output.dedup) {
            impl_->counters.dedup_hits++;
        } else {
            impl_->counters.dedup_misses++;
        }
        impl_->counters.staging_overlap_refusals =
            impl_->catalog.snapshot().staging_overlap_refusals;
        operation.reset();
        return output;
    } catch (...) {
        output.status = server_vbr_artifact_capture_status::internal_error;
        impl_->counters.internal_error++;
        operation.reset();
        return output;
    }
}

server_vbr_artifact_capture_output server_vbr_artifact_store::capture_impl(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        const std::string & tenant_key) noexcept {
    server_vbr_artifact_capture_output output;
    impl_->counters.requested++;
    try {
        if (tenant_key.empty()) {
            output.status =
                server_vbr_artifact_capture_status::unauthorized;
            impl_->counters.refused++;
            return output;
        }
        llama_cache_budget_config budget;
        if (!impl_->sample_budget(impl_->budget_context, budget)) {
            output.status =
                server_vbr_artifact_capture_status::unavailable;
            impl_->counters.unavailable++;
            return output;
        }
        request.ring = impl_->ring.get();
        request.topologies = impl_->topologies;
        request.pool_bindings = impl_->pool_bindings;
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
        };
        request.representation_context = &representation_policy;
        request.representation_identity =
            vbr_explicit_capture_representation_identity;

        vbr_explicit_capture_accounting accounting;
        accounting.budget = &budget;
        accounting.context = &impl_->catalog;
        accounting.prepare = [](
                void * context,
                const vbr_artifact_package & package) noexcept {
            return static_cast<llama_vbr_artifact_catalog *>(context)
                ->prepare_capture_package(package);
        };
        const auto result = vbr_capture_explicit_manifest(
            memory, request, impl_->catalog, accounting);
        output.library_status = result.status;
        output.phase = result.phase;
        output.inner_stream_status =
            result.inner_stream_status;
        output.generation_failure =
            result.generation_failure;
        output.size_failure =
            result.size_failure;
        output.begin_diagnostics =
            result.begin_diagnostics;
        output.pretransfer = result.pretransfer;
        if (result.status != vbr_explicit_capture_status::_count) {
            impl_->counters.capture_outcomes[size_t(result.status)]++;
        }
        output.status = map_status(result.status);
        output.controllers = result.controllers;
        output.units = result.units;
        output.companions = result.companions;
        output.companion_failure_index = result.companion_failure_index;
        output.companion_failure_kind = result.companion_failure_kind;
        output.payload_bytes = result.payload_bytes;
        output.stash_bytes = result.stash_bytes;
        output.companion_bytes = result.companion_bytes;
        output.chunks = result.chunks;
        output.backpressure_waits = result.backpressure_waits;
        output.event_completions = result.event_completions;
        output.synchronous_fallbacks = result.synchronous_fallbacks;
        if (result.status != vbr_explicit_capture_status::ok) {
            if (output.status ==
                    server_vbr_artifact_capture_status::admission_refused) {
                impl_->counters.refused++;
            } else if (output.status ==
                    server_vbr_artifact_capture_status::internal_error) {
                impl_->counters.internal_error++;
            } else {
                impl_->counters.unavailable++;
            }
            return output;
        }
        if (result.sink.reference_artifact.v == 0) {
            output.status =
                server_vbr_artifact_capture_status::internal_error;
            impl_->counters.internal_error++;
            return output;
        }
        const auto after = impl_->catalog.snapshot();
        output.dedup = result.sink.adopted;
        output.reference = opaque_reference(
            impl_->nonce, impl_->next_reference++,
            result.sink.reference_artifact, tenant_key);
        if (!impl_->references.publish(
                output.reference, tenant_key,
                result.sink.reference_artifact)) {
            (void) impl_->catalog.retire(
                result.sink.reference_artifact);
            output.reference.clear();
            output.status =
                server_vbr_artifact_capture_status::internal_error;
            impl_->counters.internal_error++;
            return output;
        }
        output.consistency = vbr_artifact_consistency_kind::capture_exact;
        impl_->counters.exact_published++;
        impl_->counters.payload_bytes += result.payload_bytes;
        impl_->counters.stash_bytes += result.stash_bytes;
        impl_->counters.companion_bytes += result.companion_bytes;
        impl_->counters.chunks += result.chunks;
        impl_->counters.event_completions +=
            result.event_completions;
        impl_->counters.synchronous_fallbacks +=
            result.synchronous_fallbacks;
        impl_->counters.backpressure_waits += result.backpressure_waits;
        if (output.dedup) {
            impl_->counters.dedup_hits++;
        } else {
            impl_->counters.dedup_misses++;
        }
        impl_->counters.staging_overlap_refusals =
            after.staging_overlap_refusals;
        return output;
    } catch (...) {
        output.status =
            server_vbr_artifact_capture_status::internal_error;
        impl_->counters.internal_error++;
        return output;
    }
}

bool server_vbr_artifact_store::publish_projected_host_batch(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        std::vector<server_vbr_projected_host_publish_result> & output,
        server_vbr_projected_host_publish_diagnostics * diagnostics) noexcept {
    return publish_projected_host_batch_impl(
        assembly, std::move(publications), {}, {}, output, diagnostics);
}

bool server_vbr_artifact_store::publish_projected_host_batch_impl(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        std::vector<llama_vbr_projected_publication_claim> && claims,
        llama_vbr_projected_publication_batch_claim && batch_claim,
        std::vector<server_vbr_projected_host_publish_result> & output,
        server_vbr_projected_host_publish_diagnostics * diagnostics) noexcept {
    output.clear();
    if (diagnostics) {
        *diagnostics = {};
    }
    const bool capacity_admitted = !claims.empty() || batch_claim.ready();
    if (!impl_ || (!capacity_admitted && !impl_->sample_budget)) {
        return false;
    }

    llama_cache_budget_config budget;
    if (!capacity_admitted &&
        !impl_->sample_budget(impl_->budget_context, budget)) {
        return false;
    }

    const size_t manifest_count = assembly.manifests().size();
    std::vector<vbr_projected_manifest_publish_result> published;
    std::vector<llama_cache_acct_artifact_id> handoff_references;
    std::vector<size_t> handoff_indices;
    std::vector<vbr_artifact_package_view> handoff_packages;
    // Allocate every cardinality-sized adapter arena before the catalog can
    // commit a row. The catalog handoff may still reject allocation or
    // structure atomically; compensating discard is allocation-free.
    try {
        output.resize(manifest_count);
        published.reserve(manifest_count);
        handoff_references.resize(manifest_count);
        handoff_indices.resize(manifest_count);
        handoff_packages.reserve(manifest_count);
    } catch (...) {
        output.clear();
        return false;
    }

    server_vbr_projected_host_publish_diagnostics measured;
    bool catalog_published = false;
    if (batch_claim.ready()) {
        catalog_published = impl_->catalog.publish_projected_batch_claimed(
            assembly, std::move(publications), std::move(batch_claim),
            published, &measured.catalog);
    } else if (!claims.empty()) {
        catalog_published = impl_->catalog.publish_projected_batch_claimed(
            assembly, std::move(publications), std::move(claims),
            published, &measured.catalog);
    } else {
        catalog_published = impl_->catalog.publish_projected_batch(
            assembly, std::move(publications), budget,
            published, &measured.catalog);
    }
    if (!catalog_published) {
        output.clear();
        return false;
    }

    const auto is_published = [](vbr_projected_manifest_publish_status status) {
        return status == vbr_projected_manifest_publish_status::published ||
               status == vbr_projected_manifest_publish_status::adopted;
    };
    if (published.size() != output.size()) {
        for (const auto & row : published) {
            if (is_published(row.status) &&
                row.publication.reference_artifact.v != 0) {
                (void) impl_->catalog.discard_unowned_reference(
                    row.publication.reference_artifact);
            }
        }
        output.clear();
        return false;
    }

    size_t handoff_count = 0;
    size_t injected_failure = SIZE_MAX;
    for (size_t i = 0; i < published.size(); ++i) {
        const auto & row = published[i];
        auto & terminal = output[i];
        terminal.manifest_id = row.manifest_id;
        terminal.status = row.status;
        if (!is_published(row.status)) {
            continue;
        }

        const auto reference = row.publication.reference_artifact;
        if (impl_->fail_projected_host_adoption_once &&
            injected_failure == SIZE_MAX) {
            injected_failure = i;
            impl_->fail_projected_host_adoption_once = false;
            continue;
        }
        if (reference.v == 0) {
            terminal.status =
                vbr_projected_manifest_publish_status::internal_error;
            measured.postpublish_retirements++;
            continue;
        }
        handoff_references[handoff_count] = reference;
        handoff_indices[handoff_count] = i;
        ++handoff_count;
    }
    handoff_references.resize(handoff_count);
    handoff_indices.resize(handoff_count);

    const bool handed_off = handoff_count == 0 ||
        impl_->catalog.claim_fresh_host_batch(
            handoff_references, handoff_packages);
    for (size_t i = 0; i < handoff_count; ++i) {
        auto & terminal = output[handoff_indices[i]];
        if (handed_off) {
            terminal.payload = server_prompt_cache_vbr_payload::adopt(
                std::move(handoff_packages[i]));
        }
        if (terminal.payload) {
            measured.host_payloads_retained++;
            continue;
        }

        // A successfully claimed view retires itself on reset; a failed atomic
        // handoff leaves the reference unowned for allocation-free discard.
        if (handed_off) {
            handoff_packages[i].reset();
        } else {
            (void) impl_->catalog.discard_unowned_reference(
                handoff_references[i]);
        }
        terminal.status =
            vbr_projected_manifest_publish_status::internal_error;
        measured.postpublish_retirements++;
    }
    if (injected_failure != SIZE_MAX) {
        auto & terminal = output[injected_failure];
        (void) impl_->catalog.discard_unowned_reference(
            published[injected_failure].publication.reference_artifact);
        terminal.status =
            vbr_projected_manifest_publish_status::internal_error;
        measured.postpublish_retirements++;
    }
    if (diagnostics) {
        *diagnostics = measured;
    }
    return true;
}

bool server_vbr_artifact_store::capture_projected_host_batch(
        llama_memory_i & memory,
        std::vector<vbr_projected_capture_manifest_request> manifests,
        uint64_t max_packed_bytes,
        std::vector<server_vbr_projected_host_publish_result> & output,
        const server_vbr_projected_capture_admission * admission,
        server_vbr_projected_host_capture_diagnostics * diagnostics) noexcept {
    output.clear();
    if (diagnostics) {
        *diagnostics = {};
    }
    if (!impl_ || !impl_->ring || manifests.empty() ||
        max_packed_bytes == 0) {
        return false;
    }

    try {
        projected_capture_resource_admission staging(
            *impl_->ledger, impl_->budget_context, impl_->sample_budget,
            impl_->domain_bindings, admission, &impl_->catalog);
        vbr_projected_capture_batch_request request;
        request.idle_decode_thread = true;
        request.max_packed_bytes = max_packed_bytes;
        if (admission) {
            request.frontier = admission->frontier;
        }
        request.manifests = std::move(manifests);
        request.ring = impl_->ring.get();
        request.topologies = impl_->topologies;
        request.pool_bindings = impl_->pool_bindings;
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
        };
        request.representation_context = &representation_policy;
        request.representation_identity =
            vbr_explicit_capture_representation_identity;
        if (admission && admission->prepare) {
            request.pretransfer_prepare_context = admission->context;
            request.pretransfer_prepare = admission->prepare;
        }
        request.pretransfer_context = &staging;
        request.pretransfer_admit = +[](
                void * opaque,
                const vbr_projected_capture_batch_request::
                    pretransfer_quote & quote) noexcept {
            auto * state =
                static_cast<projected_capture_resource_admission *>(opaque);
            return state && state->admit(quote);
        };
        request.pretransfer_shrink = request.pretransfer_admit;
        if (admission && admission->continue_capture) {
            request.continue_context = admission->context;
            request.continue_transfer = admission->continue_capture;
        }

        auto captured = vbr_capture_projected_batch(memory, request);
        server_vbr_projected_host_capture_diagnostics measured;
        measured.capture_status = captured.status;
        measured.capture_phase = captured.phase;
        measured.inner_stream_status = captured.inner_stream_status;
        measured.source_namespace = captured.source_namespace;
        measured.first_available_manifest_id =
            captured.first_available_manifest_id;
        measured.union_cells = captured.union_cells;
        measured.planned_packed_bytes = captured.planned_packed_bytes;
        measured.frontier_status = captured.frontier_status;
        measured.requested_frontier_tokens =
            captured.requested_frontier_tokens;
        measured.selected_frontier_tokens =
            captured.selected_frontier_tokens;
        measured.selected_frontier_next_position =
            captured.selected_frontier_next_position;
        measured.frontier_survey_cells = captured.frontier_survey_cells;
        measured.frontier_survey_calls = captured.frontier_survey_calls;
        measured.frontier_recapture_calls = captured.frontier_recapture_calls;
        measured.size_pass_calls = captured.size_pass_calls;
        measured.projection_calls = captured.projection_calls;
        measured.unit_transfer_calls = captured.unit_transfer_calls;
        measured.transferred_units = captured.transferred_units;
        measured.companion_d2h_bytes = captured.companion_d2h_bytes;
        measured.companion_d2h_reads = captured.companion_d2h_reads;
        measured.ring_operation_attempts =
            captured.ring_operation_attempts;
        measured.ring_operation_acquires =
            captured.ring_operation_acquires;
        measured.ring_operation_refusals =
            captured.ring_operation_refusals;
        measured.resources = staging.terminal();
        measured.durable_claims = staging.claim_count();
        measured.durable_reserved = staging.claims_ready();
        auto & staging_reservation = staging.session();
        if (staging_reservation.preparation_required()) {
            measured.staging_prepare_status =
                staging_reservation.preparation().status;
            measured.staging_admission_status =
                staging_reservation.preparation().admission_status;
            measured.staging_failed_leaf =
                staging_reservation.preparation().failed_leaf;
            measured.staging_reserved = staging_reservation.reserved();
        } else {
            measured.staging_prepare_status =
                llama_cache_prepare_status::prepared;
            measured.staging_admission_status =
                llama_cache_admission_status::admitted;
            measured.staging_failed_leaf = SIZE_MAX;
            measured.staging_reserved = false;
        }
        measured.transfer = captured.transfer;
        if (captured.status != vbr_explicit_capture_status::ok ||
            !captured.assembly ||
            captured.publications.size() !=
                captured.assembly.manifests().size()) {
            if (diagnostics) {
                *diagnostics = measured;
            }
            return false;
        }
        llama_vbr_projected_publication_batch_claim publication_claim;
        if (!staging.take_claim_for(
                captured.assembly, publication_claim)) {
            if (diagnostics) {
                *diagnostics = measured;
            }
            return false;
        }
        const bool published = publish_captured_projected_host_batch(
            captured.assembly, std::move(captured.publications),
            std::move(publication_claim), output, admission, measured);
        if (diagnostics) {
            *diagnostics = measured;
        }
        return published;
    } catch (...) {
        output.clear();
        return false;
    }
}

bool server_vbr_artifact_store::publish_captured_projected_host_batch(
        const vbr_capture_manifest_assembly & assembly,
        std::vector<vbr_projected_manifest_publication> && publications,
        llama_vbr_projected_publication_batch_claim && claim,
        std::vector<server_vbr_projected_host_publish_result> & output,
        const server_vbr_projected_capture_admission * admission,
        server_vbr_projected_host_capture_diagnostics & diagnostics) noexcept {
    // The final D2H checkpoint is not the publication linearization point.
    // Recheck the scheduler-owned session after assembly and immediately
    // before the catalog transaction. An arrival observed at this checkpoint
    // prevents host rows; a later racing arrival still preserves the live
    // source at scheduler publication.
    if (admission && admission->continue_capture &&
        !admission->continue_capture(admission->context)) {
        output.clear();
        diagnostics.capture_status = vbr_explicit_capture_status::cancelled;
        diagnostics.capture_phase = vbr_explicit_capture_phase::publication;
        diagnostics.inner_stream_status =
            vbr_capture_stream_status::cancelled;
        return false;
    }
    return publish_projected_host_batch_impl(
        assembly, std::move(publications), {}, std::move(claim), output,
        &diagnostics.publication);
}

server_vbr_artifact_import_output server_vbr_artifact_store::import(
        server_vbr_artifact_import_request request) noexcept {
    server_vbr_artifact_import_output output;
    impl_->counters.imports_requested++;
    const auto fail = [&](server_vbr_artifact_import_status status,
                          uint64_t & counter) {
        output.status = status;
        ++counter;
        return output;
    };
    try {
        llama_cache_acct_artifact_id artifact;
        if (!impl_->references.authorize(
                request.reference, request.tenant_key, artifact)) {
            return fail(server_vbr_artifact_import_status::not_found,
                        impl_->counters.imports_not_found);
        }
        if (!request.memory || request.destination < 0 ||
            request.execution_identity.empty() ||
            request.adapter_config_identity.empty() ||
            !request.prepare_publish || !request.publish) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }

        vbr_artifact_package_view package;
        const auto resolved = impl_->catalog.resolve_reference(
            artifact, package);
        if (resolved != vbr_artifact_resolve_status::ok || !package) {
            const auto status = resolved == vbr_artifact_resolve_status::not_found
                ? server_vbr_artifact_import_status::not_found
                : server_vbr_artifact_import_status::unavailable;
            return status == server_vbr_artifact_import_status::not_found
                ? fail(status, impl_->counters.imports_not_found)
                : fail(status, impl_->counters.imports_unavailable);
        }
        return import_package(std::move(request), package);
    } catch (...) {
        return fail(server_vbr_artifact_import_status::internal_error,
                    impl_->counters.imports_unavailable);
    }
}

server_vbr_artifact_import_output
server_vbr_artifact_store::import_host_payload(
        server_vbr_artifact_import_target request,
        std::shared_ptr<const server_prompt_cache_vbr_payload> payload)
        noexcept {
    server_vbr_artifact_import_output output;
    impl_->counters.imports_requested++;
    if (!payload || !payload->retirement_owned() ||
        !payload->accounted_by(impl_->ledger) || !payload->package()) {
        output.status = server_vbr_artifact_import_status::unavailable;
        impl_->counters.imports_unavailable++;
        return output;
    }
    if (!impl_->catalog.owns_host_package(payload->package())) {
        output.status = server_vbr_artifact_import_status::not_found;
        impl_->counters.imports_not_found++;
        return output;
    }
    impl_->counters.host_imports_authenticated++;
    auto imported = import_package(
        std::move(request), payload->package());
    if (imported.status == server_vbr_artifact_import_status::ok) {
        impl_->counters.host_imports_succeeded++;
    }
    return imported;
}

server_vbr_artifact_import_output
server_vbr_artifact_store::import_host_occupied_replacement(
        server_vbr_artifact_import_target request,
        std::shared_ptr<const server_prompt_cache_vbr_payload> incoming,
        std::shared_ptr<const server_prompt_cache_vbr_payload> recovery)
        noexcept {
    server_vbr_artifact_import_output output;
    impl_->counters.imports_requested++;
    const auto structurally_owned = [&](const auto & payload) {
        return payload && payload->retirement_owned() &&
            payload->accounted_by(impl_->ledger) && payload->package();
    };
    if (!structurally_owned(incoming) || !structurally_owned(recovery) ||
        incoming == recovery ||
        incoming->reference_artifact() == recovery->reference_artifact()) {
        output.status = server_vbr_artifact_import_status::unavailable;
        impl_->counters.imports_unavailable++;
        return output;
    }
    if (!impl_->catalog.owns_host_package(incoming->package()) ||
        !impl_->catalog.owns_host_package(recovery->package())) {
        output.status = server_vbr_artifact_import_status::not_found;
        impl_->counters.imports_not_found++;
        return output;
    }
    impl_->counters.host_imports_authenticated++;
    auto imported = import_package_impl(
        std::move(request), incoming->package(), &recovery->package());
    if (imported.status == server_vbr_artifact_import_status::ok) {
        impl_->counters.host_imports_succeeded++;
    }
    return imported;
}

server_vbr_artifact_import_output
server_vbr_artifact_store::complete_validated_import(
        server_vbr_artifact_import_target request,
        vbr_manifest_validation_result validated,
        vbr_adopt_stage_policy stage_policy,
        server_vbr_artifact_import_output output) noexcept {
    const auto fail = [&](server_vbr_artifact_import_status status,
                          uint64_t & counter) {
        output.status = status;
        ++counter;
        return output;
    };
    try {
        output.validation_status = validated.status;
        output.decision = validated.decision;
        if (validated.status != vbr_manifest_validation_status::_count) {
            impl_->counters.validation_outcomes[size_t(validated.status)]++;
        }
        if (validated.decision != vbr_import_decision::_count) {
            impl_->counters.import_decisions[size_t(validated.decision)]++;
        }
        const auto disposition =
            server_vbr_artifact_import_validation_disposition(
                validated.status, validated.decision);
        if (disposition != server_vbr_artifact_import_status::ok ||
            !validated.proof) {
            return fail(disposition,
                disposition == server_vbr_artifact_import_status::report_only
                    ? impl_->counters.imports_report_only
                    : impl_->counters.imports_refused);
        }
        if (!request.prepare_publish(
                request.publish_context,
                validated.proof->token_block().tokens,
                validated.proof->authenticated_identity().sequence_epoch)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        const llama_pos expected_companion_terminal =
            validated.proof->authenticated_identity().requested_frontier > 0
                ? validated.proof->authenticated_identity().
                    requested_frontier-1
                : -1;
        const auto * occupied = validated.proof->occupied_replacement();
        const llama_pos expected_recovery_terminal =
            occupied && occupied->recovery_package() &&
                    occupied->recovery_package().manifest().identity.
                        next_position > 0
                ? occupied->recovery_package().manifest().identity.
                    next_position-1
                : -1;
        if (!impl_->bind_import_transport(stage_policy)) {
            return fail(server_vbr_artifact_import_status::stage_failed,
                        impl_->counters.imports_refused);
        }
        auto staged = vbr_stage_validated_manifest(
            std::move(validated.proof), stage_policy);
        output.stage_status = staged.status;
        // The server result field retains its route/API spelling, but the
        // value now reports the one shared downward/upward transform reserve.
        output.downward_reserve_status = staged.transform_status;
        if (staged.status != vbr_adopt_stage_status::staged ||
            !staged.manifest || !staged.staged) {
            return fail(server_vbr_artifact_import_status::stage_failed,
                        impl_->counters.imports_refused);
        }

        vbr_composite_publish_hooks hooks;
        hooks.publish = [](void * opaque) noexcept {
            auto * pair = static_cast<std::pair<
                server_vbr_artifact_import_target::publish_fn,
                void *> *>(opaque);
            pair->first(pair->second);
        };
        std::pair<server_vbr_artifact_import_target::publish_fn, void *>
            publish { request.publish, request.publish_context };
        hooks.context = &publish;
        hooks.owner_token = request.memory;
        hooks.validate_owner_token = [](
                const void * opaque, const void * token,
                const llama_memory_i * target) noexcept {
            return opaque != nullptr && token == target;
        };
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(request.memory, tree)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        for (const auto & child : tree) {
            if (child.recurrent) {
                hooks.companions.push_back(
                    vbr_recurrent_companion_adoption_provider(
                        *child.recurrent));
            }
            if (child.qsa_index_owner) {
                hooks.companions.push_back(
                    vbr_qsa_index_adoption_provider(
                        *child.qsa_index_owner, child.child_id));
            }
        }
        server_vbr_draft_target draft_target {
            request.draft_context, request.destination,
            expected_companion_terminal, expected_recovery_terminal,
        };
        if (request.draft_context) {
            vbr_companion_adoption_provider provider;
            provider.kind =
                vbr_artifact_companion_kind::required_spec_payload;
            provider.target_cookie = request.draft_context;
            provider.context = &draft_target;
            provider.prepare = server_vbr_draft_image::prepare;
            provider.prepare_replacement =
                server_vbr_draft_image::prepare_replacement;
            provider.target_empty = server_vbr_draft_image::empty;
            provider.recheck = server_vbr_draft_image::recheck;
            provider.publish_swap = server_vbr_draft_image::publish;
            provider.rollback = server_vbr_draft_image::rollback;
            hooks.companions.push_back(provider);
        }
        server_vbr_accelerator_target accelerator_target {
            request.accelerator, expected_companion_terminal,
            expected_recovery_terminal,
        };
        if (request.accelerator) {
            vbr_companion_adoption_provider provider;
            provider.kind =
                vbr_artifact_companion_kind::typed_accelerator;
            provider.target_cookie = request.accelerator;
            provider.context = &accelerator_target;
            provider.prepare = server_vbr_accelerator_image::prepare;
            provider.prepare_replacement =
                server_vbr_accelerator_image::prepare_replacement;
            provider.target_empty = server_vbr_accelerator_image::empty;
            provider.recheck = server_vbr_accelerator_image::recheck;
            provider.publish_swap = server_vbr_accelerator_image::publish;
            provider.rollback = server_vbr_accelerator_image::rollback;
            hooks.companions.push_back(provider);
        }
        server_vbr_frontier_logits_target frontier_logits_target {
            request.frontier_logits, request.frontier_logits_count,
            request.previously_observed,
        };
        if (request.frontier_logits && request.frontier_logits_count > 0) {
            vbr_companion_adoption_provider provider;
            provider.kind = vbr_artifact_companion_kind::frontier_logits;
            provider.target_cookie = request.frontier_logits;
            provider.context = &frontier_logits_target;
            provider.prepare = server_vbr_frontier_logits_image::prepare;
            provider.target_empty = server_vbr_frontier_logits_image::empty;
            provider.recheck = server_vbr_frontier_logits_image::recheck;
            provider.publish_swap = server_vbr_frontier_logits_image::publish;
            provider.rollback = server_vbr_frontier_logits_image::rollback;
            hooks.companions.push_back(provider);
        }
        const auto adopted = vbr_adopt_empty_manifest(
            *request.memory, request.destination,
            std::move(*staged.manifest), std::move(*staged.staged),
            *impl_->ledger, hooks);
        output.adopt_attempted = true;
        output.adopt_status = adopted.status;
        output.recovery = adopted.recovery;
        output.phase = adopted.phase;
        output.downward_subphase = adopted.downward_subphase;
        output.downward_edge = adopted.downward_edge;
        output.h2d_bytes = adopted.h2d_bytes;
        output.h2d_chunks = adopted.h2d_chunks;
        output.rollback_count = adopted.rollback_count;
        output.decision = adopted.decision;
        output.consistency = adopted.consistency;
        output.units = adopted.units;
        output.companions = adopted.companions;
        if (adopted.status != vbr_adopt_status::adopted) {
            return fail(server_vbr_artifact_import_status::adopt_failed,
                        impl_->counters.imports_refused);
        }
        output.status = server_vbr_artifact_import_status::ok;
        impl_->counters.imports_succeeded++;
        return output;
    } catch (...) {
        return fail(server_vbr_artifact_import_status::internal_error,
                    impl_->counters.imports_unavailable);
    }
}

vbr_artifact_prefix_projection_status
server_vbr_artifact_store::prepare_host_prefix_projection(
        const std::shared_ptr<const server_prompt_cache_vbr_payload> & payload,
        const std::vector<llama_token> & request_tokens,
        uint64_t lcp_tokens,
        vbr_artifact_attention_prefix_projection & output) noexcept {
    output.reset();
    if (!payload || !payload->retirement_owned() ||
        !payload->accounted_by(impl_->ledger) || !payload->package() ||
        !impl_->catalog.owns_host_package(payload->package())) {
        return vbr_artifact_prefix_projection_status::parent_stale;
    }
    vbr_artifact_attention_prefix_request request;
    request.tokens = request_tokens.data();
    request.token_count = request_tokens.size();
    request.lcp_tokens = lcp_tokens;
    request.text_only = true;
    return impl_->catalog.project_attention_prefix(
        payload->package(), request, {}, output);
}

server_vbr_artifact_import_output
server_vbr_artifact_store::import_host_prefix_payload(
        server_vbr_artifact_import_target request,
        std::shared_ptr<const server_prompt_cache_vbr_payload> payload,
        vbr_artifact_attention_prefix_projection projection) noexcept {
    return import_host_prefix_payload_impl(
        std::move(request), std::move(payload), std::move(projection), nullptr);
}

server_vbr_artifact_import_output
server_vbr_artifact_store::import_host_occupied_prefix_replacement(
        server_vbr_artifact_import_target request,
        std::shared_ptr<const server_prompt_cache_vbr_payload> incoming,
        std::shared_ptr<const server_prompt_cache_vbr_payload> recovery,
        vbr_artifact_attention_prefix_projection projection) noexcept {
    return import_host_prefix_payload_impl(
        std::move(request), std::move(incoming), std::move(projection),
        &recovery);
}

server_vbr_artifact_import_output
server_vbr_artifact_store::import_host_prefix_payload_impl(
        server_vbr_artifact_import_target request,
        std::shared_ptr<const server_prompt_cache_vbr_payload> payload,
        vbr_artifact_attention_prefix_projection projection,
        const std::shared_ptr<const server_prompt_cache_vbr_payload> * recovery)
        noexcept {
    server_vbr_artifact_import_output output;
    impl_->counters.imports_requested++;
    const auto fail = [&](server_vbr_artifact_import_status status,
                          uint64_t & counter) {
        output.status = status;
        ++counter;
        return output;
    };
    try {
        if (!request.memory || request.destination < 0 ||
            request.execution_identity.empty() ||
            request.adapter_config_identity.empty() ||
            !request.prepare_publish || !request.publish || !payload ||
            !payload->retirement_owned() ||
            !payload->accounted_by(impl_->ledger) || !payload->package() ||
            !projection ||
            projection.parent_artifact() != payload->reference_artifact() ||
            projection.parent_manifest_digest() !=
                payload->package().manifest().manifest_digest ||
            !impl_->catalog.owns_host_package(payload->package()) ||
            (recovery &&
             (!*recovery || *recovery == payload ||
              !(*recovery)->retirement_owned() ||
              !(*recovery)->accounted_by(impl_->ledger) ||
              !(*recovery)->package() ||
              (*recovery)->reference_artifact() ==
                  payload->reference_artifact() ||
              !impl_->catalog.owns_host_package((*recovery)->package())))) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        output.payload_bytes = projection.selected_bytes();
        output.companion_bytes = 0;

        llama_cache_budget_config budget;
        if (!impl_->sample_budget(impl_->budget_context, budget)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        const auto accounting_snapshot = impl_->ledger->snapshot();
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
        };
        live_import_context context;
        context.memory = request.memory;
        context.ledger = impl_->ledger;
        context.package = &payload->package();
        context.bindings = &impl_->domain_bindings;
        context.destination = request.destination;
        context.representation_context = &representation_policy;
        context.representation_identity =
            vbr_explicit_capture_representation_identity;
        vbr_downward_policy_projection downward_projection;
        vbr_occupied_replacement_guard occupied_guard;
        bool downward = false;
        vbr_import_schedule_quote schedule_quote;
        const auto snapshot_status =
            vbr_explicit_import_target_schedule_snapshot(
                *request.memory, request.destination, payload->package(),
                impl_->domain_bindings, request.previously_observed,
                accounting_snapshot.serial, &representation_policy,
                vbr_explicit_capture_representation_identity,
                context.snapshot, downward_projection, downward,
                schedule_quote, projection.prefix_tokens().size());
        if (snapshot_status !=
                vbr_import_target_snapshot_status::actionable ||
            schedule_quote.status() == vbr_import_schedule_status::unavailable ||
            schedule_quote.status() == vbr_import_schedule_status::_count) {
            output.validation_status =
                vbr_manifest_validation_status::unavailable;
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        output.schedule_status = schedule_quote.status();
        output.destination_status = schedule_quote.destination().status;
        output.destination_policy_steps = uint32_t(std::min<size_t>(
            schedule_quote.destination().prefix.size(), UINT32_MAX));
        output.destination_logical_bytes =
            schedule_quote.destination().logical_bytes_needed;
        output.destination_physical_growth_bytes =
            schedule_quote.destination().physical_growth_needed;
        output.destination_max_deficit =
            schedule_quote.destination().max_deficit;
        impl_->counters.import_schedules[
            size_t(schedule_quote.status())]++;
        context.schedule_quote = &schedule_quote;
        context.snapshot.scheduler_idle = true;
        if (recovery) {
            const auto guard_status =
                vbr_explicit_prepare_occupied_prefix_replacement_guard(
                    *request.memory, request.destination, payload->package(),
                    projection.prefix_tokens().size(), projection.cell_runs(),
                    (*recovery)->package(), impl_->domain_bindings,
                    accounting_snapshot.serial, &representation_policy,
                    vbr_explicit_capture_representation_identity,
                    occupied_guard, schedule_quote);
            if (guard_status !=
                    vbr_occupied_replacement_guard_status::ready) {
                output.validation_status =
                    vbr_manifest_validation_status::unavailable;
                return fail(server_vbr_artifact_import_status::unavailable,
                            impl_->counters.imports_unavailable);
            }
        }

        llama_cache_budget_plan transform_budget;
        transform_budget.accounting_serial = accounting_snapshot.serial;
        vbr_adopt_policy policy;
        policy.authorized = true;
        policy.identity = {
            request.execution_identity,
            request.adapter_config_identity,
            projection.identity().media_content_identity,
            projection.identity().sequence_epoch,
            projection.identity().next_position,
            &projection.prefix_tokens(),
        };
        policy.destination_sequence = request.destination;
        policy.allow_native = false;
        policy.allow_live_rebased = true;
        policy.allow_downward = true;
        policy.allow_upward = true;
        policy.adoption_nonce = impl_->next_reference++;
        if (policy.adoption_nonce == 0) {
            policy.adoption_nonce = impl_->next_reference++;
        }
        policy.domain_bindings = impl_->policy_bindings;
        policy.accounting_snapshot = &accounting_snapshot;
        policy.budget_config = &budget;
        policy.transform_budget_plan = &transform_budget;
        policy.schedule_quote = &schedule_quote;
        if (recovery) {
            policy.occupied_replacement = &occupied_guard;
            policy.occupied_representation_context = &representation_policy;
            policy.occupied_representation_identity =
                vbr_explicit_capture_representation_identity;
        }
        if (downward) {
            policy.downward_projection = &downward_projection;
        }
        policy.context = &context;
        policy.inspect_target = import_inspect_target;
        policy.recheck_target_empty = import_target_recheck;
        policy.read_accounting_serial = import_accounting_serial;
        policy.read_policy_epoch = import_policy_epoch;
        if (schedule_quote.status() != vbr_import_schedule_status::exact) {
            policy.read_transform_tree_digest = import_transform_digest;
        }
        auto validated = vbr_validate_attention_prefix_projection(
            context.snapshot, std::move(projection), policy);
        vbr_adopt_stage_policy stage_policy;
        stage_policy.budget = &budget;
        stage_policy.transform_context = &context;
        stage_policy.reserve_transform = import_reserve_transform;
        impl_->counters.host_imports_authenticated++;
        auto imported = complete_validated_import(
            std::move(request), std::move(validated), stage_policy,
            std::move(output));
        if (imported.status == server_vbr_artifact_import_status::ok) {
            impl_->counters.host_imports_succeeded++;
        }
        return imported;
    } catch (...) {
        return fail(server_vbr_artifact_import_status::internal_error,
                    impl_->counters.imports_unavailable);
    }
}

server_vbr_artifact_import_output server_vbr_artifact_store::import_package(
        server_vbr_artifact_import_target request,
        const vbr_artifact_package_view & package) noexcept {
    return import_package_impl(std::move(request), package, nullptr);
}

server_vbr_artifact_import_output server_vbr_artifact_store::import_package_impl(
        server_vbr_artifact_import_target request,
        const vbr_artifact_package_view & package,
        const vbr_artifact_package_view * recovery) noexcept {
    server_vbr_artifact_import_output output;
    const auto fail = [&](server_vbr_artifact_import_status status,
                          uint64_t & counter) {
        output.status = status;
        ++counter;
        return output;
    };
    try {
        if (!request.memory || request.destination < 0 ||
            request.execution_identity.empty() ||
            request.adapter_config_identity.empty() ||
            !request.prepare_publish || !request.publish || !package) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        if (!package_bytes(
                package, output.payload_bytes, output.companion_bytes)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }

        llama_cache_budget_config budget;
        if (!impl_->sample_budget(impl_->budget_context, budget)) {
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        const auto accounting_snapshot = impl_->ledger->snapshot();
        const char * build_identity = llama_commit();
        const vbr_explicit_representation_policy representation_policy {
            build_identity, strlen(build_identity),
        };
        live_import_context context;
        context.memory = request.memory;
        context.ledger = impl_->ledger;
        context.package = &package;
        context.bindings = &impl_->domain_bindings;
        context.destination = request.destination;
        context.representation_context = &representation_policy;
        context.representation_identity =
            vbr_explicit_capture_representation_identity;
        vbr_occupied_replacement_guard occupied_guard;
        vbr_downward_policy_projection downward_projection;
        bool downward = false;
        vbr_import_schedule_quote schedule_quote;
        const auto snapshot_status =
            vbr_explicit_import_target_schedule_snapshot(
                *request.memory, request.destination, package,
                impl_->domain_bindings, request.previously_observed,
                accounting_snapshot.serial, &representation_policy,
                vbr_explicit_capture_representation_identity,
                context.snapshot,
                downward_projection, downward, schedule_quote);
        const auto incoming_has_companion = [&](
                vbr_artifact_companion_kind kind) {
            return std::any_of(
                package.companions().begin(), package.companions().end(),
                [&](const auto & value) {
                    return value.descriptor.kind == kind;
                });
        };
        if (request.draft_context && incoming_has_companion(
                vbr_artifact_companion_kind::required_spec_payload)) {
            server_vbr_companion_codec codec;
            if (!server_vbr_companion_codec_for(
                    vbr_artifact_companion_kind::required_spec_payload,
                    codec)) {
                return fail(server_vbr_artifact_import_status::unavailable,
                            impl_->counters.imports_unavailable);
            }
            auto * draft_memory = llama_get_memory(request.draft_context);
            context.snapshot.companions.push_back({
                codec.kind, codec.format_version,
                codec.build_identity_digest,
                recovery != nullptr || (draft_memory &&
                    llama_memory_seq_pos_min(
                        draft_memory, request.destination) < 0 &&
                    llama_memory_seq_pos_max(
                        draft_memory, request.destination) < 0),
                request.draft_context,
            });
        }
        if (request.accelerator && incoming_has_companion(
                vbr_artifact_companion_kind::typed_accelerator)) {
            server_vbr_companion_codec codec;
            if (!server_vbr_companion_codec_for(
                    vbr_artifact_companion_kind::typed_accelerator,
                    codec)) {
                return fail(server_vbr_artifact_import_status::unavailable,
                            impl_->counters.imports_unavailable);
            }
            context.snapshot.companions.push_back({
                codec.kind, codec.format_version,
                codec.build_identity_digest,
                recovery != nullptr ||
                    common_speculative_ring_state_empty(request.accelerator),
                request.accelerator,
            });
        }
        if (request.frontier_logits && request.frontier_logits_count > 0 &&
            incoming_has_companion(
                vbr_artifact_companion_kind::frontier_logits)) {
            server_vbr_companion_codec codec;
            if (!server_vbr_companion_codec_for(
                    vbr_artifact_companion_kind::frontier_logits,
                    codec)) {
                return fail(server_vbr_artifact_import_status::unavailable,
                            impl_->counters.imports_unavailable);
            }
            context.snapshot.companions.push_back({
                codec.kind, codec.format_version,
                codec.build_identity_digest,
                request.previously_observed || request.frontier_logits->empty(),
                request.frontier_logits,
            });
        }
        if (snapshot_status ==
                vbr_import_target_snapshot_status::unavailable) {
            output.validation_status =
                vbr_manifest_validation_status::unavailable;
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        output.schedule_status = schedule_quote.status();
        const auto & destination = schedule_quote.destination();
        output.destination_status = destination.status;
        output.destination_policy_steps = uint32_t(
            std::min<size_t>(destination.prefix.size(), UINT32_MAX));
        output.destination_logical_bytes =
            destination.logical_bytes_needed;
        output.destination_physical_growth_bytes =
            destination.physical_growth_needed;
        output.destination_max_deficit = destination.max_deficit;
        context.schedule_quote = &schedule_quote;
        if (schedule_quote.status() != vbr_import_schedule_status::_count) {
            impl_->counters.import_schedules[
                size_t(schedule_quote.status())]++;
        }
        if (schedule_quote.status() ==
                vbr_import_schedule_status::unavailable ||
            schedule_quote.status() == vbr_import_schedule_status::_count) {
            output.validation_status =
                vbr_manifest_validation_status::unavailable;
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        if (snapshot_status ==
                vbr_import_target_snapshot_status::report_only) {
            // Schedule classification is authenticated by the catalog-owned
            // package and target snapshot, but the full identity validator has
            // intentionally not run. Do not report it as validated.
            output.validation_status =
                vbr_manifest_validation_status::unavailable;
            output.decision = vbr_import_decision::rebuild;
            impl_->counters.import_decisions[
                size_t(vbr_import_decision::rebuild)]++;
            return fail(server_vbr_artifact_import_status::report_only,
                        impl_->counters.imports_report_only);
        }
        if (snapshot_status !=
                vbr_import_target_snapshot_status::actionable) {
            output.validation_status =
                vbr_manifest_validation_status::unavailable;
            return fail(server_vbr_artifact_import_status::unavailable,
                        impl_->counters.imports_unavailable);
        }
        if (recovery) {
            const auto guard_status =
                vbr_explicit_prepare_occupied_replacement_guard(
                    *request.memory, request.destination, package, *recovery,
                    impl_->domain_bindings, accounting_snapshot.serial,
                    &representation_policy,
                    vbr_explicit_capture_representation_identity,
                    occupied_guard, &schedule_quote,
                    &context.snapshot.companions);
            output.occupied_guard_status = guard_status;
            if (guard_status !=
                    vbr_occupied_replacement_guard_status::ready) {
                output.validation_status =
                    vbr_manifest_validation_status::unavailable;
                return fail(server_vbr_artifact_import_status::unavailable,
                            impl_->counters.imports_unavailable);
            }
        }
        // Idleness is the SCHEDULER's fact to assert, not the library's: the
        // route handler admits imports only on an idle, deferred-safe slot, so
        // the store vouches for it here on the snapshot the validator consumes.
        context.snapshot.scheduler_idle = true;

        llama_cache_budget_plan transform_budget;
        transform_budget.accounting_serial = accounting_snapshot.serial;
        vbr_adopt_policy policy;
        policy.authorized = true;
        policy.identity = {
            request.execution_identity,
            request.adapter_config_identity,
            package.manifest().identity.media_content_identity,
            package.manifest().identity.sequence_epoch,
            package.manifest().identity.next_position,
            &package.manifest().token_block.tokens,
        };
        policy.destination_sequence = request.destination;
        policy.adoption_nonce = impl_->next_reference++;
        if (policy.adoption_nonce == 0) {
            policy.adoption_nonce = impl_->next_reference++;
        }
        policy.domain_bindings = impl_->policy_bindings;
        policy.accounting_snapshot = &accounting_snapshot;
        policy.budget_config = &budget;
        policy.transform_budget_plan = &transform_budget;
        policy.allow_upward = true;
        policy.schedule_quote = &schedule_quote;
        if (recovery) {
            policy.occupied_replacement = &occupied_guard;
            policy.occupied_representation_context = &representation_policy;
            policy.occupied_representation_identity =
                vbr_explicit_capture_representation_identity;
        }
        policy.context = &context;
        policy.inspect_target = import_inspect_target;
        policy.parse_companion = import_parse_companion;
        policy.recheck_target_empty = import_target_recheck;
        policy.read_accounting_serial = import_accounting_serial;
        policy.read_policy_epoch = import_policy_epoch;
        if (schedule_quote.status() != vbr_import_schedule_status::exact) {
            policy.read_transform_tree_digest = import_transform_digest;
        }
        if (downward) {
            policy.downward_projection = &downward_projection;
        }
        auto validated = vbr_validate_unit_manifest(
            *request.memory, package, policy);
        vbr_adopt_stage_policy stage_policy;
        stage_policy.budget = &budget;
        stage_policy.transform_context = &context;
        stage_policy.reserve_transform = import_reserve_transform;
        return complete_validated_import(
            std::move(request), std::move(validated), stage_policy,
            std::move(output));
    } catch (...) {
        return fail(server_vbr_artifact_import_status::internal_error,
                    impl_->counters.imports_unavailable);
    }
}

bool server_vbr_artifact_store::resolve_control_reference(
        const std::string & reference,
        const std::string & tenant_key,
        vbr_artifact_package_view & package) noexcept {
    package.reset();
    try {
        llama_cache_acct_artifact_id artifact;
        if (!impl_->references.authorize(reference, tenant_key, artifact)) {
            return false;
        }
        return impl_->catalog.resolve_reference(artifact, package) ==
                   vbr_artifact_resolve_status::ok &&
               package && package.validate() == vbr_artifact_status::ok;
    } catch (...) {
        package.reset();
        return false;
    }
}

bool server_vbr_artifact_store::retain_host_payload(
        const std::string & reference,
        const std::string & tenant_key,
        std::shared_ptr<const server_prompt_cache_vbr_payload> & payload)
        noexcept {
    payload.reset();
    vbr_artifact_package_view package;
    llama_cache_acct_artifact_id artifact;
    // A catalog view can only name a package that passed the sealed
    // publication transaction. Retaining that immutable capability must not
    // re-read and rehash multi-GiB payloads; explicit import/control remains
    // the boundary that performs read-time validation.
    if (!impl_->references.authorize(reference, tenant_key, artifact) ||
        impl_->catalog.resolve_reference(artifact, package) !=
            vbr_artifact_resolve_status::ok ||
        !package) {
        return false;
    }
    payload = server_prompt_cache_vbr_payload::adopt_owned(
        std::move(package));
    return bool(payload);
}

const server_vbr_artifact_store_counters &
server_vbr_artifact_store::counters() const noexcept {
    return impl_->counters;
}

uint32_t server_vbr_artifact_store::attention_children() const noexcept {
    return impl_->n_attention_children;
}

const char * server_vbr_artifact_store_create_failure_name(
        server_vbr_artifact_store_create_failure failure) noexcept {
    switch (failure) {
        case server_vbr_artifact_store_create_failure::none:
            return "none";
        case server_vbr_artifact_store_create_failure::ledger_missing:
            return "ledger_missing";
        case server_vbr_artifact_store_create_failure::
                budget_sampler_missing:
            return "budget_sampler_missing";
        case server_vbr_artifact_store_create_failure::topology_missing:
            return "topology_missing";
        case server_vbr_artifact_store_create_failure::
                pool_binding_missing:
            return "pool_binding_missing";
        case server_vbr_artifact_store_create_failure::lane_missing:
            return "lane_missing";
        case server_vbr_artifact_store_create_failure::
                attention_child_missing:
            return "attention_child_missing";
        case server_vbr_artifact_store_create_failure::
                ring_size_invalid:
            return "ring_size_invalid";
        case server_vbr_artifact_store_create_failure::
                chunk_size_invalid:
            return "chunk_size_invalid";
        case server_vbr_artifact_store_create_failure::
                budget_sample_failed:
            return "budget_sample_failed";
        case server_vbr_artifact_store_create_failure::
                ring_create_failed:
            return "ring_create_failed";
        case server_vbr_artifact_store_create_failure::internal_error:
            return "internal_error";
        case server_vbr_artifact_store_create_failure::_count:
            break;
    }
    return "invalid";
}

const char * server_vbr_artifact_capture_status_name(
        server_vbr_artifact_capture_status status) noexcept {
    switch (status) {
        case server_vbr_artifact_capture_status::ok: return "ok";
        case server_vbr_artifact_capture_status::unsupported: return "unsupported";
        case server_vbr_artifact_capture_status::unavailable: return "unavailable";
        case server_vbr_artifact_capture_status::invalid_slot: return "invalid_slot";
        case server_vbr_artifact_capture_status::slot_processing: return "slot_processing";
        case server_vbr_artifact_capture_status::stale_frontier: return "stale_frontier";
        case server_vbr_artifact_capture_status::identity_unavailable: return "identity_unavailable";
        case server_vbr_artifact_capture_status::unauthorized: return "unauthorized";
        case server_vbr_artifact_capture_status::required_companion_unavailable: return "required_companion_unavailable";
        case server_vbr_artifact_capture_status::admission_refused: return "admission_refused";
        case server_vbr_artifact_capture_status::cancelled: return "cancelled";
        case server_vbr_artifact_capture_status::source_changed: return "source_changed";
        case server_vbr_artifact_capture_status::internal_error: return "internal_error";
        case server_vbr_artifact_capture_status::_count: return "_count";
    }
    return "_count";
}

const char * server_vbr_artifact_import_status_name(
        server_vbr_artifact_import_status status) noexcept {
    switch (status) {
        case server_vbr_artifact_import_status::ok: return "ok";
        case server_vbr_artifact_import_status::unsupported: return "unsupported";
        case server_vbr_artifact_import_status::not_found: return "not_found";
        case server_vbr_artifact_import_status::invalid_slot: return "invalid_slot";
        case server_vbr_artifact_import_status::slot_processing: return "slot_processing";
        case server_vbr_artifact_import_status::slot_not_empty: return "slot_not_empty";
        case server_vbr_artifact_import_status::validation_failed: return "validation_failed";
        case server_vbr_artifact_import_status::report_only: return "report_only";
        case server_vbr_artifact_import_status::stage_failed: return "stage_failed";
        case server_vbr_artifact_import_status::adopt_failed: return "adopt_failed";
        case server_vbr_artifact_import_status::unavailable: return "unavailable";
        case server_vbr_artifact_import_status::internal_error: return "internal_error";
        case server_vbr_artifact_import_status::_count: break;
    }
    return "_count";
}
