#include "llama-vbr-explicit-capture.h"

#include "llama-vbr-artifact-adopt.h"
#include "llama-vbr-artifact-validate.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-tree.h"
#include "llama-vbr-qsa-index.h"
#include "llama-sha256.h"
#include "llama-vbr-identity-digest.h"
#include "llama-vbr-operation.h"
#include "llama-vbr-upward.h"
#include "turbo-rotation-data.h"

#include "ggml-turbo-meansub.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace {

struct baked_representation_identity_cache_entry {
    int32_t type = -1;
    bool value_side = false;
    int32_t meansub_model_id = -1;
    std::array<uint8_t, 32> digest = {};
};

std::mutex g_baked_representation_identity_mutex;
std::vector<baked_representation_identity_cache_entry>
    g_baked_representation_identity_cache;
std::atomic<uint64_t> g_baked_representation_identity_hashes { 0 };

std::array<uint8_t, 32> representation_hash_file_or_marker(
        const char * tag,
        const char * path,
        uint32_t type,
        bool value_side,
        const vbr_explicit_representation_policy & policy,
        bool & ok) {
    llama_sha256_writer writer;
    writer.string(tag, strlen(tag));
    writer.u32(type);
    writer.u32(value_side);
    if (path == nullptr || path[0] == '\0') {
        static constexpr char BUILTIN[] =
            "compiled-in/build-identity";
        writer.string(BUILTIN, sizeof(BUILTIN) - 1);
        if (policy.build_identity == nullptr ||
            policy.build_identity_len == 0) {
            ok = false;
            return {};
        }
        writer.string(
            policy.build_identity, policy.build_identity_len);
        return writer.finish();
    }
    FILE * file = fopen(path, "rb");
    if (file == nullptr) {
        ok = false;
        return {};
    }
    std::array<uint8_t, 64*1024> buffer;
    for (;;) {
        const size_t size =
            fread(buffer.data(), 1, buffer.size(), file);
        if (size != 0) {
            writer.bytes(buffer.data(), size);
        }
        if (size != buffer.size()) {
            if (ferror(file)) {
                ok = false;
            }
            break;
        }
    }
    fclose(file);
    return ok ? writer.finish() : std::array<uint8_t, 32>{};
}

const char * representation_override(
        int32_t type,
        bool value_side) {
    switch (type) {
        case GGML_TYPE_TURBO8_0:
            return std::getenv("TURBO_CB_T8");
        case GGML_TYPE_TURBO4_0:
            return std::getenv("TURBO_CB_T4");
        case GGML_TYPE_TURBO3_TCQ:
        case GGML_TYPE_TURBO2_TCQ: {
            const char * side = std::getenv(value_side
                ? "TURBO_TCQ_CB_V" : "TURBO_TCQ_CB_K");
            return side ? side : std::getenv("TURBO_TCQ_CB");
        }
        case GGML_TYPE_TURBO1_TCQ: {
            const char * side = std::getenv(value_side
                ? "TURBO1_TCQ_CB_V" : "TURBO1_TCQ_CB_K");
            return side ? side : std::getenv("TURBO1_TCQ_CB");
        }
        default:
            return nullptr;
    }
}

std::array<uint8_t, 32> representation_rotation_identity(
        int32_t type,
        bool value_side) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] =
        "buun.vbr.codec-rotation/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u32(uint32_t(type));
    writer.u32(value_side);
    const auto * matrix = value_side
        ? TURBO_ROTATION_RT : TURBO_ROTATION_R;
    writer.bytes(matrix, 128*128*sizeof(matrix[0]));
    return writer.finish();
}

std::array<uint8_t, 32> representation_meansub_identity(
        int32_t type,
        bool value_side,
        int32_t meansub_model_id,
        const vbr_explicit_representation_policy & policy,
        bool & ok,
        bool & baked) {
    baked = false;
    llama_sha256_writer writer;
    static constexpr char domain_label[] =
        "buun.vbr.codec-meansub/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u32(uint32_t(type));
    writer.u32(value_side);

    const char * disabled = std::getenv("TURBO_MEANSUB_OFF");
    if (disabled != nullptr) {
        static constexpr char OFF[] = "disabled";
        writer.string(OFF, sizeof(OFF) - 1);
        return writer.finish();
    }
    const char * path = std::getenv(
        value_side ? "TURBO_VMEAN_SUB" : "TURBO_KMEAN_SUB");
    if (path != nullptr && path[0] != '\0') {
        return representation_hash_file_or_marker(
            "buun.vbr.codec-meansub-file/v1",
            path, uint32_t(type), value_side, policy, ok);
    }

    // Built-in mean tables are immutable for the life of the process. Cache
    // their exact representation digest after checking the mutable environment
    // overrides above, so request-path import validation never re-hashes the
    // full dense table. Keeping the final digest preserves the wire identity.
    std::lock_guard<std::mutex> lock(
        g_baked_representation_identity_mutex);
    const auto cached = std::find_if(
        g_baked_representation_identity_cache.begin(),
        g_baked_representation_identity_cache.end(),
        [&](const baked_representation_identity_cache_entry & entry) {
            return entry.type == type &&
                   entry.value_side == value_side &&
                   entry.meansub_model_id == meansub_model_id;
        });
    if (cached != g_baked_representation_identity_cache.end()) {
        baked = true;
        return cached->digest;
    }

    int max_layers = 0;
    int max_channels = 0;
    int live_layers = 0;
    const float * active = ggml_turbo_meansub_table(
        meansub_model_id, value_side ? 1 : 0,
        &max_layers, &max_channels, &live_layers);
    if (active == nullptr || max_layers <= 0 ||
        max_channels <= 0 || live_layers <= 0) {
        static constexpr char INACTIVE[] = "inactive";
        writer.string(INACTIVE, sizeof(INACTIVE) - 1);
        return writer.finish();
    }
    if (size_t(max_layers) >
            std::numeric_limits<size_t>::max() /
                size_t(max_channels) ||
        size_t(max_layers)*size_t(max_channels) >
            std::numeric_limits<size_t>::max() / sizeof(float)) {
        ok = false;
        return {};
    }
    writer.u32(uint32_t(max_layers));
    writer.u32(uint32_t(max_channels));
    writer.u32(uint32_t(live_layers));
    writer.bytes(
        active,
        size_t(max_layers)*size_t(max_channels)*sizeof(float));
    const auto digest = writer.finish();
    g_baked_representation_identity_cache.push_back({
        type, value_side, meansub_model_id, digest,
    });
    g_baked_representation_identity_hashes.fetch_add(
        1, std::memory_order_relaxed);
    baked = true;
    return digest;
}

bool digest_nonzero(const std::array<uint8_t, 32> & digest) {
    return std::any_of(digest.begin(), digest.end(),
        [](uint8_t value) { return value != 0; });
}

std::array<uint8_t, 32> representation_reference_digest(
        int32_t current_type,
        int32_t last_source_type,
        const vbr_explicit_representation_identity & identity) {
    llama_sha256_writer writer;
    static constexpr char domain[] = "buun.vbr.capture/representation";
    writer.string(domain, sizeof(domain) - 1);
    writer.u32(uint32_t(current_type));
    writer.u32(uint32_t(last_source_type));
    writer.u32(identity.codec_id);
    writer.u32(identity.codec_version);
    writer.bytes(identity.codebook_digest.data(),
                 identity.codebook_digest.size());
    writer.bytes(identity.rotation_digest.data(),
                 identity.rotation_digest.size());
    writer.bytes(identity.meansub_digest.data(),
                 identity.meansub_digest.size());
    return writer.finish();
}

// Shared by the capture stamp and the import target check so the recurrent
// companion identity cannot drift between the two.
constexpr char VBR_RECURRENT_CODEC_DOMAIN[] = "buun.vbr.capture/recurrent-codec";

// Nonzero head of a finished digest (0 is reserved as "absent").
uint64_t digest_head_u64(llama_sha256_writer & writer) {
    const auto digest = writer.finish();
    uint64_t value = 0;
    std::memcpy(&value, digest.data(), sizeof(value));
    return value == 0 ? 1 : value;
}

std::array<uint8_t, 32> tagged_digest(
        const char * tag,
        uint64_t a,
        uint64_t b = 0) {
    llama_sha256_writer writer;
    writer.string(tag, strlen(tag));
    writer.u64(a);
    writer.u64(b);
    return writer.finish();
}

struct capture_transfer_cancelled {};

class chain_io_writer final : public llama_io_write_i {
public:
    static constexpr size_t CHUNK_BYTES = 1024*1024;

    chain_io_writer(
            artifact_segment_chain & chain,
            uint64_t expected_bytes,
            void * continue_context = nullptr,
            vbr_projected_capture_batch_request::continue_transfer_fn
                continue_transfer = nullptr,
            uint64_t * tensor_d2h_bytes = nullptr,
            uint64_t * tensor_d2h_reads = nullptr)
        : chain_(chain), expected_bytes_(expected_bytes),
          scratch_(size_t(std::max<uint64_t>(
              1, std::min<uint64_t>(CHUNK_BYTES, expected_bytes)))),
          continue_context_(continue_context),
          continue_transfer_(continue_transfer),
          tensor_d2h_bytes_(tensor_d2h_bytes),
          tensor_d2h_reads_(tensor_d2h_reads) {
        if (expected_bytes > SIZE_MAX) {
            throw std::length_error("companion payload exceeds size_t");
        }
    }

    void write(const void * source, size_t size) override {
        const auto * cursor = static_cast<const uint8_t *>(source);
        if (cursor == nullptr && size != 0) {
            throw std::invalid_argument("null companion source");
        }
        account(size);
        while (size != 0) {
            check_continue();
            ensure_scratch();
            const size_t take = std::min(size, scratch_.size() - used_);
            std::memcpy(scratch_.data() + used_, cursor, take);
            check_continue();
            used_ += take;
            cursor += take;
            size -= take;
            if (used_ == scratch_.size()) {
                flush();
            }
        }
    }

    void write_tensor(
            ggml_tensor * tensor,
            size_t offset,
            size_t size) override {
        if (tensor == nullptr) {
            throw std::invalid_argument("null companion tensor");
        }
        account(size);
        while (size != 0) {
            check_continue();
            ensure_scratch();
            const size_t take = std::min(size, scratch_.size() - used_);
            ggml_backend_tensor_get(
                tensor, scratch_.data() + used_, offset, take);
            if (tensor_d2h_bytes_) {
                if (*tensor_d2h_bytes_ > UINT64_MAX - take) {
                    throw std::overflow_error(
                        "companion D2H byte counter overflow");
                }
                *tensor_d2h_bytes_ += take;
            }
            if (tensor_d2h_reads_) {
                if (*tensor_d2h_reads_ == UINT64_MAX) {
                    throw std::overflow_error(
                        "companion D2H read counter overflow");
                }
                ++*tensor_d2h_reads_;
            }
            // Bound cancellation latency to one synchronous tensor quantum
            // even when the queue becomes nonempty immediately after the
            // pre-read probe.
            check_continue();
            used_ += take;
            offset += take;
            size -= take;
            if (used_ == scratch_.size()) {
                flush();
            }
        }
    }

    size_t n_bytes() override {
        return size_t(written_);
    }

    bool finish() {
        check_continue();
        flush();
        return written_ == expected_bytes_ &&
               chain_.size() == expected_bytes_;
    }

private:
    void check_continue() const {
        if (continue_transfer_ &&
            !continue_transfer_(continue_context_)) {
            throw capture_transfer_cancelled {};
        }
    }

    void account(size_t size) {
        if (written_ > expected_bytes_ ||
            size > expected_bytes_ - written_ ||
            written_ > SIZE_MAX || size > SIZE_MAX - size_t(written_)) {
            throw std::overflow_error("companion payload size mismatch");
        }
        written_ += size;
    }

    void flush() {
        if (used_ != 0) {
            scratch_.resize(used_);
            if (!chain_.append_owned(std::move(scratch_))) {
                throw std::bad_alloc();
            }
            used_ = 0;
        }
    }

    void ensure_scratch() {
        if (!scratch_.empty()) {
            return;
        }
        const uint64_t remaining = expected_bytes_ - chain_.size();
        if (remaining == 0) {
            throw std::overflow_error("companion payload overrun");
        }
        scratch_.resize(size_t(std::min<uint64_t>(
            CHUNK_BYTES, remaining)));
    }

    artifact_segment_chain & chain_;
    uint64_t expected_bytes_ = 0;
    uint64_t written_ = 0;
    std::vector<uint8_t> scratch_;
    size_t used_ = 0;
    void * continue_context_ = nullptr;
    vbr_projected_capture_batch_request::continue_transfer_fn
        continue_transfer_ = nullptr;
    uint64_t * tensor_d2h_bytes_ = nullptr;
    uint64_t * tensor_d2h_reads_ = nullptr;
};

class counting_io_writer final : public llama_io_write_i {
public:
    void write(const void *, size_t size) override {
        add(size);
    }
    void write_tensor(ggml_tensor *, size_t, size_t size) override {
        add(size);
    }
    size_t n_bytes() override {
        return bytes;
    }
    void add(size_t size) {
        if (size > std::numeric_limits<size_t>::max() - bytes) {
            throw std::bad_alloc();
        }
        bytes += size;
    }
    size_t bytes = 0;
};

struct recurrent_companion_plan {
    llama_memory_recurrent * source = nullptr;
    llama_pos expected_terminal = -1;
    vbr_artifact_companion_payload descriptor;
};

constexpr uint32_t VBR_SEQUENCE_STATE_MAGIC = 0xaf143cd8;

void recurrent_companion_write(
        llama_io_write_i & writer,
        llama_memory_recurrent & source,
        llama_seq_id sequence) {
    writer.write(&VBR_SEQUENCE_STATE_MAGIC, sizeof(VBR_SEQUENCE_STATE_MAGIC));
    writer.write(&sequence, sizeof(sequence));
    source.state_write(writer, sequence, 0);
}

bool recurrent_companion_current(
        const recurrent_companion_plan & plan,
        llama_seq_id sequence) noexcept {
    return plan.source != nullptr && sequence >= 0 &&
        plan.source->seq_pos_min(sequence) == plan.expected_terminal &&
        plan.source->seq_pos_max(sequence) == plan.expected_terminal;
}

bool recurrent_companion_prepare(
        llama_memory_recurrent * source,
        llama_seq_id sequence,
        llama_pos next_position,
        recurrent_companion_plan & output,
        vbr_explicit_capture_status & status) noexcept {
    output = {};
    try {
        if (source == nullptr || sequence < 0 || next_position < 0) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        const llama_pos expected_terminal = next_position - 1;
        if (source->seq_pos_min(sequence) != expected_terminal ||
            source->seq_pos_max(sequence) != expected_terminal) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        counting_io_writer writer;
        recurrent_companion_write(writer, *source, sequence);
        if (writer.n_bytes() == 0 ||
            source->seq_pos_min(sequence) != expected_terminal ||
            source->seq_pos_max(sequence) != expected_terminal) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        output.source = source;
        output.expected_terminal = expected_terminal;
        output.descriptor.kind = vbr_artifact_companion_kind::recurrent;
        output.descriptor.format_version = 1;
        output.descriptor.build_identity_digest = tagged_digest(
            VBR_RECURRENT_CODEC_DOMAIN, 1);
        output.descriptor.domain = {
            llama_cache_acct_residency::pageable_host,
            llama_cache_acct_domain_kind::not_applicable,
            UINT32_MAX, UINT16_MAX,
        };
        output.descriptor.payload_bytes = writer.n_bytes();
        return true;
    } catch (...) {
        output = {};
        status = vbr_explicit_capture_status::
            required_companion_unavailable;
        return false;
    }
}

bool recurrent_companion_capture(
        const recurrent_companion_plan & plan,
        llama_seq_id sequence,
        std::unique_ptr<artifact_segment_chain> & output,
        std::array<uint8_t, 32> & digest,
        vbr_explicit_capture_status & status,
        void * continue_context = nullptr,
        vbr_projected_capture_batch_request::continue_transfer_fn
            continue_transfer = nullptr,
        uint64_t * d2h_bytes = nullptr,
        uint64_t * d2h_reads = nullptr) noexcept {
    output.reset();
    digest = {};
    if (d2h_bytes) {
        *d2h_bytes = 0;
    }
    if (d2h_reads) {
        *d2h_reads = 0;
    }
    try {
        if (plan.source == nullptr ||
            plan.descriptor.payload_bytes == 0 || sequence < 0) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        if (!recurrent_companion_current(plan, sequence)) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        auto chain = std::make_unique<artifact_segment_chain>(
            plan.descriptor.payload_bytes);
        chain_io_writer writer(
            *chain, plan.descriptor.payload_bytes,
            continue_context, continue_transfer,
            d2h_bytes, d2h_reads);
        recurrent_companion_write(writer, *plan.source, sequence);
        if (!writer.finish() ||
            writer.n_bytes() != plan.descriptor.payload_bytes ||
            !recurrent_companion_current(plan, sequence)) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        digest = vbr_capture_stream_digest(*chain);
        if (!digest_nonzero(digest)) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        if (!recurrent_companion_current(plan, sequence)) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        output = std::move(chain);
        return true;
    } catch (const capture_transfer_cancelled &) {
        output.reset();
        digest = {};
        status = vbr_explicit_capture_status::cancelled;
        return false;
    } catch (...) {
        output.reset();
        digest = {};
        status = vbr_explicit_capture_status::
            required_companion_unavailable;
        return false;
    }
}

struct projected_companion_plan {
    recurrent_companion_plan recurrent;
    vbr_explicit_companion_provider provider;
    vbr_artifact_companion_payload descriptor;
    bool has_provider = false;
};

bool projected_provider_current(
        const projected_companion_plan & plan,
        llama_seq_id sequence,
        llama_pos next_position) noexcept {
    if (!plan.has_provider || plan.provider.terminal_position == nullptr) {
        return true;
    }
    llama_pos terminal = -1;
    return next_position > 0 &&
        plan.provider.terminal_position(
            plan.provider.context, sequence, terminal) &&
        terminal == next_position - 1;
}

bool projected_provider_prepare(
        const vbr_explicit_companion_provider & provider,
        llama_seq_id sequence,
        llama_pos next_position,
        projected_companion_plan & output,
        vbr_explicit_capture_status & status) noexcept {
    output = {};
    try {
        if (sequence < 0 || next_position <= 0 || provider.size == nullptr ||
            (provider.capture == nullptr &&
             provider.capture_stream == nullptr) ||
            provider.format_version == 0 ||
            !digest_nonzero(provider.build_identity_digest)) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        uint64_t bytes = 0;
        if (!provider.size(provider.context, sequence, bytes) || bytes == 0 ||
            bytes > std::numeric_limits<size_t>::max()) {
            status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return false;
        }
        output.provider = provider;
        output.has_provider = true;
        output.descriptor.kind = provider.kind;
        output.descriptor.format_version = provider.format_version;
        output.descriptor.build_identity_digest =
            provider.build_identity_digest;
        output.descriptor.domain = provider.domain;
        output.descriptor.payload_bytes = bytes;
        if ((provider.kind == vbr_artifact_companion_kind::recurrent ||
             provider.terminal_position != nullptr) &&
            !projected_provider_current(output, sequence, next_position)) {
            output = {};
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        return true;
    } catch (...) {
        output = {};
        status = vbr_explicit_capture_status::
            required_companion_unavailable;
        return false;
    }
}

bool projected_provider_capture(
        const projected_companion_plan & plan,
        llama_seq_id sequence,
        llama_pos next_position,
        std::unique_ptr<artifact_segment_chain> & output,
        std::array<uint8_t, 32> & digest,
        vbr_explicit_capture_status & status,
        void * continue_context,
        vbr_projected_capture_batch_request::continue_transfer_fn
            continue_transfer) noexcept {
    output.reset();
    digest = {};
    try {
        if (!plan.has_provider || plan.descriptor.payload_bytes == 0 ||
            !projected_provider_current(plan, sequence, next_position)) {
            status = plan.has_provider
                ? vbr_explicit_capture_status::source_changed
                : vbr_explicit_capture_status::required_companion_unavailable;
            return false;
        }
        if (continue_transfer && !continue_transfer(continue_context)) {
            status = vbr_explicit_capture_status::cancelled;
            return false;
        }
        auto chain = std::make_unique<artifact_segment_chain>(
            plan.descriptor.payload_bytes);
        if (plan.provider.capture_stream != nullptr) {
            chain_io_writer writer(
                *chain, plan.descriptor.payload_bytes,
                continue_context, continue_transfer);
            if (!plan.provider.capture_stream(
                    plan.provider.context, sequence, writer) ||
                !writer.finish()) {
                status = vbr_explicit_capture_status::
                    required_companion_unavailable;
                return false;
            }
        } else {
            std::vector<uint8_t> bytes;
            if (!plan.provider.capture ||
                !plan.provider.capture(
                    plan.provider.context, sequence, bytes) ||
                bytes.size() != plan.descriptor.payload_bytes) {
                status = vbr_explicit_capture_status::source_changed;
                return false;
            }
            static constexpr size_t CHUNK_BYTES = 1024*1024;
            for (size_t offset = 0; offset < bytes.size();) {
                if (continue_transfer && !continue_transfer(continue_context)) {
                    status = vbr_explicit_capture_status::cancelled;
                    return false;
                }
                const size_t take = std::min(
                    CHUNK_BYTES, bytes.size() - offset);
                if (!chain->append(bytes.data() + offset, take)) {
                    status = vbr_explicit_capture_status::accounting_failed;
                    return false;
                }
                offset += take;
            }
        }
        if (!projected_provider_current(plan, sequence, next_position)) {
            status = vbr_explicit_capture_status::source_changed;
            return false;
        }
        digest = vbr_capture_stream_digest(*chain);
        if (!digest_nonzero(digest)) {
            status = vbr_explicit_capture_status::hash_mismatch;
            return false;
        }
        output = std::move(chain);
        return true;
    } catch (const capture_transfer_cancelled &) {
        output.reset();
        digest = {};
        status = vbr_explicit_capture_status::cancelled;
        return false;
    } catch (...) {
        output.reset();
        digest = {};
        status = vbr_explicit_capture_status::
            required_companion_unavailable;
        return false;
    }
}

void add_accounting(
        std::vector<vbr_artifact_portable_accounting_row> & rows,
        vbr_artifact_accounting_role role,
        const vbr_artifact_portable_domain & domain,
        uint64_t bytes) {
    for (auto & row : rows) {
        if (row.role == role && row.domain == domain) {
            if (bytes > UINT64_MAX - row.logical_bytes ||
                bytes > UINT64_MAX - row.resident_bytes) {
                throw std::overflow_error("artifact accounting overflow");
            }
            row.logical_bytes += bytes;
            row.resident_bytes += bytes;
            return;
        }
    }
    rows.push_back({ role, domain, bytes, bytes,
        llama_cache_acct_attr_kind::artifact });
}

vbr_artifact_portable_domain portable_domain(
        uint32_t topology,
        uint16_t ordinal) {
    return {
        llama_cache_acct_residency::device,
        llama_cache_acct_domain_kind::device_topology,
        topology,
        ordinal,
    };
}

vbr_explicit_capture_status stream_status(
        vbr_capture_stream_status status) {
    switch (status) {
        case vbr_capture_stream_status::ok:
            return vbr_explicit_capture_status::ok;
        case vbr_capture_stream_status::ring_unavailable:
            return vbr_explicit_capture_status::ring_unavailable;
        case vbr_capture_stream_status::cancelled:
            return vbr_explicit_capture_status::cancelled;
        case vbr_capture_stream_status::transfer_failed:
            return vbr_explicit_capture_status::transfer_failed;
        case vbr_capture_stream_status::short_read:
            return vbr_explicit_capture_status::short_read;
        case vbr_capture_stream_status::hash_mismatch:
            return vbr_explicit_capture_status::hash_mismatch;
        case vbr_capture_stream_status::accounting_unavailable:
        case vbr_capture_stream_status::stage_failed:
        case vbr_capture_stream_status::commit_failed:
            return vbr_explicit_capture_status::accounting_failed;
        case vbr_capture_stream_status::accounting_refused:
            return vbr_explicit_capture_status::admission_refused;
        case vbr_capture_stream_status::publication_failed:
            return vbr_explicit_capture_status::publication_failed;
        case vbr_capture_stream_status::snapshot_changed:
            return vbr_explicit_capture_status::source_changed;
        case vbr_capture_stream_status::snapshot_unavailable:
            return vbr_explicit_capture_status::generation_unavailable;
        case vbr_capture_stream_status::format_rejected:
            return vbr_explicit_capture_status::dedup_validation_failed;
        case vbr_capture_stream_status::projection_invalid:
        case vbr_capture_stream_status::invalid_argument:
        case vbr_capture_stream_status::duplicate_segment:
        case vbr_capture_stream_status::missing_segment:
        case vbr_capture_stream_status::late_segment:
        case vbr_capture_stream_status::internal_error:
        case vbr_capture_stream_status::_count:
            return vbr_explicit_capture_status::internal_error;
    }
    return vbr_explicit_capture_status::internal_error;
}

} // namespace

bool vbr_explicit_capture_representation_identity(
        const void * context,
        int32_t current_type,
        bool value_side,
        int32_t meansub_model_id,
        vbr_explicit_representation_identity & output) noexcept {
    try {
        if (context == nullptr ||
            current_type < 0 || current_type >= GGML_TYPE_COUNT ||
            meansub_model_id < 0) {
            return false;
        }
        const auto & policy =
            *static_cast<const vbr_explicit_representation_policy *>(
                context);
        output = {};
        output.codec_id = uint32_t(current_type) + 1;
        output.codec_version = 1;
        bool ok = true;
        output.codebook_digest =
            representation_hash_file_or_marker(
                "buun.vbr.codec-codebook/v1",
                representation_override(current_type, value_side),
                uint32_t(current_type), value_side, policy, ok);
        output.rotation_digest =
            representation_rotation_identity(
                current_type, value_side);
        output.meansub_digest =
            representation_meansub_identity(
                current_type, value_side, meansub_model_id,
                policy, ok, output.meansub_baked);
        return ok;
    } catch (...) {
        return false;
    }
}

vbr_explicit_representation_identity_diagnostics
vbr_explicit_representation_identity_diagnostics_snapshot() noexcept {
    return {
        g_baked_representation_identity_hashes.load(
            std::memory_order_relaxed),
    };
}

class vbr_live_capture_adapter {
public:
    struct child {
        uint32_t child_id = 0;
        checkpoint_child_dependency_mode dependency_mode =
            checkpoint_child_dependency_mode::absent;
        llama_kv_cache * cache = nullptr;
        std::vector<llama_kv_cache::vbr_capture_unit_plan> units;
        llama_kv_cache::vbr_capture_stability_token stability;
        vbr_checkpoint_generation_controller generation;
        std::vector<vbr_artifact_stream_placement> placements;
    };

    struct representation_cache_entry {
        int32_t current_type = -1;
        bool value_side = false;
        int32_t meansub_model_id = -1;
        vbr_explicit_representation_identity identity;
    };

    static bool settle(llama_kv_cache & cache) {
        return cache.vbr_capture_settle();
    }

    static bool runtime_pools(
            llama_kv_cache & cache,
            std::vector<vbr_explicit_capture_runtime_pool> & output) {
        if (cache.other != nullptr) {
            return runtime_pools(*cache.other, output);
        }
        const auto instance = cache.vbr_instance_id();
        const auto * tracker =
            cache.vbr_generation_tracker_get();
        if (!cache.vbr_operation_armed() ||
            tracker == nullptr || !tracker->active() ||
            !vbr_controller_instance_id_is_set(instance) ||
            cache.vbr_pools_.empty()) {
            return false;
        }
        for (const auto & pool : cache.vbr_pools_) {
            if (pool.vmm == nullptr || pool.buf == nullptr ||
                pool.device < 0) {
                return false;
            }
            const auto backend_device =
                ggml_backend_buft_get_device(
                    ggml_backend_buffer_get_type(pool.buf));
            if (backend_device == nullptr) {
                return false;
            }
            const auto duplicate = std::find_if(
                output.begin(), output.end(),
                [&](const auto & current) {
                    return current.instance_id == instance &&
                           current.device == pool.device;
                });
            if (duplicate != output.end()) {
                if (duplicate->backend_device != backend_device ||
                    (duplicate->backend != nullptr &&
                     pool.backend != nullptr &&
                     duplicate->backend != pool.backend)) {
                    return false;
                }
                continue;
            }
            output.push_back({
                instance, pool.device, backend_device, pool.backend,
            });
        }
        return true;
    }

    static bool empty(const llama_kv_cache & cache) noexcept {
        return cache.other == nullptr &&
            std::all_of(cache.v_cells.begin(), cache.v_cells.end(),
                [](const llama_kv_cells & cells) {
                    return cells.get_used() == 0;
                }) &&
            std::all_of(cache.vbr_pools_.begin(), cache.vbr_pools_.end(),
                [](const llama_kv_cache::vbr_pool & pool) {
                    return pool.wm_cells == 0;
                });
    }

    static bool append_policy_identity(
            const llama_kv_cache & cache,
            llama_sha256_writer & writer) noexcept {
        llama_kv_cache::vbr_capture_stability_token token;
        if (!cache.vbr_capture_policy_snapshot(token)) {
            return false;
        }
        writer.u64(cache.vbr_representation_epoch());
        writer.bytes(token.degrade_order_digest.data(),
                     token.degrade_order_digest.size());
        writer.bytes(token.policy_digest.data(),
                     token.policy_digest.size());
        writer.u64(token.degrade_cursor);
        writer.u32(uint32_t(token.floor_type));
        writer.u64(token.pressure_independent_settings);
        return true;
    }

    static bool fill_import_child(
            const llama_memory_tree_child & tree_child,
            const vbr_artifact_package_view & package,
            const std::vector<llama_vbr_artifact_domain_binding> & bindings,
            bool previously_observed,
            uint64_t policy_epoch,
            const void * representation_context,
            vbr_explicit_capture_request::representation_identity_fn
                representation_identity,
            std::vector<representation_cache_entry> & representation_cache,
            vbr_target_child_snapshot & output) noexcept {
        auto * cache = tree_child.attention;
        if (!cache || !cache->vbr_operation_armed()) {
            return false;
        }
        const auto * tracker = cache->vbr_generation_tracker_get();
        llama_kv_cache::vbr_capture_stability_token live_policy;
        if (!tracker || !tracker->active() || !tracker->stable() ||
            !cache->vbr_capture_policy_snapshot(live_policy) ||
            tree_child.child_id >=
                package.manifest().controller_policy.size()) {
            return false;
        }
        const auto & source_policy =
            package.manifest().controller_policy[tree_child.child_id];
        if (live_policy.degrade_order_digest !=
                source_policy.degrade_order_digest ||
            live_policy.floor_type != source_policy.floor_type ||
            live_policy.pressure_independent_settings !=
                source_policy.pressure_independent_settings ||
            cache->n_stream != source_policy.n_stream ||
            (cache->n_stream == 1) != source_policy.unified) {
            return false;
        }

        if (!representation_context || !representation_identity) {
            return false;
        }
        output = {};
        output.child_id = tree_child.child_id;
        output.dependency_mode = tree_child.dependency_mode;
        output.memory_cookie = cache;
        output.empty = empty(*cache);
        output.dedicated = cache->other == nullptr;
        output.armed = true;
        output.previously_observed = previously_observed;
        output.lineage_uuid = tracker->lineage_identity();
        output.instance_id = tracker->runtime_instance();
        output.state_serial = cache->vbr_representation_epoch_;
        output.policy_epoch = policy_epoch;
        output.controller_policy = source_policy;

        for (const auto & source_unit : package.units()) {
            const auto & descriptor = source_unit.descriptor;
            if (descriptor.child_id != tree_child.child_id) {
                continue;
            }
            const size_t layer = descriptor.logical_unit_id/2;
            const bool value_side =
                (descriptor.logical_unit_id & 1u) != 0;
            if (layer >= cache->layers.size()) {
                return false;
            }
            const auto & extents =
                cache->vbr_units_of(layer, value_side);
            ggml_tensor * tensor = value_side
                ? cache->layers[layer].v : cache->layers[layer].k;
            if (!tensor ||
                extents.size() != descriptor.shards.size()) {
                return false;
            }
            const auto live_generation = tracker->unit_generation(
                descriptor.logical_unit_id);
            const auto meansub_ref =
                cache->layers[layer].turbo_meansub_ref;
            auto representation = std::find_if(
                representation_cache.begin(), representation_cache.end(),
                [&](const representation_cache_entry & cached) {
                    return cached.current_type == descriptor.current_type &&
                           cached.value_side == value_side &&
                           cached.meansub_model_id == meansub_ref.model_id;
                });
            if (representation == representation_cache.end()) {
                vbr_explicit_representation_identity identity;
                if (!representation_identity(
                        representation_context, descriptor.current_type,
                        value_side, meansub_ref.model_id, identity)) {
                    return false;
                }
                representation_cache.push_back({
                    descriptor.current_type, value_side,
                    meansub_ref.model_id, std::move(identity),
                });
                representation = std::prev(representation_cache.end());
            }
            const auto & live_identity = representation->identity;
            if (live_identity.codec_id == 0 ||
                live_identity.codec_version == 0 ||
                !digest_nonzero(live_identity.codebook_digest) ||
                !digest_nonzero(live_identity.rotation_digest) ||
                !digest_nonzero(live_identity.meansub_digest)) {
                return false;
            }
            vbr_target_unit_snapshot target;
            target.child_id = descriptor.child_id;
            target.logical_unit_id = descriptor.logical_unit_id;
            target.current_type = tensor->type;
            target.last_source_type = live_generation.last_source_type;
            target.promote_hops = live_generation.promote_hops;
            target.last_transition = live_generation.last_transition;
            target.representation_kind = descriptor.representation.kind;
            target.codec_id = live_identity.codec_id;
            target.codec_version = live_identity.codec_version;
            target.representation_reference_digest =
                representation_reference_digest(
                    descriptor.current_type, descriptor.last_source_type,
                    live_identity);
            target.source_loss_history =
                descriptor.representation.source_loss_history;
            target.checkpoint_codec_hops =
                descriptor.representation.checkpoint_codec_hops;
            target.recoverability = descriptor.recoverability;
            target.side = descriptor.side;
            target.layout = descriptor.layout;
            target.row_codec_version = descriptor.row_codec_version;
            target.current_domain = live_generation.domain;
            target.codebook_digest = live_identity.codebook_digest;
            target.rotation_digest = live_identity.rotation_digest;
            target.meansub_digest = live_identity.meansub_digest;
            target.meansub_model_id = meansub_ref.model_id;
            target.meansub_layer = meansub_ref.layer;
            target.meansub_baked = live_identity.meansub_baked;
            target.n_stream = descriptor.n_stream;
            target.unified = descriptor.unified;
            target.v_trans = false;
            target.wm_cells = descriptor.wm_cells;
            target.rank = descriptor.rank;
            target.dimensions = descriptor.dimensions;
            target.row_alignment = descriptor.row_alignment;
            for (size_t i = 0; i < extents.size(); ++i) {
                const auto & source = descriptor.shards[i];
                const auto binding = std::find_if(
                    bindings.begin(), bindings.end(),
                    [&](const llama_vbr_artifact_domain_binding & value) {
                        return value.topology_index == source.topology_index &&
                               value.device_ordinal == source.device_ordinal;
                    });
                if (binding == bindings.end() ||
                    source.topology_index >= package.topologies().size() ||
                    !extents[i].first || !extents[i].second ||
                    !extents[i].second->t) {
                    return false;
                }
                target.shards.push_back({
                    uint32_t(i), extents[i].first, binding->domain,
                    source.topology_index, source.device_ordinal,
                    package.topologies()[source.topology_index].digest,
                    source.logical_offset, source.row_count,
                    uint64_t(ggml_row_size(
                        extents[i].second->t->type,
                        extents[i].second->t->ne[0])),
                    uint64_t(ggml_nbytes(extents[i].second->t)),
                });
            }
            output.units.push_back(std::move(target));
        }
        return !output.units.empty();
    }

    static bool recheck_import_child(
            const llama_kv_cache & cache,
            const vbr_child_empty_fingerprint & expected) noexcept {
        const auto * tracker = cache.vbr_generation_tracker_get();
        return expected.memory_cookie == &cache &&
               expected.state_serial == cache.vbr_representation_epoch() &&
               tracker &&
               expected.instance_id == tracker->runtime_instance() &&
               empty(cache);
    }

    static bool reserve_import_transform(
            llama_kv_cache & cache,
            const std::vector<const vbr_validated_child_plan *> & plans,
            llama_cache_acct_ledger & ledger,
            const llama_cache_budget_config & budget,
            vbr_downward_stage_reservation & output) noexcept {
        return cache.vbr_import_transform_reserve(
            plans, ledger, budget, output);
    }

    static bool apply_import_destination(
            const std::vector<llama_memory_tree_child> & tree,
            const vbr_artifact_package_view & package,
            const vbr_import_destination_projection & destination,
            const void * representation_context,
            vbr_explicit_capture_request::representation_identity_fn
                representation_identity,
            std::vector<representation_cache_entry> & representation_cache,
            vbr_target_validation_snapshot & output,
            vbr_downward_policy_projection & projection) noexcept {
        struct indexed_child {
            llama_kv_cache * cache = nullptr;
            vbr_target_child_snapshot * target = nullptr;
            std::vector<const vbr_artifact_unit_view *> units;
        };
        try {
        if (!destination.feasible() ||
            destination.prefix.size() >
                VBR_IMPORT_DESTINATION_MAX_STEPS ||
            destination.final_types.size() != output.children.size() ||
            destination.final_cursors.size() != output.children.size() ||
            destination.child_type_digests.size() != output.children.size() ||
            !vbr_digest_nonzero(destination.tree_digest)) {
            return false;
        }
        projection.status = vbr_downward_policy_status::coherent;
        projection.final_types = destination.final_types;
        projection.final_cursors = destination.final_cursors;
        projection.child_type_digests = destination.child_type_digests;
        projection.tree_digest = destination.tree_digest;

        const size_t n_controller =
            package.manifest().controller_policy.size();
        std::vector<const llama_memory_tree_child *> tree_index(
            n_controller, nullptr);
        for (const auto & child : tree) {
            if (!child.attention) {
                continue;
            }
            if (child.child_id >= tree_index.size() ||
                tree_index[child.child_id] != nullptr) {
                return false;
            }
            tree_index[child.child_id] = &child;
        }
        std::vector<size_t> child_position(n_controller, SIZE_MAX);
        std::vector<indexed_child> indexed(output.children.size());
        for (size_t i = 0; i < output.children.size(); ++i) {
            auto & target = output.children[i];
            if (target.child_id >= tree_index.size() ||
                child_position[target.child_id] != SIZE_MAX) {
                return false;
            }
            const auto * live = tree_index[target.child_id];
            if (!live || !live->attention ||
                live->attention->vbr_pools_.empty()) {
                return false;
            }
            child_position[target.child_id] = i;
            indexed[i].cache = live->attention;
            indexed[i].target = &target;
            indexed[i].units.resize(
                live->attention->layers.size()*2, nullptr);
        }
        for (const auto & unit : package.units()) {
            const auto child_id = unit.descriptor.child_id;
            const auto unit_id = unit.descriptor.logical_unit_id;
            if (child_id >= child_position.size() ||
                child_position[child_id] == SIZE_MAX) {
                return false;
            }
            auto & units = indexed[child_position[child_id]].units;
            if (unit_id >= units.size() || units[unit_id] != nullptr) {
                return false;
            }
            units[unit_id] = &unit;
        }

        for (size_t child_index = 0; child_index < indexed.size();
             ++child_index) {
            auto & child = indexed[child_index];
            if (child.target->child_id != child_index ||
                projection.final_types[child_index].size() !=
                    child.units.size() ||
                vbr_type_vector_digest(projection.final_types[child_index]) !=
                    projection.child_type_digests[child_index]) {
                return false;
            }
            for (auto & unit : child.target->units) {
                if (unit.logical_unit_id >= child.units.size() ||
                    !child.units[unit.logical_unit_id]) {
                    return false;
                }
                const auto & descriptor = child.units[
                    unit.logical_unit_id]->descriptor;
                const auto source_type = static_cast<ggml_type>(
                    descriptor.current_type);
                const auto target_type = projection.final_types[child_index]
                    [unit.logical_unit_id];
                const auto live_type = static_cast<ggml_type>(
                    unit.current_type);
                if (source_type != target_type || target_type != live_type) {
                    const bool value_side =
                        (unit.logical_unit_id & 1u) != 0;
                    const auto meansub_ref = child.cache->layers[
                        unit.logical_unit_id/2].turbo_meansub_ref;
                    auto target_representation = std::find_if(
                        representation_cache.begin(),
                        representation_cache.end(),
                        [&](const representation_cache_entry & cached) {
                            return cached.current_type == int32_t(target_type) &&
                                   cached.value_side == value_side &&
                                   cached.meansub_model_id == meansub_ref.model_id;
                        });
                    if (target_representation == representation_cache.end()) {
                        vbr_explicit_representation_identity identity;
                        if (!representation_identity ||
                            !representation_context ||
                            !representation_identity(
                                representation_context, int32_t(target_type),
                                value_side, meansub_ref.model_id, identity)) {
                            return false;
                        }
                        representation_cache.push_back({
                            int32_t(target_type), value_side,
                            meansub_ref.model_id, std::move(identity),
                        });
                        target_representation =
                            std::prev(representation_cache.end());
                    }
                    const auto & selected_identity =
                        target_representation->identity;
                    if (selected_identity.codec_id == 0 ||
                        selected_identity.codec_version == 0 ||
                        !digest_nonzero(selected_identity.codebook_digest) ||
                        !digest_nonzero(selected_identity.rotation_digest) ||
                        !digest_nonzero(selected_identity.meansub_digest)) {
                        return false;
                    }
                    const vbr_upward_representation_identity
                        selected_source_identity {
                            unit.codebook_digest,
                            unit.rotation_digest,
                            unit.meansub_digest,
                            unit.meansub_model_id,
                            unit.meansub_layer,
                            unit.meansub_baked,
                            unit.codec_id,
                            unit.codec_version,
                            unit.representation_reference_digest,
                        };
                    const vbr_upward_representation_identity
                        selected_target_identity {
                            selected_identity.codebook_digest,
                            selected_identity.rotation_digest,
                            selected_identity.meansub_digest,
                            meansub_ref.model_id, meansub_ref.layer,
                            selected_identity.meansub_baked,
                            selected_identity.codec_id,
                            selected_identity.codec_version,
                            representation_reference_digest(
                                int32_t(target_type), int32_t(source_type),
                                selected_identity),
                        };
                    if (!child.cache->vbr_import_bind_target_unit(
                            descriptor, target_type, selected_source_identity,
                            selected_target_identity, projection,
                            uint32_t(child_index), unit)) {
                        return false;
                    }
                }
                if (source_type == target_type) {
                    // This copied snapshot is the post-adoption image. Exact
                    // selected units inherit their authenticated source
                    // generation rather than the empty target's old history.
                    unit.last_source_type = descriptor.last_source_type;
                    unit.promote_hops = descriptor.promote_hops;
                    unit.last_transition = descriptor.last_transition;
                }
            }
            child.target->controller_policy.current_type_vector_digest =
                projection.child_type_digests[child_index];
            child.target->controller_policy.cursor =
                projection.final_cursors[child_index];
        }
        return vbr_type_tree_digest(
                   projection.child_type_digests,
                   VBR_DOWNWARD_RECIPE_VERSION) == projection.tree_digest;
        } catch (...) {
            projection = {};
            return false;
        }
    }

    static bool recheck_import_destination(
            const std::vector<llama_memory_tree_child> & tree,
            const vbr_target_validation_snapshot & live,
            const vbr_import_destination_projection & destination) noexcept {
        if (!destination.feasible() ||
            destination.initial_types.size() != live.children.size() ||
            destination.initial_cursors.size() != live.children.size()) {
            return false;
        }
        size_t attention_index = 0;
        for (const auto & child : tree) {
            if (!child.attention) {
                continue;
            }
            if (attention_index >= live.children.size() ||
                child.child_id != attention_index ||
                live.children[attention_index].child_id != child.child_id ||
                destination.initial_types[attention_index].size() !=
                    live.children[attention_index].units.size()) {
                return false;
            }
            llama_kv_cache::vbr_capture_stability_token policy;
            if (!child.attention->vbr_capture_policy_snapshot(policy) ||
                policy.degrade_cursor !=
                    destination.initial_cursors[attention_index]) {
                return false;
            }
            for (const auto & unit : live.children[attention_index].units) {
                if (unit.logical_unit_id >=
                        destination.initial_types[attention_index].size() ||
                    unit.current_type != int32_t(
                        destination.initial_types[attention_index]
                            [unit.logical_unit_id])) {
                    return false;
                }
            }
            attention_index++;
        }
        return attention_index == live.children.size();
    }

    static bool negotiate_import_destination(
            const std::vector<llama_memory_tree_child> & tree,
            const vbr_artifact_package_view & package,
            vbr_import_schedule_quote & quote,
            uint64_t selected_frontier) noexcept {
        quote.destination_ = {};
        try {
            struct child_state {
                llama_kv_cache * cache = nullptr;
                vbr_import_destination_child input;
                llama_kv_cache::vbr_import_destination_pricing pricing;
                std::vector<llama_memory_vbr_physical_growth> physical;
            };
            std::vector<child_state> children;
            for (const auto & tree_child : tree) {
                if (!tree_child.attention) {
                    continue;
                }
                if (tree_child.child_id >=
                        package.manifest().controller_policy.size()) {
                    return false;
                }
                const uint64_t parent_wm = package.manifest().controller_policy[
                    tree_child.child_id].wm_cells;
                const uint64_t wm = selected_frontier != 0
                    ? selected_frontier : parent_wm;
                if (parent_wm == 0 || parent_wm > UINT32_MAX || wm == 0 ||
                    wm > parent_wm || wm > UINT32_MAX) {
                    return false;
                }
                child_state state;
                state.cache = tree_child.attention;
                if (!state.cache->vbr_import_destination_input(
                        uint32_t(wm), state.input) ||
                    !state.cache->vbr_import_destination_pricing_begin(
                        state.input.initial_types, uint32_t(wm),
                        state.pricing)) {
                    return false;
                }
                state.physical.reserve(state.pricing.devices.size());
                children.push_back(std::move(state));
            }
            if (children.empty()) {
                return false;
            }

            struct measure_context {
                std::vector<child_state> * children = nullptr;
                llama_memory_vbr_preflight_tree tree;
            } context { &children, {} };
            const auto measure = [](
                    void * opaque,
                    const std::vector<std::vector<ggml_type>> & types,
                    const llama_vbr_policy::selection * selected,
                    vbr_import_destination_evidence & evidence) noexcept {
                try {
                    auto * context = static_cast<measure_context *>(
                        opaque);
                    if (!context || !context->children ||
                        types.size() != context->children->size()) {
                        return false;
                    }
                    if (!context->tree.ready()) {
                        if (!context->tree.reset(
                                context->children->size())) {
                            return false;
                        }
                        for (size_t i = 0; i <
                                context->children->size(); ++i) {
                            auto & state = (*context->children)[i];
                            const auto child = state.cache->
                                vbr_import_destination_preflight(
                                    state.pricing, &state.physical);
                            if (!child.active || !context->tree.set_leaf(
                                    i, child, state.physical)) {
                                return false;
                            }
                        }
                        if (!context->tree.build()) {
                            return false;
                        }
                    } else if (selected) {
                        if (selected->child_index >=
                                context->children->size()) {
                            return false;
                        }
                        auto & state = (*context->children)[
                            selected->child_index];
                        if (!state.cache->vbr_import_destination_pricing_apply(
                                selected->value, state.pricing)) {
                            return false;
                        }
                        const auto child = state.cache->
                            vbr_import_destination_preflight(
                                state.pricing, &state.physical);
                        if (!child.active || !context->tree.replace_leaf(
                                selected->child_index,
                                child, state.physical)) {
                            return false;
                        }
                    }
                    const auto & aggregate = context->tree.preflight();
                    evidence.active = aggregate.active;
                    evidence.fits = aggregate.fits;
                    evidence.pools = aggregate.pools;
                    evidence.logical_bytes_needed = aggregate.bytes_needed;
                    evidence.logical_bytes_available =
                        aggregate.bytes_available;
                    evidence.physical_growth_needed =
                        aggregate.physical_growth_needed;
                    evidence.physical_growth_available =
                        aggregate.physical_growth_available;
                    evidence.max_deficit = aggregate.max_deficit;
                    return true;
                } catch (...) {
                    return false;
                }
            };
            std::vector<vbr_import_destination_child> inputs;
            inputs.reserve(children.size());
            for (const auto & child : children) {
                inputs.push_back(child.input);
            }
            quote.destination_ = vbr_select_import_destination(
                inputs, &context, measure);
            return (quote.destination_.feasible() &&
                    vbr_import_destination_projection_coherent(
                        inputs, quote.destination_)) ||
                quote.destination_.status ==
                    vbr_import_destination_status::exhausted;
        } catch (...) {
            quote.destination_ = {};
            return false;
        }
    }

    static bool capture_metadata(
            llama_kv_cache & cache,
            uint32_t child_id,
            checkpoint_child_dependency_mode mode,
            llama_seq_id sequence,
            llama_pos frontier,
            const std::vector<vbr_explicit_capture_pool_binding> & bindings,
            child & output,
            vbr_explicit_generation_failure & failure,
            vbr_explicit_size_failure & size_failure) {
        failure = vbr_explicit_generation_failure::none;
        size_failure = vbr_explicit_size_failure::none;
        llama_kv_cache::vbr_capture_unit_request request;
        request.child_id = child_id;
        request.bindings = &bindings;
        output.child_id = child_id;
        output.dependency_mode = mode;
        output.cache = &cache;
        // Snapshot the byte geometry/generation token first, then capture
        // ownership.  The final stability reread binds the ownership record
        // to that exact token: a mutation between these two calls advances a
        // monotone controller serial or unit publish_seq and fails closed.
        if (!cache.vbr_capture_size_pass(
                request, output.units, output.stability,
                &size_failure)) {
            failure = vbr_explicit_generation_failure::size_pass;
            return false;
        }
        vbr_artifact_stream_placement placement;
        auto * placement_out = mode !=
                checkpoint_child_dependency_mode::absent ?
            &placement : nullptr;
        if (!cache.vbr_capture_generation_record(
                child_id, mode, sequence, frontier,
                output.generation, placement_out, &failure)) {
            if (failure == vbr_explicit_generation_failure::none) {
                failure =
                    vbr_explicit_generation_failure::internal_error;
            }
            return false;
        }
        if (placement_out != nullptr) {
            output.placements.push_back(std::move(placement));
        }
        if (!cache.vbr_capture_stability_matches(output.stability)) {
            failure =
                vbr_explicit_generation_failure::stability_reread_failed;
            return false;
        }
        return true;
    }

    static bool capture_size(
            child & output,
            const std::vector<vbr_explicit_capture_pool_binding> & bindings,
            vbr_explicit_generation_failure & failure,
            vbr_explicit_size_failure & size_failure) {
        failure = vbr_explicit_generation_failure::none;
        size_failure = vbr_explicit_size_failure::none;
        if (output.cache == nullptr) {
            failure = vbr_explicit_generation_failure::internal_error;
            return false;
        }
        llama_kv_cache::vbr_capture_unit_request request;
        request.child_id = output.child_id;
        request.bindings = &bindings;
        if (!output.cache->vbr_capture_size_pass(
                request, output.units, output.stability, &size_failure)) {
            failure = vbr_explicit_generation_failure::size_pass;
            return false;
        }
        return true;
    }

    static bool capture_generation(
            const child & value,
            llama_seq_id sequence,
            llama_pos frontier,
            vbr_checkpoint_generation_controller & generation,
            vbr_artifact_stream_placement & placement,
            vbr_explicit_generation_failure & failure) {
        failure = vbr_explicit_generation_failure::none;
        if (value.cache == nullptr ||
            !value.cache->vbr_capture_generation_record(
                value.child_id, value.dependency_mode,
                sequence, frontier, generation, &placement, &failure)) {
            if (failure == vbr_explicit_generation_failure::none) {
                failure = vbr_explicit_generation_failure::internal_error;
            }
            return false;
        }
        return true;
    }

    static vbr_controller_instance_id instance(const child & value) noexcept {
        return value.cache ? value.cache->vbr_instance_id() :
            vbr_controller_instance_id {};
    }

    static vbr_capture_stream_status transfer_projected_unit(
            const child & value,
            const llama_kv_cache::vbr_capture_unit_plan & plan,
            const vbr_capture_projection & projection,
            uint64_t source_namespace,
            vbr_pinned_chunk_ring & ring,
            vbr_capture_projected_unit & output,
            void * continue_context = nullptr,
            vbr_projected_capture_batch_request::continue_transfer_fn
                continue_transfer = nullptr,
            vbr_capture_stream_stats * attempted = nullptr,
            const vbr_pinned_ring_operation * operation = nullptr) noexcept {
        if (!value.cache) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        std::vector<vbr_capture_projected_shard_source> sources;
        if (!value.cache->vbr_capture_projected_sources(plan, sources)) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        for (auto & source : sources) {
            source.source.continue_context = continue_context;
            source.source.continue_transfer = continue_transfer;
        }
        llama_kv_cache::vbr_capture_snapshot_session session;
        if (!value.cache->vbr_capture_snapshot_bind(
                plan, sources, source_namespace, session)) {
            return vbr_capture_stream_status::snapshot_unavailable;
        }
        return vbr_capture_projected_unit_transfer(
            projection, value.child_id, 0, plan.logical_unit,
            sources, {}, session.provider(), ring, output, attempted,
            operation);
    }

    static bool capture_schema(
            const child & value,
            const void * representation_context,
            vbr_explicit_capture_request::representation_identity_fn
                representation_identity,
            bool allow_clean_stash,
            std::vector<representation_cache_entry> & representation_cache,
            vbr_artifact_controller_policy & policy,
            std::vector<vbr_artifact_unit_descriptor> & descriptors,
            vbr_explicit_capture_status & status) {
        descriptors.clear();
        if (value.units.empty() || !representation_identity) {
            status = vbr_explicit_capture_status::identity_unavailable;
            return false;
        }
        const auto & first = value.units.front();
        if (first.n_stream == 0 || first.wm_cells == 0) {
            status = vbr_explicit_capture_status::unsupported_layout;
            return false;
        }

        policy = {};
        policy.child_id = value.child_id;
        policy.dependency_mode = value.dependency_mode;
        policy.degrade_order_digest =
            value.stability.degrade_order_digest;
        policy.policy_digest = value.stability.policy_digest;
        policy.cursor = value.stability.degrade_cursor;
        policy.floor_type = value.stability.floor_type;
        policy.pressure_independent_settings =
            value.stability.pressure_independent_settings;
        policy.n_stream = first.n_stream;
        policy.unified = first.unified;
        policy.wm_cells = first.wm_cells;
        policy.completed_wave = value.stability.completed_wave;

        std::vector<ggml_type> current_types;
        current_types.reserve(value.units.size());
        descriptors.reserve(value.units.size());
        for (const auto & plan : value.units) {
            if (plan.child_id != value.child_id ||
                plan.logical_unit != descriptors.size() ||
                plan.n_stream != policy.n_stream ||
                plan.unified != policy.unified ||
                plan.wm_cells != policy.wm_cells ||
                plan.meansub_ref.model_id < 0 ||
                plan.meansub_ref.layer < 0 ||
                plan.shards.empty()) {
                status = vbr_explicit_capture_status::unsupported_layout;
                return false;
            }
            vbr_artifact_unit_descriptor descriptor;
            descriptor.child_id = value.child_id;
            descriptor.logical_unit_id = plan.logical_unit;
            descriptor.lineage_uuid = value.stability.lineage_uuid;
            descriptor.repr_gen = plan.generation.repr_gen;
            descriptor.current_type = plan.generation.current_type;
            descriptor.last_source_type =
                plan.generation.last_source_type;
            descriptor.promote_hops = plan.generation.promote_hops;
            descriptor.last_transition = plan.generation.last_transition;
            descriptor.representation.kind =
                plan.generation.current_type == GGML_TYPE_F16
                    ? vbr_artifact_representation_kind::raw
                    : vbr_artifact_representation_kind::approximate;
            auto representation = std::find_if(
                representation_cache.begin(), representation_cache.end(),
                [&](const auto & cached) {
                    return cached.current_type ==
                            plan.generation.current_type &&
                        cached.value_side == plan.is_v &&
                        cached.meansub_model_id ==
                            plan.meansub_ref.model_id;
                });
            if (representation == representation_cache.end()) {
                vbr_explicit_representation_identity identity;
                if (!representation_identity(
                        representation_context,
                        plan.generation.current_type,
                        plan.is_v, plan.meansub_ref.model_id,
                        identity)) {
                    status =
                        vbr_explicit_capture_status::identity_unavailable;
                    return false;
                }
                representation_cache.push_back({
                    plan.generation.current_type, plan.is_v,
                    plan.meansub_ref.model_id,
                    std::move(identity),
                });
                representation = std::prev(representation_cache.end());
            }
            const auto & identity = representation->identity;
            if (
                identity.codec_id == 0 ||
                identity.codec_version == 0 ||
                !digest_nonzero(identity.codebook_digest) ||
                !digest_nonzero(identity.rotation_digest) ||
                !digest_nonzero(identity.meansub_digest)) {
                status = vbr_explicit_capture_status::identity_unavailable;
                return false;
            }
            descriptor.representation.codec_id = identity.codec_id;
            descriptor.representation.codec_version =
                identity.codec_version;
            descriptor.representation.reference_digest =
                representation_reference_digest(
                    plan.generation.current_type,
                    plan.generation.last_source_type, identity);
            descriptor.representation.source_loss_history =
                plan.generation.promote_hops;
            descriptor.side = plan.is_v
                ? vbr_artifact_side::value : vbr_artifact_side::key;
            descriptor.n_stream = plan.n_stream;
            descriptor.unified = plan.unified;
            descriptor.wm_cells = plan.wm_cells;
            descriptor.rank = 2;
            uint64_t total_columns = 0;
            uint64_t logical_offset = 0;
            bool has_stash = false;
            uint32_t stash_rows = 0;
            for (const auto & shard : plan.shards) {
                if (shard.columns == 0 ||
                    shard.columns > UINT64_MAX - total_columns) {
                    status = vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                if (shard.stash_bytes != 0 && !allow_clean_stash) {
                    status = vbr_explicit_capture_status::unsupported_layout;
                    return false;
                }
                total_columns += shard.columns;
                vbr_artifact_shard_descriptor wire;
                wire.shard_index = shard.shard_index;
                wire.topology_index = shard.topology_index;
                wire.device_ordinal = shard.device_ordinal;
                wire.logical_offset = logical_offset;
                wire.row_count = plan.wm_cells;
                wire.column_count = shard.columns;
                wire.row_bytes = shard.row_bytes;
                wire.payload_bytes = shard.payload_bytes;
                descriptor.shards.push_back(wire);
                logical_offset += shard.columns;
                if (shard.stash_bytes != 0) {
                    if (shard.columns > UINT64_MAX/sizeof(uint16_t) ||
                        shard.stash_bytes %
                            (shard.columns*sizeof(uint16_t)) != 0 ||
                        shard.stash_bytes /
                            (shard.columns*sizeof(uint16_t)) > UINT32_MAX) {
                        status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return false;
                    }
                    const uint32_t rows = uint32_t(
                        shard.stash_bytes /
                            (shard.columns*sizeof(uint16_t)));
                    if (has_stash && rows != stash_rows) {
                        status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return false;
                    }
                    has_stash = true;
                    stash_rows = rows;
                }
            }
            descriptor.dimensions = { plan.wm_cells, total_columns, 0, 0 };
            descriptor.row_alignment = 1;
            descriptor.row_codec_version = 1;
            descriptor.codebook_digest = identity.codebook_digest;
            descriptor.rotation_digest = identity.rotation_digest;
            descriptor.meansub_digest = identity.meansub_digest;
            descriptor.meansub_model_id = plan.meansub_ref.model_id;
            descriptor.meansub_layer = plan.meansub_ref.layer;
            descriptor.meansub_baked = identity.meansub_baked;
            descriptor.clean_stash_state = has_stash
                ? vbr_artifact_clean_stash_state::present
                : vbr_artifact_clean_stash_state::absent_at_source;
            if (has_stash) {
                if (total_columns > UINT64_MAX/sizeof(uint16_t)) {
                    status = vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                descriptor.clean_stash.valid_rows = stash_rows;
                descriptor.clean_stash.domain = vbr_repr_domain::tapped;
                descriptor.clean_stash.row_count = stash_rows;
                descriptor.clean_stash.column_count = total_columns;
                descriptor.clean_stash.row_bytes =
                    total_columns*sizeof(uint16_t);
                logical_offset = 0;
                for (const auto & shard : plan.shards) {
                    if (shard.stash_bytes == 0) {
                        status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return false;
                    }
                    vbr_artifact_shard_descriptor wire;
                    wire.shard_index = shard.shard_index;
                    wire.topology_index = shard.topology_index;
                    wire.device_ordinal = shard.device_ordinal;
                    wire.logical_offset = logical_offset;
                    wire.row_count = stash_rows;
                    wire.column_count = shard.columns;
                    wire.row_bytes = shard.columns*sizeof(uint16_t);
                    wire.payload_bytes = shard.stash_bytes;
                    descriptor.clean_stash.shards.push_back(wire);
                    logical_offset += shard.columns;
                }
            }
            current_types.push_back(
                static_cast<ggml_type>(plan.generation.current_type));
            descriptors.push_back(std::move(descriptor));
        }
        policy.current_type_vector_digest =
            vbr_type_vector_digest(current_types);
        status = vbr_explicit_capture_status::ok;
        return true;
    }

    struct projected_target_expectation {
        uint64_t manifest_id = 0;
        llama_seq_id sequence = -1;
        llama_pos frontier = -1;
        const child * source = nullptr;
        vbr_checkpoint_generation_controller generation;
        vbr_artifact_stream_placement placement;
        vbr_capture_controller_target target;
    };

    struct projected_target_recheck_context {
        const std::vector<projected_target_expectation> * expectations =
            nullptr;
    };

    static bool projected_targets_recheck(
            void * opaque,
            uint64_t manifest_id,
            const vbr_capture_controller_target * targets,
            size_t target_count) noexcept {
        try {
            const auto * context =
                static_cast<const projected_target_recheck_context *>(opaque);
            if (!context || !context->expectations || !targets ||
                target_count == 0) {
                return false;
            }
            if (size_t(std::count_if(
                    context->expectations->begin(),
                    context->expectations->end(),
                    [&](const auto & value) {
                        return value.manifest_id == manifest_id;
                    })) != target_count) {
                return false;
            }
            for (size_t i = 0; i < target_count; ++i) {
                const auto & target = targets[i];
                for (size_t prior = 0; prior < i; ++prior) {
                    if (targets[prior].child_id == target.child_id) {
                        return false;
                    }
                }
                const auto expected = std::find_if(
                    context->expectations->begin(),
                    context->expectations->end(),
                    [&](const auto & value) {
                        return value.manifest_id == manifest_id &&
                               value.target.child_id == target.child_id;
                    });
                if (expected == context->expectations->end() ||
                    expected->source == nullptr ||
                    target.manifest_id != expected->target.manifest_id ||
                    target.source_namespace !=
                        expected->target.source_namespace ||
                    !vbr_capture_controller_representation_equal(
                        target, expected->target) ||
                    !stable(*expected->source)) {
                    return false;
                }
                vbr_checkpoint_generation_controller current;
                vbr_artifact_stream_placement placement;
                vbr_explicit_generation_failure failure;
                if (!capture_generation(
                        *expected->source, expected->sequence,
                        expected->frontier, current, placement, failure) ||
                    !(current == expected->generation) ||
                    placement.child_id != expected->placement.child_id ||
                    placement.stream_index !=
                        expected->placement.stream_index ||
                    placement.source_sequence !=
                        expected->placement.source_sequence ||
                    placement.computation_frontier !=
                        expected->placement.computation_frontier ||
                    placement.cells.size() !=
                        expected->placement.cells.size()) {
                    return false;
                }
                for (size_t cell = 0;
                     cell < placement.cells.size(); ++cell) {
                    const auto & lhs = placement.cells[cell];
                    const auto & rhs = expected->placement.cells[cell];
                    if (lhs.physical_cell != rhs.physical_cell ||
                        lhs.logical_position != rhs.logical_position ||
                        lhs.ext_x != rhs.ext_x || lhs.ext_y != rhs.ext_y) {
                        return false;
                    }
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    static bool stable(const child & value) {
        return value.cache != nullptr &&
               value.cache->vbr_capture_stability_matches(value.stability);
    }

    static bool occupied_observation(
            llama_kv_cache & cache,
            llama_seq_id destination,
            uint64_t sequence_epoch,
            std::vector<vbr_occupied_replacement_cell> & cells_out,
            std::vector<vbr_occupied_replacement_unit_currency> & units_out,
            vbr_occupied_replacement_observation & output) {
        cells_out.clear();
        units_out.clear();
        output = {};
        const auto * tracker = cache.vbr_generation_tracker_get();
        if (destination < 0 || cache.other != nullptr ||
            cache.n_stream != 1 || cache.v_cells.size() != 1 ||
            cache.v_cells[0].size() == 0 ||
            cache.v_cells[0].size() > VBR_OCCUPIED_REPLACEMENT_MAX_CELLS ||
            !cache.vbr_operation_armed() || cache.vbr_import_in_progress_ ||
            !tracker || !tracker->active() || !tracker->stable()) {
            return false;
        }
        const auto & cells = cache.v_cells[0];
        cells_out.reserve(cells.get_used());
        for (uint32_t physical = 0; physical < cells.size(); ++physical) {
            if (cells.is_empty(physical)) {
                continue;
            }
            const auto & ext = cells.ext_get(physical);
            cells_out.push_back({
                0, physical, cells.pos_get(physical), ext.x, ext.y,
                cells.seq_count(physical) == 1
                    ? cells.seq_get(physical) : -1,
                uint32_t(cells.seq_count(physical)),
            });
        }
        units_out.reserve(tracker->unit_count());
        for (uint32_t unit = 0; unit < tracker->unit_count(); ++unit) {
            units_out.push_back({ 0, unit, tracker->unit_generation(unit) });
        }
        output.destination = destination;
        output.sequence_epoch = sequence_epoch;
        output.controller_generation = tracker->controller_generation();
        output.representation_epoch = cache.vbr_representation_epoch_;
        output.cell_capacity = cells.size();
        output.cells = cells_out.data();
        output.cell_count = cells_out.size();
        output.units = units_out.data();
        output.unit_count = units_out.size();
        return true;
    }

    static bool occupied_direct_currency_digest(
            llama_kv_cache & cache,
            llama_seq_id destination,
            uint64_t accounting_serial,
            const void * representation_context,
            vbr_explicit_representation_identity_fn representation_identity,
            vbr_operation_id active_import_operation,
            std::array<uint8_t, 32> & output) noexcept {
        output = {};
        try {
            const auto * tracker = cache.vbr_generation_tracker_get();
            llama_kv_cache::vbr_capture_stability_token policy;
            if (destination < 0 || accounting_serial == 0 ||
                !representation_context || !representation_identity ||
                cache.other != nullptr || cache.n_stream != 1 ||
                cache.v_cells.size() != 1 ||
                cache.v_cells[0].size() == 0 ||
                cache.v_cells[0].size() >
                    VBR_OCCUPIED_REPLACEMENT_MAX_CELLS ||
                (cache.vbr_import_in_progress_
                    ? (!active_import_operation ||
                       cache.vbr_import_operation_ != active_import_operation)
                    : bool(active_import_operation)) ||
                !cache.vbr_operation_armed() ||
                cache.vbr_stash_dirty_ ||
                !tracker || !tracker->active() || !tracker->stable() ||
                !cache.vbr_capture_policy_snapshot(policy)) {
                return false;
            }
            for (const auto & pool : cache.vbr_pools_) {
                for (const auto * extents : { &pool.k, &pool.v }) {
                    for (const auto & extent : *extents) {
                        if (extent.t != nullptr && extent.stash_valid != 0) {
                            return false;
                        }
                    }
                }
            }
            llama_sha256_writer writer;
            static constexpr char domain[] =
                "buun.vbr.occupied-replacement/direct-currency/v1";
            writer.string(domain, sizeof(domain)-1);
            writer.u64(uint64_t(reinterpret_cast<uintptr_t>(&cache)));
            writer.u64(uint64_t(destination));
            writer.u64(accounting_serial);
            writer.u64(cache.vbr_representation_epoch_);
            writer.u64(tracker->lineage_identity().hi);
            writer.u64(tracker->lineage_identity().lo);
            writer.u64(tracker->runtime_instance().hi);
            writer.u64(tracker->runtime_instance().lo);
            writer.u64(tracker->controller_generation());
            writer.bytes(policy.degrade_order_digest.data(),
                         policy.degrade_order_digest.size());
            writer.bytes(policy.policy_digest.data(),
                         policy.policy_digest.size());
            writer.u64(uint64_t(policy.floor_type));
            writer.u64(policy.pressure_independent_settings);
            writer.u64(cache.n_stream);
            writer.u64(cache.n_stream == 1 ? 1 : 0);
            writer.u64(policy.degrade_cursor);
            writer.u64(policy.completed_wave ? 1 : 0);
            const auto & cells = cache.v_cells[0];
            writer.u64(cells.size());
            writer.u64(cells.get_used());
            for (uint32_t physical = 0; physical < cells.size(); ++physical) {
                if (cells.is_empty(physical)) {
                    continue;
                }
                if (cells.seq_count(physical) != 1 ||
                    !cells.seq_has(physical, destination)) {
                    return false;
                }
                const auto & ext = cells.ext_get(physical);
                writer.u64(physical);
                writer.u64(uint64_t(cells.pos_get(physical)));
                writer.u64(uint64_t(ext.x));
                writer.u64(uint64_t(ext.y));
            }
            writer.u64(tracker->unit_count());
            for (uint32_t unit = 0; unit < tracker->unit_count(); ++unit) {
                const auto generation = tracker->unit_generation(unit);
                writer.u64(unit);
                writer.u64(generation.repr_gen);
                writer.u64(generation.publish_seq);
                writer.u64(uint64_t(generation.current_type));
                writer.u64(uint64_t(generation.last_source_type));
                writer.u64(uint64_t(generation.domain));
                writer.u64(generation.promote_hops);
                writer.u64(uint64_t(generation.last_transition));
                const size_t layer = unit/2;
                const bool value_side = (unit & 1u) != 0;
                if (layer >= cache.layers.size()) {
                    return false;
                }
                const auto * tensor = value_side
                    ? cache.layers[layer].v : cache.layers[layer].k;
                const auto meansub = cache.layers[layer].turbo_meansub_ref;
                vbr_explicit_representation_identity identity;
                if (!tensor || tensor->type != generation.current_type ||
                    !representation_identity(
                        representation_context, generation.current_type,
                        value_side, meansub.model_id, identity)) {
                    return false;
                }
                writer.u64(identity.codec_id);
                writer.u64(identity.codec_version);
                writer.bytes(identity.codebook_digest.data(),
                             identity.codebook_digest.size());
                writer.bytes(identity.rotation_digest.data(),
                             identity.rotation_digest.size());
                writer.bytes(identity.meansub_digest.data(),
                             identity.meansub_digest.size());
                writer.u64(identity.meansub_baked ? 1 : 0);
                writer.u64(uint64_t(meansub.model_id));
                writer.u64(uint64_t(meansub.layer));
            }
            output = writer.finish();
            return vbr_digest_nonzero(output);
        } catch (...) {
            output = {};
            return false;
        }
    }

    static bool stream(
            const child & value,
            const llama_kv_cache::vbr_capture_unit_plan & unit,
            vbr_unit_build & sink,
            vbr_pinned_chunk_ring & ring,
            vbr_capture_stream_stats & stats,
            void * continue_context = nullptr,
            vbr_explicit_capture_request::continue_transfer_fn
                continue_transfer = nullptr) {
        return value.cache != nullptr &&
               value.cache->vbr_capture_stream_unit(
                   unit, sink, ring, stats,
                   continue_context, continue_transfer);
    }
};


static bool import_target_snapshot_core(
    llama_memory_i & memory,
    llama_seq_id destination,
    const vbr_artifact_package_view & package,
    const std::vector<llama_vbr_artifact_domain_binding> & bindings,
    bool previously_observed,
    uint64_t accounting_serial,
    const void * representation_context,
    vbr_explicit_capture_request::representation_identity_fn
        representation_identity,
    vbr_target_validation_snapshot & output,
    vbr_downward_policy_projection * transform_projection,
    bool * downward_required,
    vbr_import_schedule_quote * schedule_quote,
    const vbr_import_schedule_quote * authenticated_schedule,
    std::array<uint8_t, 32> * transform_tree_digest = nullptr,
    const std::vector<llama_memory_tree_child> * canonical_tree = nullptr,
    uint64_t selected_frontier = 0)
    noexcept;


vbr_occupied_replacement_guard_status
vbr_explicit_prepare_occupied_replacement_guard(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_artifact_package_view & incoming,
        const vbr_artifact_package_view & recovery,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        uint64_t accounting_serial,
        const void * representation_context,
        vbr_explicit_representation_identity_fn representation_identity,
        vbr_occupied_replacement_guard & output,
        const vbr_import_schedule_quote * authenticated_incoming,
        const std::vector<vbr_target_companion_snapshot> *
            external_companions) noexcept {
    output.reset();
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return vbr_occupied_replacement_guard_status::unsupported_tree;
        }
        const auto attention = std::find_if(
            tree.begin(), tree.end(), [](const auto & child) {
                return child.attention != nullptr && child.recurrent == nullptr;
            });
        if (attention == tree.end() ||
            std::count_if(tree.begin(), tree.end(), [](const auto & child) {
                return child.attention != nullptr;
            }) != 1 ||
            std::count_if(tree.begin(), tree.end(), [](const auto & child) {
                return child.recurrent != nullptr;
            }) > 1 ||
            std::any_of(tree.begin(), tree.end(), [](const auto & child) {
                return child.attention == nullptr && child.recurrent == nullptr;
            })) {
            return vbr_occupied_replacement_guard_status::unsupported_tree;
        }
        vbr_target_validation_snapshot target;
        vbr_import_schedule_quote recovery_quote;
        if (!import_target_snapshot_core(
                memory, destination, recovery, bindings, true,
                accounting_serial, representation_context,
                representation_identity, target,
                nullptr, nullptr, &recovery_quote, nullptr, nullptr, &tree) ||
            target.destination_sequence_absent) {
            return vbr_occupied_replacement_guard_status::unsupported_tree;
        }
        if (external_companions) {
            target.companions = *external_companions;
        }
        std::vector<vbr_occupied_replacement_cell> cells;
        std::vector<vbr_occupied_replacement_unit_currency> units;
        vbr_occupied_replacement_observation observation;
        if (!vbr_live_capture_adapter::occupied_observation(
                *attention->attention, destination,
                recovery.manifest().identity.sequence_epoch,
                cells, units, observation)) {
            return vbr_occupied_replacement_guard_status::unsupported_layout;
        }
        vbr_target_validation_snapshot selected_target = target;
        if (authenticated_incoming) {
            vbr_downward_policy_projection selected_projection;
            bool selected_downward = false;
            if (!import_target_snapshot_core(
                    memory, destination, incoming, bindings, true,
                    accounting_serial, representation_context,
                    representation_identity, selected_target,
                    &selected_projection, &selected_downward, nullptr,
                    authenticated_incoming, nullptr, &tree)) {
                return vbr_occupied_replacement_guard_status::
                    representation_mismatch;
            }
            if (external_companions) {
                selected_target.companions = *external_companions;
            }
        }
        const auto status = vbr_prepare_occupied_replacement_guard(
            target, selected_target, incoming, recovery, observation, output,
            authenticated_incoming, &recovery_quote);
        if (status != vbr_occupied_replacement_guard_status::ready) {
            return status;
        }
        std::array<uint8_t, 32> direct;
        if (!vbr_live_capture_adapter::occupied_direct_currency_digest(
                *attention->attention, destination, accounting_serial,
                representation_context, representation_identity, {}, direct)) {
            output.reset();
            return vbr_occupied_replacement_guard_status::currency_changed;
        }
        output.memory_ = &memory;
        output.cache_ = attention->attention;
        output.direct_currency_digest_ = direct;
        return status;
    } catch (...) {
        output.reset();
        return vbr_occupied_replacement_guard_status::internal_error;
    }
}

vbr_occupied_replacement_guard_status
vbr_explicit_prepare_occupied_prefix_replacement_guard(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_artifact_package_view & incoming_parent,
        uint64_t prefix_tokens,
        const std::vector<vbr_artifact_prefix_cell_run> & prefix_runs,
        const vbr_artifact_package_view & recovery,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        uint64_t accounting_serial,
        const void * representation_context,
        vbr_explicit_representation_identity_fn representation_identity,
        vbr_occupied_replacement_guard & output,
        const vbr_import_schedule_quote & authenticated_incoming) noexcept {
    output.reset();
    try {
        std::vector<llama_memory_tree_child> tree;
        if (prefix_tokens == 0 ||
            !llama_memory_tree_collect(&memory, tree) || tree.size() != 1 ||
            !tree.front().attention || tree.front().recurrent) {
            return vbr_occupied_replacement_guard_status::unsupported_tree;
        }
        vbr_target_validation_snapshot live_target;
        vbr_import_schedule_quote recovery_quote;
        if (!import_target_snapshot_core(
                memory, destination, recovery, bindings, false,
                accounting_serial, representation_context,
                representation_identity, live_target,
                nullptr, nullptr, &recovery_quote, nullptr, nullptr, &tree) ||
            live_target.destination_sequence_absent) {
            return vbr_occupied_replacement_guard_status::unsupported_tree;
        }
        vbr_target_validation_snapshot selected_target;
        vbr_downward_policy_projection selected_projection;
        bool selected_downward = false;
        if (!import_target_snapshot_core(
                memory, destination, incoming_parent, bindings, false,
                accounting_serial, representation_context,
                representation_identity, selected_target,
                &selected_projection, &selected_downward, nullptr,
                &authenticated_incoming, nullptr, &tree, prefix_tokens)) {
            return vbr_occupied_replacement_guard_status::representation_mismatch;
        }
        std::vector<vbr_occupied_replacement_cell> cells;
        std::vector<vbr_occupied_replacement_unit_currency> units;
        vbr_occupied_replacement_observation observation;
        if (!vbr_live_capture_adapter::occupied_observation(
                *tree.front().attention, destination,
                recovery.manifest().identity.sequence_epoch,
                cells, units, observation)) {
            return vbr_occupied_replacement_guard_status::unsupported_layout;
        }
        const auto status = vbr_prepare_occupied_prefix_replacement_guard(
            live_target, selected_target, incoming_parent, prefix_tokens,
            prefix_runs, recovery, observation, output,
            authenticated_incoming, &recovery_quote);
        if (status != vbr_occupied_replacement_guard_status::ready) {
            return status;
        }
        std::array<uint8_t, 32> direct;
        if (!vbr_live_capture_adapter::occupied_direct_currency_digest(
                *tree.front().attention, destination, accounting_serial,
                representation_context, representation_identity, {}, direct)) {
            output.reset();
            return vbr_occupied_replacement_guard_status::currency_changed;
        }
        output.memory_ = &memory;
        output.cache_ = tree.front().attention;
        output.direct_currency_digest_ = direct;
        return status;
    } catch (...) {
        output.reset();
        return vbr_occupied_replacement_guard_status::internal_error;
    }
}

vbr_occupied_replacement_guard_status
vbr_explicit_recheck_occupied_replacement_guard(
        llama_memory_i & memory,
        llama_seq_id destination,
        uint64_t accounting_serial,
        const void * representation_context,
        vbr_explicit_representation_identity_fn representation_identity,
        vbr_occupied_replacement_guard & guard) noexcept {
    return vbr_explicit_recheck_occupied_replacement_guard(
        memory, destination, accounting_serial, representation_context,
        representation_identity, {}, guard);
}

vbr_occupied_replacement_guard_status
vbr_explicit_recheck_occupied_replacement_guard(
        llama_memory_i & memory,
        llama_seq_id destination,
        uint64_t accounting_serial,
        const void * representation_context,
        vbr_explicit_representation_identity_fn representation_identity,
        vbr_operation_id active_import_operation,
        vbr_occupied_replacement_guard & guard) noexcept {
    if (!guard.ready() || guard.memory_ != &memory || !guard.cache_ ||
        guard.destination_ != destination ||
        guard.accounting_serial_ != accounting_serial ||
        !vbr_digest_nonzero(guard.direct_currency_digest_)) {
        guard.reset();
        return vbr_occupied_replacement_guard_status::currency_changed;
    }
    std::array<uint8_t, 32> current;
    if (!vbr_live_capture_adapter::occupied_direct_currency_digest(
            *guard.cache_, destination, accounting_serial,
            representation_context, representation_identity,
            active_import_operation, current) ||
        current != guard.direct_currency_digest_) {
        guard.reset();
        return vbr_occupied_replacement_guard_status::currency_changed;
    }
    return vbr_occupied_replacement_guard_status::ready;
}


namespace {

uint64_t import_tree_digest(
        llama_memory_i & memory,
        const std::vector<llama_memory_tree_child> & tree) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] = "buun.vbr.import-tree/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    writer.u64(uint64_t(reinterpret_cast<uintptr_t>(&memory)));
    for (const auto & child : tree) {
        writer.u32(child.child_id);
        writer.u32(uint32_t(child.dependency_mode));
        writer.u32(child.attention != nullptr);
        writer.u32(child.recurrent != nullptr);
        writer.u64(uint64_t(reinterpret_cast<uintptr_t>(
            child.attention ? static_cast<void *>(child.attention) :
            static_cast<void *>(child.recurrent))));
    }
    return digest_head_u64(writer);
}

uint64_t import_policy_epoch(
        const std::vector<llama_memory_tree_child> & tree) {
    llama_sha256_writer writer;
    static constexpr char domain_label[] = "buun.vbr.import-policy/v1";
    writer.string(domain_label, sizeof(domain_label) - 1);
    size_t n_attention = 0;
    for (const auto & child : tree) {
        if (!child.attention) {
            continue;
        }
        writer.u32(child.child_id);
        if (!vbr_live_capture_adapter::append_policy_identity(
                *child.attention, writer)) {
            return 0;
        }
        ++n_attention;
    }
    if (n_attention == 0) {
        return 0;
    }
    return digest_head_u64(writer);
}

} // namespace

namespace {

bool recurrent_target_empty(
        const llama_memory_tree_child & child) noexcept {
    if (!child.recurrent || child.attention) {
        return false;
    }
    const auto provider =
        vbr_recurrent_companion_adoption_provider(*child.recurrent);
    return provider.target_empty &&
           provider.target_empty(provider.context);
}

} // namespace

bool vbr_explicit_capture_runtime_pools(
        llama_memory_i & memory,
        std::vector<vbr_explicit_capture_runtime_pool> & pools,
        uint32_t & attention_children) noexcept {
    pools.clear();
    attention_children = 0;
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return false;
        }
        for (const auto & node : tree) {
            if (node.attention == nullptr) {
                continue;
            }
            ++attention_children;
            if (!vbr_live_capture_adapter::runtime_pools(
                    *node.attention, pools)) {
                pools.clear();
                attention_children = 0;
                return false;
            }
        }
        return attention_children != 0 && !pools.empty();
    } catch (...) {
        pools.clear();
        attention_children = 0;
        return false;
    }
}

static bool vbr_projected_host_metadata_bytes(
        uint64_t projected_units,
        uint64_t & descriptor_bytes,
        uint64_t & reference_bytes) noexcept {
    descriptor_bytes = 0;
    reference_bytes = 0;
    if (projected_units == 0 || projected_units > UINT64_MAX/256) {
        return false;
    }
    descriptor_bytes = std::max<uint64_t>(1, projected_units*256);
    reference_bytes = std::max<uint64_t>(1, projected_units*128);
    return reference_bytes <= UINT64_MAX - descriptor_bytes;
}

bool vbr_plan_attention_stem_prefix(
        uint64_t requested_token_count,
        const vbr_artifact_cell_placement * cells,
        size_t cell_count,
        uint64_t bytes_per_logical_row,
        uint64_t max_packed_bytes,
        uint64_t projected_units,
        const vbr_projected_capture_frontier_policy & policy,
        vbr_attention_stem_prefix_plan & output) noexcept {
    output = {};
    if (requested_token_count == 0 ||
        requested_token_count > VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS ||
        bytes_per_logical_row == 0 ||
        cell_count > requested_token_count ||
        (cell_count != 0 && cells == nullptr)) {
        return false;
    }
    uint64_t descriptor_bytes = 0;
    uint64_t reference_bytes = 0;
    if (!vbr_projected_host_metadata_bytes(
            projected_units, descriptor_bytes, reference_bytes)) {
        return false;
    }
    const uint64_t metadata_bytes = descriptor_bytes + reference_bytes;
    uint64_t payload_cap = std::min(
        max_packed_bytes, UINT64_MAX - metadata_bytes);
    if (policy.max_host_resident_bytes != 0) {
        payload_cap = policy.max_host_resident_bytes > metadata_bytes
            ? std::min(
                payload_cap,
                policy.max_host_resident_bytes - metadata_bytes)
            : 0;
    }
    const uint64_t token_cap = policy.max_host_resident_bytes == 0 &&
            policy.max_host_tokens != 0
        ? std::min(requested_token_count, policy.max_host_tokens)
        : requested_token_count;
    try {
        std::vector<uint8_t> present(
            size_t(requested_token_count), 0);
        for (size_t i = 0; i < cell_count; ++i) {
            const auto position = cells[i].logical_position;
            if (position < 0 ||
                uint64_t(position) >= requested_token_count ||
                present[size_t(position)] != 0) {
                return false;
            }
            present[size_t(position)] = 1;
            ++output.surveyed_cells;
        }
        for (uint64_t position = 0;
             position < token_cap; ++position) {
            if (present[size_t(position)] == 0 ||
                bytes_per_logical_row >
                    payload_cap - output.planned_packed_bytes) {
                break;
            }
            output.planned_packed_bytes += bytes_per_logical_row;
            output.selected_token_count = position + 1;
        }
        output.projected_host_resident_bytes = metadata_bytes +
            output.planned_packed_bytes;
        output.selected_next_position =
            llama_pos(output.selected_token_count);
        if (output.selected_token_count <
                std::max<uint64_t>(1, policy.minimum_tokens)) {
            output.status =
                vbr_projected_capture_frontier_status::stem_below_minimum;
        } else if (output.selected_token_count == requested_token_count) {
            output.status =
                vbr_projected_capture_frontier_status::exact;
        } else {
            output.status =
                vbr_projected_capture_frontier_status::stem_selected;
        }
        return true;
    } catch (...) {
        output = {};
        return false;
    }
}

vbr_projected_capture_batch_result vbr_capture_projected_batch(
        llama_memory_i & memory,
        const vbr_projected_capture_batch_request & request) noexcept {
    vbr_projected_capture_batch_result result;
    if (request.frontier.mode ==
            vbr_projected_capture_frontier_mode::exact) {
        result.frontier_status =
            vbr_projected_capture_frontier_status::exact;
    } else if (request.frontier.mode !=
            vbr_projected_capture_frontier_mode::longest_attention_stem) {
        result.frontier_status =
            vbr_projected_capture_frontier_status::stem_unsupported;
        result.status = vbr_explicit_capture_status::identity_unavailable;
        return result;
    }
    if (!request.idle_decode_thread || request.manifests.empty() ||
        request.manifests.size() > VBR_PROJECTED_CAPTURE_MAX_MANIFESTS ||
        request.ring == nullptr || request.topologies.empty() ||
        request.pool_bindings.empty() ||
        request.max_packed_bytes == 0 ||
        request.representation_identity == nullptr) {
        result.status = request.idle_decode_thread
            ? vbr_explicit_capture_status::identity_unavailable
            : vbr_explicit_capture_status::slot_not_idle;
        return result;
    }

    try {
        result.phase = vbr_explicit_capture_phase::memory_tree;
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            result.status = vbr_explicit_capture_status::unsupported_layout;
            return result;
        }
        std::vector<vbr_live_capture_adapter::child> children;
        std::vector<llama_memory_recurrent *> recurrent_children;
        size_t qsa_index_children = 0;
        for (const auto & node : tree) {
            qsa_index_children += node.qsa_index_owner != nullptr;
            if (node.recurrent != nullptr) {
                recurrent_children.push_back(node.recurrent);
                continue;
            }
            if (node.attention == nullptr ||
                !node.attention->vbr_operation_armed() ||
                node.dependency_mode !=
                    checkpoint_child_dependency_mode::live_guarded) {
                result.status = vbr_explicit_capture_status::unsupported_layout;
                return result;
            }
            vbr_live_capture_adapter::child child;
            child.child_id = node.child_id;
            child.dependency_mode = node.dependency_mode;
            child.cache = node.attention;
            children.push_back(std::move(child));
        }
        if (children.empty()) {
            result.status = vbr_explicit_capture_status::not_armed;
            return result;
        }
        if (qsa_index_children > 1) {
            result.status = vbr_explicit_capture_status::unsupported_layout;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::settlement;
        std::vector<vbr_controller_instance_id> instances;
        instances.reserve(children.size());
        for (auto & child : children) {
            if (!vbr_live_capture_adapter::settle(*child.cache)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            const auto instance =
                vbr_live_capture_adapter::instance(child);
            if (!vbr_controller_instance_id_is_set(instance) ||
                std::find(instances.begin(), instances.end(), instance) !=
                    instances.end() ||
                vbr_recovery_pending_for(instance)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            instances.push_back(instance);
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::metadata_and_manifest;
        struct child_schema {
            vbr_artifact_controller_policy policy;
            std::vector<vbr_artifact_unit_descriptor> descriptors;
        };
        std::vector<child_schema> schemas(children.size());
        std::vector<vbr_live_capture_adapter::representation_cache_entry>
            representation_cache;
        for (size_t i = 0; i < children.size(); ++i) {
            ++result.size_pass_calls;
            if (!vbr_live_capture_adapter::capture_size(
                    children[i], request.pool_bindings,
                    result.generation_failure, result.size_failure)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            if (children[i].dependency_mode !=
                    checkpoint_child_dependency_mode::live_guarded ||
                children[i].units.empty() ||
                children[i].units.front().n_stream != 1 ||
                !children[i].units.front().unified) {
                result.status =
                    vbr_explicit_capture_status::unsupported_layout;
                return result;
            }
            if (!vbr_live_capture_adapter::capture_schema(
                    children[i], request.representation_context,
                    request.representation_identity,
                    false,
                    representation_cache,
                    schemas[i].policy, schemas[i].descriptors,
                    result.status)) {
                if (result.status ==
                        vbr_explicit_capture_status::internal_error) {
                    result.status =
                        vbr_explicit_capture_status::generation_unavailable;
                }
                return result;
            }
        }

        // Stem planning is deliberately absent from exact capture. For its
        // narrow singleton mode, survey the authenticated live placement once
        // and price each complete logical row from the already validated
        // physical unit geometry. The selected prefix is recaptured at most
        // once below; the final union therefore remains the sole projection
        // and transfer authority.
        uint64_t selected_frontier_tokens = 0;
        llama_pos selected_frontier_next_position = -1;
        vbr_checkpoint_generation_controller surveyed_generation;
        vbr_artifact_stream_placement surveyed_placement;
        bool surveyed_full_frontier = false;
        if (request.frontier.mode ==
                vbr_projected_capture_frontier_mode::longest_attention_stem) {
            if (request.manifests.size() != 1 || children.size() != 1 ||
                !recurrent_children.empty()) {
                result.frontier_status =
                    vbr_projected_capture_frontier_status::stem_unsupported;
                result.status = recurrent_children.empty()
                    ? vbr_explicit_capture_status::unsupported_layout
                    : vbr_explicit_capture_status::
                        required_companion_unavailable;
                return result;
            }
            const auto & manifest = request.manifests.front();
            const auto & identity = manifest.identity;
            if (manifest.manifest_id == 0 || manifest.sequence < 0 ||
                identity.execution_identity.empty() ||
                identity.adapter_config_identity.empty() ||
                identity.media_content_identity.empty() ||
                identity.sequence_epoch == 0 || !manifest.text_only ||
                identity.token_count <= 0 ||
                identity.next_position != identity.token_count ||
                uint64_t(identity.token_count) >
                    VBR_PROJECTED_CAPTURE_MAX_TOKEN_IDS ||
                manifest.token_block.size() !=
                    size_t(identity.token_count) ||
                digest_nonzero(manifest.identity_policy_order_digest) ||
                std::any_of(
                    children.front().units.begin(),
                    children.front().units.end(),
                    [](const auto & unit) {
                        return unit.n_stream != 1 || !unit.unified;
                    })) {
                result.frontier_status =
                    vbr_projected_capture_frontier_status::stem_unsupported;
                result.status =
                    vbr_explicit_capture_status::unsupported_layout;
                return result;
            }
            result.requested_frontier_tokens =
                uint64_t(identity.token_count);
            uint64_t bytes_per_logical_row = 0;
            for (const auto & unit : children.front().units) {
                for (const auto & shard : unit.shards) {
                    if (shard.row_bytes == 0 ||
                        shard.row_bytes >
                            UINT64_MAX - bytes_per_logical_row) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return result;
                    }
                    bytes_per_logical_row += shard.row_bytes;
                }
            }
            if (bytes_per_logical_row == 0) {
                result.frontier_status =
                    vbr_projected_capture_frontier_status::stem_unsupported;
                result.status =
                    vbr_explicit_capture_status::unsupported_layout;
                return result;
            }
            ++result.frontier_survey_calls;
            if (!vbr_live_capture_adapter::capture_generation(
                    children.front(), manifest.sequence,
                    identity.next_position, surveyed_generation,
                    surveyed_placement, result.generation_failure)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            surveyed_full_frontier = true;
            vbr_attention_stem_prefix_plan prefix;
            if (!vbr_plan_attention_stem_prefix(
                    result.requested_frontier_tokens,
                    surveyed_placement.cells.data(),
                    surveyed_placement.cells.size(),
                    bytes_per_logical_row, request.max_packed_bytes,
                    schemas.front().descriptors.size(), request.frontier,
                    prefix)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            selected_frontier_tokens = prefix.selected_token_count;
            selected_frontier_next_position =
                prefix.selected_next_position;
            result.frontier_status = prefix.status;
            result.selected_frontier_tokens =
                prefix.selected_token_count;
            result.selected_frontier_next_position =
                prefix.selected_next_position;
            result.frontier_survey_cells = prefix.surveyed_cells;
            if (prefix.status ==
                    vbr_projected_capture_frontier_status::
                        stem_below_minimum) {
                result.status =
                    vbr_explicit_capture_status::accounting_failed;
                return result;
            }
            if (result.frontier_status ==
                    vbr_projected_capture_frontier_status::stem_selected) {
                // The selected recapture is authoritative. Drop the full
                // survey arenas before allocating that smaller generation.
                surveyed_generation = {};
                surveyed_placement = {};
                surveyed_full_frontier = false;
            }
        } else if (request.manifests.size() == 1) {
            const auto & identity = request.manifests.front().identity;
            if (identity.token_count > 0) {
                result.requested_frontier_tokens =
                    uint64_t(identity.token_count);
                result.selected_frontier_tokens =
                    uint64_t(identity.token_count);
                result.selected_frontier_next_position =
                    identity.next_position;
            }
        }

        llama_sha256_writer namespace_hash;
        static constexpr char NAMESPACE_DOMAIN[] =
            "buun.vbr.projected-capture/tree-namespace/v1";
        namespace_hash.string(
            NAMESPACE_DOMAIN, sizeof(NAMESPACE_DOMAIN) - 1);
        namespace_hash.u64(
            uint64_t(reinterpret_cast<uintptr_t>(&memory)));
        static std::atomic<uint64_t> next_capture_namespace { 1 };
        namespace_hash.u64(next_capture_namespace.fetch_add(
            1, std::memory_order_relaxed));
        for (size_t i = 0; i < children.size(); ++i) {
            namespace_hash.u32(children[i].child_id);
            namespace_hash.u64(instances[i].hi);
            namespace_hash.u64(instances[i].lo);
        }
        result.source_namespace = digest_head_u64(namespace_hash);
        if (result.source_namespace == 0) {
            result.status =
                vbr_explicit_capture_status::generation_unavailable;
            return result;
        }

        std::vector<vbr_capture_projection_manifest> projection_manifests;
        std::vector<vbr_capture_controller_target> targets;
        std::vector<vbr_live_capture_adapter::projected_target_expectation>
            expectations;
        projection_manifests.reserve(request.manifests.size());
        targets.reserve(request.manifests.size()*children.size());
        expectations.reserve(request.manifests.size()*children.size());
        std::vector<uint64_t> manifest_ids;
        std::vector<std::vector<projected_companion_plan>> companion_plans;
        std::vector<bool> manifest_dependency_available;
        manifest_ids.reserve(request.manifests.size());
        companion_plans.reserve(request.manifests.size());
        manifest_dependency_available.reserve(request.manifests.size());
        for (const auto & manifest_request : request.manifests) {
            vbr_artifact_identity_block selected_identity;
            const vbr_artifact_identity_block * identity_ptr =
                &manifest_request.identity;
            const bool stemmed = request.frontier.mode ==
                    vbr_projected_capture_frontier_mode::
                        longest_attention_stem &&
                selected_frontier_tokens !=
                    result.requested_frontier_tokens;
            if (stemmed) {
                selected_identity = manifest_request.identity;
                selected_identity.token_count =
                    int64_t(selected_frontier_tokens);
                selected_identity.next_position =
                    selected_frontier_next_position;
                identity_ptr = &selected_identity;
            }
            const auto & identity = *identity_ptr;
            if (manifest_request.manifest_id == 0 ||
                manifest_request.sequence < 0 ||
                identity.execution_identity.empty() ||
                identity.adapter_config_identity.empty() ||
                identity.media_content_identity.empty() ||
                identity.sequence_epoch == 0 || identity.token_count <= 0 ||
                identity.next_position <= 0 ||
                manifest_request.token_block.size() !=
                    size_t(manifest_request.identity.token_count) ||
                std::find(
                    manifest_ids.begin(), manifest_ids.end(),
                    manifest_request.manifest_id) != manifest_ids.end()) {
                result.status =
                    vbr_explicit_capture_status::identity_unavailable;
                return result;
            }
            manifest_ids.push_back(manifest_request.manifest_id);
            vbr_capture_projection_manifest projected;
            projected.manifest_id = manifest_request.manifest_id;
            projected.identity = identity;
            if (stemmed) {
                projected.token_block.tokens.assign(
                    manifest_request.token_block.begin(),
                    manifest_request.token_block.begin() +
                        size_t(identity.token_count));
            } else {
                projected.token_block.tokens = manifest_request.token_block;
            }
            projected.generation.version = 1;
            projected.generation.status =
                vbr_checkpoint_generation_status::complete;
            bool dependencies_available = true;
            std::vector<projected_companion_plan> manifest_companion_plans;
            manifest_companion_plans.reserve(
                recurrent_children.size() + manifest_request.companions.size());
            const size_t supplied_recurrent = size_t(std::count_if(
                manifest_request.companions.begin(),
                manifest_request.companions.end(),
                [](const vbr_explicit_companion_provider & provider) {
                    return provider.kind ==
                        vbr_artifact_companion_kind::recurrent;
                }));
            const size_t supplied_qsa = size_t(std::count_if(
                manifest_request.companions.begin(),
                manifest_request.companions.end(),
                [](const vbr_explicit_companion_provider & provider) {
                    return provider.kind ==
                        vbr_artifact_companion_kind::qsa_index;
                }));
            if (supplied_recurrent != 0 &&
                supplied_recurrent != recurrent_children.size()) {
                dependencies_available = false;
            }
            if (supplied_qsa != qsa_index_children) {
                dependencies_available = false;
            }
            for (auto * recurrent : recurrent_children) {
                if (!dependencies_available || supplied_recurrent != 0) {
                    break;
                }
                projected_companion_plan companion;
                vbr_explicit_capture_status companion_status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                if (!recurrent_companion_prepare(
                        recurrent, manifest_request.sequence,
                        identity.next_position,
                        companion.recurrent, companion_status)) {
                    dependencies_available = false;
                    projected.companions.clear();
                    manifest_companion_plans.clear();
                    break;
                }
                companion.descriptor = companion.recurrent.descriptor;
                projected.companions.push_back(companion.descriptor);
                manifest_companion_plans.push_back(std::move(companion));
            }
            for (const auto & provider : manifest_request.companions) {
                if (!dependencies_available) {
                    break;
                }
                projected_companion_plan companion;
                vbr_explicit_capture_status companion_status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                if (!projected_provider_prepare(
                        provider, manifest_request.sequence,
                        identity.next_position, companion,
                        companion_status)) {
                    if (provider.required) {
                        dependencies_available = false;
                        projected.companions.clear();
                        manifest_companion_plans.clear();
                    }
                    continue;
                }
                projected.companions.push_back(companion.descriptor);
                manifest_companion_plans.push_back(std::move(companion));
            }
            std::vector<vbr_identity_policy_digest_row> identity_policy;
            identity_policy.reserve(children.size());
            for (size_t child_index = 0;
                 child_index < children.size(); ++child_index) {
                auto & child = children[child_index];
                vbr_checkpoint_generation_controller generation;
                vbr_artifact_stream_placement placement;
                if (request.frontier.mode ==
                        vbr_projected_capture_frontier_mode::
                            longest_attention_stem &&
                    !stemmed && child_index == 0 &&
                    surveyed_full_frontier) {
                    generation = std::move(surveyed_generation);
                    placement = std::move(surveyed_placement);
                    surveyed_full_frontier = false;
                } else if (!vbr_live_capture_adapter::capture_generation(
                               child, manifest_request.sequence,
                               identity.next_position, generation, placement,
                               result.generation_failure)) {
                    result.status =
                        vbr_explicit_capture_status::generation_unavailable;
                    return result;
                } else if (request.frontier.mode ==
                        vbr_projected_capture_frontier_mode::
                            longest_attention_stem) {
                    ++result.frontier_recapture_calls;
                }
                const llama_pos terminal_position =
                    identity.next_position - 1;
                const auto terminal_cell = std::max_element(
                    placement.cells.begin(), placement.cells.end(),
                    [](const auto & lhs, const auto & rhs) {
                        return lhs.logical_position < rhs.logical_position;
                    });
                if (terminal_cell == placement.cells.end() ||
                    terminal_cell->logical_position != terminal_position) {
                    dependencies_available = false;
                }
                projected.generation.controllers.push_back(generation);
                projected.placements.push_back(placement);
                identity_policy.push_back({
                    child.child_id, child.dependency_mode,
                    child.stability.lineage_uuid,
                });

                if (!dependencies_available) {
                    continue;
                }
                vbr_capture_controller_target target;
                target.manifest_id = manifest_request.manifest_id;
                target.source_namespace = result.source_namespace;
                target.child_id = child.child_id;
                target.lineage_uuid = child.stability.lineage_uuid;
                target.controller_generation =
                    child.stability.controller_generation;
                target.policy = schemas[child_index].policy;
                target.unit_descriptors = schemas[child_index].descriptors;
                target.units.reserve(child.units.size());
                for (const auto & unit : child.units) {
                    target.units.push_back(unit.generation);
                }
                vbr_live_capture_adapter::projected_target_expectation
                    expectation;
                expectation.manifest_id = manifest_request.manifest_id;
                expectation.sequence = manifest_request.sequence;
                expectation.frontier = identity.next_position;
                expectation.source = &child;
                expectation.generation = std::move(generation);
                expectation.placement = std::move(placement);
                expectation.target = target;
                expectations.push_back(std::move(expectation));
                targets.push_back(std::move(target));
            }
            vbr_checkpoint_frontier_fields frontier;
            frontier.execution_identity = identity.execution_identity.data();
            frontier.execution_identity_len =
                identity.execution_identity.size();
            frontier.adapter_config_identity =
                identity.adapter_config_identity.data();
            frontier.adapter_config_identity_len =
                identity.adapter_config_identity.size();
            frontier.media_content_identity =
                identity.media_content_identity.data();
            frontier.media_content_identity_len =
                identity.media_content_identity.size();
            frontier.sequence_epoch = identity.sequence_epoch;
            frontier.token_count = identity.token_count;
            frontier.next_position = identity.next_position;
            const auto identity_digest =
                vbr_identity_policy_digest(frontier, identity_policy);
            if (digest_nonzero(
                    manifest_request.identity_policy_order_digest) &&
                identity_digest !=
                    manifest_request.identity_policy_order_digest) {
                result.status =
                    vbr_explicit_capture_status::identity_unavailable;
                return result;
            }
            projected.identity_policy_order_digest = identity_digest;
            projected.generation.identity_policy_order_digest =
                identity_digest;
            companion_plans.push_back(
                std::move(manifest_companion_plans));
            manifest_dependency_available.push_back(
                dependencies_available);
            projection_manifests.push_back(std::move(projected));
        }

        const auto manifest_available = [&](uint64_t manifest_id) {
            const auto manifest = std::find(
                manifest_ids.begin(), manifest_ids.end(), manifest_id);
            return manifest != manifest_ids.end() &&
                manifest_dependency_available[size_t(
                    manifest - manifest_ids.begin())];
        };
        const auto apply_dependency_availability =
                [&](std::vector<vbr_capture_projection_manifest> & manifests) {
            for (auto & projected : manifests) {
                const auto manifest = std::find(
                    manifest_ids.begin(), manifest_ids.end(),
                    projected.manifest_id);
                if (manifest == manifest_ids.end()) {
                    continue;
                }
                const size_t manifest_index = size_t(
                    manifest - manifest_ids.begin());
                if (manifest_dependency_available[manifest_index]) {
                    continue;
                }
                projected.dependencies_available = false;
                projected.placements.clear();
                projected.companions.clear();
                companion_plans[manifest_index].clear();
            }
            targets.erase(
                std::remove_if(
                    targets.begin(), targets.end(),
                    [&](const auto & target) {
                        return !manifest_available(target.manifest_id);
                    }),
                targets.end());
            expectations.erase(
                std::remove_if(
                    expectations.begin(), expectations.end(),
                    [&](const auto & expectation) {
                        return !manifest_available(expectation.manifest_id);
                    }),
                expectations.end());
        };
        apply_dependency_availability(projection_manifests);
        for (size_t i = 0; i < manifest_ids.size(); ++i) {
            if (manifest_dependency_available[i]) {
                result.first_available_manifest_id = manifest_ids[i];
                break;
            }
        }

        vbr_capture_projection projection;
        vbr_capture_projection_limits projection_limits;
        projection_limits.max_manifests =
            VBR_PROJECTED_CAPTURE_MAX_MANIFESTS;
        ++result.projection_calls;
        if (!vbr_artifact_project_capture_union(
                { result.source_namespace,
                  std::move(projection_manifests) },
                projection_limits, projection)) {
            result.status = vbr_explicit_capture_status::generation_unavailable;
            return result;
        }
        result.union_cells = projection->union_cell_count;

        std::vector<vbr_projected_capture_batch_request::
            pretransfer_quote::staging_row> staging_rows;
        std::vector<vbr_artifact_portable_accounting_row>
            attention_accounting;
        std::vector<std::vector<vbr_artifact_portable_accounting_row>>
            attention_reserve_accounting(request.manifests.size());
        uint32_t projected_unit_count = 0;
        const auto preflight_projection_bytes = [&]() {
            result.planned_packed_bytes = 0;
            staging_rows.clear();
            attention_accounting.clear();
            for (auto & rows : attention_reserve_accounting) {
                rows.clear();
            }
            projected_unit_count = 0;
            const auto add_planned_bytes = [&] (
                    const vbr_artifact_portable_domain & domain,
                    uint64_t bytes) {
                if (bytes == 0) {
                    return true;
                }
                if (bytes > UINT64_MAX - result.planned_packed_bytes) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                result.planned_packed_bytes += bytes;
                if (result.planned_packed_bytes > request.max_packed_bytes) {
                    result.status =
                        vbr_explicit_capture_status::accounting_failed;
                    return false;
                }
                const auto found = std::lower_bound(
                    staging_rows.begin(), staging_rows.end(), domain,
                    [&](const auto & row, const auto & key) {
                        return vbr_artifact_portable_domain_less(
                            row.domain, key);
                    });
                if (found != staging_rows.end()) {
                    if (found->domain != domain) {
                        staging_rows.insert(found, { domain, bytes });
                    } else if (bytes > UINT64_MAX - found->bytes) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return false;
                    } else {
                        found->bytes += bytes;
                    }
                } else {
                    staging_rows.push_back({ domain, bytes });
                }
                return true;
            };
            for (const auto & child : children) {
                const auto stream = std::find_if(
                    projection->streams.begin(), projection->streams.end(),
                    [&](const auto & value) {
                        return value.child_id == child.child_id &&
                            value.stream_index == 0;
                    });
                if (stream == projection->streams.end()) {
                    continue;
                }
                if (child.units.size() >
                        UINT32_MAX - projected_unit_count) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                projected_unit_count += uint32_t(child.units.size());
                uint64_t projected_rows = 0;
                uint64_t owner_manifest = UINT64_MAX;
                for (const auto & segment : stream->segments) {
                    if (segment.cell_count > UINT64_MAX - projected_rows) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return false;
                    }
                    projected_rows += segment.cell_count;
                    if (segment.first_dependency >
                            projection->dependent_manifest_ids.size() ||
                        segment.dependency_count >
                            projection->dependent_manifest_ids.size() -
                                segment.first_dependency) {
                        result.status =
                            vbr_explicit_capture_status::internal_error;
                        return false;
                    }
                    for (uint32_t dependency = 0;
                         dependency < segment.dependency_count;
                         ++dependency) {
                        owner_manifest = std::min(
                            owner_manifest,
                            projection->dependent_manifest_ids[
                                segment.first_dependency + dependency]);
                    }
                }
                const auto owner = std::lower_bound(
                    manifest_ids.begin(), manifest_ids.end(),
                    owner_manifest);
                if (owner_manifest == UINT64_MAX ||
                    owner == manifest_ids.end() ||
                    *owner != owner_manifest) {
                    result.status =
                        vbr_explicit_capture_status::internal_error;
                    return false;
                }
                const size_t owner_index = size_t(owner - manifest_ids.begin());
                for (const auto & unit : child.units) {
                    for (const auto & shard : unit.shards) {
                        if (shard.row_bytes != 0 &&
                            projected_rows > UINT64_MAX/shard.row_bytes) {
                            result.status =
                                vbr_explicit_capture_status::size_overflow;
                            return false;
                        }
                        const vbr_artifact_portable_domain domain {
                            llama_cache_acct_residency::device,
                            llama_cache_acct_domain_kind::device_topology,
                            shard.topology_index,
                            shard.device_ordinal,
                        };
                        if (!add_planned_bytes(
                                domain,
                                projected_rows*shard.row_bytes)) {
                            return false;
                        }
                        add_accounting(
                            attention_accounting,
                            vbr_artifact_accounting_role::unit_payload,
                            domain,
                            projected_rows*shard.row_bytes);
                        add_accounting(
                            attention_reserve_accounting[owner_index],
                            vbr_artifact_accounting_role::unit_payload,
                            domain,
                            projected_rows*shard.row_bytes);
                    }
                }
            }
            for (size_t manifest_index = 0;
                 manifest_index < companion_plans.size(); ++manifest_index) {
                if (!manifest_dependency_available[manifest_index]) {
                    continue;
                }
                for (const auto & companion :
                     companion_plans[manifest_index]) {
                    if (!add_planned_bytes(
                            companion.descriptor.domain,
                            companion.descriptor.payload_bytes)) {
                        return false;
                    }
                }
            }
            return true;
        };
        if (!preflight_projection_bytes()) {
            return result;
        }

        const auto build_pretransfer_quote = [&] (
                vbr_projected_capture_batch_request::pretransfer_quote &
                    quote) {
            quote = {};
            quote.planned_packed_bytes = result.planned_packed_bytes;
            quote.union_cells = result.union_cells;
            quote.manifests = uint32_t(std::count(
                manifest_dependency_available.begin(),
                manifest_dependency_available.end(), true));
            quote.projected_units = projected_unit_count;
            quote.staging = staging_rows;
            const auto host_domain = vbr_artifact_portable_domain {
                llama_cache_acct_residency::pageable_host,
                llama_cache_acct_domain_kind::not_applicable,
                UINT32_MAX, UINT16_MAX,
            };
            for (size_t manifest_index = 0;
                 manifest_index < manifest_ids.size(); ++manifest_index) {
                if (!manifest_dependency_available[manifest_index]) {
                    continue;
                }
                vbr_projected_capture_batch_request::pretransfer_quote::
                    durable_manifest durable;
                durable.manifest_id = manifest_ids[manifest_index];
                const auto & requested_identity =
                    request.manifests[manifest_index].identity;
                durable.requested_token_count =
                    requested_identity.token_count > 0
                    ? uint64_t(requested_identity.token_count) : 0;
                durable.selected_token_count =
                    request.frontier.mode ==
                            vbr_projected_capture_frontier_mode::
                                longest_attention_stem
                        ? result.selected_frontier_tokens
                        : durable.requested_token_count;
                durable.selected_next_position =
                    request.frontier.mode ==
                            vbr_projected_capture_frontier_mode::
                                longest_attention_stem
                        ? result.selected_frontier_next_position
                        : requested_identity.next_position;
                durable.stemmed = durable.selected_token_count !=
                    durable.requested_token_count;
                durable.accounting = attention_accounting;
                durable.reserve_accounting =
                    attention_reserve_accounting[manifest_index];
                uint64_t descriptor_bytes = 0;
                uint64_t reference_bytes = 0;
                if (!vbr_projected_host_metadata_bytes(
                        quote.projected_units, descriptor_bytes,
                        reference_bytes)) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                add_accounting(
                    durable.accounting,
                    vbr_artifact_accounting_role::descriptor_metadata,
                    host_domain, descriptor_bytes);
                add_accounting(
                    durable.accounting,
                    vbr_artifact_accounting_role::reference_metadata,
                    host_domain, reference_bytes);
                for (const auto & companion :
                     companion_plans[manifest_index]) {
                    add_accounting(
                        durable.accounting,
                        companion.descriptor.kind ==
                                vbr_artifact_companion_kind::recurrent
                            ? vbr_artifact_accounting_role::recurrent_payload
                            : vbr_artifact_accounting_role::
                                typed_accelerator_payload,
                        companion.descriptor.domain,
                        companion.descriptor.payload_bytes);
                }
                quote.durable.push_back(std::move(durable));
            }
            std::sort(
                quote.durable.begin(), quote.durable.end(),
                [](const auto & lhs, const auto & rhs) {
                    return lhs.manifest_id < rhs.manifest_id;
                });
            const auto add_host_bytes = [&](uint64_t bytes) {
                if (bytes > UINT64_MAX -
                        quote.projected_host_resident_bytes) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return false;
                }
                quote.projected_host_resident_bytes += bytes;
                return true;
            };
            for (const auto & durable : quote.durable) {
                for (const auto & row : durable.accounting) {
                    if (row.role !=
                            vbr_artifact_accounting_role::unit_payload &&
                        !add_host_bytes(row.resident_bytes)) {
                        return false;
                    }
                }
                for (const auto & row : durable.reserve_accounting) {
                    if (row.role !=
                            vbr_artifact_accounting_role::unit_payload) {
                        result.status =
                            vbr_explicit_capture_status::internal_error;
                        return false;
                    }
                    if (!add_host_bytes(row.resident_bytes)) {
                        return false;
                    }
                }
            }
            return true;
        };
        vbr_projected_capture_batch_request::pretransfer_quote current_quote;
        if (!build_pretransfer_quote(current_quote)) {
            return result;
        }

        result.phase =
            vbr_explicit_capture_phase::reservation_preparation;
        if (request.pretransfer_prepare &&
            !request.pretransfer_prepare(
                request.pretransfer_prepare_context, current_quote)) {
            result.status =
                vbr_explicit_capture_status::admission_refused;
            return result;
        }

        // A positive scheduler/resource admission is the final pre-D2H
        // readiness checkpoint. Preparation above remains outside the ring;
        // for nonzero work, hold the persistent
        // transport operation at that checkpoint as well: a busy ring then
        // refuses before budget sampling or reservation preparation, and an
        // admission refusal releases it without transferring a byte. The
        // canonical zero-work quote requires no transport operation.
        vbr_pinned_ring_operation ring_operation;
        if (result.planned_packed_bytes != 0) {
            ++result.ring_operation_attempts;
            ring_operation = request.ring->try_begin_operation();
            if (!ring_operation) {
                ++result.ring_operation_refusals;
                result.inner_stream_status =
                    vbr_capture_stream_status::ring_unavailable;
                result.status = vbr_explicit_capture_status::ring_unavailable;
                return result;
            }
            ++result.ring_operation_acquires;
        }
        if (request.pretransfer_admit) {
            if (!request.pretransfer_admit(
                    request.pretransfer_context, current_quote)) {
                result.status =
                    vbr_explicit_capture_status::admission_refused;
                return result;
            }
        }

        // Payload runway is proven before the first companion D2H byte. A
        // locally stale/failed companion removes only its manifest; rebuild
        // the physical union and re-cap it before attention transfer.
        result.phase = vbr_explicit_capture_phase::companion_capture;
        std::vector<std::vector<vbr_capture_sealed_companion>>
            sealed_companions(request.manifests.size());
        bool rebuild_projection = false;
        for (size_t manifest_index = 0;
             manifest_index < request.manifests.size(); ++manifest_index) {
            auto & sealed = sealed_companions[manifest_index];
            if (!manifest_dependency_available[manifest_index]) {
                continue;
            }
            sealed.reserve(companion_plans[manifest_index].size());
            for (size_t companion_index = 0;
                 companion_index < companion_plans[manifest_index].size();
                 ++companion_index) {
                std::unique_ptr<artifact_segment_chain> chain;
                std::array<uint8_t, 32> companion_digest;
                vbr_explicit_capture_status companion_status =
                    vbr_explicit_capture_status::
                        required_companion_unavailable;
                vbr_capture_sealed_companion capability;
                uint64_t companion_d2h_bytes = 0;
                uint64_t companion_d2h_reads = 0;
                const auto & companion =
                    companion_plans[manifest_index][companion_index];
                const bool companion_captured = companion.has_provider
                    ? projected_provider_capture(
                        companion,
                        request.manifests[manifest_index].sequence,
                        request.manifests[manifest_index].identity.next_position,
                        chain, companion_digest, companion_status,
                        request.continue_context,
                        request.continue_transfer)
                    : recurrent_companion_capture(
                        companion.recurrent,
                        request.manifests[manifest_index].sequence,
                        chain, companion_digest, companion_status,
                        request.continue_context,
                        request.continue_transfer,
                        &companion_d2h_bytes,
                        &companion_d2h_reads);
                if (companion_d2h_bytes > UINT64_MAX -
                        result.companion_d2h_bytes ||
                    companion_d2h_reads > UINT64_MAX -
                        result.companion_d2h_reads) {
                    result.status =
                        vbr_explicit_capture_status::size_overflow;
                    return result;
                }
                result.companion_d2h_bytes += companion_d2h_bytes;
                result.companion_d2h_reads += companion_d2h_reads;
                if (!companion_captured ||
                    !vbr_capture_seal_companion(
                        uint32_t(companion_index), std::move(chain),
                        capability) ||
                    capability.streaming_digest() != companion_digest ||
                    (companion.has_provider
                        ? !projected_provider_current(
                            companion,
                            request.manifests[manifest_index].sequence,
                            request.manifests[manifest_index].identity.
                                next_position)
                        : !recurrent_companion_current(
                            companion.recurrent,
                            request.manifests[manifest_index].sequence))) {
                    if (companion_status ==
                            vbr_explicit_capture_status::cancelled) {
                        result.status = companion_status;
                        return result;
                    }
                    manifest_dependency_available[manifest_index] = false;
                    sealed.clear();
                    rebuild_projection = true;
                    break;
                }
                sealed.push_back(std::move(capability));
            }
        }
        if (rebuild_projection) {
            auto revised_manifests = projection->manifests;
            apply_dependency_availability(revised_manifests);
            projection = {};
            ++result.projection_calls;
            if (!vbr_artifact_project_capture_union(
                    { result.source_namespace,
                      std::move(revised_manifests) },
                    projection_limits, projection)) {
                result.status =
                    vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            result.union_cells = projection->union_cell_count;
            if (!preflight_projection_bytes()) {
                return result;
            }
            if (!build_pretransfer_quote(current_quote)) {
                return result;
            }
            if (request.pretransfer_shrink) {
                result.phase =
                    vbr_explicit_capture_phase::reservation_preparation;
                if (!request.pretransfer_shrink(
                        request.pretransfer_context, current_quote)) {
                    result.status =
                        vbr_explicit_capture_status::admission_refused;
                    return result;
                }
            }
        }

        result.phase = vbr_explicit_capture_phase::pre_transfer_stability;
        for (const auto & child : children) {
            if (!vbr_live_capture_adapter::stable(child)) {
                result.status = vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::unit_transfer;
        std::vector<vbr_capture_projected_unit> projected_units;
        for (const auto & child : children) {
            const bool projected_child = std::any_of(
                projection->streams.begin(), projection->streams.end(),
                [&](const auto & stream) {
                    return stream.child_id == child.child_id &&
                        stream.stream_index == 0;
                });
            if (!projected_child) {
                continue;
            }
            for (const auto & plan : child.units) {
                vbr_capture_projected_unit captured;
                vbr_capture_stream_stats attempted;
                ++result.unit_transfer_calls;
                const auto transferred =
                    vbr_live_capture_adapter::transfer_projected_unit(
                        child, plan, projection, result.source_namespace,
                        *request.ring, captured,
                        request.continue_context,
                        request.continue_transfer, &attempted,
                        ring_operation ? &ring_operation : nullptr);
                if (transferred != vbr_capture_stream_status::ok) {
                    if (attempted.bytes >
                            UINT64_MAX - result.transfer.bytes ||
                        attempted.chunks >
                            UINT64_MAX - result.transfer.chunks ||
                        attempted.submitted_bytes >
                            UINT64_MAX - result.transfer.submitted_bytes ||
                        attempted.submitted_chunks >
                            UINT64_MAX - result.transfer.submitted_chunks ||
                        attempted.backpressure_waits >
                            UINT64_MAX - result.transfer.backpressure_waits ||
                        attempted.event_completions >
                            UINT64_MAX - result.transfer.event_completions ||
                        attempted.synchronous_fallbacks >
                            UINT64_MAX - result.transfer.synchronous_fallbacks) {
                        result.status =
                            vbr_explicit_capture_status::internal_error;
                        return result;
                    }
                    result.transfer.bytes += attempted.bytes;
                    result.transfer.chunks += attempted.chunks;
                    result.transfer.submitted_bytes +=
                        attempted.submitted_bytes;
                    result.transfer.submitted_chunks +=
                        attempted.submitted_chunks;
                    result.transfer.backpressure_waits +=
                        attempted.backpressure_waits;
                    result.transfer.event_completions +=
                        attempted.event_completions;
                    result.transfer.synchronous_fallbacks +=
                        attempted.synchronous_fallbacks;
                    result.transfer.max_segment_size = std::max(
                        result.transfer.max_segment_size,
                        attempted.max_segment_size);
                    if (result.inner_stream_status ==
                            vbr_capture_stream_status::_count) {
                        result.inner_stream_status = transferred;
                    }
                    if (transferred ==
                            vbr_capture_stream_status::cancelled) {
                        result.status =
                            vbr_explicit_capture_status::cancelled;
                        return result;
                    }
                    if (transferred ==
                            vbr_capture_stream_status::snapshot_unavailable ||
                        transferred ==
                            vbr_capture_stream_status::snapshot_changed ||
                        transferred ==
                            vbr_capture_stream_status::transfer_failed ||
                        transferred == vbr_capture_stream_status::short_read ||
                        transferred ==
                            vbr_capture_stream_status::hash_mismatch) {
                        continue;
                    }
                    result.status = stream_status(transferred);
                    return result;
                }
                const auto & stats = captured.transfer();
                if (stats.bytes > UINT64_MAX - result.transfer.bytes ||
                    stats.chunks > UINT64_MAX - result.transfer.chunks ||
                    stats.submitted_bytes >
                        UINT64_MAX - result.transfer.submitted_bytes ||
                    stats.submitted_chunks >
                        UINT64_MAX - result.transfer.submitted_chunks ||
                    stats.backpressure_waits >
                        UINT64_MAX - result.transfer.backpressure_waits ||
                    stats.event_completions >
                        UINT64_MAX - result.transfer.event_completions ||
                    stats.synchronous_fallbacks >
                        UINT64_MAX - result.transfer.synchronous_fallbacks) {
                    result.status = vbr_explicit_capture_status::size_overflow;
                    return result;
                }
                result.transfer.bytes += stats.bytes;
                result.transfer.chunks += stats.chunks;
                result.transfer.submitted_bytes += stats.submitted_bytes;
                result.transfer.submitted_chunks += stats.submitted_chunks;
                result.transfer.backpressure_waits +=
                    stats.backpressure_waits;
                result.transfer.event_completions +=
                    stats.event_completions;
                result.transfer.synchronous_fallbacks +=
                    stats.synchronous_fallbacks;
                result.transfer.max_segment_size = std::max(
                    result.transfer.max_segment_size,
                    stats.max_segment_size);
                ++result.transferred_units;
                projected_units.push_back(std::move(captured));
            }
        }
        // Transport ownership ends with the final D2H chunk. Metadata
        // rechecks, assembly, and catalog publication never hold the ring.
        ring_operation = {};

        // Successful preflight companions remain immutable while attention
        // bytes transfer. A changed recurrent frontier is a violated idle
        // snapshot, so fail the whole batch instead of publishing stale rows.
        for (size_t manifest_index = 0;
             manifest_index < request.manifests.size(); ++manifest_index) {
            if (!manifest_dependency_available[manifest_index]) {
                continue;
            }
            for (const auto & companion : companion_plans[manifest_index]) {
                const bool current = companion.has_provider
                    ? projected_provider_current(
                        companion,
                        request.manifests[manifest_index].sequence,
                        request.manifests[manifest_index].identity.next_position)
                    : recurrent_companion_current(
                        companion.recurrent,
                        request.manifests[manifest_index].sequence);
                if (!current) {
                    result.status =
                        vbr_explicit_capture_status::source_changed;
                    return result;
                }
            }
        }

        result.phase = vbr_explicit_capture_phase::post_transfer_stability;
        for (const auto instance : instances) {
            if (vbr_recovery_pending_for(instance)) {
                result.status =
                    vbr_explicit_capture_status::recovery_pending;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }
        vbr_live_capture_adapter::projected_target_recheck_context
            recheck_context { &expectations };
        vbr_capture_controller_target_provider provider;
        provider.context = &recheck_context;
        provider.recheck =
            vbr_live_capture_adapter::projected_targets_recheck;
        vbr_capture_manifest_assembly_limits assembly_limits;
        assembly_limits.max_manifests =
            VBR_PROJECTED_CAPTURE_MAX_MANIFESTS;
        if (!vbr_capture_assemble_manifests(
                projection, std::move(targets),
                std::move(projected_units), provider,
                assembly_limits, result.assembly)) {
            result.status = vbr_explicit_capture_status::internal_error;
            return result;
        }

        result.publications.reserve(result.assembly.manifests().size());
        for (const auto & manifest : result.assembly.manifests()) {
            vbr_projected_manifest_publication publication;
            publication.manifest_id = manifest.manifest_id;
            publication.topologies = request.topologies;
            const auto requested = std::find(
                manifest_ids.begin(), manifest_ids.end(),
                manifest.manifest_id);
            if (requested == manifest_ids.end()) {
                result.status = vbr_explicit_capture_status::internal_error;
                result.assembly = {};
                result.publications.clear();
                return result;
            }
            const size_t manifest_index = size_t(
                requested - manifest_ids.begin());
            if (manifest.state == vbr_capture_manifest_state::ready) {
                const auto durable = std::lower_bound(
                    current_quote.durable.begin(),
                    current_quote.durable.end(), manifest.manifest_id,
                    [](const auto & lhs, uint64_t id) {
                        return lhs.manifest_id < id;
                    });
                if (durable == current_quote.durable.end() ||
                    durable->manifest_id != manifest.manifest_id ||
                    manifest.unit_count != current_quote.projected_units) {
                    result.status =
                        vbr_explicit_capture_status::internal_error;
                    result.assembly = {};
                    result.publications.clear();
                    return result;
                }
                publication.accounting =
                    std::move(durable->accounting);
                publication.companions = std::move(
                    sealed_companions[manifest_index]);
            }
            result.publications.push_back(std::move(publication));
        }
        result.status = vbr_explicit_capture_status::ok;
        result.phase = vbr_explicit_capture_phase::complete;
        return result;
    } catch (...) {
        result.status = vbr_explicit_capture_status::internal_error;
        result.assembly = {};
        result.publications.clear();
        return result;
    }
}

uint64_t vbr_explicit_import_policy_epoch(
        llama_memory_i & memory) noexcept {
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return 0;
        }
        return import_policy_epoch(tree);
    } catch (...) {
        return 0;
    }
}

static vbr_import_target_snapshot_status
import_classified_schedule_actionability(
        vbr_import_schedule_status status,
        const std::vector<vbr_import_schedule_unit> & units) noexcept {
    if (units.empty()) {
        return vbr_import_target_snapshot_status::unavailable;
    }
    switch (status) {
        case vbr_import_schedule_status::exact:
        case vbr_import_schedule_status::downward:
            return vbr_import_target_snapshot_status::actionable;
        case vbr_import_schedule_status::upward_same_domain:
        case vbr_import_schedule_status::upward_cross_domain:
            break;
        case vbr_import_schedule_status::mixed_direction_unsupported:
            return vbr_import_target_snapshot_status::report_only;
        case vbr_import_schedule_status::unavailable:
        case vbr_import_schedule_status::_count:
            return vbr_import_target_snapshot_status::unavailable;
    }
    const bool supported = std::all_of(
        units.begin(), units.end(),
        [](const vbr_import_schedule_unit & unit) {
            if (unit.source_type == unit.target_type) {
                return unit.source_domain == unit.target_domain;
            }
            vbr_upward_recipe recipe;
            return vbr_upward_resolve_recipe(
                       static_cast<ggml_type>(unit.source_type),
                       static_cast<ggml_type>(unit.target_type), recipe) ==
                vbr_upward_recipe_status::resolved;
        });
    return supported
        ? vbr_import_target_snapshot_status::actionable
        : vbr_import_target_snapshot_status::report_only;
}

vbr_import_target_snapshot_status
vbr_explicit_import_schedule_actionability(
        vbr_import_schedule_status status,
        const std::vector<vbr_import_schedule_unit> & units) noexcept {
    if (units.empty() ||
        vbr_classify_import_schedule_units(units) != status) {
        return vbr_import_target_snapshot_status::unavailable;
    }
    return import_classified_schedule_actionability(status, units);
}

static bool import_upward_schedule_supported(
        const vbr_import_schedule_quote & schedule) noexcept {
    return (schedule.status() ==
                vbr_import_schedule_status::upward_same_domain ||
            schedule.status() ==
                vbr_import_schedule_status::upward_cross_domain) &&
           import_classified_schedule_actionability(
               schedule.status(), schedule.units()) ==
               vbr_import_target_snapshot_status::actionable;
}

static bool import_target_snapshot_core(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_artifact_package_view & package,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        bool previously_observed,
        uint64_t accounting_serial,
        const void * representation_context,
        vbr_explicit_capture_request::representation_identity_fn
            representation_identity,
        vbr_target_validation_snapshot & output,
        vbr_downward_policy_projection * transform_projection,
        bool * downward_required,
        vbr_import_schedule_quote * schedule_quote,
        const vbr_import_schedule_quote * authenticated_schedule,
        std::array<uint8_t, 32> * transform_tree_digest,
        const std::vector<llama_memory_tree_child> * canonical_tree,
        uint64_t selected_frontier) noexcept {
    output = {};
    if (transform_projection) {
        *transform_projection = {};
    }
    if (downward_required) {
        *downward_required = false;
    }
    if (schedule_quote) {
        *schedule_quote = {};
    }
    if (transform_tree_digest) {
        *transform_tree_digest = {};
    }
    try {
        std::vector<llama_memory_tree_child> collected_tree;
        if (destination < 0 || !package ||
            (!canonical_tree &&
             !llama_memory_tree_collect(&memory, collected_tree))) {
            return false;
        }
        const auto & tree = canonical_tree ? *canonical_tree : collected_tree;
        if (tree.empty()) {
            return false;
        }
        const uint64_t policy_epoch = import_policy_epoch(tree);
        if (policy_epoch == 0) {
            return false;
        }
        output.memory_instance_cookie =
            uint64_t(reinterpret_cast<uintptr_t>(&memory));
        output.accounting_serial = accounting_serial;
        output.policy_epoch = policy_epoch;
        // Idleness is a SCHEDULER-authority fact; the library cannot vouch for
        // it. The route owner asserts it on the snapshot after this returns.
        output.scheduler_idle = false;
        output.destination_sequence_absent =
            memory.seq_pos_min(destination) < 0 &&
            memory.seq_pos_max(destination) < 0;
        output.tree_shape_digest = import_tree_digest(memory, tree);
        output.target_state_serial = 1;
        std::vector<vbr_live_capture_adapter::representation_cache_entry>
            representation_cache;
        size_t n_attention = 0;
        size_t n_recurrent = 0;
        size_t n_qsa = 0;
        for (const auto & child : tree) {
            if (child.recurrent) {
                if (n_recurrent != 0 ||
                    (!previously_observed && !recurrent_target_empty(child))) {
                    output = {};
                    return false;
                }
                output.companions.push_back({
                    vbr_artifact_companion_kind::recurrent, 1,
                    tagged_digest(
                        VBR_RECURRENT_CODEC_DOMAIN, 1),
                    true, child.recurrent,
                });
                ++n_recurrent;
                continue;
            }
            if (!child.attention) {
                output = {};
                return false;
            }
            if (child.qsa_index_owner) {
                const auto provider = vbr_qsa_index_adoption_provider(
                    *child.qsa_index_owner, child.child_id);
                if (!provider.target_empty ||
                    (!previously_observed &&
                     !provider.target_empty(provider.context))) {
                    output = {};
                    return false;
                }
                output.companions.push_back({
                    vbr_artifact_companion_kind::qsa_index,
                    vbr_qsa_index_companion_format_version(),
                    vbr_qsa_index_companion_build_identity(), true,
                    child.qsa_index_owner,
                });
                ++n_qsa;
            }
            vbr_target_child_snapshot snapshot;
            if (!vbr_live_capture_adapter::fill_import_child(
                    child, package, bindings, previously_observed,
                    policy_epoch, representation_context,
                    representation_identity, representation_cache,
                    snapshot)) {
                output = {};
                return false;
            }
            if (output.target_state_serial >
                    UINT64_MAX - snapshot.state_serial) {
                output = {};
                return false;
            }
            output.target_state_serial += snapshot.state_serial;
            output.children.push_back(std::move(snapshot));
            ++n_attention;
        }
        if (output.children.empty() ||
            n_attention != output.children.size() ||
            output.children.size() !=
                package.manifest().generation.controllers.size() ||
            n_recurrent != size_t(std::count_if(
                package.companions().begin(), package.companions().end(),
                [](const vbr_artifact_companion_view & companion) {
                    return companion.descriptor.kind ==
                        vbr_artifact_companion_kind::recurrent;
                })) ||
            n_qsa != size_t(std::count_if(
                package.companions().begin(), package.companions().end(),
                [](const vbr_artifact_companion_view & companion) {
                    return companion.descriptor.kind ==
                        vbr_artifact_companion_kind::qsa_index;
                }))) {
            output = {};
            return false;
        }
        vbr_import_schedule_quote local_schedule;
        const vbr_import_schedule_quote * negotiated = nullptr;
        vbr_downward_policy_projection selected_projection;
        if (authenticated_schedule != nullptr) {
            if (!authenticated_schedule->destination().feasible() ||
                !vbr_live_capture_adapter::recheck_import_destination(
                    tree, output,
                    authenticated_schedule->destination()) ||
                !vbr_live_capture_adapter::apply_import_destination(
                    tree, package, authenticated_schedule->destination(),
                    representation_context, representation_identity,
                    representation_cache, output, selected_projection) ||
                !vbr_import_schedule_quote_matches(
                    *authenticated_schedule, output, package)) {
                output = {};
                return false;
            }
            negotiated = authenticated_schedule;
        } else {
            auto * destination = schedule_quote != nullptr
                ? schedule_quote : &local_schedule;
            if (!vbr_quote_import_schedule(output, package, *destination)) {
                output = {};
                return false;
            }
            if (!vbr_live_capture_adapter::negotiate_import_destination(
                    tree, package, *destination, selected_frontier)) {
                output = {};
                *destination = {};
                return false;
            }
            if (!destination->destination().feasible()) {
                // Exhausted negotiation is useful report evidence, but it is
                // never an actionable import target.
                return schedule_quote != nullptr;
            }
            const auto & selected = destination->destination();
            if (!vbr_live_capture_adapter::apply_import_destination(
                    tree, package, selected,
                    representation_context, representation_identity,
                    representation_cache, output,
                    selected_projection)) {
                output = {};
                *destination = {};
                return false;
            }
            if (!vbr_rebind_import_schedule_quote(
                    output, package, selected, *destination)) {
                output = {};
                *destination = {};
                return false;
            }
            negotiated = destination;
        }
        const auto schedule_status = negotiated->status();
        const bool upward_actionable =
            (schedule_quote != nullptr || authenticated_schedule != nullptr) &&
            import_upward_schedule_supported(*negotiated);
        if (schedule_status != vbr_import_schedule_status::exact &&
            schedule_status != vbr_import_schedule_status::downward &&
            !upward_actionable) {
            // A caller asking for the quote can report the unsupported
            // schedule without pretending it was a downward-bind failure.
            return schedule_quote != nullptr;
        }
        if (schedule_status == vbr_import_schedule_status::downward ||
            upward_actionable) {
            if (transform_projection == nullptr) {
                output = {};
                if (schedule_quote) {
                    *schedule_quote = {};
                }
                return false;
            }
            *transform_projection = std::move(selected_projection);
            if (transform_tree_digest) {
                *transform_tree_digest = transform_projection->tree_digest;
            }
            if (downward_required &&
                schedule_status == vbr_import_schedule_status::downward) {
                *downward_required = true;
            }
        }
        return true;
    } catch (...) {
        output = {};
        if (transform_projection) {
            *transform_projection = {};
        }
        if (schedule_quote) {
            *schedule_quote = {};
        }
        if (transform_tree_digest) {
            *transform_tree_digest = {};
        }
        return false;
    }
}

vbr_import_target_snapshot_status
vbr_explicit_import_target_schedule_snapshot(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_artifact_package_view & package,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        bool previously_observed,
        uint64_t accounting_serial,
        const void * representation_context,
        vbr_explicit_capture_request::representation_identity_fn
            representation_identity,
        vbr_target_validation_snapshot & output,
        vbr_downward_policy_projection & downward_projection,
        bool & downward_required,
        vbr_import_schedule_quote & schedule_quote,
        uint64_t selected_frontier) noexcept {
    downward_projection = {};
    vbr_downward_policy_projection transform_projection;
    if (!import_target_snapshot_core(
        memory, destination, package, bindings, previously_observed,
        accounting_serial, representation_context, representation_identity,
        output, &transform_projection,
            &downward_required, &schedule_quote, nullptr, nullptr, nullptr,
            selected_frontier)) {
        return vbr_import_target_snapshot_status::unavailable;
    }
    if (downward_required) {
        downward_projection = std::move(transform_projection);
    }
    if (!schedule_quote.destination().feasible()) {
        return vbr_import_target_snapshot_status::unavailable;
    }
    // The private quote was minted from these units and has already stored
    // their canonical classification. Avoid re-scanning exact/downward imports;
    // the public helper above retains its defensive relabeling check for
    // external model-free callers.
    return import_classified_schedule_actionability(
        schedule_quote.status(), schedule_quote.units());
}

bool vbr_explicit_import_transform_projection_recheck(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_artifact_package_view & package,
        const std::vector<llama_vbr_artifact_domain_binding> & bindings,
        const vbr_import_schedule_quote & authenticated_schedule,
        const void * representation_context,
        vbr_explicit_capture_request::representation_identity_fn
            representation_identity,
        std::array<uint8_t, 32> & tree_digest) noexcept {
    tree_digest = {};
    if (authenticated_schedule.status() !=
            vbr_import_schedule_status::downward &&
        !import_upward_schedule_supported(authenticated_schedule)) {
        return false;
    }
    vbr_target_validation_snapshot snapshot;
    vbr_downward_policy_projection projection;
    bool downward = false;
    if (!import_target_snapshot_core(
            memory, destination, package, bindings, false,
            authenticated_schedule.accounting_serial(),
            representation_context, representation_identity, snapshot,
            &projection, &downward,
            nullptr, &authenticated_schedule, &tree_digest) ||
        (authenticated_schedule.status() ==
             vbr_import_schedule_status::downward) != downward ||
        projection.status != vbr_downward_policy_status::coherent) {
        tree_digest = {};
        return false;
    }
    return tree_digest == projection.tree_digest && std::any_of(
        tree_digest.begin(), tree_digest.end(),
        [](uint8_t byte) { return byte != 0; });
}

bool vbr_explicit_import_target_recheck(
        llama_memory_i & memory,
        llama_seq_id destination,
        const vbr_target_empty_fingerprint & expected) noexcept {
    try {
        std::vector<llama_memory_tree_child> tree;
        if (destination < 0 ||
            expected.memory_instance_cookie !=
                uint64_t(reinterpret_cast<uintptr_t>(&memory)) ||
            !llama_memory_tree_collect(&memory, tree) ||
            expected.tree_shape_digest != import_tree_digest(memory, tree) ||
            expected.policy_epoch != import_policy_epoch(tree) ||
            memory.seq_pos_min(destination) >= 0 ||
            memory.seq_pos_max(destination) >= 0) {
            return false;
        }
        size_t n_attention = 0;
        size_t n_recurrent = 0;
        for (const auto & child : tree) {
            if (child.recurrent) {
                if (n_recurrent != 0 || !recurrent_target_empty(child)) {
                    return false;
                }
                ++n_recurrent;
                continue;
            }
            if (!child.attention) {
                return false;
            }
            ++n_attention;
            const auto item = std::find_if(
                expected.children.begin(), expected.children.end(),
                [&](const vbr_child_empty_fingerprint & value) {
                    return value.child_id == child.child_id;
                });
            if (item == expected.children.end() ||
                !vbr_live_capture_adapter::recheck_import_child(
                    *child.attention, *item)) {
                return false;
            }
        }
        return n_attention == expected.children.size();
    } catch (...) {
        return false;
    }
}

bool vbr_explicit_import_reserve_transform(
        llama_memory_i & memory,
        const std::vector<vbr_validated_child_plan> & plans,
        llama_cache_acct_ledger & ledger,
        const llama_cache_budget_config & budget,
        vbr_downward_stage_reservation & output) noexcept {
    output = {};
    try {
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            return false;
        }
        struct grouped_plan {
            const vbr_validated_child_plan * plan = nullptr;
            size_t next = SIZE_MAX;
        };
        std::unordered_map<uint32_t, size_t> child_indices;
        child_indices.reserve(tree.size());
        std::vector<size_t> heads(tree.size(), SIZE_MAX);
        std::vector<size_t> tails(tree.size(), SIZE_MAX);
        for (size_t i = 0; i < tree.size(); ++i) {
            if (tree[i].attention != nullptr &&
                !child_indices.emplace(tree[i].child_id, i).second) {
                return false;
            }
        }
        std::vector<grouped_plan> grouped;
        grouped.reserve(plans.size());
        bool any_transform = false;
        for (const auto & plan : plans) {
            const bool stash_only =
                plan.transform_kind == vbr_import_transform_kind::none &&
                plan.stash_action ==
                    vbr_validated_stash_action::restore_exact;
            if (plan.transform_kind == vbr_import_transform_kind::none &&
                !stash_only) {
                continue;
            }
            const auto child = child_indices.find(plan.child_id);
            if (child == child_indices.end()) {
                continue;
            }
            const size_t index = grouped.size();
            grouped.push_back({ &plan, SIZE_MAX });
            any_transform = any_transform ||
                plan.transform_kind != vbr_import_transform_kind::none;
            const size_t tree_index = child->second;
            if (tails[tree_index] == SIZE_MAX) {
                heads[tree_index] = index;
            } else {
                grouped[tails[tree_index]].next = index;
            }
            tails[tree_index] = index;
        }
        output.status = vbr_downward_reserve_status::reserved;
        bool any = false;
        std::vector<const vbr_validated_child_plan *> selected;
        selected.reserve(grouped.size());
        for (size_t tree_index = 0; tree_index < tree.size(); ++tree_index) {
            const auto & child = tree[tree_index];
            if (!child.attention) {
                continue;
            }
            selected.clear();
            for (size_t index = heads[tree_index]; index != SIZE_MAX;
                    index = grouped[index].next) {
                selected.push_back(grouped[index].plan);
            }
            if (selected.empty()) {
                continue;
            }
            any = true;
            vbr_downward_stage_reservation one;
            if (!vbr_live_capture_adapter::reserve_import_transform(
                    *child.attention, selected, ledger, budget, one)) {
                output = one;
                return false;
            }
            if (one.status != vbr_downward_reserve_status::reserved &&
                one.status !=
                    vbr_downward_reserve_status::reserved_stashless) {
                output = one;
                return true;
            }
            output.stashless_units.insert(
                output.stashless_units.end(),
                one.stashless_units.begin(),
                one.stashless_units.end());
            if (!one.stashless_units.empty()) {
                output.status =
                    vbr_downward_reserve_status::reserved_stashless;
            }
        }
        // Stash-only exact siblings are admitted only as part of a real
        // transform manifest. This keeps the ordinary exact import path on
        // its established staging owner while making a mixed upward tree's
        // pre-transform stash H2D fully receipt-backed.
        if (!any || !any_transform) {
            output.status =
                vbr_downward_reserve_status::projection_unavailable;
            return false;
        }
        return true;
    } catch (...) {
        output = {};
        output.status = vbr_downward_reserve_status::internal_error;
        return false;
    }
}

std::array<uint8_t, 32>
vbr_explicit_recurrent_companion_build_identity() noexcept {
    return tagged_digest(VBR_RECURRENT_CODEC_DOMAIN, 1);
}

bool vbr_explicit_recurrent_companion_terminal(
        const void * data, size_t size, llama_pos & output) noexcept {
    output = -1;
    constexpr size_t prefix = sizeof(uint32_t)+sizeof(llama_seq_id);
    if (!data || size < prefix+sizeof(uint32_t)+sizeof(llama_pos)) {
        return false;
    }
    uint32_t magic = 0;
    llama_seq_id source_sequence = -1;
    uint32_t cell_count = 0;
    llama_pos position = -1;
    std::memcpy(&magic, data, sizeof(magic));
    std::memcpy(&source_sequence,
                static_cast<const uint8_t *>(data)+sizeof(magic),
                sizeof(source_sequence));
    std::memcpy(&cell_count,
                static_cast<const uint8_t *>(data)+prefix,
                sizeof(cell_count));
    std::memcpy(&position,
                static_cast<const uint8_t *>(data)+prefix+
                    sizeof(cell_count),
                sizeof(position));
    if (magic != VBR_SEQUENCE_STATE_MAGIC || source_sequence < 0 ||
        cell_count != 1 || position < 0) {
        return false;
    }
    output = position;
    return true;
}

bool vbr_explicit_capture_pretransfer_quote_admissible(
        const vbr_explicit_capture_pretransfer_quote & quote,
        uint64_t max_packed_bytes) noexcept {
    uint64_t packed = quote.payload_bytes;
    if (quote.stash_bytes > UINT64_MAX - packed) {
        return false;
    }
    packed += quote.stash_bytes;
    if (quote.companion_bytes > UINT64_MAX - packed) {
        return false;
    }
    packed += quote.companion_bytes;
    if (quote.metadata_bytes > UINT64_MAX - packed) {
        return false;
    }
    const uint64_t host = packed + quote.metadata_bytes;
    return quote.controllers != 0 && quote.units != 0 &&
           quote.planned_packed_bytes == packed &&
           quote.conservative_host_resident_bytes == host &&
           (max_packed_bytes == 0 || packed <= max_packed_bytes);
}

struct vbr_explicit_capture_operation::impl {
    struct pending_companion {
        recurrent_companion_plan recurrent;
        vbr_explicit_companion_provider provider;
        bool has_provider = false;
        uint64_t bytes = 0;
    };

    vbr_explicit_capture_request request;
    std::vector<vbr_live_capture_adapter::child> children;
    std::vector<vbr_controller_instance_id> instances;
    vbr_artifact_package package;
    std::vector<pending_companion> pending_companions;
    std::unique_ptr<vbr_capture_build> build;
    vbr_explicit_capture_result result;
    bool transfer_started = false;
    bool transferred = false;
    bool published = false;
};

vbr_explicit_capture_operation::vbr_explicit_capture_operation() noexcept =
    default;
vbr_explicit_capture_operation::vbr_explicit_capture_operation(
        vbr_explicit_capture_operation && other) noexcept = default;
vbr_explicit_capture_operation &
vbr_explicit_capture_operation::operator=(
        vbr_explicit_capture_operation && other) noexcept = default;
vbr_explicit_capture_operation::~vbr_explicit_capture_operation() = default;

bool vbr_explicit_capture_operation::ready_for_transfer() const noexcept {
    return impl_ && impl_->build && !impl_->transfer_started &&
        !impl_->transferred && !impl_->published;
}

bool vbr_explicit_capture_operation::ready_for_publication() const noexcept {
    return impl_ && impl_->build && impl_->transferred && !impl_->published &&
        impl_->result.status == vbr_explicit_capture_status::ok;
}

void vbr_explicit_capture_operation::reset() noexcept {
    impl_.reset();
}

vbr_explicit_capture_result vbr_prepare_explicit_manifest(
        llama_memory_i & memory,
        vbr_explicit_capture_request request,
        vbr_unit_version_sink & sink,
        const vbr_explicit_capture_accounting & accounting,
        vbr_explicit_capture_operation & operation) noexcept {
    operation.reset();
    vbr_explicit_capture_result result;
    if (!request.idle_decode_thread) {
        result.status = vbr_explicit_capture_status::slot_not_idle;
        return result;
    }
    if (request.sequence < 0 || request.frontier.next_position < 0 ||
        request.ring == nullptr || accounting.budget == nullptr ||
        request.topologies.empty() || request.pool_bindings.empty() ||
        request.representation_identity == nullptr ||
        request.identity.token_count < 0 ||
        request.token_block.size() !=
            size_t(request.identity.token_count) ||
        request.identity.execution_identity.empty() ||
        request.identity.adapter_config_identity.empty() ||
        request.identity.media_content_identity.empty()) {
        result.status = vbr_explicit_capture_status::identity_unavailable;
        return result;
    }

    try {
        result.phase = vbr_explicit_capture_phase::memory_tree;
        std::vector<llama_memory_tree_child> tree;
        if (!llama_memory_tree_collect(&memory, tree)) {
            result.status = vbr_explicit_capture_status::unsupported_layout;
            return result;
        }

        std::vector<vbr_live_capture_adapter::child> children;
        std::vector<llama_memory_recurrent *> recurrent;
        size_t qsa_index_children = 0;
        for (const auto & node : tree) {
            qsa_index_children += node.qsa_index_owner != nullptr;
            if (node.attention != nullptr) {
                if (!node.attention->vbr_operation_armed()) {
                    result.status = vbr_explicit_capture_status::not_armed;
                    return result;
                }
                vbr_live_capture_adapter::child child;
                child.child_id = node.child_id;
                child.dependency_mode = node.dependency_mode;
                child.cache = node.attention;
                children.push_back(std::move(child));
            }
            if (node.recurrent != nullptr) {
                recurrent.push_back(node.recurrent);
            }
        }
        if (children.empty()) {
            result.status = vbr_explicit_capture_status::not_armed;
            return result;
        }
        const size_t supplied_qsa = size_t(std::count_if(
            request.companions.begin(), request.companions.end(),
            [](const vbr_explicit_companion_provider & provider) {
                return provider.kind ==
                    vbr_artifact_companion_kind::qsa_index;
            }));
        if (qsa_index_children > 1 || supplied_qsa != qsa_index_children) {
            result.status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return result;
        }

        result.phase = vbr_explicit_capture_phase::settlement;
        // Settlement is deliberately before both quiescence proofs. It flushes
        // only already-deferred housekeeping and dirty stash metadata.
        for (auto & child : children) {
            if (!vbr_live_capture_adapter::settle(*child.cache)) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
        }

        result.phase =
            vbr_explicit_capture_phase::pre_capture_quiescence;
        std::vector<vbr_controller_instance_id> instances;
        result.phase =
            vbr_explicit_capture_phase::metadata_and_manifest;
        for (auto & child : children) {
            const auto instance = child.cache->vbr_instance_id();
            if (!vbr_controller_instance_id_is_set(instance) ||
                std::any_of(instances.begin(), instances.end(),
                    [&](const auto & current) {
                        return current == instance;
                    })) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
            instances.push_back(instance);
            if (vbr_recovery_pending_for(instance)) {
                result.status = vbr_explicit_capture_status::recovery_pending;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        for (auto & child : children) {
            if (!vbr_live_capture_adapter::capture_metadata(
                    *child.cache, child.child_id, child.dependency_mode,
                    request.sequence, request.frontier.next_position,
                    request.pool_bindings, child,
                    result.generation_failure,
                    result.size_failure)) {
                result.status = vbr_explicit_capture_status::generation_unavailable;
                return result;
            }
        }

        std::vector<vbr_identity_policy_digest_row> identity_policy;
        identity_policy.reserve(children.size());
        for (const auto & child : children) {
            identity_policy.push_back({
                child.child_id,
                child.dependency_mode,
                child.stability.lineage_uuid,
            });
        }
        const auto identity_policy_order_digest =
            vbr_identity_policy_digest(
                request.frontier, identity_policy);
        if (digest_nonzero(request.identity_policy_order_digest) &&
            request.identity_policy_order_digest !=
                identity_policy_order_digest) {
            result.status =
                vbr_explicit_capture_status::identity_unavailable;
            return result;
        }

        // Recurrent state uses the existing exact state codec. Accelerator
        // companions use equally typed injected existing codecs.
        vbr_artifact_package package;
        package.topologies = request.topologies;
        package.manifest.identity = request.identity;
        package.manifest.token_block.tokens = std::move(request.token_block);
        package.manifest.identity_policy_order_digest =
            identity_policy_order_digest;
        package.manifest.generation.version = 1;
        package.manifest.generation.status =
            vbr_checkpoint_generation_status::complete;
        package.manifest.generation.identity_policy_order_digest =
            identity_policy_order_digest;

        uint32_t global_unit = 0;
        std::vector<vbr_live_capture_adapter::representation_cache_entry>
            representation_cache;
        for (auto & child : children) {
            package.manifest.generation.controllers.push_back(
                child.generation);
            package.manifest.stream_placements.insert(
                package.manifest.stream_placements.end(),
                child.placements.begin(), child.placements.end());
            vbr_artifact_controller_policy policy;
            std::vector<vbr_artifact_unit_descriptor> descriptors;
            if (!vbr_live_capture_adapter::capture_schema(
                    child, request.representation_context,
                    request.representation_identity, true,
                    representation_cache,
                    policy, descriptors, result.status)) {
                return result;
            }
            package.manifest.controller_policy.push_back(std::move(policy));

            for (size_t plan_index = 0;
                 plan_index < child.units.size(); ++plan_index) {
                auto & plan = child.units[plan_index];
                plan.capture_index = global_unit;
                vbr_artifact_unit_blob blob;
                blob.descriptor = std::move(descriptors[plan_index]);
                const bool has_stash = blob.descriptor.clean_stash_state ==
                    vbr_artifact_clean_stash_state::present;
                const uint32_t stash_rows = uint32_t(
                    blob.descriptor.clean_stash.valid_rows);
                const uint64_t total_columns =
                    blob.descriptor.dimensions[1];
                for (const auto & shard : plan.shards) {
                    add_accounting(
                        package.manifest.accounting,
                        vbr_artifact_accounting_role::unit_payload,
                        portable_domain(
                            shard.topology_index,
                            shard.device_ordinal),
                        shard.payload_bytes);
                    if (has_stash) {
                        add_accounting(
                            package.manifest.accounting,
                            vbr_artifact_accounting_role::
                                clean_stash_payload,
                            portable_domain(
                                shard.topology_index,
                                shard.device_ordinal),
                            shard.stash_bytes);
                    }
                }
                package.unit_blobs.push_back(std::move(blob));

                vbr_artifact_unit_reference reference;
                reference.lineage_uuid = child.stability.lineage_uuid;
                reference.logical_unit_id = plan.logical_unit;
                reference.repr_gen = plan.generation.repr_gen;
                reference.authorized_stream_refs = { 0 };
                if (has_stash) {
                    if (child.generation.streams.empty()) {
                        result.status =
                            vbr_explicit_capture_status::stash_inconsistent;
                        return result;
                    }
                    const auto & stream =
                        child.generation.streams.front();
                    reference.has_stash_reference = true;
                    reference.stash_reference.valid_rows = stash_rows;
                    reference.stash_reference.domain =
                        vbr_repr_domain::tapped;
                    reference.stash_reference.row_count = stash_rows;
                    reference.stash_reference.column_count =
                        total_columns;
                    reference.stash_reference.row_bytes =
                        total_columns*sizeof(uint16_t);
                    reference.stash_reference.captured_sink_count =
                        stream.captured_dependency_count;
                    reference.stash_reference.covered_sink_pages =
                        stream.pages;
                }
                package.manifest.unit_references.push_back(
                    std::move(reference));
                ++global_unit;
            }
        }

        std::vector<vbr_explicit_capture_operation::impl::pending_companion>
            pending_companions;
        const size_t supplied_recurrent = size_t(std::count_if(
            request.companions.begin(), request.companions.end(),
            [](const vbr_explicit_companion_provider & provider) {
                return provider.kind ==
                    vbr_artifact_companion_kind::recurrent;
            }));
        // A frontier checkpoint may carry the exact recurrent image for an
        // earlier attention prefix.  It replaces (rather than supplements)
        // the live recurrent serializer: combining both would bind two
        // different frontiers under one manifest.  Require a complete
        // one-for-one replacement so a partial recurrent tree cannot be
        // published accidentally.
        if (supplied_recurrent != 0 && supplied_recurrent != recurrent.size()) {
            result.status = vbr_explicit_capture_status::
                required_companion_unavailable;
            return result;
        }
        for (auto * memory_recurrent : recurrent) {
            if (supplied_recurrent != 0) {
                break;
            }
            recurrent_companion_plan recurrent_plan;
            if (!recurrent_companion_prepare(
                    memory_recurrent, request.sequence,
                    request.frontier.next_position,
                    recurrent_plan, result.status)) {
                return result;
            }
            pending_companions.push_back({
                recurrent_plan, {}, false,
                recurrent_plan.descriptor.payload_bytes,
            });
            package.companions.push_back(recurrent_plan.descriptor);
        }
        for (const auto & provider : request.companions) {
            if (provider.size == nullptr ||
                (provider.capture == nullptr &&
                 provider.capture_stream == nullptr) ||
                provider.format_version == 0 ||
                !digest_nonzero(provider.build_identity_digest)) {
                if (provider.required) {
                    result.status =
                        vbr_explicit_capture_status::
                            required_companion_unavailable;
                    return result;
                }
                continue;
            }
            uint64_t companion_size = 0;
            if (!provider.size(
                    provider.context, request.sequence,
                    companion_size) ||
                companion_size == 0 ||
                companion_size >
                    std::numeric_limits<size_t>::max()) {
                if (provider.required) {
                    result.status =
                        vbr_explicit_capture_status::
                            required_companion_unavailable;
                    return result;
                }
                continue;
            }
            if (provider.kind ==
                    vbr_artifact_companion_kind::recurrent ||
                provider.terminal_position != nullptr) {
                llama_pos terminal = -1;
                if (request.frontier.next_position <= 0 ||
                    provider.terminal_position == nullptr ||
                    !provider.terminal_position(
                        provider.context, request.sequence, terminal) ||
                    terminal != request.frontier.next_position-1) {
                    result.status = vbr_explicit_capture_status::
                        required_companion_unavailable;
                    return result;
                }
            }
            pending_companions.push_back({
                {}, provider, true, companion_size,
            });
            vbr_artifact_companion_payload companion;
            companion.kind = provider.kind;
            companion.format_version = provider.format_version;
            companion.build_identity_digest =
                provider.build_identity_digest;
            companion.domain = provider.domain;
            companion.payload_bytes = companion_size;
            package.companions.push_back(companion);
        }
        package.manifest.companions = package.companions;
        for (const auto & companion : package.companions) {
            add_accounting(
                package.manifest.accounting,
                companion.kind ==
                    vbr_artifact_companion_kind::recurrent
                    ? vbr_artifact_accounting_role::recurrent_payload
                    : vbr_artifact_accounting_role::
                        typed_accelerator_payload,
                companion.domain, companion.payload_bytes);
            result.companion_bytes += companion.payload_bytes;
        }
        const auto metadata_domain = vbr_artifact_portable_domain {
            llama_cache_acct_residency::pageable_host,
            llama_cache_acct_domain_kind::not_applicable,
            UINT32_MAX, UINT16_MAX,
        };
        add_accounting(
            package.manifest.accounting,
            vbr_artifact_accounting_role::descriptor_metadata,
            metadata_domain,
            std::max<uint64_t>(1, package.unit_blobs.size()*256));
        add_accounting(
            package.manifest.accounting,
            vbr_artifact_accounting_role::reference_metadata,
            metadata_domain,
            std::max<uint64_t>(1,
                package.manifest.unit_references.size()*128));
        package.manifest.consistency.kind =
            vbr_artifact_consistency_kind::capture_exact;

        // The exact companion route previously discovered its total only after
        // companion and attention D2H. Price the authenticated size/schema
        // inventory once and expose one scalar admission before transfer-side
        // allocation or byte movement.
        result.phase =
            vbr_explicit_capture_phase::reservation_preparation;
        auto & quote = result.pretransfer;
        if (children.size() > UINT32_MAX ||
            package.unit_blobs.size() > UINT32_MAX ||
            package.companions.size() > UINT32_MAX) {
            result.status = vbr_explicit_capture_status::size_overflow;
            return result;
        }
        quote.controllers = uint32_t(children.size());
        quote.units = uint32_t(package.unit_blobs.size());
        quote.companions = uint32_t(package.companions.size());
        const auto add_quote_bytes = [&] (
                uint64_t & destination, uint64_t bytes) noexcept {
            if (bytes > UINT64_MAX - destination) {
                return false;
            }
            destination += bytes;
            return true;
        };
        for (const auto & child : children) {
            for (const auto & plan : child.units) {
                for (const auto & shard : plan.shards) {
                    if (!add_quote_bytes(
                            quote.payload_bytes, shard.payload_bytes) ||
                        !add_quote_bytes(
                            quote.stash_bytes, shard.stash_bytes)) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return result;
                    }
                }
            }
        }
        quote.companion_bytes = result.companion_bytes;
        uint64_t accounted_host_bytes = 0;
        for (const auto & row : package.manifest.accounting) {
            if (!add_quote_bytes(accounted_host_bytes, row.resident_bytes)) {
                result.status = vbr_explicit_capture_status::size_overflow;
                return result;
            }
        }
        quote.planned_packed_bytes = quote.payload_bytes;
        if (!add_quote_bytes(
                quote.planned_packed_bytes, quote.stash_bytes) ||
            !add_quote_bytes(
                quote.planned_packed_bytes, quote.companion_bytes) ||
            accounted_host_bytes < quote.planned_packed_bytes) {
            result.status = vbr_explicit_capture_status::size_overflow;
            return result;
        }
        quote.metadata_bytes =
            accounted_host_bytes - quote.planned_packed_bytes;
        quote.conservative_host_resident_bytes = accounted_host_bytes;
        if (!vbr_explicit_capture_pretransfer_quote_admissible(quote, 0)) {
            result.status = vbr_explicit_capture_status::internal_error;
            return result;
        }
        if (!vbr_explicit_capture_pretransfer_quote_admissible(
                quote, request.max_packed_bytes)) {
            result.status = vbr_explicit_capture_status::admission_refused;
            return result;
        }
        if (request.pretransfer_admit &&
            !request.pretransfer_admit(
                request.pretransfer_context, quote)) {
            result.status = request.continue_transfer &&
                    !request.continue_transfer(request.continue_context)
                ? vbr_explicit_capture_status::cancelled
                : vbr_explicit_capture_status::admission_refused;
            return result;
        }
        if (request.continue_transfer &&
            !request.continue_transfer(request.continue_context)) {
            result.status = vbr_explicit_capture_status::cancelled;
            return result;
        }

        result.phase =
            vbr_explicit_capture_phase::pre_transfer_stability;
        // Exact equality immediately before the first data byte.
        for (const auto & child : children) {
            if (!vbr_live_capture_adapter::stable(child)) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.phase =
            vbr_explicit_capture_phase::accounting_configuration;
        if (accounting.prepare != nullptr &&
            !accounting.prepare(accounting.context, package)) {
            result.status = vbr_explicit_capture_status::accounting_failed;
            return result;
        }
        result.phase =
            vbr_explicit_capture_phase::reservation_preparation;
        vbr_capture_stream_status begin_status;
        auto build = sink.begin_capture(
            package, *accounting.budget, accounting.fault,
            begin_status, &result.begin_diagnostics);
        if (!build) {
            result.inner_stream_status = begin_status;
            result.status = stream_status(begin_status);
            return result;
        }

        auto prepared = std::make_unique<
            vbr_explicit_capture_operation::impl>();
        prepared->request = std::move(request);
        prepared->children = std::move(children);
        prepared->instances = std::move(instances);
        prepared->package = std::move(package);
        prepared->pending_companions = std::move(pending_companions);
        prepared->build = std::move(build);
        result.status = vbr_explicit_capture_status::ok;
        prepared->result = result;
        operation.impl_ = std::move(prepared);
        return result;
    } catch (...) {
        result.status = vbr_explicit_capture_status::internal_error;
        return result;
    }
}

vbr_explicit_capture_result vbr_transfer_explicit_manifest(
        vbr_explicit_capture_operation & operation) noexcept {
    vbr_explicit_capture_result result;
    if (!operation.ready_for_transfer()) {
        result.status = vbr_explicit_capture_status::internal_error;
        return result;
    }
    auto & state = *operation.impl_;
    auto & request = state.request;
    auto & children = state.children;
    auto & instances = state.instances;
    auto & package = state.package;
    auto & pending_companions = state.pending_companions;
    auto & build = state.build;
    result = state.result;
    state.transfer_started = true;
    try {

        result.phase =
            vbr_explicit_capture_phase::companion_capture;
        // Durable + transfer-staging claims now exist. Only at this point may
        // companion codecs allocate their pageable byte images.
        for (size_t i = 0; i < pending_companions.size(); ++i) {
            result.companion_failure_index = uint32_t(i);
            result.companion_failure_kind =
                package.companions[i].kind;
            if (request.continue_transfer &&
                !request.continue_transfer(request.continue_context)) {
                result.status = vbr_explicit_capture_status::cancelled;
                return result;
            }
            std::vector<uint8_t> bytes;
            const auto & pending = pending_companions[i];
            const auto provider_frontier_current = [&]() noexcept {
                if (!pending.has_provider ||
                    pending.provider.terminal_position == nullptr) {
                    return true;
                }
                llama_pos terminal = -1;
                return request.frontier.next_position > 0 &&
                    pending.provider.terminal_position != nullptr &&
                    pending.provider.terminal_position(
                        pending.provider.context, request.sequence,
                        terminal) &&
                    terminal == request.frontier.next_position-1;
            };
            if (!provider_frontier_current()) {
                result.status = vbr_explicit_capture_status::source_changed;
                return result;
            }
            std::unique_ptr<artifact_segment_chain> recurrent_chain;
            std::array<uint8_t, 32> recurrent_digest;
            if (pending.recurrent.source != nullptr) {
                if (!recurrent_companion_capture(
                        pending.recurrent, request.sequence,
                        recurrent_chain, recurrent_digest,
                        result.status, request.continue_context,
                        request.continue_transfer)) {
                    return result;
                }
            } else if (!pending.has_provider) {
                result.status = vbr_explicit_capture_status::
                    required_companion_unavailable;
                return result;
            } else if (pending.provider.capture_stream) {
                try {
                    recurrent_chain =
                        std::make_unique<artifact_segment_chain>(
                            pending.bytes);
                    chain_io_writer writer(
                        *recurrent_chain, pending.bytes,
                        request.continue_context,
                        request.continue_transfer);
                    if (!pending.provider.capture_stream(
                            pending.provider.context, request.sequence,
                            writer) || !writer.finish()) {
                        result.status = vbr_explicit_capture_status::
                            required_companion_unavailable;
                        return result;
                    }
                    recurrent_digest =
                        vbr_capture_stream_digest(*recurrent_chain);
                    if (!digest_nonzero(recurrent_digest)) {
                        result.status = vbr_explicit_capture_status::
                            hash_mismatch;
                        return result;
                    }
                } catch (const capture_transfer_cancelled &) {
                    result.status = vbr_explicit_capture_status::cancelled;
                    return result;
                } catch (...) {
                    result.status = vbr_explicit_capture_status::
                        required_companion_unavailable;
                    return result;
                }
            } else if (!pending.provider.capture ||
                       !pending.provider.capture(
                           pending.provider.context,
                           request.sequence, bytes)) {
                result.status = vbr_explicit_capture_status::
                    required_companion_unavailable;
                return result;
            }
            if (request.continue_transfer &&
                !request.continue_transfer(request.continue_context)) {
                result.status = vbr_explicit_capture_status::cancelled;
                return result;
            }
            if (!provider_frontier_current()) {
                result.status = vbr_explicit_capture_status::source_changed;
                return result;
            }
            // Companion size→data coherence relies on the required idle-slot,
            // no-decode route invariant. The server route enforces that invariant;
            // size equality is intentionally the capture guard, not a second
            // content-hash pass over the existing companion codecs.
            if (pending.recurrent.source == nullptr && !recurrent_chain &&
                bytes.size() != pending.bytes) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
            std::shared_ptr<const artifact_segment_chain> chain;
            std::array<uint8_t, 32> digest;
            if (recurrent_chain) {
                chain = std::shared_ptr<const artifact_segment_chain>(
                    std::move(recurrent_chain));
                digest = recurrent_digest;
            } else {
                auto provider_chain =
                    std::make_shared<artifact_segment_chain>(pending.bytes);
                static constexpr size_t CHUNK = 1024*1024;
                for (size_t offset = 0; offset < bytes.size();) {
                    if (request.continue_transfer &&
                        !request.continue_transfer(
                            request.continue_context)) {
                        result.status =
                            vbr_explicit_capture_status::cancelled;
                        return result;
                    }
                    const size_t size = std::min(
                        CHUNK, bytes.size() - offset);
                    if (!provider_chain->append(
                            bytes.data() + offset, size)) {
                        result.status =
                            vbr_explicit_capture_status::accounting_failed;
                        return result;
                    }
                    offset += size;
                }
                digest = vbr_capture_stream_digest(*provider_chain);
                if (!digest_nonzero(digest)) {
                    result.status =
                        vbr_explicit_capture_status::hash_mismatch;
                    return result;
                }
                chain = std::move(provider_chain);
            }
            vbr_verified_companion verified;
            verified.companion_index = uint32_t(i);
            verified.bytes = chain;
            verified.streaming_digest = digest;
            const auto accepted =
                build->accept_verified_companion(verified);
            if (accepted != vbr_capture_stream_status::ok) {
                result.inner_stream_status = accepted;
                result.status = stream_status(accepted);
                return result;
            }
        }
        result.companion_failure_index = UINT32_MAX;
        result.companion_failure_kind =
            vbr_artifact_companion_kind::_count;

        result.phase = vbr_explicit_capture_phase::unit_transfer;
        uint32_t unit_index = 0;
        for (const auto & child : children) {
            for (const auto & plan : child.units) {
                if (request.continue_transfer &&
                    !request.continue_transfer(request.continue_context)) {
                    result.status = vbr_explicit_capture_status::cancelled;
                    return result;
                }
                vbr_capture_stream_status unit_status;
                auto unit = build->begin_unit(unit_index, unit_status);
                if (!unit) {
                    result.inner_stream_status = unit_status;
                    result.status = stream_status(unit_status);
                    return result;
                }
                vbr_capture_stream_stats stats;
                if (!vbr_live_capture_adapter::stream(
                        child, plan, *unit, *request.ring, stats,
                        request.continue_context,
                        request.continue_transfer)) {
                    result.status = request.continue_transfer &&
                            !request.continue_transfer(
                                request.continue_context)
                        ? vbr_explicit_capture_status::cancelled
                        : vbr_explicit_capture_status::transfer_failed;
                    return result;
                }
                result.chunks += stats.chunks;
                result.backpressure_waits += stats.backpressure_waits;
                result.event_completions += stats.event_completions;
                result.synchronous_fallbacks +=
                    stats.synchronous_fallbacks;
                const auto sealed = unit->seal_unit();
                if (sealed != vbr_capture_stream_status::ok) {
                    result.inner_stream_status = sealed;
                    result.status =
                        vbr_explicit_capture_status::hash_mismatch;
                    return result;
                }
                for (const auto & shard : plan.shards) {
                    if (shard.payload_bytes >
                            UINT64_MAX - result.payload_bytes ||
                        shard.stash_bytes >
                            UINT64_MAX - result.stash_bytes) {
                        result.status =
                            vbr_explicit_capture_status::size_overflow;
                        return result;
                    }
                    result.payload_bytes += shard.payload_bytes;
                    result.stash_bytes += shard.stash_bytes;
                }
                ++unit_index;
            }
        }

        result.phase =
            vbr_explicit_capture_phase::post_transfer_stability;
        // Both levels of stability and quiescence are re-read after all D2H
        // completions, before the catalog's final reference publication.
        for (const auto & child : children) {
            const auto instance = child.cache->vbr_instance_id();
            if (!vbr_live_capture_adapter::stable(child) ||
                vbr_recovery_pending_for(instance)) {
                result.status =
                    vbr_explicit_capture_status::source_changed;
                return result;
            }
        }
        if (!vbr_operation_registry_quiescent_for(
                instances.data(), instances.size())) {
            result.status = vbr_explicit_capture_status::registry_busy;
            return result;
        }

        result.controllers = children.size();
        result.units = unit_index;
        result.companions = package.companions.size();
        result.status = vbr_explicit_capture_status::ok;
        state.result = result;
        state.transferred = true;
        return result;
    } catch (...) {
        result.status = vbr_explicit_capture_status::internal_error;
        state.result = result;
        return result;
    }
}

vbr_explicit_capture_result vbr_publish_explicit_manifest(
        vbr_explicit_capture_operation & operation) noexcept {
    vbr_explicit_capture_result result;
    if (!operation.ready_for_publication()) {
        result.status = vbr_explicit_capture_status::internal_error;
        return result;
    }
    auto & state = *operation.impl_;
    result = state.result;
    result.phase = vbr_explicit_capture_phase::publication;
    state.published = true;
    result.sink = state.build->publish_reference();
    result.inner_stream_status = result.sink.status;
    result.status = stream_status(result.sink.status);
    if (result.status == vbr_explicit_capture_status::ok) {
        result.phase = vbr_explicit_capture_phase::complete;
    }
    state.result = result;
    return result;
}

vbr_explicit_capture_result vbr_capture_explicit_manifest(
        llama_memory_i & memory,
        const vbr_explicit_capture_request & request,
        vbr_unit_version_sink & sink,
        const vbr_explicit_capture_accounting & accounting) noexcept {
    vbr_explicit_capture_operation operation;
    auto result = vbr_prepare_explicit_manifest(
        memory, request, sink, accounting, operation);
    if (result.status != vbr_explicit_capture_status::ok) {
        return result;
    }
    result = vbr_transfer_explicit_manifest(operation);
    if (result.status != vbr_explicit_capture_status::ok) {
        return result;
    }
    return vbr_publish_explicit_manifest(operation);
}

const char * vbr_explicit_capture_phase_name(
        vbr_explicit_capture_phase phase) noexcept {
    switch (phase) {
        case vbr_explicit_capture_phase::validation: return "validation";
        case vbr_explicit_capture_phase::memory_tree: return "memory_tree";
        case vbr_explicit_capture_phase::settlement: return "settlement";
        case vbr_explicit_capture_phase::pre_capture_quiescence: return "pre_capture_quiescence";
        case vbr_explicit_capture_phase::metadata_and_manifest: return "metadata_and_manifest";
        case vbr_explicit_capture_phase::pre_transfer_stability: return "pre_transfer_stability";
        case vbr_explicit_capture_phase::accounting_configuration: return "accounting_configuration";
        case vbr_explicit_capture_phase::reservation_preparation: return "reservation_preparation";
        case vbr_explicit_capture_phase::companion_capture: return "companion_capture";
        case vbr_explicit_capture_phase::unit_transfer: return "unit_transfer";
        case vbr_explicit_capture_phase::post_transfer_stability: return "post_transfer_stability";
        case vbr_explicit_capture_phase::publication: return "publication";
        case vbr_explicit_capture_phase::complete: return "complete";
        case vbr_explicit_capture_phase::_count: return "_count";
    }
    return "invalid";
}

const char * vbr_explicit_generation_failure_name(
        vbr_explicit_generation_failure failure) noexcept {
    switch (failure) {
        case vbr_explicit_generation_failure::none: return "none";
        case vbr_explicit_generation_failure::size_pass: return "size_pass";
        case vbr_explicit_generation_failure::tracker_missing: return "tracker_missing";
        case vbr_explicit_generation_failure::tracker_unstable: return "tracker_unstable";
        case vbr_explicit_generation_failure::tracker_shadow_unavailable: return "tracker_shadow_unavailable";
        case vbr_explicit_generation_failure::invalid_sequence_or_frontier: return "invalid_sequence_or_frontier";
        case vbr_explicit_generation_failure::invalid_stream: return "invalid_stream";
        case vbr_explicit_generation_failure::ownership_index_missing: return "ownership_index_missing";
        case vbr_explicit_generation_failure::ownership_view_missing: return "ownership_view_missing";
        case vbr_explicit_generation_failure::ownership_view_unavailable: return "ownership_view_unavailable";
        case vbr_explicit_generation_failure::ownership_rank_failed: return "ownership_rank_failed";
        case vbr_explicit_generation_failure::ownership_enumeration_failed: return "ownership_enumeration_failed";
        case vbr_explicit_generation_failure::ownership_cardinality_mismatch: return "ownership_cardinality_mismatch";
        case vbr_explicit_generation_failure::stream_capture_failed: return "stream_capture_failed";
        case vbr_explicit_generation_failure::controller_capture_failed: return "controller_capture_failed";
        case vbr_explicit_generation_failure::stability_reread_failed: return "stability_reread_failed";
        case vbr_explicit_generation_failure::internal_error: return "internal_error";
        case vbr_explicit_generation_failure::_count: return "_count";
    }
    return "_count";
}

const char * vbr_explicit_size_failure_name(
        vbr_explicit_size_failure failure) noexcept {
    switch (failure) {
        case vbr_explicit_size_failure::none: return "none";
        case vbr_explicit_size_failure::not_armed: return "not_armed";
        case vbr_explicit_size_failure::tracker_missing: return "tracker_missing";
        case vbr_explicit_size_failure::tracker_unstable: return "tracker_unstable";
        case vbr_explicit_size_failure::bindings_missing: return "bindings_missing";
        case vbr_explicit_size_failure::stream_layout: return "stream_layout";
        case vbr_explicit_size_failure::policy_snapshot: return "policy_snapshot";
        case vbr_explicit_size_failure::unit_index: return "unit_index";
        case vbr_explicit_size_failure::extents_empty: return "extents_empty";
        case vbr_explicit_size_failure::extent_missing: return "extent_missing";
        case vbr_explicit_size_failure::vmm_missing: return "vmm_missing";
        case vbr_explicit_size_failure::backend_unavailable: return "backend_unavailable";
        case vbr_explicit_size_failure::wm_cells_zero: return "wm_cells_zero";
        case vbr_explicit_size_failure::extent_type_mismatch: return "extent_type_mismatch";
        case vbr_explicit_size_failure::promote_hops_mismatch: return "promote_hops_mismatch";
        case vbr_explicit_size_failure::domain_mismatch: return "domain_mismatch";
        case vbr_explicit_size_failure::shard_disagreement: return "shard_disagreement";
        case vbr_explicit_size_failure::binding_missing: return "binding_missing";
        case vbr_explicit_size_failure::topology_order: return "topology_order";
        case vbr_explicit_size_failure::bounds: return "bounds";
        case vbr_explicit_size_failure::stash_bounds: return "stash_bounds";
        case vbr_explicit_size_failure::stability_reread: return "stability_reread";
        case vbr_explicit_size_failure::internal_error: return "internal_error";
        case vbr_explicit_size_failure::_count: return "_count";
    }
    return "_count";
}

vbr_explicit_size_failure vbr_explicit_capture_validate_extent_generation(
        uint32_t wm_cells,
        int32_t extent_type,
        uint8_t extent_promote_hops,
        const vbr_unit_generation & generation) noexcept {
    if (wm_cells == 0) {
        return vbr_explicit_size_failure::wm_cells_zero;
    }
    if (extent_type != generation.current_type) {
        return vbr_explicit_size_failure::extent_type_mismatch;
    }
    if (extent_promote_hops != generation.promote_hops) {
        return vbr_explicit_size_failure::promote_hops_mismatch;
    }
    const auto expected_domain =
        generation.current_type == GGML_TYPE_F16 ||
        generation.current_type == GGML_TYPE_TURBO8_0
            ? vbr_repr_domain::full
            : vbr_repr_domain::tapped;
    if (generation.domain != expected_domain) {
        return vbr_explicit_size_failure::domain_mismatch;
    }
    return vbr_explicit_size_failure::none;
}

const char * vbr_explicit_capture_status_name(
        vbr_explicit_capture_status status) noexcept {
    switch (status) {
        case vbr_explicit_capture_status::ok: return "ok";
        case vbr_explicit_capture_status::not_armed: return "not_armed";
        case vbr_explicit_capture_status::unsupported_layout: return "unsupported_layout";
        case vbr_explicit_capture_status::slot_not_idle: return "slot_not_idle";
        case vbr_explicit_capture_status::identity_unavailable: return "identity_unavailable";
        case vbr_explicit_capture_status::generation_unavailable: return "generation_unavailable";
        case vbr_explicit_capture_status::registry_busy: return "registry_busy";
        case vbr_explicit_capture_status::recovery_pending: return "recovery_pending";
        case vbr_explicit_capture_status::geometry_mismatch: return "geometry_mismatch";
        case vbr_explicit_capture_status::stash_inconsistent: return "stash_inconsistent";
        case vbr_explicit_capture_status::required_companion_unavailable: return "required_companion_unavailable";
        case vbr_explicit_capture_status::size_overflow: return "size_overflow";
        case vbr_explicit_capture_status::ring_unavailable: return "ring_unavailable";
        case vbr_explicit_capture_status::admission_refused: return "admission_refused";
        case vbr_explicit_capture_status::cancelled: return "cancelled";
        case vbr_explicit_capture_status::transfer_failed: return "transfer_failed";
        case vbr_explicit_capture_status::short_read: return "short_read";
        case vbr_explicit_capture_status::event_failed: return "event_failed";
        case vbr_explicit_capture_status::source_changed: return "source_changed";
        case vbr_explicit_capture_status::hash_mismatch: return "hash_mismatch";
        case vbr_explicit_capture_status::dedup_validation_failed: return "dedup_validation_failed";
        case vbr_explicit_capture_status::accounting_failed: return "accounting_failed";
        case vbr_explicit_capture_status::publication_failed: return "publication_failed";
        case vbr_explicit_capture_status::internal_error: return "internal_error";
        case vbr_explicit_capture_status::_count: return "_count";
    }
    return "_count";
}
