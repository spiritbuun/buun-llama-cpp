#include "llama-vbr-artifact.h"
#include "llama-bit-ops.h"

#include "llama-sha256.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t ARTIFACT_MAGIC = 0x32524256; // "VBR2", canonical little-endian
constexpr uint32_t ARTIFACT_HEADER_SIZE = 4 + 4 + 4 + 4 + 8 + 4 + 32 + 32;
constexpr uint32_t ARTIFACT_SECTION_HEADER_SIZE = 4 + 4 + 8 + 32;
constexpr uint32_t ARTIFACT_FLAGS_V1 = 0;
constexpr size_t STREAM_CHUNK_SIZE = 1024 * 1024;
constexpr uint64_t MIN_WIRE_TOPOLOGY = 84;
constexpr uint64_t MIN_WIRE_DEVICE = 36;
constexpr uint64_t MIN_WIRE_CONTROLLER = 40;
constexpr uint64_t MIN_WIRE_UNIT_GENERATION = 28;
constexpr uint64_t MIN_WIRE_STREAM = 20;
constexpr uint64_t MIN_WIRE_PAGE = 40;
constexpr uint64_t MIN_WIRE_SHARD = 84;
constexpr uint64_t MIN_WIRE_CONTROLLER_POLICY = 144;
constexpr uint64_t MIN_WIRE_STREAM_PLACEMENT = 20;
constexpr uint64_t MIN_WIRE_CELL_PLACEMENT = 16;
constexpr uint64_t MIN_WIRE_UNIT_REFERENCE = 100;
constexpr uint64_t MIN_WIRE_COMPANION_REFERENCE = 128;
constexpr uint64_t MIN_WIRE_ACCOUNTING_ROW = 40;

constexpr char DOMAIN_PACKAGE[] = "buun.vbr.artifact-package";
constexpr char DOMAIN_SECTION[] = "buun.vbr.artifact-section";
constexpr char DOMAIN_ORDERING[] = "buun.vbr.artifact-order";
constexpr char DOMAIN_UNIT[] = "buun.vbr.unit-version";
constexpr char DOMAIN_PAYLOAD[] = "buun.vbr.payload";
constexpr char DOMAIN_STASH[] = "buun.vbr.clean-stash";
constexpr char DOMAIN_MANIFEST[] = "buun.vbr.reference-manifest";
constexpr char DOMAIN_CAPTURE[] = "buun.vbr.capture-generation";
constexpr char DOMAIN_SHARD[] = "buun.vbr.shard";
constexpr char DOMAIN_COMPANION[] = "buun.vbr.companion";
constexpr char DOMAIN_TOKEN_BLOCK[] = "buun.vbr.token-block";

static_assert(llama_cache_acct_all_ids_distinct<
        vbr_unit_version_id,
        vbr_payload_digest,
        vbr_stash_payload_id,
        vbr_manifest_digest,
        vbr_capture_generation_id,
        vbr_transition_lineage_id,
        vbr_token_block_digest,
        llama_cache_acct_content_digest,
        llama_cache_acct_lineage_id>::value);
static_assert(llama_cache_acct_distinct_from_all<
        vbr_unit_version_id, uint32_t, uint64_t>);

bool checked_add(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool digest_nonzero(const std::array<uint8_t, 32> & bytes) {
    return std::any_of(bytes.begin(), bytes.end(), [](uint8_t value) {
        return value != 0;
    });
}

bool artifact_version_supported(uint32_t version) {
    return version >= VBR_UNIT_ARTIFACT_FORMAT_VERSION_MIN &&
           version <= VBR_UNIT_ARTIFACT_FORMAT_VERSION;
}

template <typename Digest>
Digest typed_digest(llama_sha256_writer & writer) {
    return Digest::from_sha256(writer.finish());
}

struct emitter {
    const vbr_artifact_stream_writer * output = nullptr;
    llama_sha256_writer * hash_a = nullptr;
    llama_sha256_writer * hash_b = nullptr;
    uint64_t count = 0;
    bool ok = true;

    bool raw(const void * data, size_t size) {
        uint64_t next;
        if (!ok || !checked_add(count, size, next)) {
            ok = false;
            return false;
        }
        if (hash_a) {
            hash_a->bytes(data, size);
        }
        if (hash_b) {
            hash_b->bytes(data, size);
        }
        if (output && size > 0 &&
            (!output->write ||
             !output->write(output->context,
                            static_cast<const uint8_t *>(data), size))) {
            ok = false;
            return false;
        }
        count = next;
        return true;
    }

    bool u32(uint32_t value) {
        uint8_t data[4];
        llama_store_le_u32(data, value);
        return raw(data, sizeof(data));
    }

    bool i32(int32_t value) {
        return u32(uint32_t(value));
    }

    bool u64(uint64_t value) {
        uint8_t data[8];
        llama_store_le_u64(data, value);
        return raw(data, sizeof(data));
    }

    bool i64(int64_t value) {
        return u64(uint64_t(value));
    }

    bool bytes(const void * data, size_t size) {
        return u64(size) && raw(data, size);
    }

    template <typename Digest>
    bool digest(const Digest & value) {
        return raw(value.bytes().data(), value.bytes().size());
    }

    bool array_digest(const std::array<uint8_t, 32> & value) {
        return raw(value.data(), value.size());
    }
};

template <typename Fn>
bool read_source_chunks(const vbr_artifact_byte_source & source, Fn && fn) {
    if (!source.valid()) {
        return false;
    }
    // Keep the 1 MiB transfer buffer off the stack. MSVC's default process
    // stack is also 1 MiB, so this otherwise overflows before the first read.
    std::vector<uint8_t> buffer(STREAM_CHUNK_SIZE);
    uint64_t offset = 0;
    while (offset < source.size) {
        const size_t n = size_t(std::min<uint64_t>(
            source.size - offset, buffer.size()));
        if (!source.read(source.context, offset, buffer.data(), n) ||
            !fn(offset, buffer.data(), n)) {
            return false;
        }
        offset += n;
    }
    return true;
}

bool stream_payload_chunks(
        emitter & out,
        const vbr_artifact_byte_source & source,
        llama_sha256_writer * hash_a = nullptr,
        llama_sha256_writer * hash_b = nullptr,
        llama_sha256_writer * hash_c = nullptr) {
    if (!out.u64(source.size)) {
        return false;
    }
    return read_source_chunks(source, [&](uint64_t, const uint8_t * data, size_t size) {
        if (hash_a) {
            hash_a->bytes(data, size);
        }
        if (hash_b) {
            hash_b->bytes(data, size);
        }
        if (hash_c) {
            hash_c->bytes(data, size);
        }
        return out.raw(data, size);
    });
}

bool digest_matches_source(
        const char * domain,
        uint32_t object_index,
        uint32_t shard_index,
        const vbr_artifact_byte_source & source,
        std::array<uint8_t, 32> & out) {
    llama_sha256_writer writer;
    writer.string(domain, strlen(domain));
    writer.u32(object_index);
    writer.u32(shard_index);
    writer.u64(source.size);
    if (!read_source_chunks(source, [&](uint64_t, const uint8_t * data, size_t size) {
            writer.bytes(data, size);
            return true;
        })) {
        return false;
    }
    out = writer.finish();
    return digest_nonzero(out);
}

bool emit_lineage(emitter & out, const vbr_lineage_uuid & lineage) {
    return out.u64(lineage.hi) && out.u64(lineage.lo);
}

bool emit_portable_domain(emitter & out, const vbr_artifact_portable_domain & domain) {
    return out.u32(uint32_t(domain.residency)) &&
           out.u32(uint32_t(domain.kind)) &&
           out.u32(domain.topology_index) &&
           out.u32(domain.device_ordinal);
}

bool emit_page_ref(emitter & out, const vbr_generation_page_ref & page) {
    if (!out.u32(page.page_index) ||
        !out.u32(page.captured_page_gen)) {
        return false;
    }
    for (uint64_t word : page.covered_mask) {
        if (!out.u64(word)) {
            return false;
        }
    }
    return true;
}

bool emit_unit_generation(
        emitter & out,
        const vbr_checkpoint_unit_generation & unit) {
    return out.u64(unit.repr_gen) &&
           out.i32(unit.current_type) &&
           out.i32(unit.last_source_type) &&
           out.u32(uint32_t(unit.domain)) &&
           out.u32(unit.promote_hops) &&
           out.u32(uint32_t(unit.last_transition));
}

bool emit_generation_record(
        emitter & out,
        const vbr_checkpoint_generation_record & generation) {
    if (!out.u32(generation.version) ||
        !out.u32(uint32_t(generation.status)) ||
        !out.array_digest(generation.identity_policy_order_digest) ||
        !out.u32(uint32_t(generation.controllers.size()))) {
        return false;
    }
    for (const auto & controller : generation.controllers) {
        if (!out.u32(controller.child_id) ||
            !out.u32(uint32_t(controller.dependency_mode)) ||
            !emit_lineage(out, controller.lineage_uuid) ||
            !out.u64(controller.global_generation) ||
            !out.u32(uint32_t(controller.units.size()))) {
            return false;
        }
        for (const auto & unit : controller.units) {
            if (!emit_unit_generation(out, unit)) {
                return false;
            }
        }
        if (!out.u32(uint32_t(controller.streams.size()))) {
            return false;
        }
        for (const auto & stream : controller.streams) {
            if (!out.u32(stream.stream_index) ||
                !out.i32(stream.dependency_seq_id) ||
                !out.i32(stream.computation_frontier) ||
                !out.u32(stream.captured_dependency_count) ||
                !out.u32(uint32_t(stream.pages.size()))) {
                return false;
            }
            for (const auto & page : stream.pages) {
                if (!emit_page_ref(out, page)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool emit_shard_descriptor(
        emitter & out,
        const vbr_artifact_shard_descriptor & shard) {
    return out.u32(shard.shard_index) &&
           out.u32(shard.topology_index) &&
           out.u32(shard.device_ordinal) &&
           out.u64(shard.logical_offset) &&
           out.u64(shard.row_count) &&
           out.u64(shard.column_count) &&
           out.u64(shard.row_bytes) &&
           out.u64(shard.payload_bytes) &&
           out.array_digest(shard.section_checksum);
}

bool emit_unit_descriptor_body(
        emitter & out,
        const vbr_artifact_unit_descriptor & descriptor,
        uint32_t format_version,
        bool include_child_id = true) {
    if ((include_child_id && !out.u32(descriptor.child_id)) ||
        !out.i32(descriptor.current_type) ||
        !out.i32(descriptor.last_source_type) ||
        !out.u32(descriptor.promote_hops) ||
        !out.u32(uint32_t(descriptor.last_transition)) ||
        !out.u32(uint32_t(descriptor.representation.kind)) ||
        !out.u32(descriptor.representation.codec_id) ||
        !out.u32(descriptor.representation.codec_version) ||
        !out.array_digest(descriptor.representation.reference_digest) ||
        !out.u32(descriptor.representation.source_loss_history) ||
        !out.u32(descriptor.representation.checkpoint_codec_hops) ||
        !out.u32(uint32_t(descriptor.recoverability)) ||
        !out.u32(uint32_t(descriptor.side)) ||
        !out.u32(uint32_t(descriptor.layout)) ||
        !out.u32(descriptor.n_stream) ||
        !out.u32(descriptor.unified ? 1 : 0) ||
        !out.u64(descriptor.wm_cells) ||
        !out.u32(descriptor.rank)) {
        return false;
    }
    for (uint64_t dimension : descriptor.dimensions) {
        if (!out.u64(dimension)) {
            return false;
        }
    }
    if (!out.u64(descriptor.row_alignment) ||
        !out.u32(descriptor.row_codec_version) ||
        !out.array_digest(descriptor.codebook_digest) ||
        !out.array_digest(descriptor.rotation_digest) ||
        !out.array_digest(descriptor.meansub_digest) ||
        (artifact_has_meansub_reference(format_version) &&
         (!out.i32(descriptor.meansub_model_id) ||
          !out.i32(descriptor.meansub_layer) ||
          !out.u32(descriptor.meansub_baked ? 1 : 0))) ||
        !out.u32(uint32_t(descriptor.shards.size()))) {
        return false;
    }
    for (const auto & shard : descriptor.shards) {
        if (!emit_shard_descriptor(out, shard)) {
            return false;
        }
    }
    return true;
}

bool emit_clean_stash_descriptor(
        emitter & out,
        const vbr_artifact_clean_stash & stash) {
    if (!out.u64(stash.valid_rows) ||
        !out.u32(uint32_t(stash.domain)) ||
        !out.u32(uint32_t(stash.layout)) ||
        !out.u64(stash.row_count) ||
        !out.u64(stash.column_count) ||
        !out.u64(stash.row_bytes) ||
        !out.digest(stash.payload_id) ||
        !out.u32(uint32_t(stash.shards.size()))) {
        return false;
    }
    for (const auto & shard : stash.shards) {
        if (!emit_shard_descriptor(out, shard)) {
            return false;
        }
    }
    return true;
}

template <typename T, typename Emit>
bool hash_sized(
        llama_sha256_writer & hash,
        const T & value,
        Emit && emit) {
    emitter count;
    if (!emit(count, value) || !count.ok) {
        return false;
    }
    hash.u64(count.count);
    emitter body;
    body.hash_a = &hash;
    return emit(body, value) && body.ok &&
           body.count == count.count;
}

bool hash_sized_descriptor(
        llama_sha256_writer & hash,
        const vbr_artifact_unit_descriptor & descriptor,
        uint32_t format_version) {
    return hash_sized(
        hash, descriptor,
        [format_version](emitter & out, const vbr_artifact_unit_descriptor & value) {
            return emit_unit_descriptor_body(
                out, value, format_version, false);
        });
}

bool hash_sized_stash_descriptor(
        llama_sha256_writer & hash,
        const vbr_artifact_clean_stash & stash) {
    return hash_sized(
        hash, stash,
        [](emitter & out, const vbr_artifact_clean_stash & value) {
            return emit_clean_stash_descriptor(out, value);
        });
}

bool hash_source_string(
        llama_sha256_writer & hash,
        const vbr_artifact_byte_source & source) {
    hash.u64(source.size);
    return read_source_chunks(source, [&](uint64_t, const uint8_t * data, size_t size) {
        hash.bytes(data, size);
        return true;
    });
}

bool hash_generation(
        const vbr_checkpoint_generation_record & generation,
        vbr_capture_generation_id & result) {
    llama_sha256_writer hash;
    hash.string(DOMAIN_CAPTURE, sizeof(DOMAIN_CAPTURE) - 1);
    emitter out;
    out.hash_a = &hash;
    if (!emit_generation_record(out, generation) || !out.ok) {
        return false;
    }
    result = typed_digest<vbr_capture_generation_id>(hash);
    return result.valid();
}

bool topology_valid(const vbr_artifact_portable_topology & topology) {
    if (topology.version != LLAMA_CACHE_ACCT_TOPOLOGY_VERSION ||
        topology.device_count == 0 ||
        topology.device_identities.empty() ||
        topology.device_count != topology.device_identities.size() ||
        topology.device_identities.size() != topology.shard_weights.size() ||
        topology.device_identities.size() > UINT16_MAX ||
        topology.main_device.v >= topology.device_identities.size()) {
        return false;
    }
    uint64_t total_weight = 0;
    for (size_t i = 0; i < topology.device_identities.size(); ++i) {
        if (!digest_nonzero(topology.device_identities[i].bytes())) {
            return false;
        }
        total_weight += topology.shard_weights[i];
    }
    if (total_weight != LLAMA_CACHE_ACCT_SHARD_WEIGHT_DENOMINATOR) {
        return false;
    }
    const auto canonical =
        llama_cache_acct_compute_topology_digest(topology);
    return canonical.valid() && canonical == topology.digest;
}

bool domain_valid(
        const vbr_artifact_portable_domain & domain,
        const std::vector<vbr_artifact_portable_topology> & topologies) {
    if (domain.residency >= llama_cache_acct_residency::_count ||
        domain.kind >= llama_cache_acct_domain_kind::_count) {
        return false;
    }
    if (domain.residency == llama_cache_acct_residency::device) {
        return domain.kind == llama_cache_acct_domain_kind::device_topology &&
               domain.topology_index < topologies.size() &&
               domain.device_ordinal <
                   topologies[domain.topology_index].device_identities.size();
    }
    return domain.kind == llama_cache_acct_domain_kind::not_applicable &&
           domain.topology_index == UINT32_MAX &&
           domain.device_ordinal == UINT16_MAX;
}

llama_cache_acct_category role_category(vbr_artifact_accounting_role role) {
    switch (role) {
        case vbr_artifact_accounting_role::unit_payload:
            return llama_cache_acct_category::unit_version_payload;
        case vbr_artifact_accounting_role::clean_stash_payload:
            return llama_cache_acct_category::clean_stash_payload;
        case vbr_artifact_accounting_role::recurrent_payload:
            return llama_cache_acct_category::full_snapshot_payload;
        case vbr_artifact_accounting_role::typed_accelerator_payload:
            return llama_cache_acct_category::typed_accelerator_payload;
        case vbr_artifact_accounting_role::descriptor_metadata:
            return llama_cache_acct_category::artifact_descriptor_metadata;
        case vbr_artifact_accounting_role::reference_metadata:
            return llama_cache_acct_category::artifact_reference_metadata;
        case vbr_artifact_accounting_role::_count:
            break;
    }
    return llama_cache_acct_category::_count;
}

bool validate_generation_record(
        const vbr_checkpoint_generation_record & generation,
        const vbr_artifact_decode_limits * limits = nullptr) {
    if (generation.version != 1 ||
        generation.status != vbr_checkpoint_generation_status::complete ||
        !digest_nonzero(generation.identity_policy_order_digest) ||
        generation.controllers.empty() ||
        (limits && generation.controllers.size() > limits->max_controllers)) {
        return false;
    }
    uint32_t expected_child = 0;
    for (const auto & controller : generation.controllers) {
        if (controller.child_id != expected_child++ ||
            uint32_t(controller.dependency_mode) >
                uint32_t(checkpoint_child_dependency_mode::live_guarded) ||
            (controller.dependency_mode != checkpoint_child_dependency_mode::absent &&
             !vbr_lineage_uuid_is_set(controller.lineage_uuid)) ||
            (controller.dependency_mode == checkpoint_child_dependency_mode::absent &&
             vbr_lineage_uuid_is_set(controller.lineage_uuid)) ||
            (limits && (controller.units.size() > limits->max_units_per_controller ||
                        controller.streams.size() > limits->max_streams_per_controller))) {
            return false;
        }
        for (const auto & unit : controller.units) {
            if (unit.repr_gen == 0 ||
                unit.current_type < 0 ||
                unit.last_source_type < 0 ||
                unit.domain > vbr_repr_domain::tapped ||
                unit.last_transition > vbr_repr_transition::recovery_invalidate) {
                return false;
            }
        }
        uint32_t expected_stream = 0;
        for (const auto & stream : controller.streams) {
            if (stream.stream_index != expected_stream++ ||
                stream.computation_frontier < 0 ||
                (limits && stream.pages.size() > limits->max_pages_per_stream)) {
                return false;
            }
            uint64_t cardinality = 0;
            uint32_t prior_page = 0;
            bool first = true;
            for (const auto & page : stream.pages) {
                if ((!first && page.page_index <= prior_page) ||
                    page.captured_page_gen == 0) {
                    return false;
                }
                first = false;
                prior_page = page.page_index;
                for (uint64_t word : page.covered_mask) {
                    cardinality += llama_popcount_u64(word);
                }
            }
            if (cardinality != stream.captured_dependency_count) {
                return false;
            }
        }
    }
    return true;
}

bool shard_metadata_valid(
        const vbr_artifact_shard_descriptor & shard,
        uint32_t expected_index,
        const std::vector<vbr_artifact_portable_topology> & topologies,
        bool require_source) {
    if (shard.shard_index != expected_index ||
        shard.topology_index >= topologies.size() ||
        shard.device_ordinal >=
            topologies[shard.topology_index].device_identities.size() ||
        shard.row_count == 0 ||
        shard.column_count == 0 ||
        shard.row_bytes == 0 ||
        shard.payload_bytes == 0 ||
        !digest_nonzero(shard.section_checksum)) {
        return false;
    }
    uint64_t expected_size;
    if (shard.row_count >
        std::numeric_limits<uint64_t>::max() / shard.row_bytes) {
        return false;
    }
    expected_size = shard.row_count * shard.row_bytes;
    return expected_size == shard.payload_bytes &&
           (!require_source ||
            (shard.payload.valid() &&
             shard.payload.size == shard.payload_bytes));
}

bool descriptor_metadata_valid(
        const vbr_artifact_unit_descriptor & descriptor,
        const std::vector<vbr_artifact_portable_topology> & topologies,
        uint32_t format_version,
        bool require_source,
        bool allow_sparse_rows = false) {
    const bool current_type_supported =
        descriptor.current_type == GGML_TYPE_F16 ||
        descriptor.current_type == GGML_TYPE_TURBO8_0 ||
        descriptor.current_type == GGML_TYPE_TURBO4_0 ||
        descriptor.current_type == GGML_TYPE_TURBO3_TCQ ||
        descriptor.current_type == GGML_TYPE_TURBO2_TCQ ||
        descriptor.current_type == GGML_TYPE_TURBO1_TCQ;
    const bool source_type_supported =
        descriptor.last_source_type == GGML_TYPE_F16 ||
        descriptor.last_source_type == GGML_TYPE_TURBO8_0 ||
        descriptor.last_source_type == GGML_TYPE_TURBO4_0 ||
        descriptor.last_source_type == GGML_TYPE_TURBO3_TCQ ||
        descriptor.last_source_type == GGML_TYPE_TURBO2_TCQ ||
        descriptor.last_source_type == GGML_TYPE_TURBO1_TCQ;
    if (!vbr_lineage_uuid_is_set(descriptor.lineage_uuid) ||
        descriptor.repr_gen == 0 ||
        !current_type_supported ||
        !source_type_supported ||
        descriptor.last_transition > vbr_repr_transition::recovery_invalidate ||
        descriptor.representation.kind >= vbr_artifact_representation_kind::_count ||
        descriptor.representation.codec_version == 0 ||
        (descriptor.current_type != GGML_TYPE_F16 &&
         descriptor.representation.kind !=
             vbr_artifact_representation_kind::approximate) ||
        (descriptor.representation.kind ==
             vbr_artifact_representation_kind::approximate &&
         !digest_nonzero(descriptor.representation.reference_digest)) ||
        descriptor.recoverability >= vbr_artifact_recoverability::_count ||
        descriptor.side >= vbr_artifact_side::_count ||
        descriptor.layout >= vbr_artifact_layout::_count ||
        descriptor.n_stream == 0 ||
        descriptor.wm_cells == 0 ||
        descriptor.rank == 0 ||
        descriptor.rank > descriptor.dimensions.size() ||
        descriptor.row_alignment == 0 ||
        descriptor.row_codec_version == 0 ||
        descriptor.shards.empty() ||
        descriptor.clean_stash_state >= vbr_artifact_clean_stash_state::_count ||
        (artifact_has_meansub_reference(format_version)
             ? (descriptor.meansub_model_id < 0 ||
                descriptor.meansub_layer < 0)
             : (descriptor.meansub_model_id != -1 ||
                descriptor.meansub_layer != -1 ||
                descriptor.meansub_baked))) {
        return false;
    }
    for (uint32_t i = 0; i < descriptor.rank; ++i) {
        if (descriptor.dimensions[i] == 0) {
            return false;
        }
    }
    for (uint32_t i = descriptor.rank; i < descriptor.dimensions.size(); ++i) {
        if (descriptor.dimensions[i] != 0) {
            return false;
        }
    }
    uint32_t topology_index = UINT32_MAX;
    uint16_t prior_ordinal = 0;
    for (uint32_t i = 0; i < descriptor.shards.size(); ++i) {
        if (!shard_metadata_valid(
                descriptor.shards[i], i, topologies, require_source) ||
            (allow_sparse_rows
                ? (descriptor.shards[i].row_count == 0 ||
                   descriptor.shards[i].row_count > descriptor.wm_cells)
                : descriptor.shards[i].row_count != descriptor.wm_cells) ||
            (i > 0 &&
             (descriptor.shards[i].topology_index != topology_index ||
              descriptor.shards[i].device_ordinal <= prior_ordinal))) {
            return false;
        }
        topology_index = descriptor.shards[i].topology_index;
        prior_ordinal = descriptor.shards[i].device_ordinal;
    }
    const auto & stash = descriptor.clean_stash;
    if (descriptor.clean_stash_state !=
        vbr_artifact_clean_stash_state::present) {
        return stash.valid_rows == 0 && stash.row_count == 0 &&
               stash.column_count == 0 && stash.row_bytes == 0 &&
               stash.shards.empty() &&
               !stash.payload_id.valid();
    }
    if (stash.valid_rows == 0 ||
        stash.valid_rows > stash.row_count ||
        stash.domain != vbr_repr_domain::tapped ||
        stash.layout >= vbr_artifact_layout::_count ||
        stash.row_count == 0 ||
        stash.column_count == 0 ||
        stash.row_bytes == 0 ||
        stash.column_count >
            std::numeric_limits<uint64_t>::max() / sizeof(uint16_t) ||
        stash.row_bytes != stash.column_count * sizeof(uint16_t) ||
        !stash.payload_id.valid() ||
        stash.shards.empty()) {
        return false;
    }
    if (stash.shards.size() != descriptor.shards.size()) {
        return false;
    }
    for (uint32_t i = 0; i < stash.shards.size(); ++i) {
        if (!shard_metadata_valid(
                stash.shards[i], i, topologies, require_source) ||
            stash.shards[i].row_count != stash.valid_rows ||
            stash.shards[i].topology_index !=
                descriptor.shards[i].topology_index ||
            stash.shards[i].device_ordinal !=
                descriptor.shards[i].device_ordinal) {
            return false;
        }
    }
    return true;
}

bool prepare_shard_checksums(
        uint32_t object_index,
        std::vector<vbr_artifact_shard_descriptor> & shards) {
    for (auto & shard : shards) {
        if (!shard.payload.valid() ||
            shard.payload.size != shard.payload_bytes ||
            !digest_matches_source(
                DOMAIN_SHARD, object_index, shard.shard_index,
                shard.payload, shard.section_checksum)) {
            return false;
        }
    }
    return true;
}

bool canonicalize_shards(
        std::vector<vbr_artifact_shard_descriptor> & shards) {
    std::sort(
        shards.begin(), shards.end(),
        [](const vbr_artifact_shard_descriptor & lhs,
           const vbr_artifact_shard_descriptor & rhs) {
            return lhs.shard_index < rhs.shard_index;
        });
    for (uint32_t i = 0; i < shards.size(); ++i) {
        if (shards[i].shard_index != i) {
            return false;
        }
    }
    return true;
}

bool prepare_payload_digest(vbr_artifact_unit_blob & blob) {
    llama_sha256_writer hash;
    hash.string(DOMAIN_PAYLOAD, sizeof(DOMAIN_PAYLOAD) - 1);
    hash.u32(uint32_t(blob.descriptor.shards.size()));
    for (const auto & shard : blob.descriptor.shards) {
        emitter desc;
        desc.hash_a = &hash;
        if (!emit_shard_descriptor(desc, shard) ||
            !hash_source_string(hash, shard.payload)) {
            return false;
        }
    }
    blob.payload_digest = typed_digest<vbr_payload_digest>(hash);
    return blob.payload_digest.valid();
}

bool prepare_stash_digest(vbr_artifact_unit_blob & blob) {
    if (blob.descriptor.clean_stash_state !=
        vbr_artifact_clean_stash_state::present) {
        blob.descriptor.clean_stash.payload_id = {};
        return true;
    }
    auto & stash = blob.descriptor.clean_stash;
    llama_sha256_writer hash;
    hash.string(DOMAIN_STASH, sizeof(DOMAIN_STASH) - 1);
    hash.u32(uint32_t(stash.layout));
    hash.u64(stash.row_count);
    hash.u64(stash.column_count);
    hash.u64(stash.row_bytes);
    hash.u64(stash.valid_rows);
    hash.u32(uint32_t(stash.shards.size()));
    for (const auto & shard : stash.shards) {
        emitter desc;
        desc.hash_a = &hash;
        if (!emit_shard_descriptor(desc, shard) ||
            !hash_source_string(hash, shard.payload)) {
            return false;
        }
    }
    stash.payload_id = typed_digest<vbr_stash_payload_id>(hash);
    return stash.payload_id.valid();
}

bool hash_lineage(
        llama_sha256_writer & hash,
        const vbr_lineage_uuid & lineage) {
    emitter out;
    out.hash_a = &hash;
    return emit_lineage(out, lineage) && out.ok;
}

bool prepare_unit_id(vbr_artifact_unit_blob & blob, uint32_t format_version) {
    const auto & descriptor = blob.descriptor;
    llama_sha256_writer hash;
    hash.string(DOMAIN_UNIT, sizeof(DOMAIN_UNIT) - 1);
    hash.u32(format_version);
    if (!hash_lineage(hash, descriptor.lineage_uuid)) {
        return false;
    }
    hash.u32(descriptor.logical_unit_id);
    hash.u64(descriptor.repr_gen);
    if (!hash_sized_descriptor(hash, descriptor, format_version)) {
        return false;
    }
    hash.u32(uint32_t(descriptor.shards.size()));
    for (const auto & shard : descriptor.shards) {
        hash.u32(shard.shard_index);
        if (!hash_source_string(hash, shard.payload)) {
            return false;
        }
    }
    hash.u32(uint32_t(descriptor.clean_stash_state));
    if (descriptor.clean_stash_state ==
        vbr_artifact_clean_stash_state::present) {
        if (!hash_sized_stash_descriptor(hash, descriptor.clean_stash)) {
            return false;
        }
        hash.u32(uint32_t(descriptor.clean_stash.shards.size()));
        for (const auto & shard : descriptor.clean_stash.shards) {
            hash.u32(shard.shard_index);
            if (!hash_source_string(hash, shard.payload)) {
                return false;
            }
        }
    }
    blob.unit_version_id = typed_digest<vbr_unit_version_id>(hash);
    return blob.unit_version_id.valid();
}

bool emit_identity(emitter & out, const vbr_artifact_identity_block & identity) {
    return out.bytes(identity.execution_identity.data(),
                     identity.execution_identity.size()) &&
           out.bytes(identity.adapter_config_identity.data(),
                     identity.adapter_config_identity.size()) &&
           out.bytes(identity.media_content_identity.data(),
                     identity.media_content_identity.size()) &&
           out.u64(identity.sequence_epoch) &&
           out.i64(identity.token_count) &&
           out.i32(identity.next_position);
}

bool emit_token_block(
        emitter & out,
        const vbr_artifact_token_block & block,
        bool include_digest) {
    if (!out.u32(block.codec_version) ||
        !out.u32(uint32_t(block.tokens.size()))) {
        return false;
    }
    for (llama_token token : block.tokens) {
        if (!out.i32(token)) {
            return false;
        }
    }
    return !include_digest || out.digest(block.digest);
}

bool prepare_token_block(
        const vbr_artifact_identity_block & identity,
        const std::array<uint8_t, 32> & identity_policy_order_digest,
        vbr_artifact_token_block & block) {
    if (block.codec_version != VBR_ARTIFACT_TOKEN_BLOCK_CODEC_VERSION ||
        block.tokens.size() != size_t(identity.token_count) ||
        block.tokens.size() > UINT32_MAX) {
        return false;
    }
    llama_sha256_writer hash;
    hash.string(DOMAIN_TOKEN_BLOCK, sizeof(DOMAIN_TOKEN_BLOCK) - 1);
    emitter out;
    out.hash_a = &hash;
    if (!emit_identity(out, identity) ||
        !out.array_digest(identity_policy_order_digest) ||
        !emit_token_block(out, block, false) || !out.ok) {
        return false;
    }
    block.digest = typed_digest<vbr_token_block_digest>(hash);
    return block.digest.valid();
}

bool emit_stream_placement(
        emitter & out,
        const vbr_artifact_stream_placement & placement) {
    if (!out.u32(placement.child_id) ||
        !out.u32(placement.stream_index) ||
        !out.i32(placement.source_sequence) ||
        !out.i32(placement.computation_frontier) ||
        !out.u32(uint32_t(placement.cells.size()))) {
        return false;
    }
    for (const auto & cell : placement.cells) {
        if (!out.u32(cell.physical_cell) ||
            !out.i32(cell.logical_position) ||
            !out.i32(cell.ext_x) ||
            !out.i32(cell.ext_y)) {
            return false;
        }
    }
    return true;
}

bool emit_controller_policy(
        emitter & out,
        const vbr_artifact_controller_policy & policy) {
    return out.u32(policy.child_id) &&
           out.u32(uint32_t(policy.dependency_mode)) &&
           out.array_digest(policy.degrade_order_digest) &&
           out.array_digest(policy.policy_digest) &&
           out.u64(policy.cursor) &&
           out.i32(policy.floor_type) &&
           out.u64(policy.pressure_independent_settings) &&
           out.u32(policy.n_stream) &&
           out.u32(policy.unified ? 1 : 0) &&
           out.u64(policy.wm_cells) &&
           out.array_digest(policy.current_type_vector_digest) &&
           out.u32(policy.completed_wave ? 1 : 0);
}

bool emit_stash_reference(
        emitter & out,
        const vbr_artifact_stash_reference & reference) {
    if (!out.u64(reference.valid_rows) ||
        !out.u32(uint32_t(reference.domain)) ||
        !out.u64(reference.row_count) ||
        !out.u64(reference.column_count) ||
        !out.u64(reference.row_bytes) ||
        !out.u32(reference.captured_sink_count) ||
        !out.u32(uint32_t(reference.covered_sink_pages.size()))) {
        return false;
    }
    for (const auto & page : reference.covered_sink_pages) {
        if (!emit_page_ref(out, page)) {
            return false;
        }
    }
    return out.digest(reference.payload_id);
}

bool emit_unit_reference(
        emitter & out,
        const vbr_artifact_unit_reference & reference) {
    if (!emit_lineage(out, reference.lineage_uuid) ||
        !out.u32(reference.logical_unit_id) ||
        !out.u64(reference.repr_gen) ||
        !out.digest(reference.unit_version_id) ||
        !out.digest(reference.payload_digest) ||
        !out.u32(uint32_t(reference.authorized_stream_refs.size()))) {
        return false;
    }
    for (uint32_t stream : reference.authorized_stream_refs) {
        if (!out.u32(stream)) {
            return false;
        }
    }
    return out.u32(reference.has_stash_reference ? 1 : 0) &&
           (!reference.has_stash_reference ||
            emit_stash_reference(out, reference.stash_reference));
}

bool emit_companion_reference(
        emitter & out,
        const vbr_artifact_companion_payload & companion) {
    return out.u32(uint32_t(companion.kind)) &&
           out.u32(companion.format_version) &&
           out.array_digest(companion.build_identity_digest) &&
           emit_portable_domain(out, companion.domain) &&
           out.digest(companion.payload_digest) &&
           out.u64(companion.payload_bytes) &&
           out.array_digest(companion.section_checksum);
}

bool emit_accounting_row(
        emitter & out,
        const vbr_artifact_portable_accounting_row & row) {
    return out.u32(uint32_t(row.role)) &&
           emit_portable_domain(out, row.domain) &&
           out.u64(row.logical_bytes) &&
           out.u64(row.resident_bytes) &&
           out.u32(uint32_t(row.attribution));
}

bool emit_consistency(
        emitter & out,
        const vbr_artifact_consistency & consistency) {
    return out.u32(uint32_t(consistency.kind)) &&
           out.digest(consistency.source_capture_generation_id) &&
           out.digest(consistency.target_capture_generation_id) &&
           out.digest(consistency.transition_lineage_id);
}

bool emit_manifest_body(
        emitter & out,
        const vbr_artifact_reference_manifest & manifest) {
    if (!out.u32(manifest.version) ||
        !out.array_digest(manifest.identity_policy_order_digest) ||
        !emit_identity(out, manifest.identity) ||
        (artifact_has_reference_placement(manifest.version) &&
         !emit_token_block(out, manifest.token_block, true)) ||
        !emit_generation_record(out, manifest.generation) ||
        !out.digest(manifest.capture_generation_id) ||
        !emit_consistency(out, manifest.consistency) ||
        !out.u32(uint32_t(manifest.controller_policy.size()))) {
        return false;
    }
    for (const auto & policy : manifest.controller_policy) {
        if (!emit_controller_policy(out, policy)) {
            return false;
        }
    }
    if (artifact_has_reference_placement(manifest.version)) {
        if (!out.u32(uint32_t(manifest.stream_placements.size()))) {
            return false;
        }
        for (const auto & placement : manifest.stream_placements) {
            if (!emit_stream_placement(out, placement)) {
                return false;
            }
        }
    }
    if (!out.u32(uint32_t(manifest.unit_references.size()))) {
        return false;
    }
    for (const auto & reference : manifest.unit_references) {
        if (!emit_unit_reference(out, reference)) {
            return false;
        }
    }
    if (!out.u32(uint32_t(manifest.companions.size()))) {
        return false;
    }
    for (const auto & companion : manifest.companions) {
        if (!emit_companion_reference(out, companion)) {
            return false;
        }
    }
    if (!out.u32(uint32_t(manifest.accounting.size()))) {
        return false;
    }
    for (const auto & row : manifest.accounting) {
        if (!emit_accounting_row(out, row)) {
            return false;
        }
    }
    return true;
}

bool prepare_manifest_digest(vbr_artifact_reference_manifest & manifest) {
    llama_sha256_writer hash;
    hash.string(DOMAIN_MANIFEST, sizeof(DOMAIN_MANIFEST) - 1);
    emitter out;
    out.hash_a = &hash;
    if (!emit_manifest_body(out, manifest) || !out.ok) {
        return false;
    }
    manifest.manifest_digest = typed_digest<vbr_manifest_digest>(hash);
    return manifest.manifest_digest.valid();
}

bool consistency_valid(
        const vbr_artifact_consistency & consistency,
        const vbr_capture_generation_id & capture_id) {
    if (consistency.kind >= vbr_artifact_consistency_kind::_count ||
        !consistency.source_capture_generation_id.valid() ||
        consistency.source_capture_generation_id != capture_id) {
        return false;
    }
    if (consistency.kind == vbr_artifact_consistency_kind::capture_exact) {
        return !consistency.target_capture_generation_id.valid() &&
               !consistency.transition_lineage_id.valid();
    }
    return consistency.target_capture_generation_id.valid() &&
           consistency.transition_lineage_id.valid();
}

bool identity_valid(
        const vbr_artifact_identity_block & identity,
        const vbr_artifact_decode_limits * limits = nullptr) {
    if (identity.execution_identity.empty() ||
        identity.adapter_config_identity.empty() ||
        identity.media_content_identity.empty() ||
        identity.token_count < 0 ||
        identity.next_position < 0) {
        return false;
    }
    return !limits ||
           (identity.execution_identity.size() <= limits->max_string_bytes &&
            identity.adapter_config_identity.size() <= limits->max_string_bytes &&
            identity.media_content_identity.size() <= limits->max_string_bytes);
}

bool token_block_valid(
        const vbr_artifact_identity_block & identity,
        const std::array<uint8_t, 32> & identity_policy_order_digest,
        const vbr_artifact_token_block & block,
        const vbr_artifact_decode_limits * limits) {
    if (block.codec_version != VBR_ARTIFACT_TOKEN_BLOCK_CODEC_VERSION ||
        block.tokens.size() != size_t(identity.token_count) ||
        (limits && block.tokens.size() > limits->max_token_ids) ||
        !block.digest.valid()) {
        return false;
    }
    auto canonical = block;
    return prepare_token_block(
               identity, identity_policy_order_digest, canonical) &&
           canonical.digest == block.digest;
}

bool reference_metadata_absent(
        const vbr_artifact_reference_manifest & manifest) {
    return manifest.token_block.tokens.empty() &&
           !manifest.token_block.digest.valid() &&
           manifest.stream_placements.empty();
}

bool placement_valid(
        const vbr_artifact_reference_manifest & manifest,
        const vbr_artifact_decode_limits * limits) {
    const auto carries_placement = [&](const auto & controller) {
        return controller.dependency_mode ==
                   checkpoint_child_dependency_mode::live_guarded ||
            std::any_of(
                manifest.stream_placements.begin(),
                manifest.stream_placements.end(),
                [&](const vbr_artifact_stream_placement & placement) {
                    return placement.child_id == controller.child_id;
                });
    };
    size_t expected_streams = 0;
    uint64_t total_cells = 0;
    for (const auto & controller : manifest.generation.controllers) {
        if (carries_placement(controller)) {
            expected_streams += controller.streams.size();
        }
    }
    if (manifest.stream_placements.size() != expected_streams ||
        (limits && manifest.stream_placements.size() >
                       limits->max_stream_placements)) {
        return false;
    }

    size_t next = 0;
    std::map<std::pair<uint32_t, llama_seq_id>, std::set<llama_pos>>
        logical_positions;
    for (const auto & controller : manifest.generation.controllers) {
        if (!carries_placement(controller)) {
            continue;
        }
        for (const auto & stream : controller.streams) {
            if (next >= manifest.stream_placements.size()) {
                return false;
            }
            const auto & placement = manifest.stream_placements[next++];
            if (placement.child_id != controller.child_id ||
                placement.stream_index != stream.stream_index ||
                placement.source_sequence != stream.dependency_seq_id ||
                placement.computation_frontier !=
                    stream.computation_frontier ||
                placement.cells.size() !=
                    stream.captured_dependency_count) {
                return false;
            }
            if (!checked_add(total_cells, placement.cells.size(), total_cells) ||
                (limits && total_cells > limits->max_placement_cells)) {
                return false;
            }

            const auto expected_cells =
                vbr_generation_production_covered_set(stream);
            if (expected_cells.size() != placement.cells.size()) {
                return false;
            }
            auto & source_positions =
                logical_positions[{
                    placement.child_id, placement.source_sequence,
                }];
            for (size_t i = 0; i < placement.cells.size(); ++i) {
                const auto & cell = placement.cells[i];
                if (cell.physical_cell != expected_cells[i] ||
                    cell.logical_position < 0 ||
                    cell.logical_position >= placement.computation_frontier ||
                    cell.logical_position >= manifest.identity.token_count ||
                    (i != 0 && placement.cells[i - 1].physical_cell >=
                                   cell.physical_cell) ||
                    !source_positions.insert(cell.logical_position).second) {
                    return false;
                }
            }
        }
    }
    return next == manifest.stream_placements.size();
}

bool controller_policy_valid(
        const vbr_artifact_controller_policy & policy,
        uint32_t expected_child) {
    return policy.child_id == expected_child &&
           uint32_t(policy.dependency_mode) <=
               uint32_t(checkpoint_child_dependency_mode::live_guarded) &&
           digest_nonzero(policy.degrade_order_digest) &&
           digest_nonzero(policy.policy_digest) &&
           policy.floor_type >= 0 &&
           policy.n_stream > 0 &&
           policy.wm_cells > 0 &&
           digest_nonzero(policy.current_type_vector_digest) &&
           policy.completed_wave;
}

bool stash_reference_valid(
        const vbr_artifact_stash_reference & reference) {
    if (reference.valid_rows == 0 ||
        reference.valid_rows > reference.row_count ||
        reference.domain > vbr_repr_domain::tapped ||
        reference.row_count == 0 ||
        reference.column_count == 0 ||
        reference.row_bytes == 0 ||
        reference.captured_sink_count == 0 ||
        reference.covered_sink_pages.empty() ||
        !reference.payload_id.valid()) {
        return false;
    }
    uint64_t cardinality = 0;
    uint32_t prior_page = 0;
    bool first = true;
    for (const auto & page : reference.covered_sink_pages) {
        if ((!first && page.page_index <= prior_page) ||
            page.captured_page_gen == 0) {
            return false;
        }
        first = false;
        prior_page = page.page_index;
        for (uint64_t word : page.covered_mask) {
            cardinality += llama_popcount_u64(word);
        }
    }
    return cardinality == reference.captured_sink_count;
}

bool accounting_payloads_match(const vbr_artifact_package & package) {
    struct expected_row {
        vbr_artifact_accounting_role role;
        vbr_artifact_portable_domain domain;
        uint64_t bytes;
    };
    std::vector<expected_row> expected;
    auto add = [&](vbr_artifact_accounting_role role,
                   const vbr_artifact_portable_domain & domain,
                   uint64_t bytes) {
        for (auto & row : expected) {
            if (row.role == role && row.domain == domain) {
                return checked_add(row.bytes, bytes, row.bytes);
            }
        }
        expected.push_back({ role, domain, bytes });
        return true;
    };
    try {
        for (const auto & blob : package.unit_blobs) {
            for (const auto & shard : blob.descriptor.shards) {
                const vbr_artifact_portable_domain domain {
                    llama_cache_acct_residency::device,
                    llama_cache_acct_domain_kind::device_topology,
                    shard.topology_index,
                    shard.device_ordinal,
                };
                if (!add(vbr_artifact_accounting_role::unit_payload,
                         domain, shard.payload_bytes)) {
                    return false;
                }
            }
            if (blob.descriptor.clean_stash_state ==
                vbr_artifact_clean_stash_state::present) {
                for (const auto & shard :
                     blob.descriptor.clean_stash.shards) {
                    const vbr_artifact_portable_domain domain {
                        llama_cache_acct_residency::device,
                        llama_cache_acct_domain_kind::device_topology,
                        shard.topology_index,
                        shard.device_ordinal,
                    };
                    if (!add(
                            vbr_artifact_accounting_role::clean_stash_payload,
                            domain, shard.payload_bytes)) {
                        return false;
                    }
                }
            }
        }
        for (const auto & companion : package.companions) {
            const auto role =
                companion.kind == vbr_artifact_companion_kind::recurrent ?
                    vbr_artifact_accounting_role::recurrent_payload :
                    vbr_artifact_accounting_role::typed_accelerator_payload;
            if (!add(role, companion.domain, companion.payload_bytes)) {
                return false;
            }
        }
    } catch (...) {
        return false;
    }

    bool descriptor_row = false;
    bool reference_row = false;
    for (const auto & row : package.manifest.accounting) {
        descriptor_row |=
            row.role == vbr_artifact_accounting_role::descriptor_metadata;
        reference_row |=
            row.role == vbr_artifact_accounting_role::reference_metadata;
    }
    if (!descriptor_row || !reference_row) {
        return false;
    }
    for (const auto & required : expected) {
        const auto found = std::find_if(
            package.manifest.accounting.begin(),
            package.manifest.accounting.end(),
            [&](const vbr_artifact_portable_accounting_row & row) {
                return row.role == required.role &&
                       row.domain == required.domain;
            });
        if (found == package.manifest.accounting.end() ||
            found->logical_bytes != required.bytes ||
            found->resident_bytes != required.bytes) {
            return false;
        }
    }
    return true;
}

bool manifest_valid(
        const vbr_artifact_package & package,
        const vbr_artifact_decode_limits * limits,
        bool require_sources) {
    const auto & manifest = package.manifest;
    if (manifest.version != package.version ||
        !artifact_version_supported(manifest.version) ||
        !digest_nonzero(manifest.identity_policy_order_digest) ||
        manifest.identity_policy_order_digest !=
            manifest.generation.identity_policy_order_digest ||
        !identity_valid(manifest.identity, limits) ||
        !validate_generation_record(manifest.generation, limits) ||
        !manifest.capture_generation_id.valid() ||
        !consistency_valid(manifest.consistency,
                           manifest.capture_generation_id) ||
        manifest.controller_policy.size() !=
            manifest.generation.controllers.size() ||
        manifest.unit_references.empty() ||
        manifest.unit_references.size() != package.unit_blobs.size() ||
        !manifest.manifest_digest.valid() ||
        (limits && (manifest.unit_references.size() >
                        limits->max_unit_blobs ||
                    manifest.companions.size() >
                        limits->max_companions ||
                    manifest.accounting.size() >
                        limits->max_accounting_rows))) {
        return false;
    }
    if (artifact_has_reference_placement(manifest.version)) {
        if (!token_block_valid(
                manifest.identity,
                manifest.identity_policy_order_digest,
                manifest.token_block, limits) ||
            !placement_valid(manifest, limits)) {
            return false;
        }
    } else if (!reference_metadata_absent(manifest)) {
        return false;
    }
    for (uint32_t i = 0; i < manifest.controller_policy.size(); ++i) {
        if (!controller_policy_valid(manifest.controller_policy[i], i) ||
            manifest.controller_policy[i].dependency_mode !=
                manifest.generation.controllers[i].dependency_mode) {
            return false;
        }
    }
    std::map<std::array<uint8_t, 32>,
             const vbr_artifact_unit_blob *> blob_by_id;
    for (const auto & blob : package.unit_blobs) {
        if (!blob_by_id.emplace(
                blob.unit_version_id.bytes(), &blob).second) {
            return false;
        }
    }
    using reference_key =
        std::tuple<uint64_t, uint64_t, uint32_t, uint64_t>;
    std::set<reference_key> reference_generations;
    for (const auto & reference : manifest.unit_references) {
        if (!vbr_lineage_uuid_is_set(reference.lineage_uuid) ||
            reference.repr_gen == 0 ||
            !reference.unit_version_id.valid() ||
            !reference.payload_digest.valid() ||
            reference.authorized_stream_refs.empty() ||
            !reference_generations.emplace(
                reference.lineage_uuid.hi, reference.lineage_uuid.lo,
                reference.logical_unit_id, reference.repr_gen).second) {
            return false;
        }
        const auto found_blob =
            blob_by_id.find(reference.unit_version_id.bytes());
        const auto * blob = found_blob == blob_by_id.end()
            ? nullptr : found_blob->second;
        if (!blob ||
            blob->payload_digest != reference.payload_digest ||
            blob->descriptor.lineage_uuid != reference.lineage_uuid ||
            blob->descriptor.logical_unit_id != reference.logical_unit_id ||
            blob->descriptor.repr_gen != reference.repr_gen) {
            return false;
        }
        if (blob->descriptor.clean_stash_state ==
                vbr_artifact_clean_stash_state::omitted_source_present &&
            manifest.consistency.kind !=
                vbr_artifact_consistency_kind::live_rebased) {
            return false;
        }
        if (blob->descriptor.child_id >=
                manifest.generation.controllers.size() ||
            blob->descriptor.child_id >=
                manifest.controller_policy.size() ||
            manifest.generation.controllers[
                blob->descriptor.child_id].child_id !=
                    blob->descriptor.child_id) {
            return false;
        }
        const auto & child = manifest.generation.controllers[
            blob->descriptor.child_id];
        if (reference.logical_unit_id >= child.units.size() ||
            manifest.controller_policy[blob->descriptor.child_id].n_stream !=
                blob->descriptor.n_stream ||
            manifest.controller_policy[blob->descriptor.child_id].unified !=
                blob->descriptor.unified ||
            manifest.controller_policy[blob->descriptor.child_id].wm_cells !=
                blob->descriptor.wm_cells ||
            child.units[reference.logical_unit_id].repr_gen !=
                reference.repr_gen ||
            child.units[reference.logical_unit_id].current_type !=
                blob->descriptor.current_type ||
            child.units[reference.logical_unit_id].last_source_type !=
                blob->descriptor.last_source_type ||
            child.units[reference.logical_unit_id].domain !=
                (blob->descriptor.current_type == GGML_TYPE_F16 ||
                 blob->descriptor.current_type == GGML_TYPE_TURBO8_0 ?
                     vbr_repr_domain::full : vbr_repr_domain::tapped) ||
            child.units[reference.logical_unit_id].promote_hops !=
                blob->descriptor.promote_hops ||
            child.units[reference.logical_unit_id].last_transition !=
                blob->descriptor.last_transition) {
            return false;
        }
        std::set<uint32_t> streams;
        for (uint32_t stream : reference.authorized_stream_refs) {
            if (stream >= child.streams.size() ||
                !streams.insert(stream).second) {
                return false;
            }
        }
        if (reference.has_stash_reference) {
            if (blob->descriptor.clean_stash_state !=
                    vbr_artifact_clean_stash_state::present ||
                !stash_reference_valid(reference.stash_reference) ||
                reference.stash_reference.payload_id !=
                    blob->descriptor.clean_stash.payload_id ||
                reference.stash_reference.valid_rows !=
                    blob->descriptor.clean_stash.valid_rows ||
                reference.stash_reference.domain !=
                    blob->descriptor.clean_stash.domain ||
                reference.stash_reference.row_count !=
                    blob->descriptor.clean_stash.row_count ||
                reference.stash_reference.column_count !=
                    blob->descriptor.clean_stash.column_count ||
                reference.stash_reference.row_bytes !=
                    blob->descriptor.clean_stash.row_bytes) {
                return false;
            }
            for (const auto & stash_page :
                 reference.stash_reference.covered_sink_pages) {
                bool authorized = false;
                for (uint32_t stream_index :
                     reference.authorized_stream_refs) {
                    const auto & stream = child.streams[stream_index];
                    const auto page = std::find_if(
                        stream.pages.begin(), stream.pages.end(),
                        [&](const vbr_generation_page_ref & candidate) {
                            return candidate.page_index ==
                                       stash_page.page_index &&
                                   candidate.captured_page_gen ==
                                       stash_page.captured_page_gen;
                        });
                    if (page == stream.pages.end()) {
                        continue;
                    }
                    authorized = true;
                    for (size_t word = 0;
                         word < stash_page.covered_mask.size(); ++word) {
                        if ((stash_page.covered_mask[word] &
                             ~page->covered_mask[word]) != 0) {
                            authorized = false;
                            break;
                        }
                    }
                    if (authorized) {
                        break;
                    }
                }
                if (!authorized) {
                    return false;
                }
            }
        } else if (blob->descriptor.clean_stash_state ==
                   vbr_artifact_clean_stash_state::present) {
            return false;
        }
    }
    for (const auto & controller : manifest.generation.controllers) {
        if (controller.dependency_mode !=
            checkpoint_child_dependency_mode::live_guarded) {
            continue;
        }
        for (uint32_t unit = 0; unit < controller.units.size(); ++unit) {
            if (reference_generations.find(reference_key {
                    controller.lineage_uuid.hi,
                    controller.lineage_uuid.lo,
                    unit, controller.units[unit].repr_gen,
                }) == reference_generations.end()) {
                return false;
            }
        }
    }
    if (manifest.companions.size() != package.companions.size()) {
        return false;
    }
    for (size_t i = 0; i < manifest.companions.size(); ++i) {
        const auto & ref = manifest.companions[i];
        const auto & payload = package.companions[i];
        if (ref.kind != payload.kind ||
            ref.format_version != payload.format_version ||
            ref.build_identity_digest != payload.build_identity_digest ||
            ref.domain != payload.domain ||
            ref.payload_digest != payload.payload_digest ||
            ref.payload_bytes != payload.payload_bytes ||
            ref.section_checksum != payload.section_checksum ||
            (require_sources &&
             (!payload.payload.valid() ||
              payload.payload.size != payload.payload_bytes))) {
            return false;
        }
    }
    return vbr_artifact_validate_portable_accounting(
               package.topologies, manifest.accounting) &&
           accounting_payloads_match(package);
}

bool package_metadata_valid(
        const vbr_artifact_package & package,
        const vbr_artifact_decode_limits * limits,
        bool require_sources,
        bool allow_sparse_rows = false) {
    if (!artifact_version_supported(package.version) ||
        package.flags != ARTIFACT_FLAGS_V1 ||
        package.topologies.empty() ||
        package.unit_blobs.empty() ||
        (limits && (package.topologies.size() > limits->max_topologies ||
                    package.unit_blobs.size() > limits->max_unit_blobs ||
                    package.companions.size() > limits->max_companions))) {
        return false;
    }
    for (const auto & topology : package.topologies) {
        if (!topology_valid(topology) ||
            (limits && topology.device_identities.size() >
                           limits->max_devices_per_topology)) {
            return false;
        }
    }
    std::set<std::array<uint8_t, 32>> ids;
    for (const auto & blob : package.unit_blobs) {
        if (!blob.unit_version_id.valid() ||
            !blob.payload_digest.valid() ||
            !ids.insert(blob.unit_version_id.bytes()).second ||
            !descriptor_metadata_valid(
                blob.descriptor, package.topologies, package.version,
                require_sources,
                allow_sparse_rows) ||
            (limits &&
             (blob.descriptor.shards.size() >
                  limits->max_shards_per_unit ||
              blob.descriptor.clean_stash.shards.size() >
                  limits->max_shards_per_unit))) {
            return false;
        }
    }
    return manifest_valid(package, limits, require_sources);
}

bool prepared_identity_equal(
        const vbr_artifact_package & expected,
        const vbr_artifact_package & canonical) {
    if (expected.version != canonical.version ||
        expected.flags != canonical.flags ||
        expected.topologies.size() != canonical.topologies.size() ||
        expected.unit_blobs.size() != canonical.unit_blobs.size() ||
        expected.companions.size() != canonical.companions.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.topologies.size(); ++i) {
        if (expected.topologies[i].digest != canonical.topologies[i].digest) {
            return false;
        }
    }
    for (size_t i = 0; i < expected.unit_blobs.size(); ++i) {
        const auto & lhs = expected.unit_blobs[i];
        const auto & rhs = canonical.unit_blobs[i];
        if (lhs.unit_version_id != rhs.unit_version_id ||
            lhs.payload_digest != rhs.payload_digest ||
            lhs.descriptor.clean_stash.payload_id !=
                rhs.descriptor.clean_stash.payload_id ||
            lhs.descriptor.shards.size() != rhs.descriptor.shards.size() ||
            lhs.descriptor.clean_stash.shards.size() !=
                rhs.descriptor.clean_stash.shards.size()) {
            return false;
        }
        for (size_t j = 0; j < lhs.descriptor.shards.size(); ++j) {
            if (lhs.descriptor.shards[j].section_checksum !=
                    rhs.descriptor.shards[j].section_checksum) {
                return false;
            }
        }
        for (size_t j = 0;
             j < lhs.descriptor.clean_stash.shards.size(); ++j) {
            if (lhs.descriptor.clean_stash.shards[j].section_checksum !=
                    rhs.descriptor.clean_stash.shards[j].section_checksum) {
                return false;
            }
        }
    }
    for (size_t i = 0; i < expected.companions.size(); ++i) {
        if (expected.companions[i].payload_digest !=
                canonical.companions[i].payload_digest ||
            expected.companions[i].section_checksum !=
                canonical.companions[i].section_checksum) {
            return false;
        }
    }
    const auto & lhs = expected.manifest;
    const auto & rhs = canonical.manifest;
    if (lhs.capture_generation_id != rhs.capture_generation_id ||
        lhs.token_block.digest != rhs.token_block.digest ||
        lhs.manifest_digest != rhs.manifest_digest ||
        lhs.unit_references.size() != rhs.unit_references.size() ||
        lhs.companions.size() != rhs.companions.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.unit_references.size(); ++i) {
        const auto & a = lhs.unit_references[i];
        const auto & b = rhs.unit_references[i];
        if (a.unit_version_id != b.unit_version_id ||
            a.payload_digest != b.payload_digest ||
            a.stash_reference.payload_id != b.stash_reference.payload_id) {
            return false;
        }
    }
    for (size_t i = 0; i < lhs.companions.size(); ++i) {
        if (lhs.companions[i].payload_digest !=
                rhs.companions[i].payload_digest ||
            lhs.companions[i].section_checksum !=
                rhs.companions[i].section_checksum) {
            return false;
        }
    }
    return true;
}

bool prepare_companion(
        uint32_t index,
        const std::vector<vbr_artifact_portable_topology> & topologies,
        vbr_artifact_companion_payload & companion) {
    if (companion.kind >= vbr_artifact_companion_kind::_count ||
        companion.format_version == 0 ||
        !digest_nonzero(companion.build_identity_digest) ||
        !domain_valid(companion.domain, topologies) ||
        companion.payload_bytes == 0 ||
        !companion.payload.valid() ||
        companion.payload.size != companion.payload_bytes) {
        return false;
    }
    if (!digest_matches_source(
            DOMAIN_COMPANION, index, 0, companion.payload,
            companion.section_checksum)) {
        return false;
    }
    llama_sha256_writer hash;
    hash.string(DOMAIN_COMPANION, sizeof(DOMAIN_COMPANION) - 1);
    hash.u32(index);
    hash.u32(uint32_t(companion.kind));
    hash.u32(companion.format_version);
    hash.string(companion.build_identity_digest.data(),
                companion.build_identity_digest.size());
    emitter domain_hash;
    domain_hash.hash_a = &hash;
    if (!emit_portable_domain(domain_hash, companion.domain)) {
        return false;
    }
    if (!hash_source_string(hash, companion.payload)) {
        return false;
    }
    companion.payload_digest = typed_digest<vbr_payload_digest>(hash);
    return companion.payload_digest.valid();
}

bool emit_topology(emitter & out, const vbr_artifact_portable_topology & topology) {
    if (!out.u32(topology.version) ||
        !out.u32(topology.device_count) ||
        !out.i32(topology.split_mode) ||
        !out.u32(topology.main_device.v)) {
        return false;
    }
    for (size_t i = 0; i < topology.device_identities.size(); ++i) {
        if (!out.raw(topology.device_identities[i].bytes().data(), 32) ||
            !out.u32(topology.shard_weights[i])) {
            return false;
        }
    }
    return out.raw(topology.digest.bytes().data(), 32);
}

bool emit_topology_section(emitter & out, const vbr_artifact_package & package) {
    if (!out.u32(uint32_t(package.topologies.size()))) {
        return false;
    }
    for (const auto & topology : package.topologies) {
        if (!emit_topology(out, topology)) {
            return false;
        }
    }
    return true;
}

bool emit_unit_section(
        emitter & out,
        const vbr_artifact_unit_blob & blob,
        uint32_t format_version) {
    const auto & descriptor = blob.descriptor;
    if (!out.digest(blob.unit_version_id) ||
        !out.digest(blob.payload_digest) ||
        !emit_lineage(out, descriptor.lineage_uuid) ||
        !out.u32(descriptor.logical_unit_id) ||
        !out.u64(descriptor.repr_gen) ||
        !emit_unit_descriptor_body(out, descriptor, format_version) ||
        !out.u32(uint32_t(descriptor.clean_stash_state)) ||
        (descriptor.clean_stash_state ==
             vbr_artifact_clean_stash_state::present &&
         !emit_clean_stash_descriptor(out, descriptor.clean_stash))) {
        return false;
    }
    for (const auto & shard : descriptor.shards) {
        if (!out.u32(shard.shard_index) ||
            !stream_payload_chunks(out, shard.payload)) {
            return false;
        }
    }
    if (descriptor.clean_stash_state ==
        vbr_artifact_clean_stash_state::present) {
        for (const auto & shard : descriptor.clean_stash.shards) {
            if (!out.u32(shard.shard_index) ||
                !stream_payload_chunks(out, shard.payload)) {
                return false;
            }
        }
    }
    return true;
}

bool emit_companion_section(
        emitter & out,
        const vbr_artifact_companion_payload & companion) {
    return emit_companion_reference(out, companion) &&
           stream_payload_chunks(out, companion.payload);
}

bool emit_manifest_section(
        emitter & out,
        const vbr_artifact_reference_manifest & manifest) {
    return emit_manifest_body(out, manifest) &&
           out.digest(manifest.manifest_digest);
}

bool emit_unit_section_verified(
        emitter & out,
        const vbr_artifact_unit_blob & blob,
        uint32_t object_index,
        uint32_t format_version) {
    const auto & descriptor = blob.descriptor;
    if (!out.digest(blob.unit_version_id) ||
        !out.digest(blob.payload_digest) ||
        !emit_lineage(out, descriptor.lineage_uuid) ||
        !out.u32(descriptor.logical_unit_id) ||
        !out.u64(descriptor.repr_gen) ||
        !emit_unit_descriptor_body(out, descriptor, format_version) ||
        !out.u32(uint32_t(descriptor.clean_stash_state)) ||
        (descriptor.clean_stash_state ==
             vbr_artifact_clean_stash_state::present &&
         !emit_clean_stash_descriptor(out, descriptor.clean_stash))) {
        return false;
    }

    llama_sha256_writer unit_hash;
    unit_hash.string(DOMAIN_UNIT, sizeof(DOMAIN_UNIT) - 1);
    unit_hash.u32(format_version);
    if (!hash_lineage(unit_hash, descriptor.lineage_uuid)) {
        return false;
    }
    unit_hash.u32(descriptor.logical_unit_id);
    unit_hash.u64(descriptor.repr_gen);
    if (!hash_sized_descriptor(unit_hash, descriptor, format_version)) {
        return false;
    }
    unit_hash.u32(uint32_t(descriptor.shards.size()));

    llama_sha256_writer payload_hash;
    payload_hash.string(DOMAIN_PAYLOAD, sizeof(DOMAIN_PAYLOAD) - 1);
    payload_hash.u32(uint32_t(descriptor.shards.size()));
    for (const auto & shard : descriptor.shards) {
        emitter descriptor_hash;
        descriptor_hash.hash_a = &payload_hash;
        if (!emit_shard_descriptor(descriptor_hash, shard) ||
            !out.u32(shard.shard_index)) {
            return false;
        }
        unit_hash.u32(shard.shard_index);
        unit_hash.u64(shard.payload.size);
        payload_hash.u64(shard.payload.size);

        llama_sha256_writer shard_hash;
        shard_hash.string(DOMAIN_SHARD, sizeof(DOMAIN_SHARD) - 1);
        shard_hash.u32(object_index);
        shard_hash.u32(shard.shard_index);
        shard_hash.u64(shard.payload.size);
        if (!stream_payload_chunks(
                out, shard.payload, &unit_hash, &payload_hash,
                &shard_hash) ||
            shard_hash.finish() != shard.section_checksum) {
            return false;
        }
    }

    unit_hash.u32(uint32_t(descriptor.clean_stash_state));
    if (descriptor.clean_stash_state ==
        vbr_artifact_clean_stash_state::present) {
        const auto & stash = descriptor.clean_stash;
        if (!hash_sized_stash_descriptor(unit_hash, stash)) {
            return false;
        }
        unit_hash.u32(uint32_t(stash.shards.size()));

        llama_sha256_writer stash_hash;
        stash_hash.string(DOMAIN_STASH, sizeof(DOMAIN_STASH) - 1);
        stash_hash.u32(uint32_t(stash.layout));
        stash_hash.u64(stash.row_count);
        stash_hash.u64(stash.column_count);
        stash_hash.u64(stash.row_bytes);
        stash_hash.u64(stash.valid_rows);
        stash_hash.u32(uint32_t(stash.shards.size()));
        for (const auto & shard : stash.shards) {
            emitter descriptor_hash;
            descriptor_hash.hash_a = &stash_hash;
            if (!emit_shard_descriptor(descriptor_hash, shard) ||
                !out.u32(shard.shard_index)) {
                return false;
            }
            unit_hash.u32(shard.shard_index);
            unit_hash.u64(shard.payload.size);
            stash_hash.u64(shard.payload.size);

            llama_sha256_writer shard_hash;
            shard_hash.string(DOMAIN_SHARD, sizeof(DOMAIN_SHARD) - 1);
            shard_hash.u32(object_index);
            shard_hash.u32(shard.shard_index);
            shard_hash.u64(shard.payload.size);
            if (!stream_payload_chunks(
                    out, shard.payload, &unit_hash, &stash_hash,
                    &shard_hash) ||
                shard_hash.finish() != shard.section_checksum) {
                return false;
            }
        }
        if (typed_digest<vbr_stash_payload_id>(stash_hash) !=
            stash.payload_id) {
            return false;
        }
    }

    return typed_digest<vbr_payload_digest>(payload_hash) ==
               blob.payload_digest &&
           typed_digest<vbr_unit_version_id>(unit_hash) ==
               blob.unit_version_id;
}

bool emit_companion_section_verified(
        emitter & out,
        const vbr_artifact_companion_payload & companion,
        uint32_t object_index) {
    if (!emit_companion_reference(out, companion)) {
        return false;
    }
    llama_sha256_writer payload_hash;
    payload_hash.string(DOMAIN_COMPANION, sizeof(DOMAIN_COMPANION) - 1);
    payload_hash.u32(object_index);
    payload_hash.u32(uint32_t(companion.kind));
    payload_hash.u32(companion.format_version);
    payload_hash.string(
        companion.build_identity_digest.data(),
        companion.build_identity_digest.size());
    emitter domain_hash;
    domain_hash.hash_a = &payload_hash;
    if (!emit_portable_domain(domain_hash, companion.domain)) {
        return false;
    }
    payload_hash.u64(companion.payload.size);

    llama_sha256_writer section_hash;
    section_hash.string(
        DOMAIN_COMPANION, sizeof(DOMAIN_COMPANION) - 1);
    section_hash.u32(object_index);
    section_hash.u32(0);
    section_hash.u64(companion.payload.size);
    return stream_payload_chunks(
               out, companion.payload, &payload_hash,
               &section_hash) &&
           section_hash.finish() == companion.section_checksum &&
           typed_digest<vbr_payload_digest>(payload_hash) ==
               companion.payload_digest;
}

struct section_descriptor {
    vbr_artifact_section_kind kind =
        vbr_artifact_section_kind::topology_table;
    uint32_t index = 0;
    uint64_t size = 0;
    std::array<uint8_t, 32> checksum = {};
};

std::vector<section_descriptor> section_inventory(
        const vbr_artifact_package & package) {
    std::vector<section_descriptor> result;
    result.reserve(2 + package.unit_blobs.size() + package.companions.size());
    result.push_back({ vbr_artifact_section_kind::topology_table, 0 });
    for (uint32_t i = 0; i < package.unit_blobs.size(); ++i) {
        result.push_back({ vbr_artifact_section_kind::unit_blob, i });
    }
    for (uint32_t i = 0; i < package.companions.size(); ++i) {
        result.push_back({ vbr_artifact_section_kind::companion_payload, i });
    }
    result.push_back({ vbr_artifact_section_kind::reference_manifest, 0 });
    return result;
}

bool emit_section_body(
        emitter & out,
        const vbr_artifact_package & package,
        const section_descriptor & section) {
    switch (section.kind) {
        case vbr_artifact_section_kind::topology_table:
            return section.index == 0 &&
                   emit_topology_section(out, package);
        case vbr_artifact_section_kind::unit_blob:
            return section.index < package.unit_blobs.size() &&
                   emit_unit_section(
                       out, package.unit_blobs[section.index],
                       package.version);
        case vbr_artifact_section_kind::companion_payload:
            return section.index < package.companions.size() &&
                   emit_companion_section(
                       out, package.companions[section.index]);
        case vbr_artifact_section_kind::reference_manifest:
            return section.index == 0 &&
                   emit_manifest_section(out, package.manifest);
        case vbr_artifact_section_kind::_count:
            break;
    }
    return false;
}

bool emit_section_body_verified(
        emitter & out,
        const vbr_artifact_package & package,
        const section_descriptor & section) {
    switch (section.kind) {
        case vbr_artifact_section_kind::unit_blob:
            return section.index < package.unit_blobs.size() &&
                   emit_unit_section_verified(
                       out, package.unit_blobs[section.index],
                       section.index, package.version);
        case vbr_artifact_section_kind::companion_payload:
            return section.index < package.companions.size() &&
                   emit_companion_section_verified(
                       out, package.companions[section.index],
                       section.index);
        case vbr_artifact_section_kind::topology_table:
        case vbr_artifact_section_kind::reference_manifest:
            return emit_section_body(out, package, section);
        case vbr_artifact_section_kind::_count:
            break;
    }
    return false;
}

bool prepare_sections(
        const vbr_artifact_package & package,
        std::vector<section_descriptor> & sections) {
    sections = section_inventory(package);
    for (auto & section : sections) {
        llama_sha256_writer hash;
        hash.string(DOMAIN_SECTION, sizeof(DOMAIN_SECTION) - 1);
        hash.u32(uint32_t(section.kind));
        hash.u32(section.index);
        emitter body;
        body.hash_a = &hash;
        if (!emit_section_body(body, package, section) || !body.ok) {
            return false;
        }
        section.size = body.count;
        section.checksum = hash.finish();
        if (!digest_nonzero(section.checksum)) {
            return false;
        }
    }
    return true;
}

std::array<uint8_t, 32> ordering_digest(
        const std::vector<section_descriptor> & sections) {
    llama_sha256_writer hash;
    hash.string(DOMAIN_ORDERING, sizeof(DOMAIN_ORDERING) - 1);
    hash.u32(uint32_t(sections.size()));
    for (const auto & section : sections) {
        hash.u32(uint32_t(section.kind));
        hash.u32(section.index);
        hash.u64(section.size);
        hash.string(section.checksum.data(), section.checksum.size());
    }
    return hash.finish();
}

bool compute_total_size(
        const std::vector<section_descriptor> & sections,
        uint64_t & total) {
    total = ARTIFACT_HEADER_SIZE;
    for (const auto & section : sections) {
        uint64_t with_header;
        if (!checked_add(ARTIFACT_SECTION_HEADER_SIZE, section.size,
                         with_header) ||
            !checked_add(total, with_header, total)) {
            return false;
        }
    }
    return true;
}

bool hash_package(
        uint32_t version,
        uint32_t flags,
        const std::vector<section_descriptor> & sections,
        uint64_t total_size,
        const std::array<uint8_t, 32> & order,
        std::array<uint8_t, 32> & result) {
    llama_sha256_writer hash;
    hash.string(DOMAIN_PACKAGE, sizeof(DOMAIN_PACKAGE) - 1);
    hash.u32(ARTIFACT_MAGIC);
    hash.u32(version);
    hash.u32(ARTIFACT_HEADER_SIZE);
    hash.u32(flags);
    hash.u64(total_size);
    hash.u32(uint32_t(sections.size()));
    hash.string(order.data(), order.size());
    for (const auto & section : sections) {
        hash.u32(uint32_t(section.kind));
        hash.u32(section.index);
        hash.u64(section.size);
        hash.string(section.checksum.data(), section.checksum.size());
    }
    result = hash.finish();
    return digest_nonzero(result);
}

bool emit_header(
        emitter & out,
        const vbr_artifact_package & package,
        uint64_t total_size,
        uint32_t section_count,
        const std::array<uint8_t, 32> & order,
        const std::array<uint8_t, 32> & checksum) {
    return out.u32(ARTIFACT_MAGIC) &&
           out.u32(package.version) &&
           out.u32(ARTIFACT_HEADER_SIZE) &&
           out.u32(package.flags) &&
           out.u64(total_size) &&
           out.u32(section_count) &&
           out.array_digest(order) &&
           out.array_digest(checksum);
}

bool emit_section_header(emitter & out, const section_descriptor & section) {
    return out.u32(uint32_t(section.kind)) &&
           out.u32(section.index) &&
           out.u64(section.size) &&
           out.array_digest(section.checksum);
}

struct input_cursor {
    const vbr_artifact_stream_reader * input = nullptr;
    uint64_t remaining = 0;

    bool raw(void * data, size_t size) {
        if (!input || !input->read || size > remaining ||
            !input->read(input->context, static_cast<uint8_t *>(data), size)) {
            return false;
        }
        remaining -= size;
        return true;
    }

    bool u32(uint32_t & value) {
        uint8_t data[4];
        if (!raw(data, sizeof(data))) {
            return false;
        }
        value = llama_load_le_u32(data);
        return true;
    }

    bool u64(uint64_t & value) {
        uint8_t data[8];
        if (!raw(data, sizeof(data))) {
            return false;
        }
        value = llama_load_le_u64(data);
        return true;
    }
};

struct bounded_reader {
    input_cursor * input = nullptr;
    uint64_t remaining = 0;
    llama_sha256_writer * package_hash = nullptr;
    llama_sha256_writer * section_hash = nullptr;
    llama_sha256_writer * auxiliary_hash = nullptr;

    bool declared_count_fits(
            uint64_t count,
            uint64_t min_wire_element_size) const noexcept {
        return min_wire_element_size != 0 &&
               count <= remaining / min_wire_element_size;
    }

    bool raw(void * data, size_t size) {
        if (!input || size > remaining || !input->raw(data, size)) {
            return false;
        }
        if (package_hash) {
            package_hash->bytes(data, size);
        }
        if (section_hash) {
            section_hash->bytes(data, size);
        }
        if (auxiliary_hash) {
            auxiliary_hash->bytes(data, size);
        }
        remaining -= size;
        return true;
    }

    bool u32(uint32_t & value) {
        uint8_t data[4];
        if (!raw(data, sizeof(data))) {
            return false;
        }
        value = llama_load_le_u32(data);
        return true;
    }

    bool i32(int32_t & value) {
        uint32_t raw_value;
        if (!u32(raw_value)) {
            return false;
        }
        value = int32_t(raw_value);
        return true;
    }

    bool u64(uint64_t & value) {
        uint8_t data[8];
        if (!raw(data, sizeof(data))) {
            return false;
        }
        value = llama_load_le_u64(data);
        return true;
    }

    bool i64(int64_t & value) {
        uint64_t raw_value;
        if (!u64(raw_value)) {
            return false;
        }
        value = int64_t(raw_value);
        return true;
    }

    bool fixed_digest(std::array<uint8_t, 32> & value) {
        return raw(value.data(), value.size());
    }

    template <typename Digest>
    bool typed_digest_value(Digest & value) {
        std::array<uint8_t, 32> bytes;
        if (!fixed_digest(bytes)) {
            return false;
        }
        value = Digest::from_sha256(std::move(bytes));
        return true;
    }

    bool string(std::string & value, const vbr_artifact_decode_limits & limits) {
        uint64_t size;
        if (!u64(size) || size > limits.max_string_bytes ||
            size > std::numeric_limits<size_t>::max() ||
            size > remaining) {
            return false;
        }
        value.resize(size_t(size));
        return size == 0 || raw(value.data(), size_t(size));
    }
};

bool read_lineage(bounded_reader & in, vbr_lineage_uuid & lineage) {
    return in.u64(lineage.hi) && in.u64(lineage.lo);
}

bool read_portable_domain(
        bounded_reader & in,
        vbr_artifact_portable_domain & domain) {
    uint32_t residency;
    uint32_t kind;
    uint32_t ordinal;
    if (!in.u32(residency) ||
        !in.u32(kind) ||
        !in.u32(domain.topology_index) ||
        !in.u32(ordinal) ||
        residency >= uint32_t(llama_cache_acct_residency::_count) ||
        kind >= uint32_t(llama_cache_acct_domain_kind::_count) ||
        ordinal > UINT16_MAX) {
        return false;
    }
    domain.residency = llama_cache_acct_residency(residency);
    domain.kind = llama_cache_acct_domain_kind(kind);
    domain.device_ordinal = uint16_t(ordinal);
    return true;
}

bool read_page_ref(bounded_reader & in, vbr_generation_page_ref & page) {
    if (!in.u32(page.page_index) ||
        !in.u32(page.captured_page_gen)) {
        return false;
    }
    for (uint64_t & word : page.covered_mask) {
        if (!in.u64(word)) {
            return false;
        }
    }
    return true;
}

bool read_unit_generation(
        bounded_reader & in,
        vbr_checkpoint_unit_generation & unit) {
    uint32_t domain;
    uint32_t promote_hops;
    uint32_t transition;
    if (!in.u64(unit.repr_gen) ||
        !in.i32(unit.current_type) ||
        !in.i32(unit.last_source_type) ||
        !in.u32(domain) ||
        !in.u32(promote_hops) ||
        !in.u32(transition) ||
        domain > uint32_t(vbr_repr_domain::tapped) ||
        promote_hops > UINT8_MAX ||
        transition > uint32_t(vbr_repr_transition::recovery_invalidate)) {
        return false;
    }
    unit.domain = vbr_repr_domain(domain);
    unit.promote_hops = uint8_t(promote_hops);
    unit.last_transition = vbr_repr_transition(transition);
    return true;
}

bool read_generation_record(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_checkpoint_generation_record & generation) {
    uint32_t status;
    uint32_t n_controllers;
    if (!in.u32(generation.version) ||
        !in.u32(status) ||
        !in.fixed_digest(generation.identity_policy_order_digest) ||
        !in.u32(n_controllers) ||
        status > uint32_t(vbr_checkpoint_generation_status::generation_unknown) ||
        n_controllers > limits.max_controllers ||
        !in.declared_count_fits(
            n_controllers, MIN_WIRE_CONTROLLER)) {
        return false;
    }
    generation.status = vbr_checkpoint_generation_status(status);
    generation.controllers.resize(n_controllers);
    for (auto & controller : generation.controllers) {
        uint32_t dependency_mode;
        uint32_t n_units;
        uint32_t n_streams;
        if (!in.u32(controller.child_id) ||
            !in.u32(dependency_mode) ||
            !read_lineage(in, controller.lineage_uuid) ||
            !in.u64(controller.global_generation) ||
            !in.u32(n_units) ||
            dependency_mode >
                uint32_t(checkpoint_child_dependency_mode::live_guarded) ||
            n_units > limits.max_units_per_controller ||
            !in.declared_count_fits(
                n_units, MIN_WIRE_UNIT_GENERATION)) {
            return false;
        }
        controller.dependency_mode =
            checkpoint_child_dependency_mode(dependency_mode);
        controller.units.resize(n_units);
        for (auto & unit : controller.units) {
            if (!read_unit_generation(in, unit)) {
                return false;
            }
        }
        if (!in.u32(n_streams) ||
            n_streams > limits.max_streams_per_controller ||
            !in.declared_count_fits(n_streams, MIN_WIRE_STREAM)) {
            return false;
        }
        controller.streams.resize(n_streams);
        for (auto & stream : controller.streams) {
            uint32_t n_pages;
            if (!in.u32(stream.stream_index) ||
                !in.i32(stream.dependency_seq_id) ||
                !in.i32(stream.computation_frontier) ||
                !in.u32(stream.captured_dependency_count) ||
                !in.u32(n_pages) ||
                n_pages > limits.max_pages_per_stream ||
                !in.declared_count_fits(n_pages, MIN_WIRE_PAGE)) {
                return false;
            }
            stream.pages.resize(n_pages);
            for (auto & page : stream.pages) {
                if (!read_page_ref(in, page)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool read_shard_descriptor(
        bounded_reader & in,
        vbr_artifact_shard_descriptor & shard) {
    uint32_t ordinal;
    if (!in.u32(shard.shard_index) ||
        !in.u32(shard.topology_index) ||
        !in.u32(ordinal) ||
        !in.u64(shard.logical_offset) ||
        !in.u64(shard.row_count) ||
        !in.u64(shard.column_count) ||
        !in.u64(shard.row_bytes) ||
        !in.u64(shard.payload_bytes) ||
        !in.fixed_digest(shard.section_checksum) ||
        ordinal > UINT16_MAX) {
        return false;
    }
    shard.device_ordinal = uint16_t(ordinal);
    return true;
}

bool read_unit_descriptor_body(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        uint32_t format_version,
        vbr_artifact_unit_descriptor & descriptor) {
    uint32_t promote_hops;
    uint32_t transition;
    uint32_t representation;
    uint32_t recoverability;
    uint32_t side;
    uint32_t layout;
    uint32_t unified;
    uint32_t n_shards;
    uint32_t meansub_baked = 0;
    if (!in.u32(descriptor.child_id) ||
        !in.i32(descriptor.current_type) ||
        !in.i32(descriptor.last_source_type) ||
        !in.u32(promote_hops) ||
        !in.u32(transition) ||
        !in.u32(representation) ||
        !in.u32(descriptor.representation.codec_id) ||
        !in.u32(descriptor.representation.codec_version) ||
        !in.fixed_digest(descriptor.representation.reference_digest) ||
        !in.u32(descriptor.representation.source_loss_history) ||
        !in.u32(descriptor.representation.checkpoint_codec_hops) ||
        !in.u32(recoverability) ||
        !in.u32(side) ||
        !in.u32(layout) ||
        !in.u32(descriptor.n_stream) ||
        !in.u32(unified) ||
        !in.u64(descriptor.wm_cells) ||
        !in.u32(descriptor.rank)) {
        return false;
    }
    for (uint64_t & dimension : descriptor.dimensions) {
        if (!in.u64(dimension)) {
            return false;
        }
    }
    if (!in.u64(descriptor.row_alignment) ||
        !in.u32(descriptor.row_codec_version) ||
        !in.fixed_digest(descriptor.codebook_digest) ||
        !in.fixed_digest(descriptor.rotation_digest) ||
        !in.fixed_digest(descriptor.meansub_digest) ||
        (artifact_has_meansub_reference(format_version) &&
         (!in.i32(descriptor.meansub_model_id) ||
          !in.i32(descriptor.meansub_layer) ||
          !in.u32(meansub_baked))) ||
        !in.u32(n_shards) ||
        promote_hops > UINT8_MAX ||
        transition > uint32_t(vbr_repr_transition::recovery_invalidate) ||
        representation >= uint32_t(vbr_artifact_representation_kind::_count) ||
        recoverability >= uint32_t(vbr_artifact_recoverability::_count) ||
        side >= uint32_t(vbr_artifact_side::_count) ||
        layout >= uint32_t(vbr_artifact_layout::_count) ||
        unified > 1 || meansub_baked > 1 ||
        n_shards > limits.max_shards_per_unit ||
        !in.declared_count_fits(n_shards, MIN_WIRE_SHARD)) {
        return false;
    }
    descriptor.promote_hops = uint8_t(promote_hops);
    descriptor.last_transition = vbr_repr_transition(transition);
    descriptor.representation.kind =
        vbr_artifact_representation_kind(representation);
    descriptor.recoverability = vbr_artifact_recoverability(recoverability);
    descriptor.side = vbr_artifact_side(side);
    descriptor.layout = vbr_artifact_layout(layout);
    descriptor.unified = unified != 0;
    descriptor.meansub_baked = meansub_baked != 0;
    descriptor.shards.resize(n_shards);
    for (auto & shard : descriptor.shards) {
        if (!read_shard_descriptor(in, shard)) {
            return false;
        }
    }
    return true;
}

bool read_clean_stash_descriptor(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_clean_stash & stash) {
    uint32_t domain;
    uint32_t layout;
    uint32_t n_shards;
    if (!in.u64(stash.valid_rows) ||
        !in.u32(domain) ||
        !in.u32(layout) ||
        !in.u64(stash.row_count) ||
        !in.u64(stash.column_count) ||
        !in.u64(stash.row_bytes) ||
        !in.typed_digest_value(stash.payload_id) ||
        !in.u32(n_shards) ||
        domain > uint32_t(vbr_repr_domain::tapped) ||
        layout >= uint32_t(vbr_artifact_layout::_count) ||
        n_shards > limits.max_shards_per_unit ||
        !in.declared_count_fits(n_shards, MIN_WIRE_SHARD)) {
        return false;
    }
    stash.domain = vbr_repr_domain(domain);
    stash.layout = vbr_artifact_layout(layout);
    stash.shards.resize(n_shards);
    for (auto & shard : stash.shards) {
        if (!read_shard_descriptor(in, shard)) {
            return false;
        }
    }
    return true;
}

bool read_payload_bytes(
        bounded_reader & in,
        const vbr_artifact_payload_consumer * consumer,
        vbr_artifact_section_kind section,
        uint32_t object_index,
        uint32_t shard_index,
        bool clean_stash,
        uint64_t expected_size,
        llama_sha256_writer * hash_a,
        llama_sha256_writer * hash_b,
        std::array<uint8_t, 32> & observed_section_checksum,
        const char * checksum_domain) {
    uint32_t encoded_index;
    uint64_t encoded_size;
    if (!in.u32(encoded_index) ||
        !in.u64(encoded_size) ||
        encoded_index != shard_index ||
        encoded_size != expected_size ||
        encoded_size > in.remaining) {
        return false;
    }
    if (hash_a) {
        hash_a->u32(encoded_index);
        hash_a->u64(encoded_size);
    }
    if (hash_b) {
        hash_b->u64(encoded_size);
    }
    llama_sha256_writer section_payload_hash;
    section_payload_hash.string(checksum_domain, strlen(checksum_domain));
    section_payload_hash.u32(object_index);
    section_payload_hash.u32(shard_index);
    section_payload_hash.u64(encoded_size);
    std::vector<uint8_t> buffer(STREAM_CHUNK_SIZE);
    uint64_t offset = 0;
    while (offset < encoded_size) {
        const size_t n = size_t(std::min<uint64_t>(
            encoded_size - offset, buffer.size()));
        if (!in.raw(buffer.data(), n)) {
            return false;
        }
        if (hash_a) {
            hash_a->bytes(buffer.data(), n);
        }
        if (hash_b) {
            hash_b->bytes(buffer.data(), n);
        }
        section_payload_hash.bytes(buffer.data(), n);
        if (consumer && consumer->consume &&
            !consumer->consume(
                consumer->context, section, object_index, shard_index,
                clean_stash, offset, encoded_size, buffer.data(), n)) {
            return false;
        }
        offset += n;
    }
    observed_section_checksum = section_payload_hash.finish();
    return digest_nonzero(observed_section_checksum);
}

bool decode_topology_section(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_package & package) {
    uint32_t n_topologies;
    if (!in.u32(n_topologies) ||
        n_topologies == 0 ||
        n_topologies > limits.max_topologies ||
        !in.declared_count_fits(
            n_topologies, MIN_WIRE_TOPOLOGY)) {
        return false;
    }
    package.topologies.resize(n_topologies);
    for (auto & topology : package.topologies) {
        int32_t split_mode;
        uint32_t main_device;
        uint32_t n_devices;
        if (!in.u32(topology.version) ||
            !in.u32(n_devices) ||
            !in.i32(split_mode) ||
            !in.u32(main_device) ||
            split_mode < INT16_MIN ||
            split_mode > INT16_MAX ||
            main_device > UINT16_MAX ||
            n_devices == 0 ||
            n_devices > limits.max_devices_per_topology ||
            n_devices > UINT16_MAX ||
            !in.declared_count_fits(
                n_devices, MIN_WIRE_DEVICE)) {
            return false;
        }
        topology.device_count = uint16_t(n_devices);
        topology.split_mode = int16_t(split_mode);
        topology.main_device = { uint16_t(main_device) };
        topology.device_identities.resize(n_devices);
        topology.shard_weights.resize(n_devices);
        for (uint32_t i = 0; i < n_devices; ++i) {
            std::array<uint8_t, 32> identity;
            if (!in.fixed_digest(identity) ||
                !in.u32(topology.shard_weights[i])) {
                return false;
            }
            topology.device_identities[i] =
                llama_cache_acct_device_digest::from_sha256(
                    std::move(identity));
        }
        std::array<uint8_t, 32> digest;
        if (!in.fixed_digest(digest)) {
            return false;
        }
        topology.digest =
            llama_cache_acct_topology_digest::from_sha256(
                std::move(digest));
        if (!topology_valid(topology)) {
            return false;
        }
    }
    return true;
}

bool decode_unit_section(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        const vbr_artifact_payload_consumer * consumer,
        uint32_t object_index,
        vbr_artifact_package & package) {
    vbr_artifact_unit_blob blob;
    if (!in.typed_digest_value(blob.unit_version_id) ||
        !in.typed_digest_value(blob.payload_digest) ||
        !read_lineage(in, blob.descriptor.lineage_uuid) ||
        !in.u32(blob.descriptor.logical_unit_id) ||
        !in.u64(blob.descriptor.repr_gen) ||
        !read_unit_descriptor_body(
            in, limits, package.version, blob.descriptor)) {
        return false;
    }
    uint32_t stash_state;
    if (!in.u32(stash_state) ||
        stash_state >= uint32_t(vbr_artifact_clean_stash_state::_count)) {
        return false;
    }
    blob.descriptor.clean_stash_state =
        vbr_artifact_clean_stash_state(stash_state);
    if (blob.descriptor.clean_stash_state ==
            vbr_artifact_clean_stash_state::present &&
        !read_clean_stash_descriptor(
            in, limits, blob.descriptor.clean_stash)) {
        return false;
    }

    llama_sha256_writer unit_hash;
    unit_hash.string(DOMAIN_UNIT, sizeof(DOMAIN_UNIT) - 1);
    unit_hash.u32(package.version);
    if (!hash_lineage(unit_hash, blob.descriptor.lineage_uuid)) {
        return false;
    }
    unit_hash.u32(blob.descriptor.logical_unit_id);
    unit_hash.u64(blob.descriptor.repr_gen);
    if (!hash_sized_descriptor(
            unit_hash, blob.descriptor, package.version)) {
        return false;
    }
    unit_hash.u32(uint32_t(blob.descriptor.shards.size()));

    llama_sha256_writer payload_hash;
    payload_hash.string(DOMAIN_PAYLOAD, sizeof(DOMAIN_PAYLOAD) - 1);
    payload_hash.u32(uint32_t(blob.descriptor.shards.size()));
    for (auto & shard : blob.descriptor.shards) {
        emitter descriptor_hash;
        descriptor_hash.hash_a = &payload_hash;
        if (!emit_shard_descriptor(descriptor_hash, shard)) {
            return false;
        }
        std::array<uint8_t, 32> observed;
        if (!read_payload_bytes(
                in, consumer, vbr_artifact_section_kind::unit_blob,
                object_index, shard.shard_index, false,
                shard.payload_bytes, &unit_hash, &payload_hash,
                observed, DOMAIN_SHARD) ||
            observed != shard.section_checksum) {
            return false;
        }
    }

    unit_hash.u32(uint32_t(blob.descriptor.clean_stash_state));
    if (blob.descriptor.clean_stash_state ==
        vbr_artifact_clean_stash_state::present) {
        if (!hash_sized_stash_descriptor(
                unit_hash, blob.descriptor.clean_stash)) {
            return false;
        }
        unit_hash.u32(uint32_t(blob.descriptor.clean_stash.shards.size()));

        const auto & stash = blob.descriptor.clean_stash;
        llama_sha256_writer stash_hash;
        stash_hash.string(DOMAIN_STASH, sizeof(DOMAIN_STASH) - 1);
        stash_hash.u32(uint32_t(stash.layout));
        stash_hash.u64(stash.row_count);
        stash_hash.u64(stash.column_count);
        stash_hash.u64(stash.row_bytes);
        stash_hash.u64(stash.valid_rows);
        stash_hash.u32(uint32_t(stash.shards.size()));
        for (auto & shard : blob.descriptor.clean_stash.shards) {
            emitter descriptor_hash;
            descriptor_hash.hash_a = &stash_hash;
            if (!emit_shard_descriptor(descriptor_hash, shard)) {
                return false;
            }
            std::array<uint8_t, 32> observed;
            if (!read_payload_bytes(
                    in, consumer, vbr_artifact_section_kind::unit_blob,
                    object_index, shard.shard_index, true,
                    shard.payload_bytes, &unit_hash, &stash_hash,
                    observed, DOMAIN_SHARD) ||
                observed != shard.section_checksum) {
                return false;
            }
        }
        const auto decoded_stash_id =
            typed_digest<vbr_stash_payload_id>(stash_hash);
        if (decoded_stash_id != stash.payload_id) {
            return false;
        }
    }
    const auto decoded_payload_id =
        typed_digest<vbr_payload_digest>(payload_hash);
    const auto decoded_unit_id =
        typed_digest<vbr_unit_version_id>(unit_hash);
    if (decoded_payload_id != blob.payload_digest ||
        decoded_unit_id != blob.unit_version_id) {
        return false;
    }
    package.unit_blobs.push_back(std::move(blob));
    return true;
}

bool read_companion_reference(
        bounded_reader & in,
        vbr_artifact_companion_payload & companion);

bool decode_companion_section(
        bounded_reader & in,
        const vbr_artifact_payload_consumer * consumer,
        uint32_t object_index,
        vbr_artifact_package & package) {
    vbr_artifact_companion_payload companion;
    if (!read_companion_reference(in, companion)) {
        return false;
    }

    uint64_t encoded_size;
    if (!in.u64(encoded_size) ||
        encoded_size != companion.payload_bytes ||
        encoded_size > in.remaining) {
        return false;
    }
    llama_sha256_writer section_payload_hash;
    section_payload_hash.string(
        DOMAIN_COMPANION, sizeof(DOMAIN_COMPANION) - 1);
    section_payload_hash.u32(object_index);
    section_payload_hash.u32(0);
    section_payload_hash.u64(encoded_size);

    llama_sha256_writer payload_hash;
    payload_hash.string(DOMAIN_COMPANION, sizeof(DOMAIN_COMPANION) - 1);
    payload_hash.u32(object_index);
    payload_hash.u32(uint32_t(companion.kind));
    payload_hash.u32(companion.format_version);
    payload_hash.string(companion.build_identity_digest.data(),
                        companion.build_identity_digest.size());
    emitter domain_hash;
    domain_hash.hash_a = &payload_hash;
    if (!emit_portable_domain(domain_hash, companion.domain)) {
        return false;
    }
    payload_hash.u64(encoded_size);

    std::vector<uint8_t> buffer(STREAM_CHUNK_SIZE);
    uint64_t offset = 0;
    while (offset < encoded_size) {
        const size_t n = size_t(std::min<uint64_t>(
            encoded_size - offset, buffer.size()));
        if (!in.raw(buffer.data(), n)) {
            return false;
        }
        section_payload_hash.bytes(buffer.data(), n);
        payload_hash.bytes(buffer.data(), n);
        if (consumer && consumer->consume &&
            !consumer->consume(
                consumer->context,
                vbr_artifact_section_kind::companion_payload,
                object_index, 0, false, offset, encoded_size,
                buffer.data(), n)) {
            return false;
        }
        offset += n;
    }
    if (section_payload_hash.finish() != companion.section_checksum ||
        typed_digest<vbr_payload_digest>(payload_hash) !=
            companion.payload_digest) {
        return false;
    }
    package.companions.push_back(std::move(companion));
    return true;
}

bool read_identity(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_identity_block & identity) {
    return in.string(identity.execution_identity, limits) &&
           in.string(identity.adapter_config_identity, limits) &&
           in.string(identity.media_content_identity, limits) &&
           in.u64(identity.sequence_epoch) &&
           in.i64(identity.token_count) &&
           in.i32(identity.next_position);
}

bool read_token_block(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_token_block & block) {
    uint32_t count;
    if (!in.u32(block.codec_version) || !in.u32(count) ||
        count > limits.max_token_ids ||
        !in.declared_count_fits(count, sizeof(uint32_t))) {
        return false;
    }
    block.tokens.resize(count);
    for (llama_token & token : block.tokens) {
        if (!in.i32(token)) {
            return false;
        }
    }
    return in.typed_digest_value(block.digest);
}

bool read_stream_placement(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_stream_placement & placement,
        uint64_t & total_cells) {
    uint32_t count;
    if (!in.u32(placement.child_id) ||
        !in.u32(placement.stream_index) ||
        !in.i32(placement.source_sequence) ||
        !in.i32(placement.computation_frontier) ||
        !in.u32(count) ||
        count > limits.max_placement_cells ||
        !checked_add(total_cells, count, total_cells) ||
        total_cells > limits.max_placement_cells ||
        !in.declared_count_fits(count, MIN_WIRE_CELL_PLACEMENT)) {
        return false;
    }
    placement.cells.resize(count);
    for (auto & cell : placement.cells) {
        if (!in.u32(cell.physical_cell) ||
            !in.i32(cell.logical_position) ||
            !in.i32(cell.ext_x) ||
            !in.i32(cell.ext_y)) {
            return false;
        }
    }
    return true;
}

bool read_controller_policy(
        bounded_reader & in,
        vbr_artifact_controller_policy & policy) {
    uint32_t dependency_mode;
    uint32_t unified;
    uint32_t completed;
    if (!in.u32(policy.child_id) ||
        !in.u32(dependency_mode) ||
        !in.fixed_digest(policy.degrade_order_digest) ||
        !in.fixed_digest(policy.policy_digest) ||
        !in.u64(policy.cursor) ||
        !in.i32(policy.floor_type) ||
        !in.u64(policy.pressure_independent_settings) ||
        !in.u32(policy.n_stream) ||
        !in.u32(unified) ||
        !in.u64(policy.wm_cells) ||
        !in.fixed_digest(policy.current_type_vector_digest) ||
        !in.u32(completed) ||
        dependency_mode >
            uint32_t(checkpoint_child_dependency_mode::live_guarded) ||
        unified > 1 ||
        completed > 1) {
        return false;
    }
    policy.dependency_mode =
        checkpoint_child_dependency_mode(dependency_mode);
    policy.unified = unified != 0;
    policy.completed_wave = completed != 0;
    return true;
}

bool read_stash_reference(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_stash_reference & reference) {
    uint32_t domain;
    uint32_t n_pages;
    if (!in.u64(reference.valid_rows) ||
        !in.u32(domain) ||
        !in.u64(reference.row_count) ||
        !in.u64(reference.column_count) ||
        !in.u64(reference.row_bytes) ||
        !in.u32(reference.captured_sink_count) ||
        !in.u32(n_pages) ||
        domain > uint32_t(vbr_repr_domain::tapped) ||
        n_pages > limits.max_pages_per_stream ||
        !in.declared_count_fits(n_pages, MIN_WIRE_PAGE)) {
        return false;
    }
    reference.domain = vbr_repr_domain(domain);
    reference.covered_sink_pages.resize(n_pages);
    for (auto & page : reference.covered_sink_pages) {
        if (!read_page_ref(in, page)) {
            return false;
        }
    }
    return in.typed_digest_value(reference.payload_id);
}

bool read_unit_reference(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_unit_reference & reference) {
    uint32_t n_streams;
    uint32_t has_stash;
    if (!read_lineage(in, reference.lineage_uuid) ||
        !in.u32(reference.logical_unit_id) ||
        !in.u64(reference.repr_gen) ||
        !in.typed_digest_value(reference.unit_version_id) ||
        !in.typed_digest_value(reference.payload_digest) ||
        !in.u32(n_streams) ||
        n_streams > limits.max_streams_per_controller ||
        !in.declared_count_fits(n_streams, sizeof(uint32_t))) {
        return false;
    }
    reference.authorized_stream_refs.resize(n_streams);
    for (uint32_t & stream : reference.authorized_stream_refs) {
        if (!in.u32(stream)) {
            return false;
        }
    }
    if (!in.u32(has_stash) || has_stash > 1) {
        return false;
    }
    reference.has_stash_reference = has_stash != 0;
    return !reference.has_stash_reference ||
           read_stash_reference(in, limits, reference.stash_reference);
}

bool read_companion_reference(
        bounded_reader & in,
        vbr_artifact_companion_payload & companion) {
    uint32_t kind;
    if (!in.u32(kind) ||
        !in.u32(companion.format_version) ||
        !in.fixed_digest(companion.build_identity_digest) ||
        !read_portable_domain(in, companion.domain) ||
        !in.typed_digest_value(companion.payload_digest) ||
        !in.u64(companion.payload_bytes) ||
        !in.fixed_digest(companion.section_checksum) ||
        kind >= uint32_t(vbr_artifact_companion_kind::_count)) {
        return false;
    }
    companion.kind = vbr_artifact_companion_kind(kind);
    return true;
}

bool read_accounting_row(
        bounded_reader & in,
        vbr_artifact_portable_accounting_row & row) {
    uint32_t role;
    uint32_t attribution;
    if (!in.u32(role) ||
        !read_portable_domain(in, row.domain) ||
        !in.u64(row.logical_bytes) ||
        !in.u64(row.resident_bytes) ||
        !in.u32(attribution) ||
        role >= uint32_t(vbr_artifact_accounting_role::_count) ||
        attribution >= uint32_t(llama_cache_acct_attr_kind::_count)) {
        return false;
    }
    row.role = vbr_artifact_accounting_role(role);
    row.attribution = llama_cache_acct_attr_kind(attribution);
    return true;
}

bool read_consistency(
        bounded_reader & in,
        vbr_artifact_consistency & consistency) {
    uint32_t kind;
    if (!in.u32(kind) ||
        !in.typed_digest_value(
            consistency.source_capture_generation_id) ||
        !in.typed_digest_value(
            consistency.target_capture_generation_id) ||
        !in.typed_digest_value(consistency.transition_lineage_id) ||
        kind >= uint32_t(vbr_artifact_consistency_kind::_count)) {
        return false;
    }
    consistency.kind = vbr_artifact_consistency_kind(kind);
    return true;
}

bool decode_manifest_section(
        bounded_reader & in,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_package & package) {
    auto & manifest = package.manifest;
    llama_sha256_writer manifest_hash;
    manifest_hash.string(DOMAIN_MANIFEST, sizeof(DOMAIN_MANIFEST) - 1);
    in.auxiliary_hash = &manifest_hash;

    uint32_t n_policies;
    uint32_t n_references;
    uint32_t n_companions;
    uint32_t n_accounting;
    if (!in.u32(manifest.version) ||
        manifest.version != package.version ||
        !artifact_version_supported(manifest.version) ||
        !in.fixed_digest(manifest.identity_policy_order_digest) ||
        !read_identity(in, limits, manifest.identity) ||
        (artifact_has_reference_placement(manifest.version) &&
         !read_token_block(in, limits, manifest.token_block)) ||
        !read_generation_record(in, limits, manifest.generation) ||
        !in.typed_digest_value(manifest.capture_generation_id) ||
        !read_consistency(in, manifest.consistency) ||
        !in.u32(n_policies) ||
        n_policies > limits.max_controllers ||
        !in.declared_count_fits(
            n_policies, MIN_WIRE_CONTROLLER_POLICY)) {
        return false;
    }
    manifest.controller_policy.resize(n_policies);
    for (auto & policy : manifest.controller_policy) {
        if (!read_controller_policy(in, policy)) {
            return false;
        }
    }
    if (artifact_has_reference_placement(manifest.version)) {
        uint32_t n_placements;
        if (!in.u32(n_placements) ||
            n_placements > limits.max_stream_placements ||
            !in.declared_count_fits(
                n_placements, MIN_WIRE_STREAM_PLACEMENT)) {
            return false;
        }
        manifest.stream_placements.resize(n_placements);
        uint64_t total_cells = 0;
        for (auto & placement : manifest.stream_placements) {
            if (!read_stream_placement(
                    in, limits, placement, total_cells)) {
                return false;
            }
        }
    }
    if (!in.u32(n_references) ||
        n_references > limits.max_unit_blobs ||
        !in.declared_count_fits(
            n_references, MIN_WIRE_UNIT_REFERENCE)) {
        return false;
    }
    manifest.unit_references.resize(n_references);
    for (auto & reference : manifest.unit_references) {
        if (!read_unit_reference(in, limits, reference)) {
            return false;
        }
    }
    if (!in.u32(n_companions) ||
        n_companions > limits.max_companions ||
        !in.declared_count_fits(
            n_companions, MIN_WIRE_COMPANION_REFERENCE)) {
        return false;
    }
    manifest.companions.resize(n_companions);
    for (auto & companion : manifest.companions) {
        if (!read_companion_reference(in, companion)) {
            return false;
        }
    }
    if (!in.u32(n_accounting) ||
        n_accounting > limits.max_accounting_rows ||
        !in.declared_count_fits(
            n_accounting, MIN_WIRE_ACCOUNTING_ROW)) {
        return false;
    }
    manifest.accounting.resize(n_accounting);
    for (auto & row : manifest.accounting) {
        if (!read_accounting_row(in, row)) {
            return false;
        }
    }

    in.auxiliary_hash = nullptr;
    if (!in.typed_digest_value(manifest.manifest_digest) ||
        typed_digest<vbr_manifest_digest>(manifest_hash) !=
            manifest.manifest_digest) {
        return false;
    }
    return true;
}

struct vector_writer_context {
    std::vector<uint8_t> * bytes = nullptr;
};

bool vector_write(
        void * context,
        const uint8_t * data,
        size_t size) noexcept {
    auto * target = static_cast<vector_writer_context *>(context);
    try {
        target->bytes->insert(target->bytes->end(), data, data + size);
        return true;
    } catch (...) {
        return false;
    }
}

struct vector_reader_context {
    const std::vector<uint8_t> * bytes = nullptr;
    size_t position = 0;
};

bool vector_read(
        void * context,
        uint8_t * data,
        size_t size) noexcept {
    auto * source = static_cast<vector_reader_context *>(context);
    if (size > source->bytes->size() - source->position) {
        return false;
    }
    memcpy(data, source->bytes->data() + source->position, size);
    source->position += size;
    return true;
}

} // namespace

bool vbr_artifact_validate_portable_accounting(
        const std::vector<vbr_artifact_portable_topology> & topologies,
        const std::vector<vbr_artifact_portable_accounting_row> & rows) noexcept {
    try {
        if (topologies.empty() || rows.empty()) {
            return false;
        }
        for (const auto & topology : topologies) {
            if (!topology_valid(topology)) {
                return false;
            }
        }
        struct accounting_key {
            vbr_artifact_accounting_role role;
            vbr_artifact_portable_domain domain;
        };
        std::vector<accounting_key> seen;
        seen.reserve(rows.size());
        uint64_t total_logical = 0;
        uint64_t total_resident = 0;
        for (const auto & row : rows) {
            if (row.role >= vbr_artifact_accounting_role::_count ||
                role_category(row.role) ==
                    llama_cache_acct_category::_count ||
                row.attribution != llama_cache_acct_attr_kind::artifact ||
                !domain_valid(row.domain, topologies) ||
                row.logical_bytes == 0 ||
                row.resident_bytes == 0) {
                return false;
            }
            if (!checked_add(total_logical, row.logical_bytes, total_logical) ||
                !checked_add(total_resident, row.resident_bytes, total_resident)) {
                return false;
            }
            seen.push_back({ row.role, row.domain });
        }
        std::sort(seen.begin(), seen.end(), [&](const accounting_key & lhs,
                                                const accounting_key & rhs) {
            if (lhs.role != rhs.role) {
                return lhs.role < rhs.role;
            }
            return vbr_artifact_portable_domain_less(
                lhs.domain, rhs.domain);
        });
        if (std::adjacent_find(
                seen.begin(), seen.end(), [](const accounting_key & lhs,
                                             const accounting_key & rhs) {
                    return lhs.role == rhs.role &&
                           lhs.domain == rhs.domain;
                }) != seen.end()) {
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

llama_cache_acct_category vbr_artifact_accounting_category(
        vbr_artifact_accounting_role role) noexcept {
    return role_category(role);
}

std::array<uint8_t, 32> vbr_artifact_logical_unit_digest(
        const vbr_artifact_unit_descriptor & descriptor) noexcept {
    llama_sha256_writer hash;
    static constexpr char domain[] = "buun.vbr.logical-unit";
    hash.string(domain, sizeof(domain) - 1);
    if (!hash_lineage(hash, descriptor.lineage_uuid)) {
        return {};
    }
    hash.u32(descriptor.logical_unit_id);
    hash.u64(descriptor.repr_gen);
    return hash.finish();
}

vbr_artifact_status vbr_artifact_prepare(
        vbr_artifact_package & package) noexcept {
    try {
        if (!artifact_version_supported(package.version) ||
            package.flags != ARTIFACT_FLAGS_V1 ||
            package.topologies.empty() ||
            package.unit_blobs.empty()) {
            return vbr_artifact_status::invalid_argument;
        }
        for (auto & topology : package.topologies) {
            topology.digest =
                llama_cache_acct_compute_topology_digest(topology);
            if (topology.version != LLAMA_CACHE_ACCT_TOPOLOGY_VERSION ||
                topology.device_identities.empty() ||
                topology.device_identities.size() !=
                    topology.shard_weights.size() ||
                !topology.digest.valid() ||
                !topology_valid(topology)) {
                return vbr_artifact_status::topology_mismatch;
            }
        }
        for (uint32_t i = 0; i < package.unit_blobs.size(); ++i) {
            auto & blob = package.unit_blobs[i];
            if (blob.descriptor.shards.empty() ||
                !canonicalize_shards(blob.descriptor.shards) ||
                !prepare_shard_checksums(i, blob.descriptor.shards)) {
                return vbr_artifact_status::malformed;
            }
            if (blob.descriptor.clean_stash_state ==
                vbr_artifact_clean_stash_state::present) {
                if (blob.descriptor.clean_stash.shards.empty() ||
                    !canonicalize_shards(
                        blob.descriptor.clean_stash.shards) ||
                    !prepare_shard_checksums(
                        i, blob.descriptor.clean_stash.shards)) {
                    return vbr_artifact_status::malformed;
                }
            }
            if (!prepare_payload_digest(blob) ||
                !prepare_stash_digest(blob) ||
                !descriptor_metadata_valid(
                    blob.descriptor, package.topologies,
                    package.version, true) ||
                !prepare_unit_id(blob, package.version)) {
                return vbr_artifact_status::content_id_mismatch;
            }
        }
        for (uint32_t i = 0; i < package.companions.size(); ++i) {
            if (!prepare_companion(
                    i, package.topologies, package.companions[i])) {
                return vbr_artifact_status::content_id_mismatch;
            }
        }

        vbr_capture_generation_id capture;
        if (!hash_generation(package.manifest.generation, capture)) {
            return vbr_artifact_status::generation_mismatch;
        }
        package.manifest.capture_generation_id = capture;
        if (package.manifest.consistency.kind ==
            vbr_artifact_consistency_kind::capture_exact) {
            package.manifest.consistency.source_capture_generation_id =
                capture;
            package.manifest.consistency.target_capture_generation_id = {};
            package.manifest.consistency.transition_lineage_id = {};
        }

        package.manifest.version = package.version;
        if (artifact_has_reference_placement(package.version)) {
            if (!prepare_token_block(
                    package.manifest.identity,
                    package.manifest.identity_policy_order_digest,
                    package.manifest.token_block)) {
                return vbr_artifact_status::malformed;
            }
        } else if (!reference_metadata_absent(package.manifest)) {
            return vbr_artifact_status::malformed;
        }

        // Bind reference rows by their logical tuple; content IDs are computed from bytes and
        // never accepted from a caller as a substitute for that tuple.
        for (auto & reference : package.manifest.unit_references) {
            const auto found = std::find_if(
                package.unit_blobs.begin(), package.unit_blobs.end(),
                [&](const vbr_artifact_unit_blob & blob) {
                    return blob.descriptor.lineage_uuid == reference.lineage_uuid &&
                           blob.descriptor.logical_unit_id ==
                               reference.logical_unit_id &&
                           blob.descriptor.repr_gen == reference.repr_gen;
                });
            if (found == package.unit_blobs.end()) {
                return vbr_artifact_status::generation_mismatch;
            }
            reference.unit_version_id = found->unit_version_id;
            reference.payload_digest = found->payload_digest;
            if (reference.has_stash_reference) {
                reference.stash_reference.payload_id =
                    found->descriptor.clean_stash.payload_id;
            }
        }
        package.manifest.companions = package.companions;
        if (!vbr_artifact_validate_portable_accounting(
                package.topologies, package.manifest.accounting) ||
            !accounting_payloads_match(package)) {
            return vbr_artifact_status::accounting_unavailable;
        }
        if (!prepare_manifest_digest(package.manifest) ||
            !package_metadata_valid(package, nullptr, true)) {
            return vbr_artifact_status::malformed;
        }
        return vbr_artifact_status::ok;
    } catch (...) {
        return vbr_artifact_status::internal_error;
    }
}

vbr_artifact_status vbr_artifact_prepare_projected_metadata(
        vbr_artifact_package & package) noexcept {
    try {
        if (!artifact_version_supported(package.version) ||
            package.flags != ARTIFACT_FLAGS_V1 ||
            package.topologies.empty() || package.unit_blobs.empty()) {
            return vbr_artifact_status::invalid_argument;
        }
        for (auto & topology : package.topologies) {
            topology.digest =
                llama_cache_acct_compute_topology_digest(topology);
            if (topology.version != LLAMA_CACHE_ACCT_TOPOLOGY_VERSION ||
                topology.device_identities.empty() ||
                topology.device_identities.size() !=
                    topology.shard_weights.size() ||
                !topology.digest.valid() || !topology_valid(topology)) {
                return vbr_artifact_status::topology_mismatch;
            }
        }
        for (auto & blob : package.unit_blobs) {
            if (!blob.unit_version_id.valid() ||
                !blob.payload_digest.valid() ||
                blob.descriptor.shards.empty() ||
                !canonicalize_shards(blob.descriptor.shards) ||
                blob.descriptor.clean_stash_state !=
                    vbr_artifact_clean_stash_state::absent_at_source ||
                !descriptor_metadata_valid(
                    blob.descriptor, package.topologies,
                    package.version, false, true)) {
                return vbr_artifact_status::content_id_mismatch;
            }
        }
        for (uint32_t i = 0; i < package.companions.size(); ++i) {
            if (!prepare_companion(
                    i, package.topologies, package.companions[i])) {
                return vbr_artifact_status::content_id_mismatch;
            }
        }

        vbr_capture_generation_id capture;
        if (!hash_generation(package.manifest.generation, capture)) {
            return vbr_artifact_status::generation_mismatch;
        }
        package.manifest.capture_generation_id = capture;
        package.manifest.consistency.kind =
            vbr_artifact_consistency_kind::capture_exact;
        package.manifest.consistency.source_capture_generation_id = capture;
        package.manifest.consistency.target_capture_generation_id = {};
        package.manifest.consistency.transition_lineage_id = {};
        package.manifest.version = package.version;
        if (!prepare_token_block(
                package.manifest.identity,
                package.manifest.identity_policy_order_digest,
                package.manifest.token_block)) {
            return vbr_artifact_status::malformed;
        }
        if (package.manifest.unit_references.size() !=
                package.unit_blobs.size()) {
            return vbr_artifact_status::generation_mismatch;
        }
        for (size_t i = 0;
             i < package.manifest.unit_references.size(); ++i) {
            auto & reference = package.manifest.unit_references[i];
            const auto & found = package.unit_blobs[i];
            if (found.descriptor.lineage_uuid != reference.lineage_uuid ||
                found.descriptor.logical_unit_id !=
                    reference.logical_unit_id ||
                found.descriptor.repr_gen != reference.repr_gen) {
                return vbr_artifact_status::generation_mismatch;
            }
            reference.unit_version_id = found.unit_version_id;
            reference.payload_digest = found.payload_digest;
            reference.has_stash_reference = false;
            reference.stash_reference = {};
        }
        package.manifest.companions = package.companions;
        if (!vbr_artifact_validate_portable_accounting(
                package.topologies, package.manifest.accounting) ||
            !accounting_payloads_match(package)) {
            return vbr_artifact_status::accounting_unavailable;
        }
        if (!prepare_manifest_digest(package.manifest) ||
            !package_metadata_valid(package, nullptr, false, true)) {
            return vbr_artifact_status::malformed;
        }
        return vbr_artifact_status::ok;
    } catch (...) {
        return vbr_artifact_status::internal_error;
    }
}

vbr_artifact_status vbr_artifact_validate_prepared_package(
        const vbr_artifact_package & package) noexcept {
    try {
        auto canonical = package;
        const auto status = vbr_artifact_prepare(canonical);
        if (status != vbr_artifact_status::ok) {
            return status;
        }
        return prepared_identity_equal(package, canonical) ?
            vbr_artifact_status::ok :
            vbr_artifact_status::content_id_mismatch;
    } catch (...) {
        return vbr_artifact_status::internal_error;
    }
}

vbr_artifact_status vbr_artifact_encode(
        vbr_artifact_package & package,
        const vbr_artifact_stream_writer & output,
        uint64_t max_total_bytes,
        uint64_t * encoded_size) noexcept {
    if (encoded_size) {
        *encoded_size = 0;
    }
    try {
        if (!output.write || max_total_bytes == 0) {
            return vbr_artifact_status::invalid_argument;
        }
        const auto prepared = vbr_artifact_prepare(package);
        if (prepared != vbr_artifact_status::ok) {
            return prepared;
        }

        std::vector<section_descriptor> sections;
        if (!prepare_sections(package, sections)) {
            return vbr_artifact_status::internal_error;
        }
        uint64_t total_size;
        if (!compute_total_size(sections, total_size) ||
            total_size > max_total_bytes) {
            return vbr_artifact_status::out_of_bounds;
        }
        const auto order = ordering_digest(sections);
        std::array<uint8_t, 32> expected_checksum;
        if (!hash_package(
                package.version, package.flags, sections, total_size, order,
                expected_checksum)) {
            return vbr_artifact_status::internal_error;
        }

        emitter wire;
        wire.output = &output;
        if (!emit_header(
                wire, package, total_size, uint32_t(sections.size()),
                order, expected_checksum)) {
            return vbr_artifact_status::internal_error;
        }

        llama_sha256_writer verify_package;
        verify_package.string(
            DOMAIN_PACKAGE, sizeof(DOMAIN_PACKAGE) - 1);
        verify_package.u32(ARTIFACT_MAGIC);
        verify_package.u32(package.version);
        verify_package.u32(ARTIFACT_HEADER_SIZE);
        verify_package.u32(package.flags);
        verify_package.u64(total_size);
        verify_package.u32(uint32_t(sections.size()));
        verify_package.string(order.data(), order.size());

        for (const auto & section : sections) {
            if (!emit_section_header(wire, section)) {
                return vbr_artifact_status::internal_error;
            }
            verify_package.u32(uint32_t(section.kind));
            verify_package.u32(section.index);
            verify_package.u64(section.size);
            verify_package.string(
                section.checksum.data(), section.checksum.size());

            llama_sha256_writer verify_section;
            verify_section.string(
                DOMAIN_SECTION, sizeof(DOMAIN_SECTION) - 1);
            verify_section.u32(uint32_t(section.kind));
            verify_section.u32(section.index);
            emitter body;
            body.output = &output;
            body.hash_b = &verify_section;
            if (!emit_section_body_verified(body, package, section) ||
                !body.ok ||
                body.count != section.size ||
                verify_section.finish() != section.checksum) {
                return vbr_artifact_status::content_id_mismatch;
            }
            uint64_t next;
            if (!checked_add(wire.count, body.count, next)) {
                return vbr_artifact_status::out_of_bounds;
            }
            wire.count = next;
        }
        if (wire.count != total_size ||
            verify_package.finish() != expected_checksum) {
            return vbr_artifact_status::content_id_mismatch;
        }
        if (encoded_size) {
            *encoded_size = total_size;
        }
        return vbr_artifact_status::ok;
    } catch (...) {
        return vbr_artifact_status::internal_error;
    }
}

vbr_artifact_status vbr_artifact_decode(
        const vbr_artifact_stream_reader & input,
        uint64_t encoded_size,
        const vbr_artifact_decode_limits & limits,
        const vbr_artifact_payload_consumer * payload_consumer,
        vbr_artifact_package & output) noexcept {
    output = {};
    output.version = 0;
    struct consumer_finish_guard {
        const vbr_artifact_payload_consumer * consumer = nullptr;
        bool verified = false;

        ~consumer_finish_guard() {
            if (consumer) {
                consumer->finish(consumer->context, verified);
            }
        }
    };
    if (payload_consumer &&
        ((!payload_consumer->consume && payload_consumer->finish) ||
         (payload_consumer->consume && !payload_consumer->finish))) {
        return vbr_artifact_status::invalid_argument;
    }
    consumer_finish_guard consumer_guard {
        payload_consumer && payload_consumer->consume ?
            payload_consumer : nullptr,
        false,
    };
    try {
        if (!input.read ||
            limits.max_total_bytes == 0 ||
            encoded_size < ARTIFACT_HEADER_SIZE ||
            encoded_size > limits.max_total_bytes) {
            return vbr_artifact_status::out_of_bounds;
        }
        input_cursor cursor { &input, encoded_size };
        uint32_t magic;
        uint32_t version;
        uint32_t header_size;
        uint32_t flags;
        uint64_t declared_size;
        uint32_t section_count;
        std::array<uint8_t, 32> expected_order;
        std::array<uint8_t, 32> expected_package_checksum;
        if (!cursor.u32(magic) ||
            !cursor.u32(version) ||
            !cursor.u32(header_size) ||
            !cursor.u32(flags) ||
            !cursor.u64(declared_size) ||
            !cursor.u32(section_count) ||
            !cursor.raw(expected_order.data(), expected_order.size()) ||
            !cursor.raw(expected_package_checksum.data(),
                        expected_package_checksum.size())) {
            return vbr_artifact_status::malformed;
        }
        if (magic != ARTIFACT_MAGIC) {
            return vbr_artifact_status::malformed;
        }
        if (!artifact_version_supported(version)) {
            return vbr_artifact_status::unsupported_version;
        }
        const uint64_t max_sections =
            uint64_t(limits.max_unit_blobs) +
            uint64_t(limits.max_companions) + 2;
        if (header_size != ARTIFACT_HEADER_SIZE ||
            flags != ARTIFACT_FLAGS_V1 ||
            declared_size != encoded_size ||
            section_count < 3 ||
            section_count > max_sections ||
            !digest_nonzero(expected_order) ||
            !digest_nonzero(expected_package_checksum)) {
            return vbr_artifact_status::malformed;
        }

        vbr_artifact_package decoded;
        decoded.version = version;
        decoded.flags = flags;
        llama_sha256_writer package_hash;
        package_hash.string(
            DOMAIN_PACKAGE, sizeof(DOMAIN_PACKAGE) - 1);
        package_hash.u32(magic);
        package_hash.u32(version);
        package_hash.u32(header_size);
        package_hash.u32(flags);
        package_hash.u64(declared_size);
        package_hash.u32(section_count);
        package_hash.string(expected_order.data(), expected_order.size());

        if (section_count >
            cursor.remaining / ARTIFACT_SECTION_HEADER_SIZE) {
            return vbr_artifact_status::malformed;
        }
        std::vector<section_descriptor> sections;
        sections.reserve(section_count);
        uint32_t expected_unit = 0;
        uint32_t expected_companion = 0;
        bool saw_topology = false;
        bool saw_manifest = false;
        for (uint32_t ordinal = 0; ordinal < section_count; ++ordinal) {
            uint32_t kind_value;
            section_descriptor section;
            if (!cursor.u32(kind_value) ||
                !cursor.u32(section.index) ||
                !cursor.u64(section.size) ||
                !cursor.raw(section.checksum.data(),
                            section.checksum.size()) ||
                kind_value >= uint32_t(vbr_artifact_section_kind::_count) ||
                section.size > cursor.remaining ||
                !digest_nonzero(section.checksum)) {
                return vbr_artifact_status::malformed;
            }
            section.kind = vbr_artifact_section_kind(kind_value);
            bool canonical = false;
            switch (section.kind) {
                case vbr_artifact_section_kind::topology_table:
                    canonical = ordinal == 0 &&
                                section.index == 0 &&
                                !saw_topology;
                    saw_topology = true;
                    break;
                case vbr_artifact_section_kind::unit_blob:
                    canonical = saw_topology &&
                                !saw_manifest &&
                                expected_companion == 0 &&
                                section.index == expected_unit++;
                    break;
                case vbr_artifact_section_kind::companion_payload:
                    canonical = saw_topology &&
                                expected_unit > 0 &&
                                !saw_manifest &&
                                section.index == expected_companion++;
                    break;
                case vbr_artifact_section_kind::reference_manifest:
                    canonical = ordinal + 1 == section_count &&
                                section.index == 0 &&
                                expected_unit > 0 &&
                                !saw_manifest;
                    saw_manifest = true;
                    break;
                case vbr_artifact_section_kind::_count:
                    break;
            }
            if (!canonical) {
                return vbr_artifact_status::malformed;
            }

            package_hash.u32(kind_value);
            package_hash.u32(section.index);
            package_hash.u64(section.size);
            package_hash.string(
                section.checksum.data(), section.checksum.size());
            llama_sha256_writer section_hash;
            section_hash.string(
                DOMAIN_SECTION, sizeof(DOMAIN_SECTION) - 1);
            section_hash.u32(kind_value);
            section_hash.u32(section.index);
            bounded_reader body {
                &cursor, section.size, nullptr, &section_hash, nullptr,
            };
            bool ok = false;
            switch (section.kind) {
                case vbr_artifact_section_kind::topology_table:
                    ok = decode_topology_section(body, limits, decoded);
                    break;
                case vbr_artifact_section_kind::unit_blob:
                    ok = decode_unit_section(
                        body, limits, payload_consumer,
                        section.index, decoded);
                    break;
                case vbr_artifact_section_kind::companion_payload:
                    ok = decode_companion_section(
                        body, payload_consumer, section.index, decoded);
                    break;
                case vbr_artifact_section_kind::reference_manifest:
                    ok = decode_manifest_section(body, limits, decoded);
                    break;
                case vbr_artifact_section_kind::_count:
                    break;
            }
            if (!ok || body.remaining != 0) {
                return vbr_artifact_status::malformed;
            }
            if (section_hash.finish() != section.checksum) {
                return vbr_artifact_status::checksum_mismatch;
            }
            sections.push_back(section);
        }
        if (!saw_topology ||
            !saw_manifest ||
            expected_unit == 0 ||
            cursor.remaining != 0 ||
            ordering_digest(sections) != expected_order ||
            package_hash.finish() != expected_package_checksum) {
            return vbr_artifact_status::checksum_mismatch;
        }
        vbr_capture_generation_id capture;
        if (!hash_generation(decoded.manifest.generation, capture) ||
            capture != decoded.manifest.capture_generation_id) {
            return vbr_artifact_status::generation_mismatch;
        }
        if (!vbr_artifact_validate_portable_accounting(
                decoded.topologies, decoded.manifest.accounting) ||
            !accounting_payloads_match(decoded)) {
            return vbr_artifact_status::accounting_unavailable;
        }
        if (!package_metadata_valid(decoded, &limits, false)) {
            return vbr_artifact_status::malformed;
        }
        output = std::move(decoded);
        consumer_guard.verified = true;
        return vbr_artifact_status::ok;
    } catch (...) {
        output = {};
        output.version = 0;
        return vbr_artifact_status::internal_error;
    }
}

vbr_artifact_status vbr_artifact_encode_vector(
        vbr_artifact_package & package,
        std::vector<uint8_t> & output,
        uint64_t max_total_bytes) noexcept {
    output.clear();
    try {
        vector_writer_context context { &output };
        const vbr_artifact_stream_writer writer {
            &context, vector_write,
        };
        const auto result = vbr_artifact_encode(
            package, writer, max_total_bytes, nullptr);
        if (result != vbr_artifact_status::ok) {
            output.clear();
        }
        return result;
    } catch (...) {
        output.clear();
        return vbr_artifact_status::internal_error;
    }
}

vbr_artifact_status vbr_artifact_decode_vector(
        const std::vector<uint8_t> & input,
        const vbr_artifact_decode_limits & limits,
        vbr_artifact_package & output) noexcept {
    vector_reader_context context { &input, 0 };
    const vbr_artifact_stream_reader reader {
        &context, vector_read,
    };
    return vbr_artifact_decode(
        reader, input.size(), limits, nullptr, output);
}

const char * vbr_artifact_status_name(
        vbr_artifact_status status) noexcept {
    switch (status) {
        case vbr_artifact_status::ok:                    return "vbr_artifact_ok";
        case vbr_artifact_status::invalid_argument:      return "vbr_artifact_invalid_argument";
        case vbr_artifact_status::unsupported_version:   return "vbr_artifact_unsupported_version";
        case vbr_artifact_status::malformed:              return "vbr_artifact_malformed";
        case vbr_artifact_status::out_of_bounds:          return "vbr_artifact_out_of_bounds";
        case vbr_artifact_status::checksum_mismatch:      return "vbr_artifact_checksum_mismatch";
        case vbr_artifact_status::content_id_mismatch:    return "vbr_artifact_content_id_mismatch";
        case vbr_artifact_status::topology_mismatch:      return "vbr_artifact_topology_mismatch";
        case vbr_artifact_status::generation_mismatch:    return "vbr_artifact_generation_mismatch";
        case vbr_artifact_status::accounting_unavailable: return "vbr_artifact_accounting_unavailable";
        case vbr_artifact_status::internal_error:         return "vbr_artifact_internal_error";
        case vbr_artifact_status::_count:                 return "vbr_artifact_invalid";
    }
    return "vbr_artifact_invalid";
}
