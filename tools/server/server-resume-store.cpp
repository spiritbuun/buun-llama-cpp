#include "server-resume-store.h"

#include "hash/xxhash/xxhash.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

static constexpr char     MANIFEST_MAGIC[8] = { 'B', 'U', 'U', 'N', 'R', 'S', 'M', 'M' };
static constexpr char     OBJECT_MAGIC[8]   = { 'B', 'U', 'U', 'N', 'R', 'S', 'M', 'O' };
static constexpr char     ENTRY_MAGIC[8]    = { 'B', 'U', 'U', 'N', 'R', 'S', 'M', 'E' };
static constexpr uint32_t ENTRY_VERSION     = 1;
static constexpr uint32_t HEADER_SIZE       = 64;
static constexpr size_t   MAX_STRING        = 256;
static constexpr size_t   MAX_ADAPTERS      = 32;

bool server_resume_object_record::holds(const uint8_t * payload, size_t size) const {
    return bytes == size && xxh3 == XXH3_64bits(payload, size);
}

const char * server_resume_reason_name(server_resume_reason reason) noexcept {
    switch (reason) {
        case server_resume_reason::ok:                       return "ok";
        case server_resume_reason::format_unsupported:       return "format_unsupported";
        case server_resume_reason::manifest_corrupt:         return "manifest_corrupt";
        case server_resume_reason::object_missing:           return "object_missing";
        case server_resume_reason::object_size_mismatch:     return "object_size_mismatch";
        case server_resume_reason::object_checksum_mismatch: return "object_checksum_mismatch";
        case server_resume_reason::store_locked:             return "store_locked";
        case server_resume_reason::store_unwritable:         return "store_unwritable";
        case server_resume_reason::no_space:                 return "no_space";
        case server_resume_reason::io_error:                 return "io_error";
        case server_resume_reason::_count:                   break;
    }
    return "unknown";
}

//
// little-endian fields
//

static void put_u32(uint8_t * dst, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        dst[i] = (uint8_t) (value >> (8 * i));
    }
}

static void put_u64(uint8_t * dst, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = (uint8_t) (value >> (8 * i));
    }
}

static uint32_t get_u32(const uint8_t * src) {
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
        value |= (uint32_t) src[i] << (8 * i);
    }
    return value;
}

static uint64_t get_u64(const uint8_t * src) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= (uint64_t) src[i] << (8 * i);
    }
    return value;
}

// the last field of both headers is the checksum of the bytes before it
static void header_seal(uint8_t * header) {
    put_u64(header + HEADER_SIZE - 8, XXH3_64bits(header, HEADER_SIZE - 8));
}

static bool header_sealed(const uint8_t * header) {
    return get_u64(header + HEADER_SIZE - 8) == XXH3_64bits(header, HEADER_SIZE - 8);
}

//
// manifest
//

bool server_resume_producer::operator==(const server_resume_producer & other) const {
    // the host time says when, not who
    return model_name == other.model_name && model_file == other.model_file &&
        mmproj_file == other.mmproj_file && weight_type == other.weight_type &&
        build == other.build && cache_type_k == other.cache_type_k && cache_type_v == other.cache_type_v &&
        adapters == other.adapters;
}

uint32_t server_resume_manifest::producer_index(const server_resume_producer & producer) {
    const auto it = std::find(producers.begin(), producers.end(), producer);
    if (it != producers.end()) {
        return (uint32_t) (it - producers.begin());
    }
    if (producers.size() >= server_resume_limits::max_producers) {
        throw std::length_error("resume producer table is full; retaining the previous entry");
    }
    producers.push_back(producer);
    return (uint32_t) producers.size() - 1;
}

void server_resume_manifest::retain_producers(const std::vector<server_resume_object_record *> & records) {
    constexpr uint32_t unused = UINT32_MAX;
    std::vector<uint32_t>               index(producers.size(), unused);
    std::vector<server_resume_producer> retained;
    for (server_resume_object_record * record : records) {
        uint32_t & at = index.at(record->producer);
        if (at == unused) {
            at = (uint32_t) retained.size();
            retained.push_back(std::move(producers[record->producer]));
        }
        record->producer = at;
    }
    producers = std::move(retained);
}

static bool is_hex(const std::string & text, size_t max_size) {
    return !text.empty() && text.size() <= max_size &&
        std::all_of(text.begin(), text.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

static bool object_record_valid(const server_resume_object_record & record, size_t n_producers, std::string & error) {
    if (record.bytes == 0 || record.bytes > server_resume_limits::max_bytes(record.kind)) {
        error = "object size out of bounds";
        return false;
    }
    if (!is_hex(record.prefix_digest, 64)) {
        error = "prefix digest is not hex";
        return false;
    }
    if (record.producer >= n_producers) {
        error = "producer index out of range";
        return false;
    }
    return true;
}

bool server_resume_manifest_validate(const server_resume_manifest & manifest, std::string & error) {
    if (manifest.shared_positions && !manifest.artifact) {
        error = "shared positions require a whole sequence object";
        return false;
    }
    using limits = server_resume_limits;

    if (manifest.n_tokens <= 0 || manifest.n_tokens > limits::max_tokens) {
        error = "token count out of bounds";
        return false;
    }
    if (manifest.chunk_tokens <= 0 || manifest.chunk_tokens > limits::max_tokens) {
        error = "chunk size out of bounds";
        return false;
    }
    if (manifest.chunks.empty() == !manifest.artifact || manifest.chunks.size() > limits::max_chunks ||
        manifest.tail_states.size() > limits::max_tail_states ||
        manifest.producers.empty() || manifest.producers.size() > limits::max_producers) {
        error = "record count out of bounds";
        return false;
    }
    if (!is_hex(manifest.resume_key, 64) || !is_hex(manifest.family_digest, 64) || !is_hex(manifest.adapter_identity, 64)) {
        error = "identity is not hex";
        return false;
    }
    if (manifest.ledger.empty() || manifest.ledger.size() > limits::max_manifest_bytes) {
        error = "ledger size out of bounds";
        return false;
    }

    if (manifest.artifact) {
        const auto & artifact = *manifest.artifact;
        // an artifact holds its partial state itself, a placement has the frontier's beside it;
        // either may carry the ring's two checkpoints below its end
        size_t n_frontier = 0;
        bool   roles_hold = true;
        for (const auto & tail : manifest.tail_states) {
            const bool frontier = tail.p0 == manifest.n_tokens;
            n_frontier += frontier;
            roles_hold = roles_hold && frontier == (tail.role == "frontier");
        }
        if ((artifact.kind != server_resume_object_kind::artifact && !manifest.placed() && !manifest.whole_sequence()) ||
            (manifest.whole_sequence() && !manifest.shared_positions) ||
            artifact.p0 != 0 || artifact.p1 != manifest.n_tokens ||
            artifact.gen == 0 || artifact.gen > manifest.generation || !roles_hold ||
            n_frontier > (manifest.placed() ? 1u : 0u) || manifest.tail_states.size() - n_frontier > 2) {
            error = "artifact does not hold the token range";
            return false;
        }
        if (!object_record_valid(artifact, manifest.producers.size(), error)) {
            return false;
        }
    }
    if (manifest.placed() ? !server_resume_store::entry_id_valid(manifest.pool_entry) || manifest.pool_generation == 0
                          : !manifest.pool_entry.empty() || manifest.pool_generation != 0 || manifest.pool_xxh3 != 0) {
        error = "placement and pool do not go together";
        return false;
    }
    // an artifact is bound to its epoch and chunks have none; a placement may carry one
    const bool needs_epoch = manifest.artifact && !manifest.placed() && !manifest.whole_sequence();
    if (manifest.sequence_epoch != 0 ? !manifest.artifact : needs_epoch) {
        error = "sequence epoch and artifact do not go together";
        return false;
    }

    int32_t next = manifest.artifact ? manifest.n_tokens : 0;
    for (const auto & chunk : manifest.chunks) {
        if (chunk.kind != server_resume_object_kind::base_chunk || chunk.p0 != next || chunk.p1 <= chunk.p0 ||
            chunk.p1 > manifest.n_tokens || chunk.gen == 0 || chunk.gen > manifest.generation) {
            error = "chunks do not tile the token range";
            return false;
        }
        if (!object_record_valid(chunk, manifest.producers.size(), error)) {
            return false;
        }
        next = chunk.p1;
    }
    if (next != manifest.n_tokens) {
        error = "chunks do not cover the token range";
        return false;
    }

    std::set<int32_t> positions;
    for (const auto & tail : manifest.tail_states) {
        if (tail.kind != server_resume_object_kind::tail_state || tail.p1 != 0 || tail.p0 <= 0 ||
            tail.p0 > manifest.n_tokens || tail.n_tokens != tail.p0 ||
            (manifest.shared_positions ? tail.pos_max >= tail.p0 : tail.pos_max != tail.p0 - 1) ||
            tail.pos_min < 0 || tail.pos_min > tail.pos_max || tail.gen == 0 || tail.gen > manifest.generation ||
            !positions.insert(tail.p0).second) {
            error = "tail state position is invalid";
            return false;
        }
        if (tail.role != "frontier" && tail.role != "turn" && tail.role != "early") {
            error = "tail state role is unknown";
            return false;
        }
        if (!object_record_valid(tail, manifest.producers.size(), error)) {
            return false;
        }
    }

    for (const auto & producer : manifest.producers) {
        const std::string * texts[] = { &producer.model_name, &producer.model_file, &producer.mmproj_file, &producer.weight_type,
            &producer.build, &producer.cache_type_k, &producer.cache_type_v };
        for (const auto * text : texts) {
            if (text->size() > MAX_STRING) {
                error = "producer text too long";
                return false;
            }
        }
        if (producer.adapters.size() > MAX_ADAPTERS ||
            std::any_of(producer.adapters.begin(), producer.adapters.end(), [](const std::string & a) { return a.size() > MAX_STRING; })) {
            error = "producer adapters out of bounds";
            return false;
        }
    }

    return true;
}

static json object_record_to_json(const server_resume_object_record & record) {
    json out = {
        { "gen",           record.gen },
        { "bytes",         record.bytes },
        { "xxh3",          record.xxh3 },
        { "prefix_digest", record.prefix_digest },
        { "producer",      record.producer },
    };
    if (record.kind != server_resume_object_kind::tail_state) {
        out["p0"] = record.p0;
        out["p1"] = record.p1;
    } else {
        out["pos"]      = record.p0;
        out["pos_min"]  = record.pos_min;
        out["pos_max"]  = record.pos_max;
        out["n_tokens"] = record.n_tokens;
        out["role"]     = record.role;
    }
    return out;
}

// an integer that fits the field. the library's own conversion wraps silently, and a float is not a count
template <typename T>
static T get_int(const json & in, const char * key) {
    const json & value = in.at(key);
    if (value.is_number_unsigned()) {
        const uint64_t u = value.get<uint64_t>();
        if (u <= (uint64_t) std::numeric_limits<T>::max()) {
            return (T) u;
        }
    } else if (value.is_number_integer() && std::is_signed<T>::value) {
        const int64_t i = value.get<int64_t>();
        if (i >= (int64_t) std::numeric_limits<T>::min()) {
            return (T) i; // negative here
        }
    }
    throw std::out_of_range(std::string(key) + " is not an integer of its range");
}

static server_resume_object_record object_record_from_json(const json & in, server_resume_object_kind kind) {
    server_resume_object_record record;
    record.kind          = kind;
    record.gen           = get_int<uint64_t>(in, "gen");
    record.bytes         = get_int<uint64_t>(in, "bytes");
    record.xxh3          = get_int<uint64_t>(in, "xxh3");
    record.prefix_digest = in.at("prefix_digest").get<std::string>();
    record.producer      = get_int<uint32_t>(in, "producer");
    if (kind != server_resume_object_kind::tail_state) {
        record.p0 = get_int<int32_t>(in, "p0");
        record.p1 = get_int<int32_t>(in, "p1");
    } else {
        record.p0       = get_int<int32_t>(in, "pos");
        record.pos_min  = get_int<int32_t>(in, "pos_min");
        record.pos_max  = get_int<int32_t>(in, "pos_max");
        record.n_tokens = get_int<int32_t>(in, "n_tokens");
        record.role     = in.at("role").get<std::string>();
    }
    return record;
}

std::vector<uint8_t> server_resume_manifest_encode(const server_resume_manifest & manifest) {
    json doc = {
        { "resume_key",        manifest.resume_key },
        { "family_digest",     manifest.family_digest },
        { "adapter_identity",  manifest.adapter_identity },
        { "n_tokens",          manifest.n_tokens },
        { "chunk_tokens",      manifest.chunk_tokens },
        { "saved_unix_ms",     manifest.saved_unix_ms },
        { "last_used_unix_ms", manifest.last_used_unix_ms },
        { "slot_hint",         manifest.slot_hint },
        { "chunks",            json::array() },
        { "tail_states",       json::array() },
        { "producers",         json::array() },
    };
    for (const auto & chunk : manifest.chunks) {
        doc["chunks"].push_back(object_record_to_json(chunk));
    }
    for (const auto & tail : manifest.tail_states) {
        doc["tail_states"].push_back(object_record_to_json(tail));
    }
    if (manifest.artifact) {
        doc[manifest.placed() ? "placement" : manifest.whole_sequence() ? "sequence" : "artifact"] = object_record_to_json(*manifest.artifact);
        doc["sequence_epoch"] = manifest.sequence_epoch;
    }
    if (manifest.shared_positions) {
        doc["shared_positions"] = true;
    }
    if (manifest.placed()) {
        doc["pool_entry"]      = manifest.pool_entry;
        doc["pool_generation"] = manifest.pool_generation;
        doc["pool_xxh3"]       = manifest.pool_xxh3;
    }
    for (const auto & producer : manifest.producers) {
        doc["producers"].push_back({
            { "model_name",   producer.model_name },
            { "model_file",   producer.model_file },
            { "mmproj_file",  producer.mmproj_file },
            { "weight_type",  producer.weight_type },
            { "build",        producer.build },
            { "cache_type_k", producer.cache_type_k },
            { "cache_type_v", producer.cache_type_v },
            { "adapters",     producer.adapters },
            { "host_unix_ms", producer.host_unix_ms },
        });
    }

    // provenance text comes from model files: never let a bad byte sequence throw here
    const std::string text = doc.dump(-1, ' ', false, json::error_handler_t::replace);

    std::vector<uint8_t> out(HEADER_SIZE + text.size() + manifest.ledger.size());
    std::memcpy(out.data() + HEADER_SIZE, text.data(), text.size());
    std::memcpy(out.data() + HEADER_SIZE + text.size(), manifest.ledger.data(), manifest.ledger.size());

    uint8_t * header = out.data();
    std::memcpy(header, MANIFEST_MAGIC, 8);
    put_u32(header +  8, manifest.shared_positions ? SERVER_RESUME_MEDIA_MANIFEST_VERSION : SERVER_RESUME_MANIFEST_VERSION);
    put_u32(header + 12, HEADER_SIZE);
    put_u32(header + 16, 0); // flags
    put_u32(header + 20, 0);
    put_u64(header + 24, text.size());
    put_u64(header + 32, manifest.ledger.size());
    put_u64(header + 40, manifest.generation);
    put_u64(header + 48, XXH3_64bits(out.data() + HEADER_SIZE, out.size() - HEADER_SIZE));
    header_seal(header);

    return out;
}

server_resume_reason server_resume_manifest_decode(
        const uint8_t * data, size_t size, server_resume_manifest & manifest, std::string & error) {
    using limits = server_resume_limits;

    manifest = {};

    if (size < HEADER_SIZE || size > limits::max_manifest_bytes || std::memcmp(data, MANIFEST_MAGIC, 8) != 0) {
        error = "not a resume manifest";
        return server_resume_reason::manifest_corrupt;
    }
    if (!header_sealed(data)) {
        error = "manifest header checksum";
        return server_resume_reason::manifest_corrupt;
    }
    const uint32_t version = get_u32(data + 8);
    if ((version != SERVER_RESUME_MANIFEST_VERSION && version != SERVER_RESUME_MEDIA_MANIFEST_VERSION) || get_u32(data + 12) != HEADER_SIZE ||
        get_u32(data + 16) != 0 || get_u32(data + 20) != 0) {
        error = "manifest version or flags";
        return server_resume_reason::format_unsupported;
    }

    const uint64_t json_bytes   = get_u64(data + 24);
    const uint64_t ledger_bytes = get_u64(data + 32);
    if (json_bytes == 0 || json_bytes > limits::max_json_bytes || ledger_bytes > limits::max_manifest_bytes ||
        HEADER_SIZE + json_bytes + ledger_bytes != size) {
        error = "manifest sizes";
        return server_resume_reason::manifest_corrupt;
    }
    if (get_u64(data + 48) != XXH3_64bits(data + HEADER_SIZE, size - HEADER_SIZE)) {
        error = "manifest checksum";
        return server_resume_reason::manifest_corrupt;
    }

    const char * text = (const char *) data + HEADER_SIZE;

    bool too_deep = false;
    const json::parser_callback_t depth_limit = [&](int depth, json::parse_event_t, json &) {
        too_deep |= depth > limits::max_json_depth;
        return !too_deep;
    };
    const json doc = json::parse(text, text + json_bytes, depth_limit, false);
    if (too_deep || !doc.is_object()) {
        error = "manifest json";
        return server_resume_reason::manifest_corrupt;
    }

    try {
        manifest.generation        = get_u64(data + 40);
        manifest.resume_key        = doc.at("resume_key").get<std::string>();
        manifest.family_digest     = doc.at("family_digest").get<std::string>();
        manifest.adapter_identity  = doc.at("adapter_identity").get<std::string>();
        manifest.n_tokens          = get_int<int32_t>(doc, "n_tokens");
        manifest.chunk_tokens      = get_int<int32_t>(doc, "chunk_tokens");
        manifest.saved_unix_ms     = get_int<int64_t>(doc, "saved_unix_ms");
        manifest.last_used_unix_ms = get_int<int64_t>(doc, "last_used_unix_ms");
        manifest.slot_hint         = get_int<int32_t>(doc, "slot_hint");
        manifest.shared_positions  = doc.value("shared_positions", false);
        if (manifest.shared_positions != (version == SERVER_RESUME_MEDIA_MANIFEST_VERSION)) {
            error = "position mode does not match manifest version";
            return server_resume_reason::manifest_corrupt;
        }

        const json & chunks      = doc.at("chunks");
        const json & tail_states = doc.at("tail_states");
        const json & producers   = doc.at("producers");
        if (!chunks.is_array() || !tail_states.is_array() || !producers.is_array() ||
            chunks.size() > limits::max_chunks || tail_states.size() > limits::max_tail_states ||
            producers.size() > limits::max_producers) {
            error = "record count out of bounds";
            return server_resume_reason::manifest_corrupt;
        }

        for (const auto & chunk : chunks) {
            manifest.chunks.push_back(object_record_from_json(chunk, server_resume_object_kind::base_chunk));
        }
        for (const auto & tail : tail_states) {
            manifest.tail_states.push_back(object_record_from_json(tail, server_resume_object_kind::tail_state));
        }
        if (doc.contains("artifact")) {
            manifest.artifact       = object_record_from_json(doc.at("artifact"), server_resume_object_kind::artifact);
            manifest.sequence_epoch = get_int<uint64_t>(doc, "sequence_epoch");
        } else if (doc.contains("sequence")) {
            manifest.artifact = object_record_from_json(doc.at("sequence"), server_resume_object_kind::sequence);
            manifest.sequence_epoch = get_int<uint64_t>(doc, "sequence_epoch");
        } else if (doc.contains("placement")) {
            manifest.artifact        = object_record_from_json(doc.at("placement"), server_resume_object_kind::placement);
            manifest.sequence_epoch  = get_int<uint64_t>(doc, "sequence_epoch");
            manifest.pool_entry      = doc.at("pool_entry").get<std::string>();
            manifest.pool_generation = get_int<uint64_t>(doc, "pool_generation");
            manifest.pool_xxh3       = get_int<uint64_t>(doc, "pool_xxh3");
        }
        for (const auto & in : producers) {
            server_resume_producer producer;
            producer.model_name   = in.at("model_name").get<std::string>();
            producer.model_file   = in.at("model_file").get<std::string>();
            producer.mmproj_file  = in.value("mmproj_file", std::string()); // absent before media was stored
            producer.weight_type = in.at("weight_type").get<std::string>();
            producer.build        = in.at("build").get<std::string>();
            producer.cache_type_k = in.at("cache_type_k").get<std::string>();
            producer.cache_type_v = in.at("cache_type_v").get<std::string>();
            producer.host_unix_ms = get_int<int64_t>(in, "host_unix_ms");

            const json & adapters = in.at("adapters");
            if (!adapters.is_array() || adapters.size() > MAX_ADAPTERS) {
                error = "producer adapters out of bounds";
                return server_resume_reason::manifest_corrupt;
            }
            producer.adapters = adapters.get<std::vector<std::string>>();

            manifest.producers.push_back(std::move(producer));
        }
    } catch (const std::exception & e) {
        error = std::string("manifest record: ") + e.what();
        return server_resume_reason::manifest_corrupt;
    }

    manifest.ledger.assign(data + HEADER_SIZE + json_bytes, data + size);

    if (!server_resume_manifest_validate(manifest, error)) {
        return server_resume_reason::manifest_corrupt;
    }

    return server_resume_reason::ok;
}

//
// store
//

static std::atomic<server_resume_fault_fn> g_fault{nullptr};

void server_resume_store_set_fault(server_resume_fault_fn fn) noexcept {
    g_fault.store(fn);
}

static server_resume_reason fault_at(const char * point) {
    const server_resume_fault_fn fn = g_fault.load();
    return fn ? fn(point) : server_resume_reason::ok;
}

std::string server_resume_store::new_entry_id() {
    std::random_device rd;
    std::string id;
    for (int i = 0; i < 4; ++i) {
        char part[9];
        std::snprintf(part, sizeof(part), "%08x", (unsigned) rd());
        id += part;
    }
    return id;
}

bool server_resume_store::entry_id_valid(const std::string & id) {
    return id.size() == 32 && is_hex(id, 32);
}

std::string server_resume_store::entry_dir(const std::string & id) const {
    return dir + "/entries/" + id;
}

static std::string object_name(const server_resume_object_record & record) {
    char name[96];
    if (record.kind == server_resume_object_kind::base_chunk) {
        std::snprintf(name, sizeof(name), "c-%" PRId32 "-%" PRId32 "-%" PRIu64, record.p0, record.p1, record.gen);
    } else if (record.kind == server_resume_object_kind::artifact) {
        std::snprintf(name, sizeof(name), "v-%" PRIu64, record.gen);
    } else if (record.kind == server_resume_object_kind::placement) {
        std::snprintf(name, sizeof(name), "p-%" PRIu64, record.gen);
    } else if (record.kind == server_resume_object_kind::sequence) {
        std::snprintf(name, sizeof(name), "s-%" PRIu64, record.gen);
    } else {
        std::snprintf(name, sizeof(name), "t-%" PRId32 "-%" PRIu64, record.p0, record.gen);
    }
    return name;
}

// the objects a manifest names, in the order an entry file holds them
static std::vector<const server_resume_object_record *> manifest_objects(const server_resume_manifest & manifest) {
    std::vector<const server_resume_object_record *> out;
    for (const auto & chunk : manifest.chunks) {
        out.push_back(&chunk);
    }
    for (const auto & tail : manifest.tail_states) {
        out.push_back(&tail);
    }
    if (manifest.artifact) {
        out.push_back(&*manifest.artifact);
    }
    return out;
}

#if defined(_WIN32)

std::unique_ptr<server_resume_store> server_resume_store::open(
        const std::string &, const std::string &, server_resume_reason & reason, std::string & error, bool) {
    reason = server_resume_reason::store_unwritable;
    error  = "the resume store is not implemented on this platform";
    return nullptr;
}

server_resume_store::~server_resume_store() = default;

std::vector<server_resume_entry> server_resume_store::list() const { return {}; }

server_resume_reason server_resume_store::read_manifest(const std::string &, server_resume_manifest &, std::string &) const {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::read_object(
        const std::string &, const server_resume_object_record &, std::vector<uint8_t> &, std::string &) const {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::write_object(
        const std::string &, server_resume_object_record &, const uint8_t *, size_t, std::string &) {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::write_object_stream(
        const std::string &, server_resume_object_record &, const std::function<bool(const put_fn &)> &, std::string &) {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::read_object_stream(
        const std::string &, const server_resume_object_record &, const std::function<bool(const get_fn &)> &, std::string &) const {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::commit(const std::string &, const server_resume_manifest &, std::string &) {
    return server_resume_reason::io_error;
}

void server_resume_store::sweep(const std::string &, const server_resume_manifest &) const {}

server_resume_reason server_resume_store::uncommit(const std::string &, std::string &) {
    return server_resume_reason::io_error;
}

void server_resume_store::remove_entry(const std::string &) const {}
std::set<std::string> server_resume_store::victims(const std::string &, size_t, size_t, const std::set<std::string> &, const std::set<std::string> &) const { return {}; }
void server_resume_store::prune(const std::string &, size_t, size_t, const std::set<std::string> &, const std::set<std::string> &) const {}
uint64_t server_resume_store::free_bytes() const { return 0; }

std::string server_resume_store::keep_value(const std::string &, const std::string &, std::string &) { return {}; }

server_resume_reason server_resume_store::export_entry(const std::string &, const std::string &, uint64_t &, std::string &) const {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::take_entry(const std::string &, std::string &) {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::export_taken(const std::string &, const std::string &, uint64_t &, std::string &) const {
    return server_resume_reason::io_error;
}

server_resume_reason server_resume_store::import_entry(
        const std::string &, std::string &, server_resume_manifest &, uint64_t &, std::string &) {
    return server_resume_reason::io_error;
}

bool server_resume_store::is_entry_file(const std::string &) { return false; }

#else

struct fd_guard {
    int fd = -1;

    explicit fd_guard(int fd) : fd(fd) {}
    ~fd_guard() { reset(); }

    fd_guard(const fd_guard &) = delete;
    fd_guard & operator=(const fd_guard &) = delete;

    void reset() {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }
};

// An imported entry file: its manifest and where each object's header is. The descriptor keeps
// the file that was verified even if the path is replaced meanwhile.
struct server_resume_store::mounted_entry {
    std::string                     path;
    fd_guard                        file{-1};
    server_resume_manifest          manifest;
    std::map<std::string, uint64_t> offsets; // by object name
};

const server_resume_store::mounted_entry * server_resume_store::mounted(const std::string & id) const {
    const auto it = mounts.find(id);
    return it == mounts.end() ? nullptr : it->second.get();
}

static server_resume_reason reason_from_errno(int err) {
    return err == ENOSPC || err == EDQUOT ? server_resume_reason::no_space : server_resume_reason::io_error;
}

static std::string errno_text(const char * what, const std::string & path, int err) {
    return std::string(what) + " " + path + ": " + std::strerror(err);
}

// a directory only its owner can enter. an existing one must be ours and is narrowed to 0700
static bool make_private_dir(const std::string & path, std::string & error) {
    if (mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
        error = errno_text("cannot create", path, errno);
        return false;
    }
    struct stat st;
    if (lstat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != geteuid()) {
        error = path + " is not a directory of this user";
        return false;
    }
    if ((st.st_mode & 0777) != 0700 && chmod(path.c_str(), 0700) != 0) {
        error = errno_text("cannot restrict", path, errno);
        return false;
    }
    return true;
}

static bool sync_dir(const std::string & path) {
    fd_guard dir(open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    return dir.fd >= 0 && fsync(dir.fd) == 0;
}

bool server_resume_store::sync_path(const std::string & path) const {
    return !durable || sync_dir(path);
}

// the payload checksum of an object that is never in one buffer
struct xxh3_stream {
    XXH3_state_t * state = XXH3_createState();

    xxh3_stream() {
        if (state && XXH3_64bits_reset(state) != XXH_OK) {
            XXH3_freeState(state);
            state = nullptr;
        }
    }
    ~xxh3_stream() { XXH3_freeState(state); }

    xxh3_stream(const xxh3_stream &) = delete;
    xxh3_stream & operator=(const xxh3_stream &) = delete;

    bool update(const uint8_t * data, size_t size) {
        return state && XXH3_64bits_update(state, data, size) == XXH_OK;
    }
    uint64_t digest() const { return XXH3_64bits_digest(state); }
};

static bool write_all(int fd, const uint8_t * data, size_t size) {
    while (size > 0) {
        const ssize_t n = write(fd, data, std::min<size_t>(size, 16u * 1024 * 1024));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        data += n;
        size -= (size_t) n;
    }
    return true;
}

static bool read_all(int fd, uint8_t * data, size_t size) {
    while (size > 0) {
        const ssize_t n = read(fd, data, size);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        data += n;
        size -= (size_t) n;
    }
    return true;
}

// A file written under a staging name, synced, then renamed to its name: a reader never sees a part.
// Whatever is not published is removed, except after an injected fault, which is a kill.
struct staged_file {
    const std::string directory;
    const std::string path;
    const bool        durable; // else it is renamed unsynced
    fd_guard          file;
    size_t            unsynced = 0;
    bool              keep     = false;
    int               err      = 0; // of the write that failed

    // a streamed producer puts field by field: small puts leave as one write
    static constexpr size_t coalesce = 1u << 20;
    std::vector<uint8_t>    pending;

    explicit staged_file(const std::string & directory, bool durable = true) :
        directory(directory), path(directory + "/tmp-" + server_resume_store::new_entry_id()), durable(durable),
        file(open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600)) {}

    ~staged_file() {
        file.reset();
        if (!keep) {
            unlink(path.c_str());
        }
    }

    // of the write that failed, else of the call that just failed
    server_resume_reason failed(const char * what, std::string & error) const {
        const int e = err != 0 ? err : errno;
        error = errno_text(what, path, e);
        return reason_from_errno(e);
    }

    bool put(const uint8_t * data, size_t size) {
        if (pending.size() + size > coalesce && !flush()) {
            return false;
        }
        if (size >= coalesce) {
            return write_synced(data, size);
        }
        pending.insert(pending.end(), data, data + size);
        return true;
    }

    bool flush() {
        const bool ok = write_synced(pending.data(), pending.size());
        pending.clear();
        return ok;
    }

    // bound the dirty pages of a large object instead of leaving them all to the final sync
    static constexpr size_t sync_every = 64u * 1024 * 1024;

    bool written(size_t n) {
        unsynced += n;
        if (unsynced == sync_every) {
            if (durable && fdatasync(file.fd) != 0) {
                err = errno;
                return false;
            }
            unsynced = 0;
        }
        return true;
    }

    bool write_synced(const uint8_t * data, size_t size) {
        while (size > 0) {
            const size_t n = std::min(size, sync_every - unsynced);
            if (!write_all(file.fd, data, n)) {
                err = errno;
                return false;
            }
            data += n;
            size -= n;
            if (!written(n)) {
                return false;
            }
        }
        return true;
    }

    // What it can of `size` bytes from where `from` is, copied by the kernel after a flush(): no
    // pass through user memory, and a clone where the file system shares extents. `size` is left
    // at what the caller still has to copy; a failed copy is left to that one to report.
    bool copy_in_kernel(int from, uint64_t & size) {
#if defined(__linux__)
        while (size > 0) {
            const size_t  n    = (size_t) std::min<uint64_t>(size, sync_every - unsynced);
            const ssize_t done = copy_file_range(from, nullptr, file.fd, nullptr, n, 0);
            if (done < 0 && errno == EINTR) {
                continue;
            }
            if (done <= 0) {
                break;
            }
            size -= (uint64_t) done;
            if (!written((size_t) done)) {
                return false;
            }
        }
#else
        (void) from;
        (void) size;
#endif
        return true;
    }

    // over the bytes reserved for it at the start of the file, after a flush
    bool put_header(const uint8_t * header) {
        for (size_t done = 0; done < HEADER_SIZE; ) {
            const ssize_t n = pwrite(file.fd, header + done, HEADER_SIZE - done, (off_t) done);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                return false;
            }
            done += (size_t) n;
        }
        return true;
    }

    server_resume_reason publish(const std::string & name, const char * fault_point, std::string & error) {
        if (!flush()) {
            return failed("cannot write", error);
        }
        if (durable && fsync(file.fd) != 0) {
            return failed("cannot sync", error);
        }
        file.reset();

        const server_resume_reason injected = fault_at(fault_point);
        if (injected != server_resume_reason::ok) {
            keep  = true;
            error = std::string("injected fault at ") + fault_point;
            return injected;
        }

        if (rename(path.c_str(), (directory + "/" + name).c_str()) != 0) {
            return failed("cannot publish", error);
        }
        keep = true;
        return server_resume_reason::ok;
    }
};

std::unique_ptr<server_resume_store> server_resume_store::open(
        const std::string & cache_root, const std::string & family_digest,
        server_resume_reason & reason, std::string & error, bool durable) {
    reason = server_resume_reason::store_unwritable;

    if (!is_hex(family_digest, 64)) {
        error = "family digest is not hex";
        return nullptr;
    }

    std::string root = cache_root;
    while (root.size() > 1 && root.back() == '/') {
        root.pop_back();
    }

    std::error_code ec;
    fs::create_directories(root, ec);

    std::unique_ptr<server_resume_store> store(new server_resume_store());
    store->dir     = root + "/resume/" + family_digest;
    store->durable = durable;

    if (!make_private_dir(root + "/resume", error) || !make_private_dir(store->dir, error) ||
        !make_private_dir(store->dir + "/entries", error)) {
        return nullptr;
    }

    const std::string lock_path = store->dir + "/writer.lock";
    store->lock_fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (store->lock_fd < 0) {
        error = errno_text("cannot open", lock_path, errno);
        return nullptr;
    }
    // the kernel drops the lock with its owner, so a killed server leaves nothing stale behind
    if (flock(store->lock_fd, LOCK_EX | LOCK_NB) != 0) {
        reason = errno == EWOULDBLOCK ? server_resume_reason::store_locked : server_resume_reason::store_unwritable;
        error  = errno_text("cannot lock", lock_path, errno);
        return nullptr;
    }

    // who holds it, for a person looking at the directory
    const std::string owner = std::to_string((long long) getpid()) + "\n";
    if (ftruncate(store->lock_fd, 0) == 0) {
        (void) !write(store->lock_fd, owner.data(), owner.size());
    }

    // entries taken out by a writer that died before exporting them
    fs::remove_all(store->dir + "/taken", ec);

    reason = server_resume_reason::ok;
    return store;
}

server_resume_store::~server_resume_store() {
    if (lock_fd >= 0) {
        close(lock_fd);
    }
}

// the manifest the entry directory commits
static server_resume_reason read_commit(
        const std::string & directory, server_resume_manifest & manifest, std::string & error) {
    const std::string path = directory + "/commit";

    fd_guard file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
    if (file.fd < 0) {
        error = errno_text("cannot open", path, errno);
        return errno == ENOENT ? server_resume_reason::object_missing : server_resume_reason::io_error;
    }

    struct stat st;
    if (fstat(file.fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        error = "cannot stat " + path;
        return server_resume_reason::io_error;
    }
    if (st.st_size < (off_t) HEADER_SIZE || (uint64_t) st.st_size > server_resume_limits::max_manifest_bytes) {
        error = "manifest file size out of bounds";
        return server_resume_reason::manifest_corrupt;
    }

    std::vector<uint8_t> data((size_t) st.st_size);
    if (!read_all(file.fd, data.data(), data.size())) {
        error = "cannot read " + path;
        return server_resume_reason::io_error;
    }

    return server_resume_manifest_decode(data.data(), data.size(), manifest, error);
}

server_resume_reason server_resume_store::read_manifest(
        const std::string & id, server_resume_manifest & manifest, std::string & error) const {
    if (!entry_id_valid(id)) {
        error = "entry id is not valid";
        return server_resume_reason::object_missing;
    }
    if (const auto entry = mounted(id)) {
        manifest = entry->manifest;
        return server_resume_reason::ok;
    }
    return read_commit(entry_dir(id), manifest, error);
}

std::vector<server_resume_entry> server_resume_store::list() const {
    std::vector<server_resume_entry> entries;

    std::error_code ec;
    size_t n_seen = 0;
    for (fs::directory_iterator it(dir + "/entries", ec), end; !ec && it != end; it.increment(ec)) {
        if (++n_seen > server_resume_limits::max_entries_listed) {
            break;
        }

        server_resume_entry entry;
        entry.id = it->path().filename().string();
        if (!entry_id_valid(entry.id) || !it->is_directory(ec) || it->is_symlink(ec)) {
            continue;
        }

        entry.reason = read_manifest(entry.id, entry.manifest, entry.error);
        if (entry.reason == server_resume_reason::object_missing) {
            continue; // never committed, or taken out for a replacement
        }
        entries.push_back(std::move(entry));
    }

    std::stable_sort(entries.begin(), entries.end(), [](const server_resume_entry & a, const server_resume_entry & b) {
        const bool a_ok = a.reason == server_resume_reason::ok;
        const bool b_ok = b.reason == server_resume_reason::ok;
        if (a_ok != b_ok) {
            return a_ok;
        }
        return a.manifest.last_used_unix_ms > b.manifest.last_used_unix_ms;
    });

    return entries;
}

static bool pread_all(int fd, uint8_t * data, size_t size, uint64_t offset) {
    while (size > 0) {
        const ssize_t n = pread(fd, data, size, (off_t) offset);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        data   += n;
        size   -= (size_t) n;
        offset += (uint64_t) n;
    }
    return true;
}

// the header of the record's object at `offset` of the file, verified against the record
static server_resume_reason object_at(
        int fd, uint64_t offset, const server_resume_object_record & record, const std::string & path,
        std::string & error) {
    uint8_t header[HEADER_SIZE];
    if (!pread_all(fd, header, HEADER_SIZE, offset)) {
        error = "cannot read " + path;
        return server_resume_reason::io_error;
    }
    if (std::memcmp(header, OBJECT_MAGIC, 8) != 0 || !header_sealed(header)) {
        error = "object header: " + path;
        return server_resume_reason::object_checksum_mismatch;
    }
    if (get_u32(header + 8) != SERVER_RESUME_OBJECT_VERSION || get_u32(header + 12) != HEADER_SIZE || get_u32(header + 20) != 0) {
        error = "object version or flags: " + path;
        return server_resume_reason::format_unsupported;
    }
    if (get_u64(header + 24) != record.bytes) {
        error = "payload size differs from the record: " + path;
        return server_resume_reason::object_size_mismatch;
    }
    if (get_u32(header + 16) != (uint32_t) record.kind || get_u64(header + 32) != record.xxh3 ||
        get_u32(header + 40) != (uint32_t) record.p0 || get_u32(header + 44) != (uint32_t) record.p1 ||
        get_u64(header + 48) != record.gen) {
        error = "object header differs from the record: " + path;
        return server_resume_reason::object_checksum_mismatch;
    }

    return server_resume_reason::ok;
}

// the object of the record in its own file, its header verified against it
static server_resume_reason open_object(
        const std::string & path, const server_resume_object_record & record, fd_guard & file, std::string & error) {
    file.reset();
    file.fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (file.fd < 0) {
        error = errno_text("cannot open", path, errno);
        return errno == ENOENT ? server_resume_reason::object_missing : server_resume_reason::io_error;
    }

    struct stat st;
    if (fstat(file.fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        error = "cannot stat " + path;
        return server_resume_reason::io_error;
    }
    if ((uint64_t) st.st_size != HEADER_SIZE + record.bytes) {
        error = "file size differs from the record: " + path;
        return server_resume_reason::object_size_mismatch;
    }
    return object_at(file.fd, 0, record, path, error);
}

server_resume_reason server_resume_store::read_object(
        const std::string & id, const server_resume_object_record & record,
        std::vector<uint8_t> & payload, std::string & error) const {
    if (record.bytes > server_resume_limits::max_object_bytes) {
        error = "object record out of bounds";
        return server_resume_reason::object_size_mismatch;
    }
    return read_object_stream(id, record, [&](const get_fn & get) {
        payload.resize((size_t) record.bytes);
        return get(payload.data(), payload.size());
    }, error);
}

server_resume_reason server_resume_store::read_object_stream(
        const std::string & id, const server_resume_object_record & record,
        const std::function<bool(const get_fn & get)> & consume, std::string & error) const {
    if (!entry_id_valid(id) || record.bytes == 0 || record.bytes > server_resume_limits::max_bytes(record.kind)) {
        error = "object record out of bounds";
        return server_resume_reason::object_size_mismatch;
    }

    const auto entry = mounted(id);
    std::string path;
    fd_guard file(-1);
    int fd = -1;
    uint64_t offset = 0; // of the object's header
    server_resume_reason opened;
    if (entry) {
        path = entry->path;
        const auto at = entry->offsets.find(object_name(record));
        if (at == entry->offsets.end()) {
            error = "the entry file holds no such object: " + path;
            return server_resume_reason::object_missing;
        }
        fd     = entry->file.fd;
        offset = at->second;
        opened = object_at(fd, offset, record, path, error);
    } else {
        path   = entry_dir(id) + "/" + object_name(record);
        opened = open_object(path, record, file, error);
        fd     = file.fd;
    }
    if (opened != server_resume_reason::ok) {
        return opened;
    }
    offset += HEADER_SIZE;

    xxh3_stream checksum;
    if (!checksum.state) {
        error = "out of memory";
        return server_resume_reason::io_error;
    }
    // a streamed consumer gets field by field: small gets come out of one read ahead
    constexpr size_t     read_ahead = 1u << 20;
    std::vector<uint8_t> ahead;
    size_t               ahead_pos = 0;
    uint64_t             unread    = record.bytes; // of the file
    uint64_t             left      = record.bytes; // of the consumer
    bool                 read_fail = false;

    const auto fill = [&](uint8_t * data, size_t size) {
        read_fail = !pread_all(fd, data, size, offset);
        offset += size;
        unread -= size;
        return !read_fail;
    };
    const get_fn get = [&](uint8_t * data, size_t size) {
        if (size > left) {
            return false;
        }
        left -= size;
        uint8_t * out = data;
        for (size_t need = size; need > 0; ) {
            if (ahead_pos == ahead.size()) {
                if (need >= read_ahead) {
                    if (!fill(out, need)) {
                        return false;
                    }
                    break;
                }
                ahead.resize((size_t) std::min<uint64_t>(read_ahead, unread));
                ahead_pos = 0;
                if (!fill(ahead.data(), ahead.size())) {
                    return false;
                }
            }
            const size_t n = std::min(need, ahead.size() - ahead_pos);
            std::memcpy(out, ahead.data() + ahead_pos, n);
            ahead_pos += n;
            out       += n;
            need      -= n;
        }
        return checksum.update(data, size);
    };

    const bool consumed = consume(get);
    if (read_fail) {
        error = "cannot read " + path;
        return server_resume_reason::io_error;
    }
    if (!consumed || left != 0) {
        error = "payload not accepted: " + path;
        return server_resume_reason::object_checksum_mismatch;
    }
    if (checksum.digest() != record.xxh3) {
        error = "payload checksum: " + path;
        return server_resume_reason::object_checksum_mismatch;
    }

    return server_resume_reason::ok;
}

server_resume_reason server_resume_store::write_object(
        const std::string & id, server_resume_object_record & record,
        const uint8_t * payload, size_t size, std::string & error) {
    if (size == 0 || size > server_resume_limits::max_object_bytes) {
        error = "object out of bounds";
        return server_resume_reason::io_error;
    }
    return write_object_stream(id, record, [&](const put_fn & put) { return put(payload, size); }, error);
}

server_resume_reason server_resume_store::write_object_stream(
        const std::string & id, server_resume_object_record & record,
        const std::function<bool(const put_fn & put)> & produce, std::string & error) {
    if (!entry_id_valid(id) || record.gen == 0) {
        error = "object out of bounds";
        return server_resume_reason::io_error;
    }

    const std::string directory = entry_dir(id);
    struct stat st;
    const bool is_new = lstat(directory.c_str(), &st) != 0;
    if (!make_private_dir(directory, error)) {
        return server_resume_reason::store_unwritable;
    }
    if (is_new && !sync_path(dir + "/entries")) {
        error = "cannot sync " + dir + "/entries";
        return server_resume_reason::io_error;
    }

    staged_file staged(directory, durable);
    if (staged.file.fd < 0) {
        return staged.failed("cannot create", error);
    }

    // the header holds the size and the checksum, so it is written over its place once they are known
    uint8_t header[HEADER_SIZE] = {};
    if (!staged.put(header, HEADER_SIZE)) {
        return staged.failed("cannot write", error);
    }

    xxh3_stream checksum;
    if (!checksum.state) {
        error = "out of memory";
        return server_resume_reason::io_error;
    }
    uint64_t bytes = 0;

    const put_fn put = [&](const uint8_t * data, size_t size) {
        if (size > server_resume_limits::max_bytes(record.kind) - bytes || !staged.put(data, size)) {
            return false;
        }
        bytes += size;
        return checksum.update(data, size);
    };

    const bool produced = produce(put) && staged.flush();
    if (staged.err != 0) {
        return staged.failed("cannot write", error);
    }
    if (!produced || bytes == 0) {
        error = "object payload was not produced";
        return server_resume_reason::io_error;
    }

    // the seam for a write that stops part-way: what is on disk is a staging file without its header
    const server_resume_reason injected = fault_at("object_write");
    if (injected != server_resume_reason::ok) {
        staged.keep = true;
        error = "injected fault at object_write";
        return injected;
    }

    record.bytes = bytes;
    record.xxh3  = checksum.digest();

    std::memcpy(header, OBJECT_MAGIC, 8);
    put_u32(header +  8, SERVER_RESUME_OBJECT_VERSION);
    put_u32(header + 12, HEADER_SIZE);
    put_u32(header + 16, (uint32_t) record.kind);
    put_u32(header + 20, 0); // flags
    put_u64(header + 24, record.bytes);
    put_u64(header + 32, record.xxh3);
    put_u32(header + 40, (uint32_t) record.p0);
    put_u32(header + 44, (uint32_t) record.p1);
    put_u64(header + 48, record.gen);
    header_seal(header);

    if (!staged.put_header(header)) {
        return staged.failed("cannot write", error);
    }
    return staged.publish(object_name(record), "object_publish", error);
}

server_resume_reason server_resume_store::commit(
        const std::string & id, const server_resume_manifest & manifest, std::string & error) {
    if (!entry_id_valid(id) || mounted(id)) {
        error = "entry id is not valid";
        return server_resume_reason::io_error;
    }
    if (!server_resume_manifest_validate(manifest, error)) {
        return server_resume_reason::manifest_corrupt;
    }

    const std::string directory = entry_dir(id);

    // the objects have to be durable under their names before a manifest names them
    if (!sync_path(directory)) {
        error = "cannot sync " + directory;
        return server_resume_reason::io_error;
    }

    const std::vector<uint8_t> data = server_resume_manifest_encode(manifest);
    if (data.size() > server_resume_limits::max_manifest_bytes) {
        error = "manifest too large";
        return server_resume_reason::manifest_corrupt;
    }

    staged_file staged(directory, durable);
    if (staged.file.fd < 0) {
        return staged.failed("cannot create", error);
    }
    if (!staged.put(data.data(), data.size())) {
        return staged.failed("cannot write", error);
    }
    const server_resume_reason reason = staged.publish("commit", "manifest_publish", error);
    if (reason != server_resume_reason::ok) {
        return reason;
    }
    if (!sync_path(directory)) {
        error = "cannot sync " + directory;
        return server_resume_reason::io_error;
    }
    return server_resume_reason::ok;
}

void server_resume_store::sweep(const std::string & id, const server_resume_manifest & manifest) const {
    if (!entry_id_valid(id)) {
        return;
    }

    std::set<std::string> keep = { "commit" };
    for (const auto * object : manifest_objects(manifest)) {
        keep.insert(object_name(*object));
    }

    std::error_code ec;
    std::vector<fs::path> drop;
    for (fs::directory_iterator it(entry_dir(id), ec), end; !ec && it != end; it.increment(ec)) {
        if (!keep.count(it->path().filename().string())) {
            drop.push_back(it->path());
        }
    }
    for (const auto & path : drop) {
        fs::remove(path, ec);
    }
}

server_resume_reason server_resume_store::uncommit(const std::string & id, std::string & error) {
    if (!entry_id_valid(id)) {
        error = "entry id is not valid";
        return server_resume_reason::io_error;
    }
    if (mounted(id)) {
        return server_resume_reason::ok; // nothing of it is in the store
    }
    const std::string directory = entry_dir(id);
    const auto injected = fault_at("uncommit");
    if (injected != server_resume_reason::ok) {
        error = "injected fault at uncommit";
        return injected;
    }
    if (unlink((directory + "/commit").c_str()) != 0 && errno != ENOENT) {
        error = errno_text("cannot remove", directory + "/commit", errno);
        return server_resume_reason::io_error;
    }
    const auto sync_failure = fault_at("uncommit_sync");
    if (sync_failure != server_resume_reason::ok) {
        error = "injected fault at uncommit_sync";
        return sync_failure;
    }
    if (!sync_path(directory)) {
        // A known entry may already have been removed by retention. Erasing it is
        // idempotent, but still make that absence durable before reporting success.
        if (errno == ENOENT && sync_path(dir + "/entries")) {
            return server_resume_reason::ok;
        }
        error = "cannot sync " + directory;
        return server_resume_reason::io_error;
    }
    return server_resume_reason::ok;
}

void server_resume_store::remove_entry(const std::string & id) const {
    if (!entry_id_valid(id)) {
        return;
    }
    if (mounts.erase(id) != 0) {
        return;
    }
    // the manifest first: an interrupted removal must not leave an entry that names missing objects
    const std::string directory = entry_dir(id);
    unlink((directory + "/commit").c_str());
    sync_path(directory);

    std::error_code ec;
    fs::remove_all(directory, ec);
}

// what prune() keeps of the listed entries. The held ones first: the overall bound takes others
static std::set<std::string> retained(
        const std::vector<server_resume_entry> & entries,
        const std::string & resume_key, size_t n_keep_key, size_t n_keep_total,
        const std::set<std::string> & held, const std::set<std::string> & live) {
    std::set<std::string> keep;
    size_t n_of_key = 0;
    for (const int priority : {0, 1, 2}) {
        for (const auto & entry : entries) {
            const int entry_priority = live.count(entry.id) ? 0 : held.count(entry.id) ? 1 : 2;
            // a manifest of a newer format may be of use to the server that wrote it, a damaged one to nobody
            if (entry.reason == server_resume_reason::manifest_corrupt || keep.size() >= n_keep_total ||
                priority != entry_priority) {
                continue;
            }
            if (priority < 2 || entry.manifest.resume_key != resume_key || n_of_key++ < n_keep_key) {
                keep.insert(entry.id);
            }
        }
    }
    return keep;
}

std::set<std::string> server_resume_store::victims(
        const std::string & resume_key, size_t n_keep_key, size_t n_keep_total,
        const std::set<std::string> & held, const std::set<std::string> & live) const {
    const auto entries = list();
    const auto keep = retained(entries, resume_key, n_keep_key, n_keep_total, held, live);
    std::set<std::string> drop;
    for (const auto & entry : entries) {
        if (!keep.count(entry.id)) {
            drop.insert(entry.id);
        }
    }
    return drop;
}

void server_resume_store::prune(
        const std::string & resume_key, size_t n_keep_key, size_t n_keep_total,
        const std::set<std::string> & held, const std::set<std::string> & live) const {
    const auto keep = retained(list(), resume_key, n_keep_key, n_keep_total, held, live);

    std::error_code ec;
    std::vector<std::string> drop;
    size_t n_seen = 0;
    for (fs::directory_iterator it(dir + "/entries", ec), end; !ec && it != end; it.increment(ec)) {
        if (++n_seen > server_resume_limits::max_entries_listed) {
            break;
        }
        const std::string id = it->path().filename().string();
        if (entry_id_valid(id) && !keep.count(id)) {
            drop.push_back(id);
        }
    }
    for (const auto & id : drop) {
        remove_entry(id);
    }
}

uint64_t server_resume_store::free_bytes() const {
    struct statvfs vfs;
    if (statvfs(dir.c_str(), &vfs) != 0) {
        return 0;
    }
    return (uint64_t) vfs.f_bavail * vfs.f_frsize;
}

std::string server_resume_store::keep_value(const std::string & name, const std::string & fresh, std::string & error) {
    const auto valid = [](const std::string & value) {
        return !value.empty() && value.size() <= MAX_STRING &&
            std::all_of(value.begin(), value.end(), [](char c) { return c > ' ' && c < 0x7f; });
    };
    if (name.empty() || !std::all_of(name.begin(), name.end(), [](char c) { return (c >= 'a' && c <= 'z') || c == '-'; })) {
        error = "value name is not valid";
        return {};
    }

    const std::string path = dir + "/" + name;
    {
        fd_guard file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
        char text[MAX_STRING + 1];
        const ssize_t n = file.fd < 0 ? -1 : read(file.fd, text, sizeof(text));
        std::string stored(text, (size_t) std::max<ssize_t>(n, 0));
        if (valid(stored)) {
            return stored;
        }
    }
    if (!valid(fresh)) {
        error = "value is not storable";
        return {};
    }

    staged_file staged(dir);
    if (staged.file.fd < 0) {
        staged.failed("cannot create", error);
        return {};
    }
    if (!staged.put((const uint8_t *) fresh.data(), fresh.size())) {
        staged.failed("cannot write", error);
        return {};
    }
    if (staged.publish(name, "value_publish", error) != server_resume_reason::ok || !sync_dir(dir)) {
        if (error.empty()) {
            error = "cannot sync " + dir;
        }
        return {};
    }
    return fresh;
}

// `size` bytes from where `from` is into the staged file
static bool copy_range(int from, staged_file & to, uint64_t size) {
    if (!to.flush() || !to.copy_in_kernel(from, size)) {
        return false;
    }
    std::vector<uint8_t> buffer((size_t) std::min<uint64_t>(size, 8u << 20));
    while (size > 0) {
        const size_t n = (size_t) std::min<uint64_t>(size, buffer.size());
        if (!read_all(from, buffer.data(), n) || !to.put(buffer.data(), n)) {
            return false;
        }
        size -= n;
    }
    return true;
}

// what a failed copy_range() says: a write fails in the staged file, else it was the read
static server_resume_reason copy_failed(const staged_file & to, const std::string & from, std::string & error) {
    if (to.err != 0) {
        return to.failed("cannot write", error);
    }
    error = "cannot read " + from;
    return server_resume_reason::io_error;
}

// Entry file: a header, the manifest as a commit holds it, then each object of manifest_objects()
// as its file in the entry holds it, header and payload.
//
//   0  magic "BUUNRSME"   8 u32 version   12 u32 header size   16 u32 objects   20 u32 flags (0)
//  24  u64 manifest bytes  32 u64 file bytes  40..55 zero  56 u64 xxh3 of bytes 0..55
static uint64_t entry_file_bytes(uint64_t manifest_bytes, const std::vector<const server_resume_object_record *> & objects) {
    uint64_t bytes = HEADER_SIZE + manifest_bytes;
    for (const auto * object : objects) {
        bytes += HEADER_SIZE + object->bytes;
    }
    return bytes;
}

static bool entry_file_whole(const server_resume_manifest & manifest, std::string & error) {
    if (manifest.placed()) {
        error = "a placed entry is not whole without the entry of its pool";
        return false;
    }
    return true;
}

// the entry committed in `entry`, as one file at `path`
static server_resume_reason export_directory(
        const std::string & entry, const std::string & path, uint64_t & bytes, std::string & error) {
    server_resume_manifest manifest;
    auto reason = read_commit(entry, manifest, error);
    if (reason != server_resume_reason::ok) {
        return reason;
    }
    if (!entry_file_whole(manifest, error)) {
        return server_resume_reason::format_unsupported;
    }
    const std::vector<uint8_t> encoded = server_resume_manifest_encode(manifest);
    const auto objects = manifest_objects(manifest);

    bytes = entry_file_bytes(encoded.size(), objects);
    uint8_t header[HEADER_SIZE] = {};
    std::memcpy(header, ENTRY_MAGIC, 8);
    put_u32(header +  8, ENTRY_VERSION);
    put_u32(header + 12, HEADER_SIZE);
    put_u32(header + 16, (uint32_t) objects.size());
    put_u64(header + 24, encoded.size());
    put_u64(header + 32, bytes);
    header_seal(header);

    const fs::path target(path);
    const std::string directory = target.has_parent_path() ? target.parent_path().string() : ".";
    staged_file staged(directory);
    if (staged.file.fd < 0) {
        return staged.failed("cannot create", error);
    }
    if (!staged.put(header, HEADER_SIZE) || !staged.put(encoded.data(), encoded.size())) {
        return staged.failed("cannot write", error);
    }
    for (const auto * object : objects) {
        const std::string from = entry + "/" + object_name(*object);
        fd_guard file(-1);
        reason = open_object(from, *object, file, error);
        if (reason != server_resume_reason::ok) {
            return reason;
        }
        if (lseek(file.fd, 0, SEEK_SET) != 0 || !copy_range(file.fd, staged, HEADER_SIZE + object->bytes)) {
            return copy_failed(staged, from, error);
        }
    }
    reason = staged.publish(target.filename().string(), "entry_export", error);
    if (reason == server_resume_reason::ok && !sync_dir(directory)) {
        error = "cannot sync " + directory;
        return server_resume_reason::io_error;
    }
    return reason;
}

server_resume_reason server_resume_store::export_entry(
        const std::string & id, const std::string & path, uint64_t & bytes, std::string & error) const {
    if (!entry_id_valid(id) || mounted(id)) {
        error = "no such entry " + id;
        return server_resume_reason::object_missing;
    }
    return export_directory(entry_dir(id), path, bytes, error);
}

server_resume_reason server_resume_store::take_entry(const std::string & id, std::string & error) {
    if (!entry_id_valid(id) || mounted(id)) {
        error = "no such entry " + id;
        return server_resume_reason::object_missing;
    }
    if (!make_private_dir(dir + "/taken", error)) {
        return server_resume_reason::store_unwritable;
    }
    if (rename(entry_dir(id).c_str(), (dir + "/taken/" + id).c_str()) != 0) {
        const int e = errno;
        error = errno_text("cannot take", entry_dir(id), e);
        return e == ENOENT ? server_resume_reason::object_missing : reason_from_errno(e);
    }
    return server_resume_reason::ok;
}

server_resume_reason server_resume_store::export_taken(
        const std::string & id, const std::string & path, uint64_t & bytes, std::string & error) const {
    if (!entry_id_valid(id)) {
        error = "no such entry " + id;
        return server_resume_reason::object_missing;
    }
    const std::string taken = dir + "/taken/" + id;
    const auto reason = export_directory(taken, path, bytes, error);
    std::error_code ec;
    fs::remove_all(taken, ec);
    return reason;
}

server_resume_reason server_resume_store::import_entry(
        const std::string & path, std::string & id, server_resume_manifest & manifest, uint64_t & bytes,
        std::string & error) {
    id.clear();
    bytes = 0;
    fd_guard file(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (file.fd < 0) {
        error = errno_text("cannot open", path, errno);
        return errno == ENOENT ? server_resume_reason::object_missing : server_resume_reason::io_error;
    }
    struct stat st;
    if (fstat(file.fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        error = "cannot stat " + path;
        return server_resume_reason::io_error;
    }
    const uint64_t size = (uint64_t) st.st_size;

    uint8_t header[HEADER_SIZE];
    if (size < HEADER_SIZE || !read_all(file.fd, header, HEADER_SIZE) ||
        std::memcmp(header, ENTRY_MAGIC, 8) != 0 || !header_sealed(header)) {
        error = "entry file header: " + path;
        return server_resume_reason::manifest_corrupt;
    }
    if (get_u32(header + 8) != ENTRY_VERSION || get_u32(header + 12) != HEADER_SIZE || get_u32(header + 20) != 0) {
        error = "entry file version or flags: " + path;
        return server_resume_reason::format_unsupported;
    }
    const uint64_t manifest_bytes = get_u64(header + 24);
    if (get_u64(header + 32) != size || manifest_bytes > server_resume_limits::max_manifest_bytes ||
        manifest_bytes > size - HEADER_SIZE) {
        error = "entry file size differs from its header: " + path;
        return server_resume_reason::object_size_mismatch;
    }
    std::vector<uint8_t> encoded((size_t) manifest_bytes);
    if (!read_all(file.fd, encoded.data(), encoded.size())) {
        error = "cannot read " + path;
        return server_resume_reason::io_error;
    }
    auto reason = server_resume_manifest_decode(encoded.data(), encoded.size(), manifest, error);
    if (reason != server_resume_reason::ok) {
        return reason;
    }
    if (!entry_file_whole(manifest, error)) {
        return server_resume_reason::format_unsupported;
    }
    const auto objects = manifest_objects(manifest);
    if (get_u32(header + 16) != objects.size() || entry_file_bytes(manifest_bytes, objects) != size) {
        error = "entry file size differs from its manifest: " + path;
        return server_resume_reason::object_size_mismatch;
    }

    auto entry = std::make_unique<mounted_entry>();
    uint64_t offset = HEADER_SIZE + manifest_bytes;
    for (const auto * object : objects) {
        // the header against the record; the payload is checked as it is read
        reason = object_at(file.fd, offset, *object, path, error);
        if (reason != server_resume_reason::ok) {
            return reason;
        }
        if (!entry->offsets.emplace(object_name(*object), offset).second) {
            error = "entry file names an object twice: " + path;
            return server_resume_reason::manifest_corrupt;
        }
        offset += HEADER_SIZE + object->bytes;
    }
    entry->path     = path;
    entry->manifest = manifest;
    std::swap(entry->file.fd, file.fd);

    id = new_entry_id();
    mounts.emplace(id, std::move(entry));
    bytes = size;
    return server_resume_reason::ok;
}

bool server_resume_store::is_entry_file(const std::string & path) {
    fd_guard file(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    uint8_t magic[8];
    return file.fd >= 0 && read_all(file.fd, magic, sizeof(magic)) && std::memcmp(magic, ENTRY_MAGIC, 8) == 0;
}

#endif
