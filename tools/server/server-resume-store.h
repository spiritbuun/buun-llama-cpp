#pragma once

// Durable store of the persistent server resume: one directory of objects per conversation,
// the manifest published last. Format: docs/development/server-resume-format.md §2.
//
// The store knows files, checksums and bounds. It does not know tokens, slots or the model: the
// ledger is opaque bytes here, and what a record means is the server's business.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#define SERVER_RESUME_MANIFEST_VERSION 1
#define SERVER_RESUME_MEDIA_MANIFEST_VERSION 2
#define SERVER_RESUME_OBJECT_VERSION   1

enum class server_resume_reason : uint8_t {
    ok = 0,
    format_unsupported,
    manifest_corrupt,
    object_missing,
    object_size_mismatch,
    object_checksum_mismatch,
    store_locked,
    store_unwritable,
    no_space,
    io_error,
    _count,
};

const char * server_resume_reason_name(server_resume_reason reason) noexcept;

enum class server_resume_object_kind : uint32_t {
    base_chunk = 1, // base state of the token positions [p0, p1)
    tail_state = 2, // partial state taken when the sequence ended at p0; p1 is 0
    artifact   = 3, // one self-verifying envelope of the whole sequence [0, p1), streamed
    placement  = 4, // where the sequence [0, p1) lies in the pool image of another entry's artifact
    sequence   = 5, // complete fixed-KV sequence with explicit cell positions (M-RoPE media)
};

// bounds of a manifest, checked before anything is allocated from its numbers
struct server_resume_limits {
    static constexpr uint64_t max_manifest_bytes = 64ull * 1024 * 1024;
    static constexpr uint64_t max_json_bytes     = 1ull * 1024 * 1024;
    static constexpr size_t   max_chunks         = 4096;
    static constexpr size_t   max_tail_states    = 64;
    static constexpr size_t   max_producers      = 16;
    static constexpr int32_t  max_tokens         = 1 << 24;
    static constexpr int      max_json_depth     = 6;
    static constexpr size_t   max_entries_listed = 1024;
    static constexpr uint64_t max_object_bytes   = 8ull * 1024 * 1024 * 1024;
    // The store streams an artifact in chunks and allocates nothing from this number; the host
    // cache that takes it holds it whole in host memory, admitted against its budget before any
    // of it is read.
    static constexpr uint64_t max_artifact_bytes = 256ull * 1024 * 1024 * 1024;

    static constexpr uint64_t max_bytes(server_resume_object_kind kind) {
        return kind == server_resume_object_kind::artifact ? max_artifact_bytes : max_object_bytes;
    }
};

struct server_resume_object_record {
    server_resume_object_kind kind = server_resume_object_kind::base_chunk;
    int32_t  p0    = 0;
    int32_t  p1    = 0;
    uint64_t gen   = 0;
    uint64_t bytes = 0; // payload bytes
    uint64_t xxh3  = 0; // of the payload

    std::string prefix_digest; // hex, bound by the server to the ledger's tokens [0, p1) or [0, pos)
    uint32_t    producer = 0;

    // tail states only
    int32_t     pos_min  = 0;
    int32_t     pos_max  = 0;
    int32_t     n_tokens = 0;
    std::string role; // frontier, turn, early

    int32_t pos() const { return p0; }

    // the object on disk is this payload, by size and checksum
    bool holds(const uint8_t * payload, size_t size) const;
};

// provenance, never a gate
struct server_resume_producer {
    std::string model_name;
    std::string model_file; // basename
    std::string mmproj_file; // basename, empty without a projector
    std::string weight_type;
    std::string build;
    std::string cache_type_k;
    std::string cache_type_v;
    std::vector<std::string> adapters;
    int64_t host_unix_ms = 0;

    bool operator==(const server_resume_producer & other) const;
};

struct server_resume_manifest {
    uint64_t generation = 0;

    std::string resume_key; // hex
    std::string family_digest;
    std::string adapter_identity;

    int32_t n_tokens          = 0;
    int32_t chunk_tokens      = 0;
    int64_t saved_unix_ms     = 0;
    int64_t last_used_unix_ms = 0;
    int32_t slot_hint         = -1;
    bool shared_positions    = false; // v2: cell counts and temporal positions are distinct

    // Either the chunks tile [0, n_tokens), or one object holds the whole sequence and there are
    // no chunks. That object is a VBR artifact, a full fixed-KV sequence, or a placement
    // in the artifact of `pool_entry`. Whole objects may also carry early/turn checkpoints;
    // a placement carries its own recurrent frontier when needed.
    std::vector<server_resume_object_record> chunks;
    std::vector<server_resume_object_record> tail_states;
    std::optional<server_resume_object_record> artifact;
    std::vector<server_resume_producer>      producers;

    // placements: the artifact object the rows are in, by entry, generation and checksum
    std::string pool_entry;
    uint64_t    pool_generation = 0;
    uint64_t    pool_xxh3       = 0;

    bool placed() const { return artifact && artifact->kind == server_resume_object_kind::placement; }
    bool whole_sequence() const { return artifact && artifact->kind == server_resume_object_kind::sequence; }

    // the rows are in the image `pool` is, as the entry `id` holds it now
    bool placed_in(const std::string & id, const server_resume_object_record & pool) const {
        return placed() && pool_entry == id && pool_generation == pool.gen && pool_xxh3 == pool.xxh3;
    }

    void place_in(const std::string & id, const server_resume_object_record & pool) {
        pool_entry      = id;
        pool_generation = pool.gen;
        pool_xxh3       = pool.xxh3;
    }

    // The sequence epoch the state was captured under, so a restarted server can start its own
    // counter above every epoch the store still holds. An artifact is bound to it.
    uint64_t sequence_epoch = 0;

    std::vector<uint8_t> ledger;

    // Index of an equal producer, or append it. Throws length_error if full rather
    // than attributing newly written state to an unrelated producer.
    uint32_t producer_index(const server_resume_producer & producer);

    // Keep only the producers `records` refer to and renumber those records. A save calls this
    // with the records it carries over, so the table holds who wrote the entry's current objects.
    void retain_producers(const std::vector<server_resume_object_record *> & records);
};

// structure only: bounds, gapless tiling of [0, n_tokens), tail state positions, producer indices
bool server_resume_manifest_validate(const server_resume_manifest & manifest, std::string & error);

std::vector<uint8_t> server_resume_manifest_encode(const server_resume_manifest & manifest);

server_resume_reason server_resume_manifest_decode(
    const uint8_t * data, size_t size, server_resume_manifest & manifest, std::string & error);

struct server_resume_entry {
    std::string            id;
    server_resume_reason   reason = server_resume_reason::ok; // of the manifest
    std::string            error;
    server_resume_manifest manifest;
};

// test seam: called at named points of a write, a returned reason other than ok is injected there.
// points: object_write (the payload is on disk without its header), object_publish, manifest_publish,
// value_publish, uncommit, uncommit_sync
using server_resume_fault_fn = server_resume_reason (*)(const char * point);
void server_resume_store_set_fault(server_resume_fault_fn fn) noexcept;

class server_resume_store {
public:
    // <cache root>/resume/<family digest>/, directories 0700, files 0600. takes the writer lock of the
    // namespace: a second server gets store_locked and runs without persistence. A store that is not
    // durable syncs nothing: its entries do not outlive the request that wrote them
    static std::unique_ptr<server_resume_store> open(
        const std::string & cache_root, const std::string & family_digest,
        server_resume_reason & reason, std::string & error, bool durable = true);

    ~server_resume_store();

    server_resume_store(const server_resume_store &) = delete;
    server_resume_store & operator=(const server_resume_store &) = delete;

    const std::string & directory() const { return dir; }

    static std::string new_entry_id();
    static bool        entry_id_valid(const std::string & id);

    // committed entries, newest last_used first. an unreadable manifest is listed with its reason
    std::vector<server_resume_entry> list() const;

    server_resume_reason read_manifest(const std::string & id, server_resume_manifest & manifest, std::string & error) const;

    // verifies the header against the record, the file size and the payload checksum.
    // the payload buffer is reused between calls: one chunk of staging
    server_resume_reason read_object(
        const std::string & id, const server_resume_object_record & record,
        std::vector<uint8_t> & payload, std::string & error) const;

    // stages, syncs and renames one object. fills bytes and xxh3 of the record
    server_resume_reason write_object(
        const std::string & id, server_resume_object_record & record,
        const uint8_t * payload, size_t size, std::string & error);

    // The same object without one buffer of its size. `produce` pushes the payload through `put`
    // in order; `consume` pulls it through `get`. Both return false to give up. The checksum of a
    // read is known only after the last byte, so `consume` has to hold what it read privately
    // until this returns ok.
    using put_fn = std::function<bool(const uint8_t * data, size_t size)>;
    using get_fn = std::function<bool(uint8_t * data, size_t size)>;

    server_resume_reason write_object_stream(
        const std::string & id, server_resume_object_record & record,
        const std::function<bool(const put_fn & put)> & produce, std::string & error);

    server_resume_reason read_object_stream(
        const std::string & id, const server_resume_object_record & record,
        const std::function<bool(const get_fn & get)> & consume, std::string & error) const;

    // publishes the manifest: the entry is this generation from here on
    server_resume_reason commit(const std::string & id, const server_resume_manifest & manifest, std::string & error);

    // removes what the manifest does not name: replaced objects and staging leftovers
    void sweep(const std::string & id, const server_resume_manifest & manifest) const;

    // space-bounded replacement: take the entry out of the store before its objects go
    server_resume_reason uncommit(const std::string & id, std::string & error);

    void remove_entry(const std::string & id) const;

    // keeps the entries used last, so many of the given resume key and so many overall, and drops
    // damaged manifests and directories that never got one. A server cannot read the entries of
    // another key, so only the overall bound ends those, never the number of its own slots.
    // Live-slot entries have first priority, then held overflow entries, then ordinary entries.
    // Both live and held entries are excluded from the per-key count, but obey the overall cap.
    void prune(
            const std::string & resume_key, size_t n_keep_key, size_t n_keep_total,
            const std::set<std::string> & held = {}, const std::set<std::string> & live = {}) const;

    // the committed entries prune() would drop
    std::set<std::string> victims(
            const std::string & resume_key, size_t n_keep_key, size_t n_keep_total,
            const std::set<std::string> & held = {}, const std::set<std::string> & live = {}) const;

    uint64_t free_bytes() const;

    // One committed entry as a single file, its manifest and then its objects as the store holds
    // them, and back into a new entry of this store. The file is replaced whole or not at all. A
    // placed entry is not whole without the entry of its pool and is refused both ways. An import
    // copies nothing: the entry reads the file where it lies, open from here on, until
    // remove_entry(). It is not listed and cannot be committed over.
    server_resume_reason export_entry(
        const std::string & id, const std::string & path, uint64_t & bytes, std::string & error) const;
    // An export off the thread that writes the store: take_entry() moves the committed entry out
    // of it, so nothing lists, prunes or reuses it, and export_taken() exports it and removes it,
    // either way. It touches nothing else of the store and may run on another thread. A store
    // opened again drops what was taken and never exported.
    server_resume_reason take_entry(const std::string & id, std::string & error);
    server_resume_reason export_taken(
        const std::string & id, const std::string & path, uint64_t & bytes, std::string & error) const;
    server_resume_reason import_entry(
        const std::string & path, std::string & id, server_resume_manifest & manifest, uint64_t & bytes,
        std::string & error);
    // whether the file begins as an exported entry does
    static bool is_entry_file(const std::string & path);

    // A short printable value of the namespace that outlives the process: the stored one, else
    // `fresh` is stored and returned. Empty when it can be neither read nor stored.
    std::string keep_value(const std::string & name, const std::string & fresh, std::string & error);

private:
    server_resume_store() = default;

    std::string dir; // the family namespace
    int lock_fd = -1;
    bool durable = true;

    // imported entry files, by entry id; export_taken() does not look at them
    struct mounted_entry;
    mutable std::map<std::string, std::unique_ptr<const mounted_entry>> mounts;
    const mounted_entry * mounted(const std::string & id) const;

    std::string entry_dir(const std::string & id) const;
    bool sync_path(const std::string & path) const;
};
