// Durable store of the persistent server resume: manifest encoding and its refusals, object
// verification, the writer lock, and what a reader finds after a write that stopped part-way.

#include "server-resume-store.h"

#include "hash/xxhash/xxhash.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

static int g_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            g_failed++;                                                          \
        }                                                                        \
    } while (0)

static const std::string FAMILY(64, 'a');
static const std::string KEY(64, 'b');

static std::vector<uint8_t> pattern(size_t size, uint8_t seed) {
    std::vector<uint8_t> out(size);
    for (size_t i = 0; i < size; ++i) {
        out[i] = (uint8_t) (seed + i * 31);
    }
    return out;
}

static server_resume_producer producer_of(const std::string & name) {
    server_resume_producer producer;
    producer.model_name   = name;
    producer.model_file   = name + ".gguf";
    producer.weight_type  = "Q5_K_M";
    producer.build        = "test";
    producer.cache_type_k = "turbo3_tcq";
    producer.cache_type_v = "turbo3_tcq";
    producer.adapters     = { "adapter-a" };
    producer.host_unix_ms = 1000;
    return producer;
}

static server_resume_object_record chunk_of(int32_t p0, int32_t p1, uint64_t gen) {
    server_resume_object_record record;
    record.kind          = server_resume_object_kind::base_chunk;
    record.p0            = p0;
    record.p1            = p1;
    record.gen           = gen;
    record.bytes         = 1;
    record.prefix_digest = std::string(32, 'c');
    return record;
}

static server_resume_object_record tail_of(int32_t pos, uint64_t gen, const char * role) {
    server_resume_object_record record;
    record.kind          = server_resume_object_kind::tail_state;
    record.p0            = pos;
    record.gen           = gen;
    record.bytes         = 1;
    record.prefix_digest = std::string(32, 'd');
    record.pos_min       = 0;
    record.pos_max       = pos - 1;
    record.n_tokens      = pos;
    record.role          = role;
    return record;
}

static server_resume_manifest manifest_of(int32_t n_tokens, uint64_t generation) {
    server_resume_manifest manifest;
    manifest.generation        = generation;
    manifest.resume_key        = KEY;
    manifest.family_digest     = FAMILY;
    manifest.adapter_identity  = std::string(16, 'e');
    manifest.n_tokens          = n_tokens;
    manifest.chunk_tokens      = 16;
    manifest.saved_unix_ms     = 5000;
    manifest.last_used_unix_ms = 6000;
    manifest.slot_hint         = 1;
    manifest.ledger            = pattern(100, 7);
    manifest.producer_index(producer_of("model"));
    return manifest;
}

// a manifest file around any text, with sums that hold: what is refused is the content
static std::vector<uint8_t> raw_manifest(const std::string & text, uint32_t version = SERVER_RESUME_MANIFEST_VERSION) {
    const std::vector<uint8_t> ledger = pattern(8, 1);

    std::vector<uint8_t> out(64 + text.size() + ledger.size());
    std::memcpy(out.data() + 64, text.data(), text.size());
    std::memcpy(out.data() + 64 + text.size(), ledger.data(), ledger.size());

    const auto put = [&](size_t at, uint64_t value, int n) {
        for (int i = 0; i < n; ++i) {
            out[at + i] = (uint8_t) (value >> (8 * i));
        }
    };
    std::memcpy(out.data(), "BUUNRSMM", 8);
    put( 8, version, 4);
    put(12, 64, 4);
    put(24, text.size(), 8);
    put(32, ledger.size(), 8);
    put(40, 2, 8); // generation
    put(48, XXH3_64bits(out.data() + 64, out.size() - 64), 8);
    put(56, XXH3_64bits(out.data(), 56), 8);
    return out;
}

static void test_manifest_round_trip() {
    server_resume_manifest manifest = manifest_of(40, 3);
    manifest.chunks      = { chunk_of(0, 16, 1), chunk_of(16, 32, 1), chunk_of(32, 40, 3) };
    manifest.tail_states = { tail_of(40, 3, "frontier"), tail_of(24, 1, "turn") };
    manifest.chunks[2].producer = manifest.producer_index(producer_of("fine-tune"));
    manifest.chunks[1].xxh3     = 0xfedcba9876543210ull; // above the signed range
    CHECK(manifest.producers.size() == 2);
    CHECK(manifest.producer_index(producer_of("model")) == 0);

    std::string error;
    CHECK(server_resume_manifest_validate(manifest, error));

    const std::vector<uint8_t> data = server_resume_manifest_encode(manifest);

    server_resume_manifest back;
    CHECK(server_resume_manifest_decode(data.data(), data.size(), back, error) == server_resume_reason::ok);
    CHECK(back.generation == 3 && back.resume_key == KEY && back.family_digest == FAMILY);
    CHECK(back.n_tokens == 40 && back.chunk_tokens == 16 && back.slot_hint == 1);
    CHECK(back.saved_unix_ms == 5000 && back.last_used_unix_ms == 6000);
    CHECK(back.ledger == manifest.ledger);
    CHECK(back.chunks.size() == 3 && back.tail_states.size() == 2 && back.producers.size() == 2);
    CHECK(back.chunks[1].xxh3 == 0xfedcba9876543210ull);
    CHECK(back.chunks[2].p0 == 32 && back.chunks[2].p1 == 40 && back.chunks[2].gen == 3 && back.chunks[2].producer == 1);
    CHECK(back.tail_states[1].pos() == 24 && back.tail_states[1].pos_max == 23 && back.tail_states[1].role == "turn");
    CHECK(back.producers[1] == producer_of("fine-tune"));
    CHECK(back.producers[0].adapters == std::vector<std::string>{ "adapter-a" });

    // every single flipped byte is caught by one of the two sums, or is a field the header refuses
    for (size_t i = 0; i < data.size(); i += 7) {
        std::vector<uint8_t> bad = data;
        bad[i] ^= 0x40;
        CHECK(server_resume_manifest_decode(bad.data(), bad.size(), back, error) != server_resume_reason::ok);
    }
    for (size_t size : { (size_t) 0, (size_t) 63, data.size() - 1 }) {
        CHECK(server_resume_manifest_decode(data.data(), size, back, error) == server_resume_reason::manifest_corrupt);
    }
}

static void test_manifest_refusals() {
    std::string error;

    const auto valid = [&](const server_resume_manifest & manifest) {
        return server_resume_manifest_validate(manifest, error);
    };
    const auto base = []() {
        server_resume_manifest manifest = manifest_of(32, 2);
        manifest.chunks      = { chunk_of(0, 16, 1), chunk_of(16, 32, 2) };
        manifest.tail_states = { tail_of(32, 2, "frontier") };
        return manifest;
    };
    CHECK(valid(base()));

    { auto m = base(); m.chunks[1].p0 = 17;                 CHECK(!valid(m)); } // gap
    { auto m = base(); m.chunks[1].p0 = 15;                 CHECK(!valid(m)); } // overlap
    { auto m = base(); m.chunks.pop_back();                 CHECK(!valid(m)); } // short of n_tokens
    { auto m = base(); m.chunks[1].p1 = 33;                 CHECK(!valid(m)); } // past n_tokens
    { auto m = base(); m.chunks[1].gen = 3;                 CHECK(!valid(m)); } // from the future
    { auto m = base(); m.chunks[0].gen = 0;                 CHECK(!valid(m)); }
    { auto m = base(); m.chunks[0].bytes = 0;               CHECK(!valid(m)); }
    { auto m = base(); m.chunks[0].producer = 1;            CHECK(!valid(m)); }
    { auto m = base(); m.chunks[0].prefix_digest = "XYZ";   CHECK(!valid(m)); }
    { auto m = base(); m.chunks[0].kind = server_resume_object_kind::tail_state; CHECK(!valid(m)); }
    { auto m = base(); m.tail_states[0].p0 = 33; m.tail_states[0].n_tokens = 33; m.tail_states[0].pos_max = 32; CHECK(!valid(m)); }
    { auto m = base(); m.tail_states[0].pos_max = 30;       CHECK(!valid(m)); }
    { auto m = base(); m.tail_states[0].pos_max = std::numeric_limits<int32_t>::max(); CHECK(!valid(m)); }
    { auto m = base(); m.tail_states[0].n_tokens = 31;      CHECK(!valid(m)); }
    { auto m = base(); m.tail_states[0].pos_min = 32;       CHECK(!valid(m)); }
    { auto m = base(); m.tail_states[0].role = "other";     CHECK(!valid(m)); }
    { auto m = base(); m.tail_states.push_back(m.tail_states[0]); CHECK(!valid(m)); } // same position twice
    { auto m = base(); m.resume_key = "not hex";            CHECK(!valid(m)); }
    { auto m = base(); m.family_digest.clear();             CHECK(!valid(m)); }
    { auto m = base(); m.n_tokens = 0;                      CHECK(!valid(m)); }
    { auto m = base(); m.n_tokens = server_resume_limits::max_tokens + 1; CHECK(!valid(m)); }
    { auto m = base(); m.ledger.clear();                    CHECK(!valid(m)); }
    { auto m = base(); m.producers.clear();                 CHECK(!valid(m)); }
    { auto m = base(); m.producers[0].model_name = std::string(1000, 'x'); CHECK(!valid(m)); }
    {
        auto m = base();
        m.tail_states.clear();
        for (int32_t pos = 1; pos <= (int32_t) server_resume_limits::max_tail_states + 1; ++pos) {
            m.tail_states.push_back(tail_of(pos, 1, "early"));
        }
        m.n_tokens = 128;
        m.chunks   = { chunk_of(0, 128, 1) };
        CHECK(!valid(m));
        m.tail_states.pop_back();
        CHECK(valid(m));
    }

    // A full provenance table must never attribute new bytes to an unrelated producer.
    {
        auto m = base();
        for (size_t i = 1; i < server_resume_limits::max_producers; ++i) {
            m.producer_index(producer_of("model-" + std::to_string(i)));
        }
        CHECK(m.producers.size() == server_resume_limits::max_producers);
        bool refused = false;
        try {
            m.producer_index(producer_of("one more"));
        } catch (const std::length_error &) {
            refused = true;
        }
        CHECK(refused);
        CHECK(m.producer_index(producer_of("model")) == 0);
        CHECK(valid(m));

        // a save that carries over only the last model's objects frees the rest of the table
        m.chunks[0].producer = server_resume_limits::max_producers - 1;
        m.retain_producers({ &m.chunks[0] });
        CHECK(m.producers.size() == 1 && m.chunks[0].producer == 0);
        CHECK(m.producers[0] == producer_of("model-" + std::to_string(server_resume_limits::max_producers - 1)));
        CHECK(m.producer_index(producer_of("one more")) == 1);
    }

    // content the sums vouch for, refused on its own
    server_resume_manifest out;
    const auto decode = [&](const std::vector<uint8_t> & data) {
        return server_resume_manifest_decode(data.data(), data.size(), out, error);
    };

    const std::vector<uint8_t> good = server_resume_manifest_encode(base());
    const std::string text((const char *) good.data() + 64, good.size() - 64 - base().ledger.size());
    CHECK(decode(raw_manifest(text)) == server_resume_reason::ok);
    CHECK(decode(raw_manifest(text, SERVER_RESUME_MEDIA_MANIFEST_VERSION + 1)) == server_resume_reason::format_unsupported);
    CHECK(decode(raw_manifest(text.substr(0, text.size() - 1) +
        ",\"unknown\":" + std::string(20, '[') + "0" + std::string(20, ']') + "}")) ==
        server_resume_reason::manifest_corrupt);

    CHECK(decode(raw_manifest("not json")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(raw_manifest("[1,2,3]")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(raw_manifest("{}")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(raw_manifest(std::string(200000, '[') + std::string(200000, ']'))) == server_resume_reason::manifest_corrupt);
    CHECK(decode(raw_manifest("{\"chunks\":" + std::string(100000, '[') + std::string(100000, ']') + "}")) == server_resume_reason::manifest_corrupt);

    const auto replaced = [&](const std::string & from, const std::string & to) {
        std::string changed = text;
        const size_t at = changed.find(from);
        CHECK(at != std::string::npos);
        if (at != std::string::npos) {
            changed.replace(at, from.size(), to);
        }
        return raw_manifest(changed);
    };
    CHECK(decode(replaced("\"n_tokens\":32", "\"n_tokens\":32.5")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(replaced("\"n_tokens\":32", "\"n_tokens\":4294967328")) == server_resume_reason::manifest_corrupt); // 32 after a wrap
    CHECK(decode(replaced("\"n_tokens\":32", "\"n_tokens\":\"32\"")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(replaced("\"n_tokens\":32", "\"n_tokens\":-32")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(replaced("\"p1\":16", "\"p1\":17")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(replaced("\"role\":\"frontier\"", "\"role\":7")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(replaced("\"adapters\":[\"adapter-a\"]", "\"adapters\":[1]")) == server_resume_reason::manifest_corrupt);
    CHECK(decode(replaced("\"producers\":[", "\"producers\":[[],")) == server_resume_reason::manifest_corrupt);
}

static const char *         g_fault_point  = nullptr;
static server_resume_reason g_fault_reason = server_resume_reason::ok;

static server_resume_reason fault(const char * point) {
    return g_fault_point && std::strcmp(point, g_fault_point) == 0 ? g_fault_reason : server_resume_reason::ok;
}

static void set_fault(const char * point, server_resume_reason reason = server_resume_reason::io_error) {
    g_fault_point  = point;
    g_fault_reason = reason;
    server_resume_store_set_fault(point ? fault : nullptr);
}

static size_t n_files(const std::string & directory, const std::string & prefix = "") {
    size_t n = 0;
    for (const auto & item : fs::directory_iterator(directory)) {
        n += item.path().filename().string().rfind(prefix, 0) == 0;
    }
    return n;
}

static unsigned mode_of(const std::string & path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 ? st.st_mode & 0777 : 0;
}

// writes the objects of one generation over [p0, p1) and names them in the manifest
static server_resume_reason add_generation(
        server_resume_store & store, const std::string & id, server_resume_manifest & manifest,
        int32_t p0, int32_t p1, bool do_commit = true) {
    std::string error;
    manifest.generation++;
    manifest.n_tokens = p1;

    server_resume_object_record chunk = chunk_of(p0, p1, manifest.generation);
    const std::vector<uint8_t> base = pattern(3000 + p1, (uint8_t) p1);
    server_resume_reason reason = store.write_object(id, chunk, base.data(), base.size(), error);
    if (reason != server_resume_reason::ok) {
        return reason;
    }
    manifest.chunks.push_back(chunk);

    server_resume_object_record tail = tail_of(p1, manifest.generation, "frontier");
    const std::vector<uint8_t> partial = pattern(500, (uint8_t) (p1 + 1));
    reason = store.write_object(id, tail, partial.data(), partial.size(), error);
    if (reason != server_resume_reason::ok) {
        return reason;
    }
    manifest.tail_states = { tail };

    return do_commit ? store.commit(id, manifest, error) : server_resume_reason::ok;
}

static bool entry_verifies(const server_resume_store & store, const server_resume_entry & entry) {
    std::string error;
    std::vector<uint8_t> payload;
    for (const auto & chunk : entry.manifest.chunks) {
        if (store.read_object(entry.id, chunk, payload, error) != server_resume_reason::ok ||
            payload != pattern(3000 + chunk.p1, (uint8_t) chunk.p1)) {
            return false;
        }
    }
    for (const auto & tail : entry.manifest.tail_states) {
        if (store.read_object(entry.id, tail, payload, error) != server_resume_reason::ok ||
            payload != pattern(500, (uint8_t) (tail.pos() + 1))) {
            return false;
        }
    }
    return true;
}

static void test_store(const std::string & root) {
    server_resume_reason reason;
    std::string error;

    CHECK(!server_resume_store::open(root, "../escape", reason, error));
    CHECK(reason == server_resume_reason::store_unwritable);

    auto store = server_resume_store::open(root, FAMILY, reason, error);
    CHECK(store && reason == server_resume_reason::ok);
    if (!store) {
        std::fprintf(stderr, "open: %s\n", error.c_str());
        return;
    }
    CHECK(mode_of(root + "/resume") == 0700);
    CHECK(mode_of(store->directory()) == 0700);
    CHECK(mode_of(store->directory() + "/entries") == 0700);
    CHECK(mode_of(store->directory() + "/writer.lock") == 0600);
    CHECK(store->free_bytes() > 0);
    CHECK(store->list().empty());

    // one writer per namespace; another family is another namespace
    CHECK(!server_resume_store::open(root, FAMILY, reason, error));
    CHECK(reason == server_resume_reason::store_locked);
    CHECK(server_resume_store::open(root, std::string(64, 'f'), reason, error) != nullptr);

    const std::string id = server_resume_store::new_entry_id();
    CHECK(server_resume_store::entry_id_valid(id));
    CHECK(!server_resume_store::entry_id_valid("../../etc"));
    CHECK(!server_resume_store::entry_id_valid(id + "0"));
    const std::string entry_dir = store->directory() + "/entries/" + id;

    // generation 1, then an append as generation 2
    server_resume_manifest manifest = manifest_of(16, 0);
    manifest.chunks.clear();
    CHECK(add_generation(*store, id, manifest, 0, 16) == server_resume_reason::ok);
    CHECK(mode_of(entry_dir) == 0700);
    CHECK(mode_of(entry_dir + "/commit") == 0600);
    CHECK(mode_of(entry_dir + "/c-0-16-1") == 0600);
    CHECK(n_files(entry_dir) == 3);

    CHECK(add_generation(*store, id, manifest, 16, 32) == server_resume_reason::ok);
    CHECK(n_files(entry_dir) == 5); // the old tail state is still there
    store->sweep(id, manifest);
    CHECK(n_files(entry_dir) == 4);
    CHECK(!fs::exists(entry_dir + "/t-16-1"));

    {
        const auto entries = store->list();
        CHECK(entries.size() == 1);
        CHECK(entries.size() == 1 && entries[0].id == id && entries[0].reason == server_resume_reason::ok);
        CHECK(entries.size() == 1 && entries[0].manifest.generation == 2 && entries[0].manifest.n_tokens == 32);
        CHECK(entries.size() == 1 && entry_verifies(*store, entries[0]));
    }

    // a write that stops: nothing it left behind is visible, and generation 2 still verifies
    const server_resume_manifest committed = manifest;
    for (const char * point : { "object_write", "object_publish", "manifest_publish" }) {
        server_resume_manifest next = committed;
        set_fault(point, std::strcmp(point, "object_publish") == 0 ? server_resume_reason::no_space : server_resume_reason::io_error);
        const server_resume_reason stopped = add_generation(*store, id, next, 32, 48);
        set_fault(nullptr);
        CHECK(stopped == (std::strcmp(point, "object_publish") == 0 ? server_resume_reason::no_space : server_resume_reason::io_error));
        CHECK(n_files(entry_dir, "tmp-") == 1);

        // what the next server finds
        store.reset();
        store = server_resume_store::open(root, FAMILY, reason, error);
        CHECK(store != nullptr);
        if (!store) {
            return;
        }
        const auto entries = store->list();
        CHECK(entries.size() == 1);
        CHECK(entries.size() == 1 && entries[0].manifest.generation == 2 && entry_verifies(*store, entries[0]));

        store->sweep(id, committed);
        CHECK(n_files(entry_dir) == 4);
    }

    // damage to a published object
    {
        std::vector<uint8_t> payload;
        const server_resume_object_record & chunk = committed.chunks[1];
        const std::string path = entry_dir + "/c-16-32-2";
        const std::vector<uint8_t> original = [&]() {
            std::ifstream in(path, std::ios::binary);
            return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
        }();
        const auto rewrite = [&](const std::vector<uint8_t> & data) {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write((const char *) data.data(), data.size());
        };
        CHECK(original.size() == 64 + chunk.bytes);

        std::vector<uint8_t> bad = original;
        bad[64 + 1000] ^= 1;
        rewrite(bad);
        CHECK(store->read_object(id, chunk, payload, error) == server_resume_reason::object_checksum_mismatch);

        bad = original;
        bad[40] ^= 1; // p0 of the header
        rewrite(bad);
        CHECK(store->read_object(id, chunk, payload, error) == server_resume_reason::object_checksum_mismatch);

        bad = original;
        bad.pop_back();
        rewrite(bad);
        CHECK(store->read_object(id, chunk, payload, error) == server_resume_reason::object_size_mismatch);

        bad = original;
        bad.push_back(0);
        rewrite(bad);
        CHECK(store->read_object(id, chunk, payload, error) == server_resume_reason::object_size_mismatch);

        // the object of another range under this name
        {
            std::ifstream in(entry_dir + "/c-0-16-1", std::ios::binary);
            bad.assign(std::istreambuf_iterator<char>(in), {});
        }
        rewrite(bad);
        CHECK(store->read_object(id, chunk, payload, error) != server_resume_reason::ok);

        fs::remove(path);
        CHECK(store->read_object(id, chunk, payload, error) == server_resume_reason::object_missing);

        rewrite(original);
        CHECK(store->read_object(id, chunk, payload, error) == server_resume_reason::ok);

        // a record the manifest bounds would have refused never sizes a buffer
        server_resume_object_record huge = chunk;
        huge.bytes = server_resume_limits::max_object_bytes + 1;
        CHECK(store->read_object(id, huge, payload, error) == server_resume_reason::object_size_mismatch);
    }

    // a damaged manifest is listed with its reason, after the good ones
    const std::string id_bad = server_resume_store::new_entry_id();
    {
        server_resume_manifest other = manifest_of(16, 0);
        other.chunks.clear();
        other.last_used_unix_ms = 9000;
        CHECK(add_generation(*store, id_bad, other, 0, 16) == server_resume_reason::ok);

        std::fstream file(store->directory() + "/entries/" + id_bad + "/commit", std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(80);
        file.put('!');
    }
    {
        const auto entries = store->list();
        CHECK(entries.size() == 2);
        CHECK(entries.size() == 2 && entries[0].id == id && entries[1].id == id_bad);
        CHECK(entries.size() == 2 && entries[1].reason == server_resume_reason::manifest_corrupt);
    }
    store->prune(KEY, 8, 8);
    CHECK(store->list().size() == 1);
    CHECK(!fs::exists(store->directory() + "/entries/" + id_bad));

    // retention: newest use first, and a directory that never got a manifest goes as well
    const std::string id_new = server_resume_store::new_entry_id();
    const std::string id_open = server_resume_store::new_entry_id();
    {
        server_resume_manifest other = manifest_of(16, 0);
        other.chunks.clear();
        other.last_used_unix_ms = 7000;
        CHECK(add_generation(*store, id_new, other, 0, 16) == server_resume_reason::ok);

        server_resume_manifest open_entry = manifest_of(16, 0);
        open_entry.chunks.clear();
        CHECK(add_generation(*store, id_open, open_entry, 0, 16, false) == server_resume_reason::ok);
    }
    const std::string id_old = server_resume_store::new_entry_id();
    {
        server_resume_manifest other = manifest_of(16, 0);
        other.chunks.clear();
        other.last_used_unix_ms = 100;
        other.resume_key = std::string(64, 'e');
        CHECK(add_generation(*store, id_old, other, 0, 16) == server_resume_reason::ok);
    }
    CHECK(n_files(store->directory() + "/entries") == 4);
    CHECK(store->list().size() == 3);
    // the bound of this key leaves the entry of the other key alone, the overall bound does not
    store->prune(KEY, 2, 8);
    CHECK(n_files(store->directory() + "/entries") == 3);
    CHECK(store->list().size() == 3);
    // an entry that is held for a conversation without a slot is not counted against the slots
    store->prune(KEY, 1, 8, {id});
    CHECK(store->list().size() == 3);
    // what a prune would take is known before it, and the overall bound takes held entries last
    CHECK(store->victims(KEY, 1, 8) == std::set<std::string>{id});
    CHECK(store->victims(KEY, 1, 8, {id}).empty());
    CHECK((store->victims(KEY, 2, 1, {id_old}) == std::set<std::string>{id, id_new}));
    // Shrinking the slot count must not let newer overflow evict the oldest live slot.
    CHECK((store->victims(KEY, 0, 1, {id, id_new}, {id_old}) == std::set<std::string>{id, id_new}));
    CHECK((store->victims(KEY, 0, 2, {id, id_new}, {id_old}) == std::set<std::string>{id}));
    CHECK(store->list().size() == 3);
    store->prune(KEY, 1, 8);
    {
        const auto entries = store->list();
        CHECK(entries.size() == 2);
        CHECK(entries.size() == 2 && entries[0].id == id_new && entries[1].id == id_old);
    }
    {
        server_resume_manifest other = manifest_of(16, 0);
        other.chunks.clear();
        other.last_used_unix_ms = 6000;
        CHECK(add_generation(*store, id, other, 0, 16) == server_resume_reason::ok);
    }
    store->prune(KEY, 2, 2);
    {
        const auto entries = store->list();
        CHECK(n_files(store->directory() + "/entries") == 2);
        CHECK(entries.size() == 2);
        CHECK(entries.size() == 2 && entries[0].id == id_new && entries[1].id == id);
    }

    // space-bounded replacement takes the entry out first
    set_fault("uncommit", server_resume_reason::io_error);
    CHECK(store->uncommit(id, error) == server_resume_reason::io_error);
    set_fault(nullptr);
    server_resume_manifest still_present;
    CHECK(store->read_manifest(id, still_present, error) == server_resume_reason::ok);
    set_fault("uncommit_sync", server_resume_reason::io_error);
    CHECK(store->uncommit(id, error) == server_resume_reason::io_error);
    set_fault(nullptr);
    CHECK(store->uncommit(id, error) == server_resume_reason::ok);
    CHECK(store->uncommit(id, error) == server_resume_reason::ok);
    CHECK(store->list().size() == 1);
    server_resume_manifest gone;
    CHECK(store->read_manifest(id, gone, error) == server_resume_reason::object_missing);

    store->remove_entry(id_new);
    store->prune(KEY, 0, 0);
    CHECK(n_files(store->directory() + "/entries") == 0);
    // An explicit erase remains successful if retention already removed the entry.
    CHECK(store->uncommit(id, error) == server_resume_reason::ok);

    // an entries directory that is a link somewhere else is not followed
    store.reset();
    fs::remove_all(root + "/resume/" + FAMILY + "/entries");
    fs::create_directories(root + "/elsewhere");
    fs::create_directory_symlink(root + "/elsewhere", root + "/resume/" + FAMILY + "/entries");
    CHECK(!server_resume_store::open(root, FAMILY, reason, error));
    CHECK(reason == server_resume_reason::store_unwritable);
}

// one streamed object in place of the chunks: never in one buffer, its checksum known at the end
static void test_artifact(const std::string & root) {
    server_resume_reason reason;
    std::string error;
    auto store = server_resume_store::open(root, FAMILY, reason, error);
    CHECK(store);
    if (!store) {
        return;
    }

    const std::string id = server_resume_store::new_entry_id();
    const std::vector<uint8_t> payload = pattern(300000, 9);

    server_resume_object_record record;
    record.kind          = server_resume_object_kind::artifact;
    record.p1            = 48;
    record.gen           = 1;
    record.prefix_digest = std::string(32, 'c');

    const auto produce = [&](const server_resume_store::put_fn & put) {
        for (size_t done = 0; done < payload.size(); done += 4099) {
            if (!put(payload.data() + done, std::min<size_t>(4099, payload.size() - done))) {
                return false;
            }
        }
        return true;
    };
    CHECK(store->write_object_stream(id, record, produce, error) == server_resume_reason::ok);
    CHECK(record.holds(payload.data(), payload.size()));

    auto manifest = manifest_of(48, 1);
    manifest.artifact       = record;
    manifest.sequence_epoch = 7;
    CHECK(store->commit(id, manifest, error) == server_resume_reason::ok);
    store->sweep(id, manifest);

    server_resume_manifest read;
    CHECK(store->read_manifest(id, read, error) == server_resume_reason::ok);
    CHECK(read.artifact && read.artifact->kind == server_resume_object_kind::artifact && read.chunks.empty());
    CHECK(read.artifact->bytes == payload.size() && read.artifact->xxh3 == record.xxh3 && read.sequence_epoch == 7);

    std::vector<uint8_t> got;
    const auto consume_all = [&](const server_resume_store::get_fn & get) {
        got.assign(payload.size(), 0);
        return get(got.data(), 1000) && get(got.data() + 1000, got.size() - 1000);
    };
    CHECK(store->read_object_stream(id, *read.artifact, consume_all, error) == server_resume_reason::ok);
    CHECK(got == payload);

    // a consumer that stops early, or asks for more than the object holds, has not read the object
    CHECK(store->read_object_stream(id, *read.artifact,
        [&](const server_resume_store::get_fn & get) { return get(got.data(), 1000); }, error) != server_resume_reason::ok);
    CHECK(store->read_object_stream(id, *read.artifact,
        [&](const server_resume_store::get_fn & get) { got.resize(payload.size() + 1); return get(got.data(), got.size()); }, error) !=
        server_resume_reason::ok);

    // one changed byte is found at the end of the stream
    const std::string path = store->directory() + "/entries/" + id + "/v-1";
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(64 + 200000);
        file.put((char) (payload[200000] ^ 1));
    }
    CHECK(store->read_object_stream(id, *read.artifact, consume_all, error) == server_resume_reason::object_checksum_mismatch);

    // a producer that gives up leaves nothing behind
    const std::string id_failed = server_resume_store::new_entry_id();
    CHECK(store->write_object_stream(id_failed, record,
        [&](const server_resume_store::put_fn & put) { put(payload.data(), 100); return false; }, error) != server_resume_reason::ok);
    CHECK(n_files(store->directory() + "/entries/" + id_failed) == 0);

    // an artifact stands in for the chunks, never beside them, and carries no frontier tail state:
    // only the ring's two checkpoints below its end
    const auto valid = [](const server_resume_manifest & m) { std::string e; return server_resume_manifest_validate(m, e); };
    CHECK(valid(manifest));
    { auto m = manifest; m.chunks.push_back(chunk_of(0, 48, 1)); CHECK(!valid(m)); }
    { auto m = manifest; m.tail_states.push_back(tail_of(48, 1, "frontier")); CHECK(!valid(m)); }
    { auto m = manifest; m.tail_states.push_back(tail_of(16, 1, "early")); m.tail_states.push_back(tail_of(40, 1, "turn")); CHECK(valid(m)); }
    { auto m = manifest; m.tail_states.push_back(tail_of(40, 1, "frontier")); CHECK(!valid(m)); }
    { auto m = manifest; m.tail_states.push_back(tail_of(48, 1, "turn")); CHECK(!valid(m)); }
    { auto m = manifest; for (int32_t pos : {8, 16, 40}) { m.tail_states.push_back(tail_of(pos, 1, "turn")); } CHECK(!valid(m)); }
    { auto m = manifest; m.artifact->p1 = 47; CHECK(!valid(m)); }
    { auto m = manifest; m.artifact->kind = server_resume_object_kind::base_chunk; CHECK(!valid(m)); }
    { auto m = manifest; m.artifact->gen = 2; CHECK(!valid(m)); }
    { auto m = manifest; m.sequence_epoch = 0; CHECK(!valid(m)); }
    { auto m = manifest_of(48, 1); m.chunks.push_back(chunk_of(0, 48, 1)); m.sequence_epoch = 1; CHECK(!valid(m)); }
    { auto m = manifest; m.pool_entry = id; m.pool_generation = 1; CHECK(!valid(m)); }

    // a placement names the artifact its rows are in, and may carry the one tail state of its end
    const std::string id_placed = server_resume_store::new_entry_id();
    const std::vector<uint8_t> cells = pattern(640, 3);
    server_resume_object_record placement;
    placement.kind          = server_resume_object_kind::placement;
    placement.p1            = 32;
    placement.gen           = 1;
    placement.prefix_digest = std::string(32, 'd');
    CHECK(store->write_object(id_placed, placement, cells.data(), cells.size(), error) == server_resume_reason::ok);
    CHECK(n_files(store->directory() + "/entries/" + id_placed) == 1);
    CHECK(fs::exists(store->directory() + "/entries/" + id_placed + "/p-1"));

    auto placed = manifest_of(32, 1);
    placed.artifact        = placement;
    placed.pool_entry      = id;
    placed.pool_generation = record.gen;
    placed.pool_xxh3       = record.xxh3;
    placed.sequence_epoch  = 9;
    CHECK(valid(placed) && placed.placed() && !manifest.placed());
    CHECK(store->commit(id_placed, placed, error) == server_resume_reason::ok);
    store->sweep(id_placed, placed);
    CHECK(store->read_manifest(id_placed, read, error) == server_resume_reason::ok);
    CHECK(read.placed() && read.pool_entry == id && read.pool_generation == record.gen &&
          read.pool_xxh3 == record.xxh3 && read.sequence_epoch == 9);
    CHECK(store->read_object(id_placed, *read.artifact, got, error) == server_resume_reason::ok && got == cells);

    { auto m = placed; m.sequence_epoch = 0; CHECK(valid(m)); }
    { auto m = placed; m.tail_states.push_back(tail_of(32, 1, "frontier")); CHECK(valid(m)); }
    { auto m = placed; m.tail_states.push_back(tail_of(32, 1, "frontier")); m.tail_states.push_back(tail_of(16, 1, "turn")); CHECK(valid(m)); }
    { auto m = placed; m.tail_states.push_back(tail_of(32, 1, "frontier")); m.tail_states.push_back(tail_of(32, 1, "frontier")); CHECK(!valid(m)); }
    { auto m = placed; m.tail_states.push_back(tail_of(16, 1, "frontier")); CHECK(!valid(m)); }
    { auto m = placed; m.pool_entry.clear(); CHECK(!valid(m)); }
    { auto m = placed; m.pool_entry = "../escape"; CHECK(!valid(m)); }
    { auto m = placed; m.pool_generation = 0; CHECK(!valid(m)); }
    { auto m = placed; m.chunks.push_back(chunk_of(0, 32, 1)); CHECK(!valid(m)); }

    // a kept value is the first one stored, also for the next process
    CHECK(store->keep_value("execution-identity", "first", error) == "first");
    CHECK(store->keep_value("execution-identity", "second", error) == "first");
    CHECK(store->keep_value("../escape", "x", error).empty());
    CHECK(store->keep_value("other", "not printable", error).empty());
    store.reset();
    store = server_resume_store::open(root, FAMILY, reason, error);
    CHECK(store != nullptr);
    CHECK(store->keep_value("execution-identity", "third", error) == "first");
    CHECK(store->list().size() == 2);
}

static std::vector<uint8_t> file_bytes(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

static void put_file(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream(path, std::ios::binary | std::ios::trunc).write((const char *) bytes.data(), bytes.size());
}

// an entry as one file, out of one store and into another
static void test_entry_file(const std::string & root) {
    server_resume_reason reason;
    std::string error;
    auto from = server_resume_store::open(root + "/a", FAMILY, reason, error);
    auto into = server_resume_store::open(root + "/b", FAMILY, reason, error);
    CHECK(from && into);
    if (!from || !into) {
        return;
    }
    const std::string entries = into->directory() + "/entries";
    const std::string path    = root + "/saved.bin";

    // chunks and tail states
    const std::string id = server_resume_store::new_entry_id();
    server_resume_manifest manifest = manifest_of(16, 0);
    manifest.chunks.clear();
    CHECK(add_generation(*from, id, manifest, 0, 16) == server_resume_reason::ok);
    CHECK(add_generation(*from, id, manifest, 16, 32) == server_resume_reason::ok);
    from->sweep(id, manifest);

    uint64_t bytes = 0;
    CHECK(from->export_entry(id, path, bytes, error) == server_resume_reason::ok);
    CHECK(bytes == fs::file_size(path) && server_resume_store::is_entry_file(path));
    CHECK(n_files(root) == 3); // a, b and the file: nothing staged is left beside it

    std::string got_id;
    server_resume_manifest got;
    uint64_t n_read = 0;
    CHECK(into->import_entry(path, got_id, got, n_read, error) == server_resume_reason::ok && n_read == bytes);
    CHECK(server_resume_store::entry_id_valid(got_id) && got_id != id);
    CHECK(got.generation == manifest.generation && got.chunks.size() == 2 && got.tail_states.size() == 1 &&
          got.ledger == manifest.ledger);
    {
        // read where it lies: nothing is copied into the store, and nothing is listed
        server_resume_entry mounted;
        mounted.id = got_id;
        CHECK(into->read_manifest(got_id, mounted.manifest, error) == server_resume_reason::ok &&
              mounted.manifest.ledger == manifest.ledger && entry_verifies(*into, mounted));
        CHECK(into->list().empty() && n_files(entries) == 0);
        CHECK(into->commit(got_id, got, error) != server_resume_reason::ok);
    }
    // the same file again is another entry
    std::string again;
    CHECK(into->import_entry(path, again, got, n_read, error) == server_resume_reason::ok && again != got_id);

    // an export replaces the file whole; an entry already imported keeps reading the file it opened
    const auto first = file_bytes(path);
    CHECK(add_generation(*from, id, manifest, 32, 48) == server_resume_reason::ok);
    CHECK(from->export_entry(id, path, bytes, error) == server_resume_reason::ok);
    CHECK(bytes == fs::file_size(path) && file_bytes(path) != first);
    {
        server_resume_entry mounted;
        mounted.id = again;
        CHECK(into->read_manifest(again, mounted.manifest, error) == server_resume_reason::ok &&
              mounted.manifest.chunks.size() == 2 && entry_verifies(*into, mounted));
    }
    std::string drop_error;
    CHECK(into->uncommit(got_id, drop_error) == server_resume_reason::ok);
    into->remove_entry(got_id);
    into->remove_entry(again);
    CHECK(into->read_manifest(got_id, got, error) == server_resume_reason::object_missing);

    // what is refused leaves no entry behind
    const size_t n_entries = n_files(entries);
    const auto refused = [&](const std::vector<uint8_t> & content, server_resume_reason want) {
        put_file(root + "/bad.bin", content);
        std::string bad_id = "x";
        const auto why = into->import_entry(root + "/bad.bin", bad_id, got, n_read, error);
        return why == want && bad_id.empty() && n_files(entries) == n_entries;
    };
    const auto whole = file_bytes(path);
    {
        auto cut = whole; cut.resize(cut.size() - 1);
        CHECK(refused(cut, server_resume_reason::object_size_mismatch)); // the header holds the size
        auto grown = whole; grown.push_back(0);
        CHECK(refused(grown, server_resume_reason::object_size_mismatch));
        CHECK(refused(std::vector<uint8_t>(whole.begin(), whole.begin() + 40), server_resume_reason::manifest_corrupt));
    }
    {
        auto flipped = whole; flipped[20] ^= 1; // the flags, under the seal
        CHECK(refused(flipped, server_resume_reason::manifest_corrupt));
        auto manifest_byte = whole; manifest_byte[64 + 70] ^= 1;
        CHECK(refused(manifest_byte, server_resume_reason::manifest_corrupt));
        // the header of the last object, the frontier tail state
        auto object_header = whole; object_header[whole.size() - 500 - 64 + 30] ^= 1;
        CHECK(refused(object_header, server_resume_reason::object_checksum_mismatch));
    }
    {
        // a changed payload byte comes in, and is found where every object is read
        auto payload_byte = whole; payload_byte[whole.size() - 10] ^= 1;
        put_file(root + "/bad.bin", payload_byte);
        std::string bad_id;
        CHECK(into->import_entry(root + "/bad.bin", bad_id, got, n_read, error) == server_resume_reason::ok);
        std::vector<uint8_t> payload;
        CHECK(into->read_object(bad_id, got.tail_states[0], payload, error) == server_resume_reason::object_checksum_mismatch);
        into->remove_entry(bad_id);
    }
    // a file of another format is not an entry file
    put_file(root + "/bad.bin", pattern(4096, 5));
    CHECK(!server_resume_store::is_entry_file(root + "/bad.bin"));
    CHECK(refused(pattern(4096, 5), server_resume_reason::manifest_corrupt));
    CHECK(!server_resume_store::is_entry_file(root + "/missing.bin"));
    CHECK(into->import_entry(root + "/missing.bin", got_id, got, n_read, error) == server_resume_reason::object_missing);
    CHECK(from->export_entry(server_resume_store::new_entry_id(), root + "/none.bin", bytes, error) != server_resume_reason::ok);
    CHECK(!fs::exists(root + "/none.bin"));

    // an artifact, streamed through
    const std::string id_artifact = server_resume_store::new_entry_id();
    const std::vector<uint8_t> payload = pattern(300000, 9);
    server_resume_object_record record;
    record.kind          = server_resume_object_kind::artifact;
    record.p1            = 48;
    record.gen           = 1;
    record.prefix_digest = std::string(32, 'c');
    CHECK(from->write_object(id_artifact, record, payload.data(), payload.size(), error) == server_resume_reason::ok);
    auto with_artifact = manifest_of(48, 1);
    with_artifact.artifact       = record;
    with_artifact.sequence_epoch = 7;
    with_artifact.tail_states.push_back(tail_of(40, 1, "turn"));
    const std::vector<uint8_t> ring = pattern(500, 41);
    CHECK(from->write_object(id_artifact, with_artifact.tail_states[0], ring.data(), ring.size(), error) == server_resume_reason::ok);
    CHECK(from->commit(id_artifact, with_artifact, error) == server_resume_reason::ok);

    CHECK(from->export_entry(id_artifact, path, bytes, error) == server_resume_reason::ok);
    CHECK(into->import_entry(path, got_id, got, n_read, error) == server_resume_reason::ok);
    CHECK(got.artifact && got.artifact->xxh3 == record.xxh3 && got.sequence_epoch == 7 && got.tail_states.size() == 1);
    std::vector<uint8_t> read;
    CHECK(into->read_object(got_id, *got.artifact, read, error) == server_resume_reason::ok && read == payload);
    CHECK(into->read_object(got_id, got.tail_states[0], read, error) == server_resume_reason::ok && read == ring);

    // a placement is not whole without its pool's entry
    const std::string id_placed = server_resume_store::new_entry_id();
    server_resume_object_record placement;
    placement.kind          = server_resume_object_kind::placement;
    placement.p1            = 32;
    placement.gen           = 1;
    placement.prefix_digest = std::string(32, 'd');
    const std::vector<uint8_t> cells = pattern(640, 3);
    CHECK(from->write_object(id_placed, placement, cells.data(), cells.size(), error) == server_resume_reason::ok);
    auto placed = manifest_of(32, 1);
    placed.artifact = placement;
    placed.place_in(id_artifact, record);
    CHECK(from->commit(id_placed, placed, error) == server_resume_reason::ok);
    CHECK(from->export_entry(id_placed, root + "/placed.bin", bytes, error) == server_resume_reason::format_unsupported);
    CHECK(!fs::exists(root + "/placed.bin"));

    // taken out of the store, then exported: the same file, and nothing of it is left
    const std::string taken = from->directory() + "/taken";
    CHECK(from->export_entry(id_artifact, path, bytes, error) == server_resume_reason::ok);
    CHECK(from->take_entry(id_artifact, error) == server_resume_reason::ok);
    CHECK(from->take_entry(id_artifact, error) == server_resume_reason::object_missing);
    CHECK(from->read_manifest(id_artifact, got, error) == server_resume_reason::object_missing);
    for (const auto & entry : from->list()) {
        CHECK(entry.id != id_artifact);
    }
    uint64_t taken_bytes = 0;
    CHECK(from->export_taken(id_artifact, root + "/taken.bin", taken_bytes, error) == server_resume_reason::ok);
    CHECK(taken_bytes == bytes && file_bytes(root + "/taken.bin") == file_bytes(path));
    CHECK(n_files(taken) == 0);
    // a refused export removes it all the same
    CHECK(from->take_entry(id_placed, error) == server_resume_reason::ok);
    CHECK(from->export_taken(id_placed, root + "/placed.bin", bytes, error) == server_resume_reason::format_unsupported);
    CHECK(n_files(taken) == 0 && !fs::exists(root + "/placed.bin"));
    // an imported entry is not in the store to take
    CHECK(into->take_entry(got_id, error) == server_resume_reason::object_missing);
    // taken and never exported: gone when the store is opened again
    CHECK(from->take_entry(id, error) == server_resume_reason::ok && n_files(taken) == 1);
    from.reset();
    from = server_resume_store::open(root + "/a", FAMILY, reason, error);
    CHECK(from && !fs::exists(taken));
}

static void test_shared_positions() {
    auto manifest = manifest_of(20, 1);
    manifest.shared_positions = true;
    auto record = chunk_of(0, 20, 1);
    record.kind = server_resume_object_kind::sequence;
    manifest.artifact = record;
    auto checkpoint = tail_of(15, 1, "turn");
    checkpoint.pos_max = 6; // 15 cells but only 7 temporal positions
    manifest.tail_states.push_back(checkpoint);
    std::string error;
    CHECK(server_resume_manifest_validate(manifest, error));
    auto bytes = server_resume_manifest_encode(manifest);
    server_resume_manifest decoded;
    CHECK(server_resume_manifest_decode(bytes.data(), bytes.size(), decoded, error) == server_resume_reason::ok);
    CHECK(decoded.shared_positions && decoded.whole_sequence());
    CHECK(decoded.tail_states[0].pos_max == 6 && decoded.tail_states[0].n_tokens == 15);
    // A valid header checksum cannot make shared-position semantics into v1.
    auto wrong_version = bytes;
    wrong_version[8] = SERVER_RESUME_MANIFEST_VERSION;
    const auto seal = XXH3_64bits(wrong_version.data(), 56);
    for (size_t i = 0; i < 8; ++i) { wrong_version[56 + i] = uint8_t(seal >> (8*i)); }
    CHECK(server_resume_manifest_decode(wrong_version.data(), wrong_version.size(), decoded, error) ==
        server_resume_reason::manifest_corrupt);
    manifest.shared_positions = false;
    CHECK(!server_resume_manifest_validate(manifest, error));
    manifest.shared_positions = true;
    manifest.tail_states[0].pos_max = 15;
    CHECK(!server_resume_manifest_validate(manifest, error));
    manifest.tail_states.clear();
    manifest.artifact.reset();
    manifest.chunks.push_back(chunk_of(0, 20, 1));
    CHECK(!server_resume_manifest_validate(manifest, error));
}

int main() {
    test_shared_positions();
    test_manifest_round_trip();
    test_manifest_refusals();

    {
        const std::string root = (fs::temp_directory_path() / ("test-server-resume-entry-" + server_resume_store::new_entry_id())).string();
        fs::create_directories(root);
        test_entry_file(root);
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    {
        const std::string root = (fs::temp_directory_path() / ("test-server-resume-artifact-" + server_resume_store::new_entry_id())).string();
        test_artifact(root);
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    const std::string root = (fs::temp_directory_path() / ("test-server-resume-store-" + server_resume_store::new_entry_id())).string();
    test_store(root);

    std::error_code ec;
    fs::remove_all(root, ec);

    for (int i = 0; i < (int) server_resume_reason::_count; ++i) {
        CHECK(std::strcmp(server_resume_reason_name((server_resume_reason) i), "unknown") != 0);
    }

    if (g_failed) {
        std::fprintf(stderr, "%d checks failed\n", g_failed);
        return 1;
    }
    std::printf("server resume store: all checks passed\n");
    return 0;
}
