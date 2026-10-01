#include "server-cache-disk.h"

#include "../../src/llama-sha256.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

constexpr char kObjectMagic[4] = { 'L', 'C', 'P', 'C' };
constexpr char kIndexMagic[4]  = { 'L', 'C', 'P', 'I' };
constexpr uint32_t kFormatVersion = 1;

constexpr uint64_t kMaxObjectBytes = 256ull * 1024 * 1024 * 1024; // 256 GiB
constexpr uint64_t kMaxTokens      = 1ull << 28;                  // 256M tokens
constexpr uint64_t kMaxCheckpoints = 4096;
constexpr size_t   kReadChunkBytes = 1ull << 20;                  // 1 MiB buffered reads
constexpr size_t   kIndexBodyLimit = 32ull * 1024 * 1024;         // an index never needs more

struct le_writer {
    std::vector<uint8_t> out;

    void put(const void * data, size_t size) {
        const auto * p = static_cast<const uint8_t *>(data);
        out.insert(out.end(), p, p + size);
    }
    template <typename T>
    void put(T value) {
        static_assert(std::is_trivially_copyable<T>::value, "le_writer needs trivial types");
        put(&value, sizeof(T));
    }
    void put_u64_string(const std::string & value) {
        put(uint32_t(value.size()));
        put(value.data(), value.size());
    }
    void put_bytes(const void * data, size_t size) {
        put(uint64_t(size));
        if (size > 0) {
            put(data, size);
        }
    }
};

struct le_reader {
    const uint8_t * data;
    size_t size;
    size_t pos = 0;

    bool need(size_t n) const { return pos + n <= size; }

    bool get(void * data, size_t n) {
        if (!need(n)) {
            return false;
        }
        std::memcpy(data, this->data + pos, n);
        pos += n;
        return true;
    }
    template <typename T>
    bool get(T & value) {
        static_assert(std::is_trivially_copyable<T>::value, "le_reader needs trivial types");
        return get(&value, sizeof(T));
    }
    bool get_string(std::string & value) {
        uint32_t len = 0;
        if (!get(len) || !need(len)) {
            return false;
        }
        value.assign(reinterpret_cast<const char *>(data + pos), len);
        pos += len;
        return true;
    }
    bool get_bytes(std::vector<uint8_t> & value) {
        uint64_t len = 0;
        if (!get(len) || len > size - pos) {
            return false;
        }
        value.assign(data + pos, data + pos + len);
        pos += len;
        return true;
    }
};

// the 40-byte trailer every LCPC/LCPI body ends with: its size, then the sha256 of everything
// before the trailer
bool seal_body(std::vector<uint8_t> & body) {
    const auto digest = llama_sha256_digest(body.data(), body.size());
    le_writer writer;
    writer.put(body.size());
    writer.put(digest.data(), digest.size());
    body.insert(body.end(), writer.out.begin(), writer.out.end());
    return true;
}

bool verify_body(const uint8_t * data, size_t size) {
    if (size < 40) {
        return false;
    }
    // the 40-byte trailer is the final 40 bytes of the body: its size, then the sha256 of
    // everything before it
    le_reader r { data + size - 40, 40 };
    uint64_t body_size = 0;
    std::array<uint8_t, 32> digest {};
    if (!r.get(body_size) || body_size != size - 40 || !r.get(digest.data(), digest.size())) {
        return false;
    }
    const auto actual = llama_sha256_digest(data, body_size);
    return std::memcmp(digest.data(), actual.data(), digest.size()) == 0;
}

std::string hex(const std::array<uint8_t, 32> & digest) {
    static const char * digits = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const uint8_t b : digest) {
        out.push_back(digits[b >> 4]);
        out.push_back(digits[b & 0xF]);
    }
    return out;
}

bool encode_checkpoint(const common_prompt_checkpoint & ckpt, le_writer & w) {
    w.put(ckpt.n_tokens);
    w.put(ckpt.id_task);
    w.put(ckpt.pos_min);
    w.put(ckpt.pos_max);
    w.put(ckpt.checkpoint_epoch);
    w.put(ckpt.checkpoint_epoch_swa);
    const auto & frontier = ckpt.computation_frontier;
    w.put(frontier.version);
    w.put(frontier.sequence_epoch);
    w.put(frontier.token_count);
    w.put(frontier.next_position);
    w.put_u64_string(frontier.execution_identity);
    w.put_u64_string(frontier.adapter_config_identity);
    w.put_u64_string(frontier.media_content_identity);
    w.put(uint8_t(ckpt.data_dft_full_sequence ? 1 : 0));
    w.put_bytes(ckpt.data_tgt.data(), ckpt.data_tgt.size());
    w.put_bytes(ckpt.data_dft.data(), ckpt.data_dft.size());
    w.put_bytes(ckpt.data_qsa.data(), ckpt.data_qsa.size());
    w.put_bytes(ckpt.accel.ring.data(), ckpt.accel.ring.size());
    w.put_bytes(ckpt.accel.spec.data(), ckpt.accel.spec.size());
    return true;
}

bool decode_checkpoint(le_reader & r, common_prompt_checkpoint & ckpt) {
    ckpt = common_prompt_checkpoint();
    if (!r.get(ckpt.n_tokens) || ckpt.n_tokens < 0 || ckpt.n_tokens > int64_t(kMaxTokens) ||
        !r.get(ckpt.id_task) || !r.get(ckpt.pos_min) || !r.get(ckpt.pos_max) ||
        !r.get(ckpt.checkpoint_epoch) || !r.get(ckpt.checkpoint_epoch_swa)) {
        return false;
    }
    auto & frontier = ckpt.computation_frontier;
    if (!r.get(frontier.version) || frontier.version > common_computation_frontier::VERSION ||
        !r.get(frontier.sequence_epoch) || !r.get(frontier.token_count) ||
        !r.get(frontier.next_position) ||
        !r.get_string(frontier.execution_identity) ||
        !r.get_string(frontier.adapter_config_identity) ||
        !r.get_string(frontier.media_content_identity)) {
        return false;
    }
    uint8_t dft_full = 0;
    if (!r.get(dft_full)) {
        return false;
    }
    ckpt.data_dft_full_sequence = dft_full != 0;

    const auto fill = [&](common_shared_byte_buffer & target) -> bool {
        std::vector<uint8_t> bytes;
        if (!r.get_bytes(bytes)) {
            return false;
        }
        target.overwrite(bytes.size(), [&](uint8_t * data, size_t size) {
            std::memcpy(data, bytes.data(), size);
        });
        return true;
    };
    return fill(ckpt.data_tgt) && fill(ckpt.data_dft) && fill(ckpt.data_qsa) &&
           fill(ckpt.accel.ring) && fill(ckpt.accel.spec);
}

} // namespace

const char * server_cache_disk_status_name(server_cache_disk_status status) noexcept {
    switch (status) {
        case server_cache_disk_status::ok:              return "ok";
        case server_cache_disk_status::dir_missing:     return "dir_missing";
        case server_cache_disk_status::dir_unwritable:  return "dir_unwritable";
        case server_cache_disk_status::index_corrupt:   return "index_corrupt";
        case server_cache_disk_status::io_error:        return "io_error";
        default:                                        return "unknown";
    }
}

server_cache_disk_tier::server_cache_disk_tier(std::string dir, std::string producer_identity,
                                               uint64_t limit)
    : dir_(dir), producer_identity(std::move(producer_identity)), limit(limit) {}

std::string server_cache_disk_tier::object_path(const std::string & name) const {
    return dir_ + "/" + name;
}

std::string server_cache_disk_tier::new_object_name(uint64_t now_us) {
    return "pc-" + std::to_string(now_us) + "-" + hex(llama_sha256_digest(
        &now_us, sizeof(now_us))).substr(0, 8) + ".bin";
}

std::unique_ptr<server_cache_disk_tier> server_cache_disk_tier::open(
        const std::string & dir,
        const std::string & producer_identity,
        uint64_t limit_bytes,
        server_cache_disk_status & status,
        std::string & error) {
    status = server_cache_disk_status::ok;
    error.clear();

    std::error_code ec;
    if (dir.empty()) {
        status = server_cache_disk_status::dir_missing;
        error = "empty directory";
        return nullptr;
    }
    if (!fs::is_directory(dir, ec)) {
        if (!ec && !fs::create_directories(dir, ec) && !ec) {
            // created it
        }
        if (ec || !fs::is_directory(dir, ec)) {
            status = server_cache_disk_status::dir_missing;
            error = ec ? ec.message() : "not a directory";
            return nullptr;
        }
    }

    // refuse a directory that is not writable before parking anything into it
    {
        const std::string probe = dir + "/.write-test";
        std::ofstream out(probe, std::ios::binary | std::ios::trunc);
        if (!out) {
            status = server_cache_disk_status::dir_unwritable;
            error = "cannot create " + probe;
            return nullptr;
        }
        out.put('0');
        out.close();
        if (!out) {
            status = server_cache_disk_status::dir_unwritable;
            error = "cannot write " + probe;
            fs::remove(probe, ec);
            return nullptr;
        }
        fs::remove(probe, ec);
    }

    auto tier = std::unique_ptr<server_cache_disk_tier>(
        new server_cache_disk_tier(dir, producer_identity, limit_bytes));
    if (!tier->read_index(error)) {
        status = server_cache_disk_status::index_corrupt;
        return nullptr;
    }
    if (!tier->build_index_from_files(error)) {
        status = server_cache_disk_status::io_error;
        return nullptr;
    }
    tier->prune();
    tier->publish_index();
    return tier;
}

server_cache_disk_tier::~server_cache_disk_tier() = default;

bool server_cache_disk_tier::contains(const llama_tokens & tokens,
                                      const std::string & adapter_config_key) const {
    for (const auto & entry : entries_) {
        if (entry.adapter_config_key == adapter_config_key && entry.tokens == tokens) {
            return true;
        }
    }
    return false;
}

void server_cache_disk_tier::drop(const std::list<server_cache_disk_entry>::iterator it) {
    std::error_code ec;
    fs::remove(object_path(it->file), ec);
    total_size_ = it->size > total_size_ ? 0 : total_size_ - it->size;
    entries_.erase(it);
}

bool server_cache_disk_tier::spill(
        const llama_tokens & tokens,
        const std::string & adapter_config_key,
        const uint8_t * main, size_t main_size,
        const uint8_t * drft, size_t drft_size,
        const std::list<common_prompt_checkpoint> & checkpoints,
        uint64_t sequence_epoch) {
    if (tokens.empty() || main_size == 0 ||
        main_size + drft_size + checkpoints.size() * 40 > kMaxObjectBytes) {
        return false;
    }

    // a duplicate (tokens, adapter) pair is already parked; a parked strict prefix of the new
    // conversation is superseded by it
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->adapter_config_key != adapter_config_key) {
            ++it;
            continue;
        }
        if (it->tokens == tokens) {
            return true; // duplicate: the conversation is already parked
        }
        const size_t lcp = [this, &tokens](const llama_tokens & a) {
            size_t n = 0;
            for (; n < a.size() && n < tokens.size() && a[n] == tokens[n]; ++n) {}
            return n;
        }(it->tokens);
        if (lcp == it->tokens.size()) {
            drop(it++);
        } else {
            ++it;
        }
    }

    le_writer body;
    body.put(kObjectMagic, 4);
    body.put(kFormatVersion);
    body.put(uint32_t(tokens.size()));
    body.put(tokens.data(), tokens.size() * sizeof(llama_token));
    body.put_u64_string(producer_identity);
    body.put_u64_string(adapter_config_key);
    body.put(sequence_epoch);
    body.put_bytes(main, main_size);
    body.put_bytes(drft, drft_size);
    body.put(uint32_t(checkpoints.size()));
    for (const auto & ckpt : checkpoints) {
        encode_checkpoint(ckpt, body);
    }
    if (!seal_body(body.out)) {
        return false;
    }

    const uint64_t now_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    const std::string name = new_object_name(now_us);
    const std::string path = object_path(name);

    // stage then rename: a torn object file is never visible to a restarted reader
    const std::string staged = path + ".tmp";
    {
        std::ofstream out(staged, std::ios::binary | std::ios::trunc);
        if (!out) {
            return false;
        }
        out.write(reinterpret_cast<const char *>(body.out.data()), body.out.size());
        out.flush();
        if (!out) {
            out.close();
            std::error_code ec;
            fs::remove(staged, ec);
            return false;
        }
        out.close();
    }
    std::error_code ec;
    if (fs::exists(path, ec)) {
        fs::remove(staged, ec);
        return false;
    }
    fs::rename(staged, path, ec);
    if (ec || !fs::exists(path, ec) || fs::exists(staged, ec)) {
        fs::remove(staged, ec);
        return false;
    }

    server_cache_disk_entry entry;
    entry.file = name;
    entry.size = body.out.size();
    entry.n_tokens = tokens.size();
    entry.tokens = tokens;
    entry.producer_identity = producer_identity;
    entry.adapter_config_key = adapter_config_key;
    entry.saved_unix_ms = now_us / 1000;
    entry.sha256 = llama_sha256_digest(body.out.data(), body.out.size() - 40);
    entries_.push_back(std::move(entry));
    total_size_ += body.out.size();

    prune();
    publish_index();
    return true;
}

bool server_cache_disk_tier::read(const server_cache_disk_entry & entry,
                                  server_cache_disk_image & out) const {
    const std::string path = object_path(entry.file);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (file_size < 0 || size_t(file_size) < 40 + 4 + 4 || size_t(file_size) != entry.size) {
        return false;
    }

    // read the whole object into one buffer: objects are bounded by kMaxObjectBytes and the caller
    // has already committed to restoring this conversation. The single sha256 pass is the
    // whole-file checksum.
    std::vector<uint8_t> data;
    data.resize(size_t(file_size));
    in.read(reinterpret_cast<char *>(data.data()), file_size);
    if (!in) {
        return false;
    }
    if (!verify_body(data.data(), data.size())) {
        return false;
    }

    le_reader r { data.data(), data.size() };
    char magic[4] = { 0 };
    uint32_t version = 0;
    uint32_t n_tokens = 0;
    if (!r.get(magic, 4) || std::memcmp(magic, kObjectMagic, 4) != 0 ||
        !r.get(version) || version != kFormatVersion ||
        !r.get(n_tokens) || n_tokens > kMaxTokens) {
        return false;
    }
    out.tokens.resize(n_tokens);
    if (n_tokens > 0 && !r.get(out.tokens.data(), n_tokens * sizeof(llama_token))) {
        return false;
    }
    std::string producer, adapter;
    if (!r.get_string(producer) || producer != producer_identity ||
        !r.get_string(adapter)) {
        return false;
    }
    if (!r.get(out.sequence_epoch)) {
        return false;
    }
    if (!r.get_bytes(out.main) || out.main.empty()) {
        return false;
    }
    if (!r.get_bytes(out.drft)) {
        return false;
    }
    uint32_t n_checkpoints = 0;
    if (!r.get(n_checkpoints) || n_checkpoints > kMaxCheckpoints) {
        return false;
    }
    out.checkpoints.clear();
    for (uint32_t i = 0; i < n_checkpoints; ++i) {
        common_prompt_checkpoint ckpt;
        if (!decode_checkpoint(r, ckpt)) {
            return false;
        }
        out.checkpoints.push_back(std::move(ckpt));
    }
    return true;
}

void server_cache_disk_tier::remove(const server_cache_disk_entry & entry) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->file == entry.file && it->size == entry.size) {
            drop(it);
            publish_index();
            return;
        }
    }
}

void server_cache_disk_tier::prune() {
    if (limit == 0) {
        return;
    }
    while (!entries_.empty() && total_size_ > limit) {
        drop(entries_.begin());
    }
}

bool server_cache_disk_tier::read_index(std::string & error) {
    const std::string path = dir_ + "/index.bin";
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return true; // no index yet: the caller builds one from the files
    }
    in.seekg(0, std::ios::end);
    const std::streamoff file_size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (file_size < 0 || size_t(file_size) > kIndexBodyLimit) {
        error = "index size " + std::to_string(file_size) + " exceeds the bound";
        return false;
    }
    std::vector<uint8_t> data;
    data.resize(size_t(file_size));
    in.read(reinterpret_cast<char *>(data.data()), file_size);
    if (!in) {
        error = "short index read";
        return false;
    }
    if (!verify_body(data.data(), data.size())) {
        error = "index checksum mismatch";
        return false;
    }
    le_reader r { data.data(), data.size() };
    char magic[4] = { 0 };
    uint32_t version = 0, n = 0;
    if (!r.get(magic, 4) || std::memcmp(magic, kIndexMagic, 4) != 0 ||
        !r.get(version) || version != kFormatVersion || !r.get(n)) {
        error = "index header mismatch";
        return false;
    }
    std::vector<server_cache_disk_entry> kept;
    kept.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        server_cache_disk_entry entry;
        std::string producer;
        if (!r.get_string(entry.file) || !r.get(entry.size) || !r.get(entry.n_tokens) ||
            !r.get(entry.saved_unix_ms) || !r.get_string(producer) ||
            !r.get_string(entry.adapter_config_key) ||
            !r.get(entry.sha256.data(), entry.sha256.size())) {
            error = "index row " + std::to_string(i) + " is truncated";
            return false;
        }
        if (producer != producer_identity) {
            continue; // another model's conversation: dropped below
        }
        if (entry.file.empty() || entry.file.find('/') != std::string::npos ||
            entry.file.rfind("pc-", 0) != 0 ||
            entry.file.size() < 5 || entry.file.compare(entry.file.size() - 4, 4, ".bin") != 0) {
            error = "index row " + std::to_string(i) + " names a bad file";
            return false;
        }
        // the row does not carry the tokens; read them back from the object header so selection
        // never has to open an object for a prefix scan (a short/corrupt file drops the row)
        {
            std::ifstream tin(object_path(entry.file), std::ios::binary);
            std::vector<uint8_t> head(4 + 4 + 4);
            if (!tin.read(reinterpret_cast<char *>(head.data()), head.size()) ||
                std::memcmp(head.data(), kObjectMagic, 4) != 0) {
                continue;
            }
            le_reader h { head.data() + 8, 4 };
            uint32_t n = 0;
            if (!h.get(n) || n != entry.n_tokens || n > kMaxTokens) {
                continue;
            }
            entry.tokens.resize(n);
            if (n > 0 && !tin.read(reinterpret_cast<char *>(entry.tokens.data()),
                                   n * sizeof(llama_token))) {
                continue;
            }
        }
        kept.push_back(std::move(entry));
    }
    if (kept.empty()) {
        return true;
    }
    // the index is authoritative: drop any object the index no longer names (including rows of
    // another model's producer identity, which were skipped above)
    for (auto it = entries_.begin(); it != entries_.end();) {
        bool found = false;
        for (const auto & kept_entry : kept) {
            if (kept_entry.file == it->file) {
                found = true;
                break;
            }
        }
        it = found ? std::next(it) : entries_.erase(it);
    }
    for (const auto & entry : kept) {
        entries_.push_back(entry);
        total_size_ += entry.size;
    }
    return true;
}

bool server_cache_disk_tier::build_index_from_files(std::string & error) {
    (void) error;
    std::vector<std::string> files;
    std::error_code ec;
    for (const auto & item : fs::directory_iterator(dir_, ec)) {
        const std::string name = item.path().filename().string();
        if (!item.is_regular_file(ec) || name.rfind("pc-", 0) != 0 ||
            name.size() < 12 || name.compare(name.size() - 4, 4, ".bin") != 0 ||
            name.compare(name.size() - 8, 4, ".tmp") == 0) {
            continue;
        }
        files.push_back(item.path().string());
    }
    std::sort(files.begin(), files.end());
    bool changed = false;
    for (const auto & path : files) {
        const std::string name = fs::path(path).filename().string();
        bool known = false;
        for (const auto & entry : entries_) {
            if (entry.file == name) {
                known = true;
                break;
            }
        }
        if (known) {
            continue;
        }
        // an object the index never named: read its header to recover the tokens
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            continue;
        }
        in.seekg(0, std::ios::end);
        const std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (file_size < 0 || size_t(file_size) > kMaxObjectBytes) {
            continue;
        }
        std::vector<uint8_t> header(4 + 4 + 4);
        in.read(reinterpret_cast<char *>(header.data()), header.size());
        if (!in || std::memcmp(header.data(), kObjectMagic, 4) != 0) {
            continue;
        }
        le_reader h { header.data() + 8, 4 };
        uint32_t n_tokens = 0;
        if (!h.get(n_tokens) || n_tokens > kMaxTokens) {
            continue;
        }
        std::vector<uint8_t> token_bytes(n_tokens * sizeof(llama_token));
        in.read(reinterpret_cast<char *>(token_bytes.data()), token_bytes.size());
        if (!in) {
            continue;
        }

        // whole-file checksum for the index row
        std::ifstream fin(path, std::ios::binary);
        llama_sha256 hash;
        std::vector<uint8_t> scratch(kReadChunkBytes);
        size_t consumed = 0;
        while (consumed < size_t(file_size)) {
            fin.read(reinterpret_cast<char *>(scratch.data()),
                     std::streamsize(std::min<size_t>(scratch.size(),
                                                      size_t(file_size) - consumed)));
            const auto n = fin.gcount();
            if (n <= 0) {
                break;
            }
            hash.update(scratch.data(), n);
            consumed += n;
        }
        if (consumed != size_t(file_size)) {
            continue;
        }

        server_cache_disk_entry entry;
        entry.file = name;
        entry.size = size_t(file_size);
        entry.n_tokens = n_tokens;
        entry.tokens.resize(n_tokens);
        if (n_tokens > 0) {
            std::memcpy(entry.tokens.data(), token_bytes.data(), token_bytes.size());
        }
        entry.saved_unix_ms = 0;
        entry.sha256 = hash.finish();
        entries_.push_back(std::move(entry));
        total_size_ += size_t(file_size);
        changed = true;
    }
    return true;
}

void server_cache_disk_tier::publish_index() {
    le_writer body;
    body.put(kIndexMagic, 4);
    body.put(kFormatVersion);
    body.put(uint32_t(entries_.size()));
    for (const auto & entry : entries_) {
        body.put_u64_string(entry.file);
        body.put(entry.size);
        body.put(entry.n_tokens);
        body.put(entry.saved_unix_ms);
        body.put_u64_string(entry.producer_identity);
        body.put_u64_string(entry.adapter_config_key);
        body.put(entry.sha256.data(), entry.sha256.size());
    }
    if (!seal_body(body.out)) {
        return;
    }
    const std::string path = dir_ + "/index.bin";
    const std::string staged = path + ".tmp";
    {
        std::ofstream out(staged, std::ios::binary | std::ios::trunc);
        if (!out) {
            return;
        }
        out.write(reinterpret_cast<const char *>(body.out.data()), body.out.size());
        out.flush();
        if (!out) {
            out.close();
            std::error_code ec;
            fs::remove(staged, ec);
            return;
        }
        out.close();
    }
    std::error_code ec;
    fs::rename(staged, path, ec);
}
