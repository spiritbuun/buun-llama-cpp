#include "llama-repack-cache.h"
#include "llama-impl.h"
#include "llama-sha256.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {
std::string hex(const std::array<uint8_t, 32> & bytes) {
    const char digits[] = "0123456789abcdef";
    std::string result;
    for (auto b : bytes) { result += digits[b >> 4]; result += digits[b & 15]; }
    return result;
}

#if defined(__linux__)
json stamp(const struct stat & s) {
    return {s.st_dev, s.st_ino, s.st_size, s.st_mtim.tv_sec, s.st_mtim.tv_nsec,
            s.st_ctim.tv_sec, s.st_ctim.tv_nsec};
}
json stamp(const fs::path & path) {
    struct stat s;
    if (stat(path.c_str(), &s) || !S_ISREG(s.st_mode)) {
        throw std::runtime_error("cannot identify repack source/cache file: " + path.string());
    }
    return stamp(s);
}
json stamp(const llama_file & file) {
    struct stat s;
    if (fstat(file.file_id(), &s)) throw std::runtime_error("cannot identify open repack cache file");
    return stamp(s);
}

std::string index_identity(const fs::path & root) {
    const auto index = root / "model.safetensors.index.json";
    return fs::exists(index) ? stamp(index).dump() : "absent";
}

std::vector<fs::path> index_shards(const fs::path & root) {
    std::vector<fs::path> paths;
    std::ifstream index(root / "model.safetensors.index.json");
    if (index) {
        // weight_map can have many thousands of names. ordered_json would make
        // insertion quadratic. Parse once; later checks only stat these files.
        nlohmann::json parsed;
        index >> parsed;
        for (const auto & item : parsed.at("weight_map").items()) {
            paths.push_back(root / item.value().get<std::string>());
        }
        std::sort(paths.begin(), paths.end());
        paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    }
    return paths;
}

std::string source_identity(const fs::path & root, const std::vector<fs::path> & indexed) {
    std::vector<fs::path> paths;
    // Match the importer's input families, not unrelated logs/download receipts
    // a user may keep beside a model. New import sidecars must join this list.
    static const char * metadata[] = {"config.json", "quantization_config.json", "hf_quant_config.json",
        "quanto_qmap.json", "model.safetensors.index.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "chat_template.jinja"};
    for (const auto * name : metadata) {
        if (fs::is_regular_file(root / name)) paths.push_back(root / name);
    }
    for (const auto & entry : fs::directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") paths.push_back(entry.path());
    }
    // Indexed shards can be nested, including below symlinked directories.
    paths.insert(paths.end(), indexed.begin(), indexed.end());
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    llama_sha256_writer hash;
    const auto base = root.string();
    hash.string(base.data(), base.size());
    for (const auto & path : paths) {
        const auto name = path.lexically_relative(root).string();
        const auto metadata = stamp(path).dump();
        hash.string(name.data(), name.size());
        hash.string(metadata.data(), metadata.size());
    }
    return hex(hash.finish());
}

struct fd_guard {
    int fd;
    ~fd_guard() { if (fd >= 0) close(fd); }
};

// Only remove the two files this helper creates. Never recursively delete a
// user-selected cache directory or unknown contents. Open mappings survive unlink.
void remove_entry(const fs::path & path) {
    std::error_code ec;
    fs::remove(path / "weights", ec);
    fs::remove(path / "receipt.json", ec);
    fs::remove(path, ec);
}
struct staged_entry {
    fs::path path;
    ~staged_entry() { remove_entry(path); }
};
#else
std::string source_identity(const fs::path &, const std::vector<fs::path> &) {
    throw std::runtime_error("persistent repack cache currently requires Linux; omit --repack-cache");
}
#endif
} // namespace

llama_repack_cache::llama_repack_cache(const fs::path & source, const fs::path & directory) :
    source_(fs::canonical(source)), directory_(fs::weakly_canonical(directory)) {
    const auto relative = directory_.lexically_relative(source_);
    if (relative.empty() || *relative.begin() != "..") {
        throw std::runtime_error("--repack-cache must be outside the source model directory");
    }
#if defined(__linux__)
    index_identity_ = index_identity(source_);
    indexed_shards_ = index_shards(source_);
#endif
    identity_ = source_identity(source_, indexed_shards_);
    validate_source();
}

void llama_repack_cache::validate_source() const {
#if defined(__linux__)
    if (index_identity(source_) != index_identity_) {
        throw std::runtime_error("safetensors index changed while loading cached weights");
    }
#endif
    if (source_identity(source_, indexed_shards_) != identity_) {
        throw std::runtime_error("safetensors source changed while preparing/loading cached weights");
    }
}

std::unique_ptr<llama_file> llama_repack_cache::get(
        const std::string & layout, size_t expected, bool check,
        const std::function<void()> & poll,
        const std::function<void(const std::function<void(const void *, size_t)> &)> & produce) const {
#if defined(__linux__)
    poll();
    validate_source();
    // Bump when canonical repacking semantics change. A build commit is not an
    // identity: unrelated rebuilds should not create another model-sized cache.
    llama_sha256_writer key;
    const std::string domain = "llama-prepared-weights-v1";
    key.string(domain.data(), domain.size());
    key.string(identity_.data(), identity_.size());
    key.string(layout.data(), layout.size());
    key.u64(expected);
    const uint32_t native_endian = 1;
    key.bytes(&native_endian, sizeof(native_endian));
    const auto id = hex(key.finish());
    const fs::path entry = directory_ / id;
    fs::create_directories(directory_);
    fd_guard lock{open((directory_ / (id + ".lock")).c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600)};
    if (lock.fd < 0) throw std::runtime_error("cannot open repack cache lock");
    // Cancellation remains responsive while another loader prepares this entry.
    while (flock(lock.fd, LOCK_EX | LOCK_NB)) {
        if (errno != EWOULDBLOCK && errno != EINTR) throw std::runtime_error("cannot lock repack cache");
        poll();
        usleep(10000);
    }
    std::unique_ptr<llama_file> hit;
    json receipt;
    try {
        std::ifstream input(entry / "receipt.json");
        if (input) {
            input >> receipt;
            hit = std::make_unique<llama_file>((entry / "weights").c_str(), "rb");
            if (receipt.at("key") != id || hit->size() != expected || receipt.at("stat") != stamp(*hit)) hit.reset();
        }
    } catch (const std::exception &) { hit.reset(); }
    if (hit && check) {
        llama_sha256 hash;
        std::vector<uint8_t> buffer(1024 * 1024);
        for (size_t off = 0; off < expected;) {
            poll();
            const auto size = std::min(buffer.size(), expected - off);
            hit->read_raw(buffer.data(), size);
            hash.update(buffer.data(), size);
            off += size;
        }
        if (receipt.value("sha256", std::string()) != hex(hash.finish())) hit.reset();
        else hit->seek(0, SEEK_SET);
    }
    if (hit) {
        validate_source();
        LLAMA_LOG_INFO("repack cache: reuse %s (%.2f MiB)\n", layout.c_str(), expected / 1048576.0);
        return hit;
    }
    LLAMA_LOG_INFO("repack cache: retain %s (%.2f MiB) in %s\n", layout.c_str(), expected / 1048576.0,
                   directory_.string().c_str());
    std::string pattern = (directory_ / ".preparing-XXXXXX").string();
    if (!mkdtemp(pattern.data())) throw std::runtime_error("cannot create repack staging directory");
    staged_entry staging{pattern};
    auto file = std::make_unique<llama_file>((staging.path / "weights").c_str(), "w+b");
    llama_sha256 hash;
    size_t written = 0, unsynced = 0;
    produce([&](const void * data, size_t size) {
        poll();
        if (size > expected - written) throw std::runtime_error("cached tensor exceeds destination");
        file->write_raw(data, size);
        hash.update(data, size);
        written += size;
        unsynced += size;
        if (unsynced >= 64 * 1024 * 1024) { file->sync_write(); unsynced = 0; }
    });
    if (written != expected) throw std::runtime_error("cached tensor is incomplete");
    file->finish_write();
    validate_source();
    receipt = {{"key", id}, {"layout", layout}, {"stat", stamp(*file)}, {"sha256", hex(hash.finish())}};
    {
        llama_file manifest((staging.path / "receipt.json").c_str(), "wb");
        const auto text = receipt.dump(2);
        manifest.write_raw(text.data(), text.size());
        manifest.finish_write();
    }
    poll();
    fd_guard staged_dir{open(staging.path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
    if (staged_dir.fd < 0 || fsync(staged_dir.fd)) throw std::runtime_error("cannot sync repack staging directory");
    remove_entry(entry);
    fs::rename(staging.path, entry);
    fd_guard dir{open(directory_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
    if (dir.fd < 0 || fsync(dir.fd)) throw std::runtime_error("cannot sync repack cache directory");
    return file;
#else
    (void) layout; (void) expected; (void) check; (void) poll; (void) produce;
    throw std::runtime_error("persistent repack cache requires Linux");
#endif
}
