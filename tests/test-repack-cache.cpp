#include "llama-repack-cache.h"
#include "ggml.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

int main() {
#if !defined(__linux__)
    return 77;
#else
    std::string pattern = (fs::temp_directory_path() / "repack-test-XXXXXX").string();
    require(mkdtemp(pattern.data()), "mkdtemp");
    const fs::path root(pattern), source = root / "source", directory = root / "cache";
    struct cleanup { fs::path p; ~cleanup() { fs::remove_all(p); } } cleanup{root};
    try {
        for (int type = GGML_TYPE_EXL3_1; type <= GGML_TYPE_EXL3N_8; ++type) {
            const auto t = static_cast<ggml_type>(type);
            std::vector<uint8_t> row(ggml_type_size(t), 0);
            require(ggml_validate_row_data(t, row.data(), row.size()), "valid EXL3 row rejected");
            require(!ggml_validate_row_data(t, row.data(), row.size()-1), "short EXL3 row accepted");
            if (ggml_type_is_exl3_ngram(t)) {
                const uint16_t nan = 0x7e00;
                std::memcpy(row.data(), &nan, 2);
                require(!ggml_validate_row_data(t, row.data(), row.size()), "EXL3 ngram NaN accepted");
                const uint16_t inf = 0x7c00;
                std::memcpy(row.data(), &inf, 2);
                require(!ggml_validate_row_data(t, row.data(), row.size()), "EXL3 ngram infinity accepted");
            }
        }
        fs::create_directory(source);
        { std::ofstream out(source / "model.safetensors"); out << "fixture"; }
        { std::ofstream out(source / "config.json"); out << "{}"; }
        // Production-sized indexes must not be reparsed for every cache tensor.
        nlohmann::json index = {{"weight_map", nlohmann::json::object()}};
        for (int i = 0; i < 12000; ++i) index["weight_map"]["tensor." + std::to_string(i)] = "model.safetensors";
        { std::ofstream out(source / "model.safetensors.index.json"); out << index.dump(); }
        llama_repack_cache cache(source, directory);
        require(!fs::exists(directory), "metadata probe creates cache");
        { std::ofstream log(source / "server.log"); log << "unrelated output"; }
        cache.validate_source();
        std::string bytes(65536, 'A');
        int calls = 0;
        const auto produce = [&](const auto & write) { ++calls; write(bytes.data(), bytes.size()); };
        const auto poll = []{};
        auto first = cache.get("tensor:type:shape", bytes.size(), false, poll, produce);
        auto hit = cache.get("tensor:type:shape", bytes.size(), true, poll, produce);
        require(calls == 1, "cache hit repacked");
        std::string actual(bytes.size(), 0);
        hit->read_raw(actual.data(), actual.size());
        require(actual == bytes, "cache bytes differ");
        fs::path entry;
        for (const auto & p : fs::directory_iterator(directory)) if (p.is_directory()) entry = p.path();
        require(!entry.empty(), "entry missing");

        // An ordinary in-place edit is invalidated without reading every cache byte.
        { std::fstream f(entry / "weights", std::ios::in | std::ios::out | std::ios::binary); f.put('B'); }
        cache.get("tensor:type:shape", bytes.size(), false, poll, produce);
        require(calls == 2, "edited cache reused");
        fs::resize_file(entry / "weights", 10);
        cache.get("tensor:type:shape", bytes.size(), false, poll, produce);
        require(calls == 3, "truncated cache reused");

        // Force the checksum path by simulating damage with unchanged receipt stat.
        { std::ifstream f(entry / "receipt.json"); nlohmann::ordered_json r; f >> r;
          r["sha256"] = "wrong"; std::ofstream o(entry / "receipt.json"); o << r.dump(); }
        cache.get("tensor:type:shape", bytes.size(), true, poll, produce);
        require(calls == 4, "checksum mismatch ignored");

        bool cancelled = false;
        try {
            cache.get("cancelled", bytes.size(), false, poll, [&](const auto & write) {
                write(bytes.data(), 11); throw std::runtime_error("cancel");
            });
        } catch (const std::runtime_error &) { cancelled = true; }
        require(cancelled, "cancellation swallowed");
        for (const auto & p : fs::directory_iterator(directory))
            require(p.path().filename().string().find(".preparing-") != 0, "cancel left staging");
        bool short_write = false;
        try { cache.get("short", bytes.size(), false, poll, [](const auto & write) { write("x", 1); }); }
        catch (const std::runtime_error &) { short_write = true; }
        require(short_write, "incomplete tensor published");
        bool disk_failure = false;
        try { cache.get("failed-write", bytes.size(), false, poll, [](const auto &) { throw std::runtime_error("I/O"); }); }
        catch (const std::runtime_error &) { disk_failure = true; }
        require(disk_failure, "producer I/O failure swallowed");

        // Forked loaders contend on the same entry; only one producer is called.
        const auto parallel = [&](const auto & write) {
            std::ofstream marker(root / "producer-count", std::ios::app); marker << "one\n"; marker.close();
            usleep(100000);
            write(bytes.data(), bytes.size());
        };
        pid_t children[2];
        for (auto & pid : children) {
            pid = fork(); require(pid >= 0, "fork");
            if (pid == 0) {
                try { cache.get("concurrent", bytes.size(), true, poll, parallel); _exit(0); }
                catch (...) { _exit(1); }
            }
        }
        for (auto pid : children) { int status; waitpid(pid, &status, 0); require(status == 0, "parallel loader failed"); }
        std::ifstream count(root / "producer-count");
        std::string line; int count_lines = 0; while (std::getline(count, line)) ++count_lines;
        require(count_lines == 1, "concurrent duplicate producer");

        const auto crashed = fork(); require(crashed >= 0, "fork crash fixture");
        if (crashed == 0) {
            cache.get("interrupted", bytes.size(), false, poll, [&](const auto & write) {
                write(bytes.data(), 8); _exit(42);
            });
            _exit(1);
        }
        int crashed_status; waitpid(crashed, &crashed_status, 0);
        require(WIFEXITED(crashed_status) && WEXITSTATUS(crashed_status) == 42, "crash fixture failed");
        auto recovered = cache.get("interrupted", bytes.size(), true, poll, [&](const auto & write) {
            write(bytes.data(), bytes.size());
        });
        recovered->read_raw(actual.data(), actual.size());
        require(actual == bytes, "partial crash entry reused");

        bool oversized = false;
        try { cache.get("oversized", 1, false, poll, [&](const auto & write) { write(bytes.data(), 2); }); }
        catch (const std::runtime_error &) { oversized = true; }
        require(oversized, "oversized producer accepted");

        const auto limited = fork(); require(limited >= 0, "fork write-limit fixture");
        if (limited == 0) {
            const rlimit limit{4096,4096};
            if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(2);
            signal(SIGXFSZ, SIG_IGN);
            try {
                cache.get("write-limit", bytes.size(), false, poll, [&](const auto & write) {
                    write(bytes.data(), bytes.size());
                });
                _exit(1);
            } catch (const std::runtime_error &) { _exit(0); }
        }
        int limited_status; waitpid(limited, &limited_status, 0);
        require(limited_status == 0, "real short-write failure was not propagated");

        // Remove published names while a reader owns an open handle/mapping.
        llama_mmap mapping(first.get(), 0);
        fs::remove(entry / "weights"); fs::remove(entry / "receipt.json"); fs::remove(entry);
        // first refers to an earlier generation, unlinked by the corruption rebuild.
        // It was deliberately edited to B above; its remaining bytes stay accessible.
        require(static_cast<char *>(mapping.addr())[1] == 'A', "cleanup broke live mapping");
        cache.get("tensor:type:shape", bytes.size(), false, poll, produce);
        require(calls == 5, "removed entry not rebuilt");
        cache.get("new-layout", bytes.size(), false, poll, produce);
        require(calls == 6, "layout identity ignored");

        { std::ofstream out(source / "config.json"); out << "{\"new\":true}"; }
        bool changed = false; try { cache.validate_source(); } catch (...) { changed = true; }
        require(changed, "config edit ignored");
        llama_repack_cache next(source, directory);
        next.get("tensor:type:shape", bytes.size(), false, poll, produce);
        require(calls == 7, "changed config reused stale entry");
        { std::ofstream out(source / "model.safetensors"); out << "changed"; }
        changed = false; try { next.validate_source(); } catch (...) { changed = true; }
        require(changed, "weight edit ignored");
        llama_repack_cache indexed(source, directory);
        { std::ofstream out(source / "model.safetensors.index.json"); out << "{\"weight_map\":{}}"; }
        changed = false; try { indexed.validate_source(); } catch (...) { changed = true; }
        require(changed, "index edit ignored");
        std::cout << "repack-cache: reuse/bytes/corruption/cancellation/concurrency/cleanup/identity PASS\n";
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
#endif
}
