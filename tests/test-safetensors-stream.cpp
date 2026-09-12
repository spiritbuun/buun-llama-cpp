#include "llama-safetensors-quant.h"
#include "llama-mmap.h"
#include "ggml.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#if defined(__linux__)
#include <sys/resource.h>
#endif

// --large --legacy reproduces the whole-tensor allocation under an external
// memory limit. Fixture generation and streamed verification stay bounded.
static void run(bool large, bool legacy, int bits, int codebook, llama_safetensors_io_mode io_mode, bool wide = false) {
    const auto path = std::filesystem::temp_directory_path() /
        ("llama-exl3-stream-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(path);
    struct cleanup {
        std::filesystem::path path;
        ~cleanup() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    } cleanup {path};
    const size_t kt = wide ? (8 * 1024 * 1024 / (bits * 32) + 8) : (large ? 1024 : 24);
    const size_t nt = wide ? 8 : (large ? 4096 : 40);
    const size_t tile = bits * 32, bytes = kt * nt * tile;
    auto fields = llama_safetensors_json {
        {"module.trellis", {{"dtype", "I16"}, {"shape", {kt, nt, size_t(bits * 16)}},
                            {"data_offsets", {0, bytes}}}},
        {"module.suh", {{"dtype", "F16"}, {"shape", {kt * 16}}, {"data_offsets", {bytes, bytes + kt * 32}}}},
        {"module.svh", {{"dtype", "F16"}, {"shape", {nt * 16}},
                        {"data_offsets", {bytes + kt * 32, bytes + (kt + nt) * 32}}}}
    };
    const size_t end = bytes + (kt + nt) * 32;
    if (codebook) fields[codebook == 2 ? "module.mul1" : "module.mcg"] = {
        {"dtype", "I32"}, {"shape", {1}}, {"data_offsets", {end, end + 4}}};
    const auto header = fields.dump();
    {
        std::ofstream out(path / "model.safetensors", std::ios::binary);
        const uint64_t n = header.size();
        out.write(reinterpret_cast<const char *>(&n), sizeof(n));
        out.write(header.data(), header.size());
        std::vector<uint8_t> chunk(65536);
        for (size_t off = 0; off < bytes; off += chunk.size()) {
            const size_t count = std::min(chunk.size(), bytes - off);
            for (size_t j = 0; j < count; ++j) chunk[j] = uint8_t(((off + j) / tile) * 17 + (off + j) % tile);
            out.write(reinterpret_cast<const char *>(chunk.data()), count);
        }
        const uint16_t one = 0x3c00;
        for (size_t i = 0; i < (kt + nt) * 16; ++i) out.write(reinterpret_cast<const char *>(&one), sizeof(one));
        if (codebook) { const uint32_t marker = 1; out.write(reinterpret_cast<const char *>(&marker), sizeof(marker)); }
        if (!out) throw std::runtime_error("fixture write failed");
    }
    const auto registry = llama_safetensors_registry::load(path, io_mode);
    const llama_safetensors_json config = {{"quantization_config", {{"quant_method", "exl3"}}}};
    llama_safetensors_quant_adapters quant(config, registry);
    const auto binding = quant.bind("module", llama_safetensors_quant_role::WEIGHT);
    if (!binding) throw std::runtime_error("missing EXL3 binding");
    if (binding->target_type != ggml_exl3_type(bits, codebook)) throw std::runtime_error("wrong EXL3 type");
    std::vector<uint8_t> result;
    if (!large || legacy) result = quant.finalize(*binding, quant.read(*binding));
    auto backing = large && llama_file::TEMP_SUPPORTED ? llama_file::create_temp(path.string()) : nullptr;
    size_t written = 0, largest = 0;
    uint64_t checksum = 0;
    quant.stream_exl3(*binding, [&](const void * ptr, size_t size) {
        const auto * data = static_cast<const uint8_t *>(ptr);
        if (size > bytes - written) throw std::runtime_error("excess stream bytes");
        largest = std::max(largest, size);
        if (!result.empty() && std::memcmp(data, result.data() + written, size)) {
            throw std::runtime_error("stream differs from old repacker");
        }
        for (size_t i = 0; i < size; ++i) {
            const size_t pos = written + i, n = pos / (kt * tile), k = pos / tile % kt;
            if (data[i] != uint8_t((k * nt + n) * 17 + pos % tile)) {
                throw std::runtime_error("incorrect tile permutation");
            }
            checksum = checksum * 31 + data[i];
        }
        if (backing) backing->write_raw(data, size);
        written += size;
    });
    if (written != bytes || largest > 8 * 1024 * 1024) throw std::runtime_error("stream bounds violated");
    if (backing) {
        backing->finish_write();
        if (backing->size() != bytes) throw std::runtime_error("backing file has wrong size");
        std::vector<uint8_t> block(65536);
        uint64_t actual = 0;
        for (size_t off = 0; off < bytes; off += block.size()) {
            const size_t count = std::min(block.size(), bytes - off);
            backing->read_raw(block.data(), count);
            for (size_t i = 0; i < count; ++i) actual = actual * 31 + block[i];
        }
        if (actual != checksum) throw std::runtime_error("backing bytes changed");
    }
    bool cancelled = false;
    try {
        quant.stream_exl3(*binding, [](const void *, size_t) { throw std::runtime_error("cancel"); });
    } catch (const std::runtime_error &) { cancelled = true; }
    if (!cancelled) throw std::runtime_error("writer failure swallowed");
    std::cout << "PASS: repacked " << written << " bytes, largest chunk " << largest << " bytes\n";
#if defined(__linux__)
    if (large) {
        struct rusage usage {};
        if (getrusage(RUSAGE_SELF, &usage) == 0) std::cout << "peak_rss_kib=" << usage.ru_maxrss << '\n';
    }
#endif
}

int main(int argc, char ** argv) try {
    const bool large = argc > 1 && std::string(argv[1]) == "--large";
    const bool legacy = argc > 2 && std::string(argv[2]) == "--legacy";
    if (large) {
        run(true, legacy, 2, 0, llama_safetensors_io_mode::BUFFERED);
    } else {
        for (auto mode : {llama_safetensors_io_mode::BUFFERED, llama_safetensors_io_mode::MMAP}) {
            for (int bits = 1; bits <= 8; ++bits) {
                for (int codebook = 0; codebook < 3; ++codebook) run(false, false, bits, codebook, mode);
            }
            run(false, false, 8, 0, mode, true); // one row exceeds the staging capacity
        }
    }
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
