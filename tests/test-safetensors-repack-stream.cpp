#include "llama-safetensors-quant.h"
#include "llama-mmap.h"
#include "ggml.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#if defined(__linux__)
#include <sys/resource.h>
#endif

using json = llama_safetensors_json;
namespace fs = std::filesystem;
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

struct fixture {
    fs::path dir;
    json config, fields = json::object();
    struct field { size_t size; std::function<uint8_t(size_t)> byte; };
    std::vector<field> payload;
    size_t bytes = 0;
    std::string module = "module";

    void add(const std::string & name, const std::string & dtype, std::vector<size_t> shape,
             size_t element, std::function<uint8_t(size_t)> byte) {
        size_t count = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
        const size_t size = count * element;
        fields[module + name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {bytes, bytes + size}}};
        payload.push_back({size, std::move(byte)});
        bytes += size;
    }
    fixture(const fs::path & path, const std::string & format, bool large, bool bf16, bool asymmetric,
            size_t group, bool expert = false, bool tail = false) : dir(path) {
        fs::create_directories(dir);
        if (expert) module = "model.layers.0.mlp.experts.0.gate_proj";
        const size_t k = large ? 8192 : 1024, n = large ? 65536 : (tail ? 4103 : 4104);
        const size_t g = group ? group : k;
        const std::string scale_type = bf16 ? "BF16" : "F16";
        const auto codes = [](size_t i) { return uint8_t(i * 13 + i / 257); };
        const auto scales = [bf16](size_t i) {
            const float value = float(1 + (i / 2) % 7) / 32;
            uint16_t b = bf16 ? ggml_fp32_to_bf16(value).bits : ggml_fp32_to_fp16(value);
            return uint8_t(b >> ((i % 2) * 8));
        };
        if (format == "awq" || format == "gptq4" || format == "gptq8") {
            const bool awq = format == "awq", eight = format == "gptq8";
            const size_t pack = eight ? 4 : 8;
            config = {{"quantization_config", {{"quant_method", awq ? "awq" : "gptq"},
                {"bits", eight ? 8 : 4}, {"group_size", group ? int(group) : -1},
                {"version", "gemm"}, {"zero_point", true}, {"desc_act", false}, {"sym", eight}}}};
            add(".qweight", "I32", awq ? std::vector<size_t>{k, n / 8} : std::vector<size_t>{k / pack, n}, 4, codes);
            add(".qzeros", "I32", {k / g, n / pack}, 4, eight ? std::function<uint8_t(size_t)>([](size_t) { return 0x7f; }) : codes);
            add(".scales", scale_type, {k / g, n}, 2, scales);
        } else {
            const bool nv = format == "nvfp4", eight = format == "int8";
            const size_t bits = eight ? 8 : 4;
            json weights = {{"type", nv ? "float" : "int"}, {"num_bits", bits},
                {"strategy", nv ? "tensor_group" : "group"}, {"group_size", nv ? 16 : g},
                {"symmetric", !asymmetric}, {"dynamic", false}, {"actorder", nullptr},
                {"block_structure", nullptr}, {"scale_dtype", nullptr}, {"zp_dtype", nullptr}};
            if (nv) weights["scale_dtype"] = "torch.float8_e4m3fn";
            if (asymmetric) weights["zp_dtype"] = "torch.int8";
            const std::string fmt = nv ? "nvfp4-pack-quantized" : "pack-quantized";
            config = {{"quantization_config", {{"quant_method", "compressed-tensors"}, {"format", fmt},
                {"config_groups", {{"quant", {{"format", fmt}, {"targets", {module}}, {"weights", weights},
                    {"input_activations", nullptr}, {"output_activations", nullptr}}}}}}}};
            if (nv) {
                config["quantization_config"]["format"] = "mixed-precision";
                config["quantization_config"]["config_groups"]["quant"]["weights"]["actorder"] = "static";
                auto activations = weights;
                activations["dynamic"] = "local";
                config["quantization_config"]["config_groups"]["quant"]["input_activations"] = activations;
            }
            if (nv) {
                add(".weight_packed", "U8", expert ? std::vector<size_t>{3, n / 3, k / 2} :
                    std::vector<size_t>{n, k / 2}, 1, codes);
                add(".weight_scale", "F8_E4M3", expert ? std::vector<size_t>{3, n / 3, k / 16} :
                    std::vector<size_t>{n, k / 16}, 1, [](size_t i) { return uint8_t(0x20 + i % 16); });
                // The adapter validates the complete NVFP4 module contract.
                add(".weight_global_scale", "F32", {1}, 4, [](size_t i) { return uint8_t(0x3f800000u >> (8 * i)); });
                add(".input_global_scale", "F32", {1}, 4, [](size_t i) { return uint8_t(0x3f800000u >> (8 * i)); });
            } else {
                add(".weight_packed", "I32", {n, k / (32 / bits)}, 4, codes);
                add(".weight_scale", scale_type, {n, k / g}, 2, scales);
                add(".weight_shape", "I64", {2}, 8, [n,k](size_t i) {
                    return uint8_t(uint64_t(i < 8 ? n : k) >> (8 * (i % 8)));
                });
                if (asymmetric) add(".weight_zero_point", "I32", {(n + 7) / 8, k / g}, 4, codes);
            }
        }
        std::string header = fields.dump();
        header.append((8 - header.size() % 8) % 8, ' ');
        const uint64_t size = header.size();
        llama_file out((dir / "model.safetensors").string().c_str(), "wb");
        out.write_raw(&size, 8);
        out.write_raw(header.data(), header.size());
        std::vector<uint8_t> block(65536);
        size_t unsynced = 0;
        for (const auto & f : payload) {
            for (size_t off = 0; off < f.size; off += block.size()) {
                const size_t count = std::min(block.size(), f.size - off);
                for (size_t j = 0; j < count; ++j) block[j] = f.byte(off + j);
                out.write_raw(block.data(), count);
                unsynced += count;
                if (unsynced >= 32 * 1024 * 1024) { out.sync_write(); unsynced = 0; }
            }
        }
        out.finish_write();
    }
};

int main(int argc, char ** argv) try {
    const bool large = argc > 1;
    const std::string format = large ? argv[1] : "int8";
    const auto dir = fs::temp_directory_path() / ("llama-repack-stream-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct cleanup { fs::path dir; ~cleanup() { std::error_code ec; fs::remove_all(dir, ec); } } cleanup{dir};
    const auto run = [&](const std::string & fmt, bool bf16, bool asymmetric, size_t group, bool expert, bool tail) {
        fixture f(dir, fmt, large, bf16, asymmetric, group, expert, tail);
        for (auto mode : {llama_safetensors_io_mode::BUFFERED, llama_safetensors_io_mode::MMAP}) {
            const bool mapped_input = argc > 2 && std::string(argv[2]) == "--mmap";
            if (large && (mode == llama_safetensors_io_mode::MMAP) != mapped_input) continue;
            const auto registry = llama_safetensors_registry::load(dir, mode);
            llama_safetensors_quant_adapters quant(f.config, registry);
            const auto binding = quant.bind(f.module, llama_safetensors_quant_role::WEIGHT);
            require(bool(binding), "missing binding");
            require(quant.can_stream(*binding), "binding not streamable");
            if (argc > 2 && std::string(argv[2]) == "--legacy") {
                const auto bytes = quant.finalize(*binding, quant.read(*binding));
                std::cout << "legacy_bytes=" << bytes.size() << '\n';
                continue;
            }
            std::vector<uint8_t> expected;
            if (!large) expected = quant.finalize(*binding, quant.read(*binding));
            size_t offset = 0, chunks = 0, largest = 0;
            const auto start = std::chrono::steady_clock::now();
#if defined(__linux__)
            rusage before {}, after {};
            getrusage(RUSAGE_SELF, &before);
#endif
            quant.stream(*binding, [&](const void * data, size_t size) {
                require(size > 0 && size <= 4 * 1024 * 1024, "unbounded output strip");
                if (!large) {
                    require(offset <= expected.size() && size <= expected.size() - offset, "excess output");
                    require(std::memcmp(data, expected.data() + offset, size) == 0, "repacked bytes differ");
                }
                offset += size;
                largest = std::max(largest, size);
                ++chunks;
            });
            if (large) {
                std::cout << "stream_ms=" << std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
#if defined(__linux__)
                getrusage(RUSAGE_SELF, &after);
                std::cout << " input_blocks_512=" << after.ru_inblock - before.ru_inblock
                          << " major_faults=" << after.ru_majflt - before.ru_majflt
                          << " minor_faults=" << after.ru_minflt - before.ru_minflt
                          << " peak_rss_kib=" << after.ru_maxrss;
#endif
                std::cout << '\n';
            }
            size_t rows = 1;
            for (size_t i = 1; i < binding->target_shape.size(); ++i) rows *= binding->target_shape[i];
            require(offset == rows * ggml_row_size(binding->target_type, binding->target_shape[0]), "wrong total size");
            require(chunks > 1, "fixture did not exercise multiple strips");
            if (!large) {
                bool cancelled = false;
                try { quant.stream(*binding, [](const void *, size_t) { throw std::runtime_error("stop"); }); }
                catch (const std::runtime_error & e) { cancelled = std::string(e.what()) == "stop"; }
                require(cancelled, "writer cancellation swallowed");
                size_t resumed = 0;
                quant.stream(*binding, [&](const void * data, size_t size) {
                    require(resumed <= expected.size() && size <= expected.size() - resumed,
                            "excess output after cancellation");
                    require(std::memcmp(data, expected.data() + resumed, size) == 0,
                            "repacked bytes differ after cancellation");
                    resumed += size;
                });
                require(resumed == expected.size(), "incomplete stream after cancellation");
                auto malformed = *binding;
                --malformed.target_shape[1];
                bool rejected = false;
                try { quant.stream(malformed, [](const void *, size_t) {}); }
                catch (const std::runtime_error &) { rejected = true; }
                require(rejected, "inconsistent full-tensor shape accepted");
            }
            std::cout << fmt << " bf16=" << bf16 << " asymmetric=" << asymmetric << " group=" << group
                      << " expert=" << expert << " tail=" << tail << " mode=" << int(mode)
                      << " bytes=" << offset << " chunks=" << chunks << " largest=" << largest << " PASS\n";
        }
    };
    if (large) {
        run(format, true, false, 128, false, false);
    } else {
        for (const auto & fmt : {"nvfp4", "awq", "gptq4", "gptq8", "int4", "int8"}) {
            for (bool bf16 : {false, true}) {
                if (std::string(fmt) == "int8" && !bf16) continue;
                run(fmt, bf16, false, 128, false, false);
            }
        }
        for (const auto & fmt : {"awq", "gptq4", "gptq8"}) run(fmt, false, false, 0, false, false);
        run("nvfp4", true, false, 128, true, false);
        for (bool expert : {false, true}) {
            for (bool bf16 : {false, true}) run("int4", bf16, true, 32, expert, true);
        }
        // Invalid scales in the final strip must not escape validation just
        // because the preceding strips have already been written.
        for (const auto & fmt : {"nvfp4", "awq", "gptq4", "gptq8", "int4", "int8"}) {
            fixture f(dir, fmt, false, true, false, 128);
            {
                const auto registry = llama_safetensors_registry::load(dir, llama_safetensors_io_mode::BUFFERED);
                const auto * scale = registry.find(f.module +
                    ((std::string(fmt) == "awq" || std::string(fmt).find("gptq") == 0) ? ".scales" : ".weight_scale"));
                require(scale != nullptr, "missing test scale");
                std::fstream file(dir / "model.safetensors", std::ios::in | std::ios::out | std::ios::binary);
                file.seekp(scale->offset + scale->size - (std::string(fmt) == "nvfp4" ? 1 : 2));
                const uint8_t bad[2] = {0xff, 0x7f}; // non-finite E4M3 / BF16
                file.write(reinterpret_cast<const char *>(bad), std::string(fmt) == "nvfp4" ? 1 : 2);
                require(bool(file), "invalid-scale fixture write failed");
            }
            for (auto mode : {llama_safetensors_io_mode::BUFFERED, llama_safetensors_io_mode::MMAP}) {
                const auto registry = llama_safetensors_registry::load(dir, mode);
                llama_safetensors_quant_adapters quant(f.config, registry);
                const auto binding = quant.bind(f.module, llama_safetensors_quant_role::WEIGHT);
                bool old_rejected = false, stream_rejected = false;
                try { quant.read(*binding); } catch (const std::runtime_error &) { old_rejected = true; }
                try { quant.stream(*binding, [](const void *, size_t) {}); }
                catch (const std::runtime_error &) { stream_rejected = true; }
                require(old_rejected && stream_rejected, "invalid tail scale accepted");
            }
            std::cout << fmt << " invalid final scale rejected PASS\n";
        }
    }
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
