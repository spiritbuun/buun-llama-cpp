#include "llama-safetensors.h"
#include "llama-safetensors-qwen4exp.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-cpp.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

// A complete, small Qwen4 model: real metadata importer, PLE hash/gather,
// recurrent convolution and MoE graph. No trained-model downloads required.
struct fixture {
    fs::path dir;
    json header = json::object();
    json config;
    std::vector<uint8_t> data, ple;

    void add(const std::string & name, const std::string & dtype,
             const std::vector<size_t> & shape, const void * bytes, size_t size) {
        header[name] = {{"dtype", dtype}, {"shape", shape}, {"data_offsets", {data.size(), data.size() + size}}};
        const auto * p = static_cast<const uint8_t *>(bytes);
        data.insert(data.end(), p, p + size);
    }
    void floats(const std::string & name, std::vector<size_t> shape, bool norm = false) {
        const size_t count = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) values[i] = norm ? 0.0f : float(int((i * 7 + data.size()/4) % 29) - 14) / 512;
        add(name, "F32", shape, values.data(), values.size() * sizeof(float));
    }
    void u64s(const std::string & name, std::vector<uint64_t> values) {
        add(name, "U64", {values.size()}, values.data(), values.size() * sizeof(uint64_t));
    }
    static void write_json(const fs::path & path, const json & value) {
        std::ofstream f(path);
        f << value.dump();
        require(bool(f), "writing fixture JSON failed");
    }
    fixture(fs::path path, bool trellis, bool large = false) : dir(std::move(path)) {
        fs::create_directories(dir);
        config = {
            {"model_type", "qwen4_exp"}, {"num_hidden_layers", 2}, {"hidden_size", 128},
            {"vocab_size", 32}, {"max_position_embeddings", 128}, {"mtp_num_hidden_layers", 0},
            {"num_attention_heads", 1}, {"num_key_value_heads", 1}, {"head_dim", 128},
            {"linear_num_key_heads", 1}, {"linear_num_value_heads", 2},
            {"linear_key_head_dim", 128}, {"linear_value_head_dim", 128}, {"linear_conv_kernel_dim", 4},
            {"full_attention_interval", 2}, {"layer_types", {"linear_attention", "full_attention"}},
            {"indexer_n_heads", 1}, {"indexer_head_dim", 128}, {"indexer_budget", 4},
            {"indexer_compress_ratio", 1}, {"hc_count", 2}, {"hc_lowrank", 8},
            {"num_experts", 2}, {"num_experts_per_tok", 1}, {"moe_intermediate_size", 128},
            {"shared_expert_intermediate_size", 128}, {"rms_norm_eps", 1e-5},
            {"rope_parameters", {{"rope_theta", 10000.0}, {"partial_rotary_factor", 0.25}, {"mrope_section", {4, 4, 8}}}},
            {"split_ngram_parts", 2}, {"ple_layer_ids", {1}}, {"ngram_size", 2},
            {"heads_per_ngram", 2}, {"ple_conv_kernel_size", 2}, {"eos_token_id", 0},
        };
        write_json(dir / "config.json", config);
        json vocab = json::object();
        for (int i = 0; i < 32; ++i) vocab[i == 0 ? "<|vision_pad|>" : "t" + std::to_string(i)] = i;
        write_json(dir / "tokenizer.json", {{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", json::array()}}}});
        floats("model.embed_tokens.weight", {32, 128});
        floats("lm_head.weight", {32, 128});
        const auto hc = [&](const std::string & prefix, bool inject) {
            floats(prefix + "hc_norm.weight", {256}, true);
            floats(prefix + "input_mix_weight_down.weight", {8, 256});
            floats(prefix + "input_mix_weight_up.weight", {256, 8});
            if (inject) floats(prefix + "block_inject_weight.weight", {2, 256});
        };
        hc("model.hyper_connection_mixer.", false);
        const std::string layer = "model.layers.0.";
        hc(layer + "attn_hyper_connection.", true);
        hc(layer + "mlp_hyper_connection.", true);
        const std::string lin = layer + "linear_attn.";
        floats(lin + "in_proj_qkv.weight", {512, 128});
        floats(lin + "in_proj_z.weight", {256, 128});
        floats(lin + "in_proj_a.weight", {2, 128});
        floats(lin + "in_proj_b.weight", {2, 128});
        floats(lin + "conv1d.weight", {512, 1, 4});
        floats(lin + "A_log", {2});
        floats(lin + "dt_bias", {2});
        floats(lin + "norm.weight", {128});
        floats(lin + "out_proj.weight", {128, 256});
        floats(layer + "mlp.gate.weight", {2, 128});
        floats(layer + "mlp.experts.gate_up_proj", {2, 256, 128});
        floats(layer + "mlp.experts.down_proj", {2, 128, 128});
        floats(layer + "mlp.shared_expert_gate.weight", {128});
        for (const char * proj : {"gate", "up", "down"}) floats(layer + "mlp.shared_expert." + proj + "_proj.weight", {128, 128});
        floats(layer + "ple.key_proj.weight", {256, 320});
        floats(layer + "ple.value_proj.weight", {128, 320});
        for (const char * norm : {"key", "query", "conv"}) floats(layer + "ple.norm_" + norm + ".weight", {256}, true);
        floats(layer + "ple.conv1d.weight", {256, 1, 2});
        // Keep an attention layer as well: the real hybrid model maintains
        // attention cells for PLE predecessor history, even on recurrent layers.
        const std::string full = "model.layers.1.";
        hc(full + "attn_hyper_connection.", true);
        hc(full + "mlp_hyper_connection.", true);
        floats(full + "self_attn.q_proj.weight", {256, 128});
        floats(full + "self_attn.k_proj.weight", {128, 128});
        floats(full + "self_attn.v_proj.weight", {128, 128});
        floats(full + "self_attn.o_proj.weight", {128, 128});
        floats(full + "self_attn.q_norm.weight", {128}, true);
        floats(full + "self_attn.k_norm.weight", {128}, true);
        floats(full + "self_attn.indexer.index_qk_proj.weight", {256, 128});
        floats(full + "self_attn.indexer.q_layernorm.weight", {128}, true);
        floats(full + "self_attn.indexer.k_layernorm.weight", {128}, true);
        floats(full + "mlp.gate.weight", {2, 128});
        floats(full + "mlp.experts.gate_up_proj", {2, 256, 128});
        floats(full + "mlp.experts.down_proj", {2, 128, 128});
        floats(full + "mlp.shared_expert_gate.weight", {128});
        for (const char * proj : {"gate", "up", "down"}) floats(full + "mlp.shared_expert." + proj + "_proj.weight", {128, 128});
        const std::string base = layer + "ple.ple_embedding.";
        const std::string aux = base + (trellis ? "ngram_embedding." : "");
        u64s(aux + "layer_multipliers", {3, 5});
        u64s(aux + (trellis ? "head_offsets" : "ngram_heads_offsets"), {0, 8});
        u64s(aux + (trellis ? "head_vocab_sizes" : "ngram_heads_vocab_sizes"), {8, 8});
        const size_t words = trellis ? 21 : 160;
        std::vector<uint16_t> table(16 * words);
        for (size_t row = 0; row < 16; ++row) {
            for (size_t col = 0; col < words; ++col) {
                table[row * words + col] = trellis ? uint16_t(row * 137 + col * 7919) :
                    ggml_fp32_to_bf16(float(int((row * 7 + col) % 31) - 15) / 32).bits;
            }
            if (trellis) table[row * words] = ggml_fp32_to_fp16(0.125f + row / 128.0f);
        }
        const auto * bytes = reinterpret_cast<const uint8_t *>(table.data());
        ple.assign(bytes, bytes + table.size() * 2);
        // Deliberately split INSIDE a head's row range, not at the hash-head boundary.
        add(base + "ngram_embedding.shard_0" + (trellis ? ".trellis" : ".weight"), trellis ? "I16" : "BF16", {5, words}, bytes, 5 * words * 2);
        add(base + "ngram_embedding.shard_1" + (trellis ? ".trellis" : ".weight"), trellis ? "I16" : "BF16", {11, words}, bytes + 5 * words * 2, 11 * words * 2);
        // Sparse padding supplies a real >4-GiB descriptor and streamed payload
        // without constructing a table-sized vector. The first 16 rows (the
        // active hash ranges) remain identical to the small control.
        size_t padding = 0;
        if (large) {
            const size_t rows = (size_t(1) << 32) / (words * 2) + 1;
            padding = (rows - 16) * words * 2;
            auto & last = header[base + "ngram_embedding.shard_1" + (trellis ? ".trellis" : ".weight")];
            last["shape"][0] = rows - 5;
            last["data_offsets"][1] = data.size() + padding;
        }
        std::string h = header.dump();
        h.append((8 - h.size() % 8) % 8, ' ');
        uint64_t len = h.size();
        std::ofstream f(dir / "model.safetensors", std::ios::binary);
        f.write(reinterpret_cast<const char *>(&len), sizeof(len));
        f.write(h.data(), h.size());
        f.write(reinterpret_cast<const char *>(data.data()), data.size());
        if (padding) {
            f.seekp(padding - 1, std::ios::cur);
            f.put(0);
        }
        require(bool(f), "writing fixture tensors failed");
    }
};

static std::vector<float> logits(llama_model * model) {
    auto p = llama_context_default_params();
    p.n_ctx = 128; p.n_batch = p.n_ubatch = 32;
    p.n_threads = p.n_threads_batch = 2;
    p.type_k = p.type_v = GGML_TYPE_F16;
    p.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context_ptr ctx(llama_init_from_model(model, p));
    require(bool(ctx), "context init failed");
    std::vector<float> result;
    // Two prefill batches followed by individual decode steps. EOS exercises
    // the hash-history reset; rows from both PLE shards are used.
    for (auto tokens : {std::vector<llama_token>{2, 7, 3}, {4, 0, 6}, {11}, {5}}) {
        require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), tokens.size())) == 0, "decode failed");
        const auto * out = llama_get_logits_ith(ctx.get(), -1);
        require(out != nullptr, "no logits");
        for (int i = 0; i < 32; ++i) {
            require(std::isfinite(out[i]), "non-finite fixture logits");
            result.push_back(out[i]);
        }
    }
    return result;
}

static bool prepared_mapping(const ggml_tensor * tensor) {
#if defined(__linux__)
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long long begin, end;
        if (sscanf(line.c_str(), "%llx-%llx", &begin, &end) != 2) continue;
        const auto ptr = reinterpret_cast<uintptr_t>(tensor->data);
        if (ptr >= begin && ptr < end) return line.find("llama-weights-") != std::string::npos;
    }
#else
    (void) tensor;
#endif
    return false;
}

int main(int argc, char ** argv) try {
    ggml_backend_load_all();
    if (!llama_mmap::SUPPORTED || !llama_file::TEMP_SUPPORTED) return 77;
    const int ngl = argc > 1 && std::string(argv[1]) == "--gpu" ? 99 : 0;
    const bool large = argc > 1 && std::string(argv[1]) == "--large";
    const auto root = fs::temp_directory_path() / ("llama-ple-gate-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct cleanup { fs::path path; ~cleanup() { std::error_code ec; fs::remove_all(path, ec); } } cleanup {root};

    // Exercise the precise AUTO threshold without allocating 4 GiB.
    ggml_context_ptr sizes(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
    llama_model_loader::lazy_read policy;
    policy.mode = LLAMA_LAZY_MODE_AUTO;
    for (int delta : {-1, 0, 1}) {
        auto * t = ggml_new_tensor_1d(sizes.get(), GGML_TYPE_F32, (int64_t(1) << 30) + delta);
        require(policy.add("boundary", t, nullptr) == (delta > 0), "AUTO 4-GiB boundary mismatch");
        require(!policy.any(), "fit probe recorded real mappings");
    }
    for (bool trellis : {false, true}) {
        fixture f(root / (trellis ? "exl3n" : "bf16"), trellis);
        auto p = llama_model_default_params();
        p.n_gpu_layers = 0; p.use_extra_bufts = false;
        p.load_mode = LLAMA_LOAD_MODE_NONE; p.lazy_mode = LLAMA_LAZY_MODE_OFF;
        p.mmap_prefetch = LLAMA_MMAP_PREFETCH_MODE_OFF;
        llama_model_ptr allocated(llama_model_load_from_safetensors_dir(f.dir, p));
        require(bool(allocated), "allocated native load failed");
        auto * ple = allocated->get_tensor("per_layer_token_embd.weight");
        require(ple && ggml_nbytes(ple) == f.ple.size(), "PLE shape mismatch");
        require(std::memcmp(ple->data, f.ple.data(), f.ple.size()) == 0, "PLE source rows changed");

        llama_safetensors_qwen4exp_importer importer(f.dir, f.config);
        gguf_context_ptr meta(importer.build_metadata());
        for (const auto & [name, t] : allocated->tensors_by_name) gguf_add_tensor(meta.get(), t);
        const auto control_path = f.dir / "control.gguf";
        require(gguf_write_to_file(meta.get(), control_path.c_str(), false), "writing GGUF control failed");
        // Verify the synthetic graph actually depends on the table: exact
        // agreement is meaningless if a disconnected/zero path ignores PLE.
        const auto nonzero = logits(allocated.get());
        std::vector<uint8_t> zero(f.ple.size(), 0);
        ggml_backend_tensor_set(allocated->per_layer_tok_embd, zero.data(), 0, zero.size());
        require(logits(allocated.get()) != nonzero, "fixture logits do not depend on PLE");
        allocated.reset();
        p.n_gpu_layers = ngl;
        llama_model_ptr control(llama_model_load_from_file(control_path.c_str(), p));
        require(bool(control), "GGUF control load failed");
        const auto expected = logits(control.get());
        control.reset();
        if (large) {
            // This optional disk-pressure gate is intentionally not part of
            // normal CTest. Run with TMPDIR on a disk filesystem, not tmpfs.
            fixture padded(root / "large", trellis, true);
            for (auto mode : {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP}) {
                p.load_mode = mode; p.lazy_mode = LLAMA_LAZY_MODE_AUTO;
                p.no_alloc = true;
                llama_model_ptr probe(llama_model_load_from_safetensors_dir(padded.dir, p));
                require(probe && probe->per_layer_tok_embd && !probe->per_layer_tok_embd->data,
                        "large AUTO fit allocated table data");
                probe.reset();
                p.no_alloc = false;
                llama_model_ptr model(llama_model_load_from_safetensors_dir(padded.dir, p));
                require(bool(model), "large AUTO native load failed");
                auto * t = model->get_tensor("per_layer_token_embd.weight");
                require(t && ggml_nbytes(t) > (size_t(1) << 32), "large table below AUTO threshold");
#if defined(__linux__)
                require(prepared_mapping(t), "AUTO did not prepare a mapped large table");
#endif
                require(std::memcmp(t->data, f.ple.data(), f.ple.size()) == 0, "large table active rows changed");
                std::vector<uint8_t> tail(ggml_row_size(t->type, t->ne[0]), 1);
                ggml_backend_tensor_get(t, tail.data(), ggml_nbytes(t) - tail.size(), tail.size());
                require(tail == std::vector<uint8_t>(tail.size(), 0), "large table final row changed");
                require(logits(model.get()) == expected, "large AUTO logits differ from small GGUF");
                std::cout << "PASS large bytes=" << ggml_nbytes(t) << " mode=" << int(mode) << std::endl;
            }
            break; // BF16 alone qualifies the size policy; EXL3 uses the same loader.
        }
        for (auto mode : {LLAMA_LOAD_MODE_NONE, LLAMA_LOAD_MODE_MMAP}) {
            for (auto lazy : {LLAMA_LAZY_MODE_OFF, LLAMA_LAZY_MODE_AUTO, LLAMA_LAZY_MODE_ON}) {
                for (bool no_alloc : {true, false}) {
                    p.load_mode = mode; p.lazy_mode = lazy; p.no_alloc = no_alloc;
                    llama_model_ptr model(llama_model_load_from_safetensors_dir(f.dir, p));
                    require(bool(model), "native lazy-mode load failed");
                    auto * t = model->get_tensor("per_layer_token_embd.weight");
                    require(t != nullptr, "missing PLE table");
                    if (no_alloc) {
                        require(t->data == nullptr, "fit allocated PLE data");
                        continue;
                    }
#if defined(__linux__)
                    // Input embeddings default to CPU even with all repeating
                    // layers on GPU. lazy-on maps them even with mmap disabled.
                    const bool mapped = lazy == LLAMA_LAZY_MODE_ON || mode == LLAMA_LOAD_MODE_MMAP;
                    require(prepared_mapping(t) == mapped, "wrong PLE backing/placement");
#endif
                    std::vector<uint8_t> bytes(f.ple.size());
                    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
                    require(bytes == f.ple, "loaded PLE bytes changed");
                    require(logits(model.get()) == expected, "lazy/native logits differ from GGUF");
                    std::cout << "PASS type=" << (trellis ? "exl3n2" : "bf16") << " mode=" << int(mode)
                              << " lazy=" << int(lazy) << " ngl=" << ngl << '\n';
                }
            }
        }
#if defined(__linux__)
        const auto cache_dir = (root / (trellis ? "cache-exl3" : "cache-bf16")).string();
        p.repack_cache = cache_dir.c_str();
        p.load_mode = LLAMA_LOAD_MODE_MMAP; p.lazy_mode = LLAMA_LAZY_MODE_ON;
        p.no_alloc = true;
        llama_model_ptr probe(llama_model_load_from_safetensors_dir(f.dir, p));
        require(bool(probe) && !fs::exists(cache_dir), "cache fit probe wrote files");
        probe.reset();
        p.no_alloc = false;
        std::vector<fs::file_time_type> published;
        for (int repeat = 0; repeat < 2; ++repeat) {
            p.check_tensors = repeat != 0;
            llama_model_ptr model(llama_model_load_from_safetensors_dir(f.dir, p));
            require(bool(model), "persistent native load failed");
            require(logits(model.get()) == expected, "persistent/native logits differ from GGUF");
            std::vector<fs::file_time_type> times;
            for (const auto & entry : fs::recursive_directory_iterator(cache_dir)) {
                if (entry.path().filename() == "weights") times.push_back(fs::last_write_time(entry.path()));
            }
            std::sort(times.begin(), times.end());
            require(!times.empty(), "persistent cache did not publish weights");
            if (repeat == 0) published = times;
            else require(times == published, "cache hit rewrote prepared weights");
        }
        p.repack_cache = nullptr;
#endif
    }
    std::cout << "PASS: Qwen4 PLE native/GGUF logits, mmap/lazy matrix, source lifetime, no_alloc, AUTO boundary\n";
    return 0;
} catch (const std::exception & e) {
    std::cerr << e.what() << '\n';
    return 1;
}
