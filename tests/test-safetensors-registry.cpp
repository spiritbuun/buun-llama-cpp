#include "ggml-backend.h"
#include "gguf.h"
#include "llama-safetensors-qwen35.h"
#include "llama-safetensors-quant.h"
#include "llama-safetensors.h"
#include "llama.h"

#include "nlohmann/json.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void write_shard(const std::filesystem::path & path, const std::string & header, const std::vector<uint8_t> & data) {
    std::ofstream out(path, std::ios::binary);
    require(static_cast<bool>(out), "failed to create safetensors test shard");
    const uint64_t         n = header.size();
    std::array<uint8_t, 8> prefix{};
    for (size_t i = 0; i < prefix.size(); ++i) {
        prefix[i] = uint8_t(n >> (8 * i));
    }
    out.write(reinterpret_cast<const char *>(prefix.data()), prefix.size());
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char *>(data.data()), data.size());
    require(static_cast<bool>(out), "failed to write safetensors test shard");
}

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream out(path);
    require(static_cast<bool>(out), "failed to create safetensors test metadata");
    out << text;
    require(static_cast<bool>(out), "failed to write safetensors test metadata");
}

struct tensor_fixture {
    std::string          name;
    std::string          dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t> data;
};

void write_single_shard_model(
        const std::filesystem::path & path,
        const std::vector<tensor_fixture> & tensors) {
    std::filesystem::create_directories(path);
    json                 header = json::object();
    std::vector<uint8_t> data;
    for (const tensor_fixture & tensor : tensors) {
        const size_t begin = data.size();
        data.insert(data.end(), tensor.data.begin(), tensor.data.end());
        header[tensor.name] = {
            { "dtype", tensor.dtype },
            { "shape", tensor.shape },
            { "data_offsets", { begin, data.size() } },
        };
    }
    write_shard(path / "model.safetensors", header.dump(), data);
}

template<typename Fn>
void require_rejected(Fn && fn, const char * message) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, message);
}

const char * fp8_channel_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","config_groups":{"fp8":{"format":"float-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"token","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"float","zp_dtype":null}}}}})";

const char * fp8_block_config =
    R"({"quantization_config":{"quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic","weight_block_size":[128,128]}})";

const char * nvfp4_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","config_groups":{"nvfp4":{"format":"nvfp4-pack-quantized","targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":"local","group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null}}}}})";

std::vector<uint8_t> f32_bytes(float value) {
    std::vector<uint8_t> result(sizeof(value));
    std::memcpy(result.data(), &value, sizeof(value));
    return result;
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream in(path);
    require(static_cast<bool>(in), "failed to open safetensors test metadata");
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

struct temp_dir {
    std::filesystem::path path;

    temp_dir() {
        path = std::filesystem::temp_directory_path() / "llama-safetensors-registry-test";
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }

    ~temp_dir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 4 && std::string(argv[1]) == "metadata") {
        llama_safetensors_qwen35_importer importer(argv[2]);
        gguf_context *                    actual              = importer.build_metadata();
        ggml_context *                    expected_tensor_ctx = nullptr;
        gguf_init_params                  params              = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &expected_tensor_ctx,
        };
        gguf_context * expected = gguf_init_from_file(argv[3], params);
        if (actual == nullptr || expected == nullptr) {
            throw std::runtime_error("failed to build or read metadata");
        }
        size_t mismatches = gguf_get_n_kv(actual) == gguf_get_n_kv(expected) ? 0 : 1;
        for (int64_t i = 0; i < gguf_get_n_kv(actual); ++i) {
            const char *  key      = gguf_get_key(actual, i);
            const int64_t j        = gguf_find_key(expected, key);
            bool          mismatch = j < 0 || gguf_get_kv_type(actual, i) != gguf_get_kv_type(expected, j);
            if (!mismatch) {
                switch (gguf_get_kv_type(actual, i)) {
                    case GGUF_TYPE_STRING:
                        mismatch = std::string(gguf_get_val_str(actual, i)) != gguf_get_val_str(expected, j);
                        break;
                    case GGUF_TYPE_UINT32:
                        mismatch = gguf_get_val_u32(actual, i) != gguf_get_val_u32(expected, j);
                        break;
                    case GGUF_TYPE_INT32:
                        mismatch = gguf_get_val_i32(actual, i) != gguf_get_val_i32(expected, j);
                        break;
                    case GGUF_TYPE_FLOAT32:
                        mismatch = gguf_get_val_f32(actual, i) != gguf_get_val_f32(expected, j);
                        break;
                    case GGUF_TYPE_ARRAY:
                        {
                            const gguf_type at = gguf_get_arr_type(actual, i);
                            mismatch           = at != gguf_get_arr_type(expected, j) ||
                                       gguf_get_arr_n(actual, i) != gguf_get_arr_n(expected, j);
                            if (!mismatch && at == GGUF_TYPE_STRING) {
                                for (size_t k = 0; k < gguf_get_arr_n(actual, i); ++k) {
                                    if (std::string(gguf_get_arr_str(actual, i, k)) !=
                                        gguf_get_arr_str(expected, j, k)) {
                                        mismatch = true;
                                        break;
                                    }
                                }
                            } else if (!mismatch && at == GGUF_TYPE_INT32) {
                                mismatch = std::memcmp(gguf_get_arr_data(actual, i), gguf_get_arr_data(expected, j),
                                                       gguf_get_arr_n(actual, i) * sizeof(int32_t)) != 0;
                            }
                        }
                        break;
                    default:
                        std::cerr << "unchecked metadata type for " << key << '\n';
                        mismatch = true;
                        break;
                }
            }
            if (mismatch) {
                std::cerr << "metadata mismatch " << key << '\n';
                ++mismatches;
            }
        }
        for (int64_t i = 0; i < gguf_get_n_kv(expected); ++i) {
            if (gguf_find_key(actual, gguf_get_key(expected, i)) < 0) {
                std::cerr << "missing metadata key " << gguf_get_key(expected, i) << '\n';
                ++mismatches;
            }
        }
        if (gguf_get_n_tensors(actual) != 0) {
            std::cerr << "native metadata unexpectedly contains a tensor manifest\n";
            ++mismatches;
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(expected); ++i) {
            const char *  name = gguf_get_tensor_name(expected, i);
            ggml_type type;
            std::array<int64_t, GGML_MAX_DIMS> ne;
            if (!importer.describe(name, type, ne)) {
                std::cerr << "source cannot describe tensor " << name << '\n';
                ++mismatches;
                continue;
            }
            bool            mismatch = type != gguf_get_tensor_type(expected, i);
            const int64_t * ene      = gguf_get_tensor_ne(expected, i);
            for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
                mismatch |= ne[dim] != ene[dim];
            }
            if (mismatch) {
                std::cerr << "tensor contract mismatch " << name
                          << " actual_type=" << ggml_type_name(type)
                          << " expected_type=" << ggml_type_name(gguf_get_tensor_type(expected, i))
                          << " actual_ne=" << ne[0] << ',' << ne[1] << ',' << ne[2] << ',' << ne[3]
                          << " expected_ne=" << ene[0] << ',' << ene[1] << ',' << ene[2] << ',' << ene[3] << '\n';
                ++mismatches;
            }
        }
        std::cout << "metadata_tensors=" << gguf_get_n_tensors(actual)
                  << " source_contracts=" << gguf_get_n_tensors(expected) << " mismatches=" << mismatches << '\n';
        gguf_free(actual);
        gguf_free(expected);
        ggml_free(expected_tensor_ctx);
        return mismatches == 0 ? 0 : 1;
    }
    if (argc == 4 && std::string(argv[1]) == "load") {
        llama_safetensors_qwen35_importer importer(argv[2]);
        ggml_context *                    tensor_ctx = nullptr;
        gguf_context *                    metadata   = nullptr;
        const bool                        native     = std::string(argv[3]) == "native";
        if (!native) {
            gguf_init_params init_params = {
                /*.no_alloc = */ true,
                /*.ctx      = */ &tensor_ctx,
            };
            metadata = gguf_init_from_file(argv[3], init_params);
            if (metadata == nullptr || tensor_ctx == nullptr) {
                throw std::runtime_error("failed to open GGUF metadata control");
            }
        }

        struct callback_state {
            llama_safetensors_qwen35_importer * importer;
            size_t                              tensors = 0;
            size_t                              bytes   = 0;
        } state{ &importer };

        const auto set_tensor_data = [](ggml_tensor * tensor, void * userdata) {
            auto *               state = static_cast<callback_state *>(userdata);
            std::vector<uint8_t> data  = state->importer->materialize(tensor->name, tensor->type, ggml_nbytes(tensor));
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
            ++state->tensors;
            state->bytes += data.size();
        };

        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 99;
        llama_model * model = native ?
            llama_model_load_from_safetensors_dir(argv[2], model_params) :
            llama_model_init_from_user(metadata, set_tensor_data, &state, model_params);
        if (model == nullptr) {
            throw std::runtime_error("direct safetensors model load failed");
        }
        if (!native) {
            std::cout << "loaded_tensors=" << state.tensors << " loaded_bytes=" << state.bytes << '\n';
        }

        const llama_vocab * vocab    = llama_model_get_vocab(model);
        const std::string   prompt   = "The capital of France is";
        const int           n_prompt = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n_prompt);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) < 0) {
            throw std::runtime_error("failed to tokenize direct-load smoke prompt");
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx                = 512;
        context_params.n_batch              = 512;
        context_params.n_ubatch             = 512;
        context_params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * context             = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            throw std::runtime_error("failed to create direct safetensors context");
        }
        llama_sampler * sampler = llama_sampler_init_greedy();
        llama_batch     batch   = llama_batch_get_one(tokens.data(), tokens.size());
        std::string     generated;
        for (int i = 0; i < 8; ++i) {
            if (llama_decode(context, batch) != 0) {
                throw std::runtime_error("direct safetensors decode failed");
            }
            llama_token token = llama_sampler_sample(sampler, context, -1);
            if (llama_vocab_is_eog(vocab, token)) {
                break;
            }
            std::array<char, 256> piece{};
            const int             n_piece = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
            if (n_piece < 0) {
                throw std::runtime_error("failed to render direct-load token");
            }
            generated.append(piece.data(), static_cast<size_t>(n_piece));
            batch = llama_batch_get_one(&token, 1);
        }
        std::cout << "generated=" << generated << '\n';
        llama_sampler_free(sampler);
        llama_free(context);
        llama_model_free(model);
        if (metadata != nullptr) {
            gguf_free(metadata);
        }
        if (tensor_ctx != nullptr) {
            ggml_free(tensor_ctx);
        }
        return 0;
    }
    if (argc == 3) {
        llama_safetensors_qwen35_importer importer(argv[1]);
        ggml_context *                    tensor_ctx = nullptr;
        gguf_init_params                  params     = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &tensor_ctx,
        };
        gguf_context * gguf = gguf_init_from_file(argv[2], params);
        if (gguf == nullptr || tensor_ctx == nullptr) {
            throw std::runtime_error("failed to open GGUF control");
        }
        std::ifstream control(argv[2], std::ios::binary);
        if (!control) {
            throw std::runtime_error("failed to open GGUF control data");
        }

        const size_t  data_offset    = gguf_get_data_offset(gguf);
        const int64_t n_tensors      = gguf_get_n_tensors(gguf);
        size_t        compared_bytes = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char *  name   = gguf_get_tensor_name(gguf, i);
            ggml_tensor * tensor = ggml_get_tensor(tensor_ctx, name);
            require(tensor != nullptr, "GGUF tensor descriptor is missing from its metadata context");
            const size_t         size   = ggml_nbytes(tensor);
            std::vector<uint8_t> actual = importer.materialize(name, tensor->type, size);
            std::vector<uint8_t> expected(size);
            control.seekg(static_cast<std::streamoff>(data_offset + gguf_get_tensor_offset(gguf, i)));
            control.read(reinterpret_cast<char *>(expected.data()), static_cast<std::streamsize>(expected.size()));
            if (!control) {
                throw std::runtime_error(std::string("failed to read GGUF control tensor '") + name + "'");
            }
            bool equal = actual == expected;
            if (!equal) {
                size_t first = 0;
                while (first < size && actual[first] == expected[first]) {
                    ++first;
                }
                throw std::runtime_error(std::string("import mismatch for '") + name + "' at byte " +
                                         std::to_string(first));
            }
            compared_bytes += size;
        }
        std::cout << "matched_tensors=" << n_tensors << " matched_bytes=" << compared_bytes << '\n';
        gguf_free(gguf);
        ggml_free(tensor_ctx);
        return 0;
    }
    if (argc == 2) {
        const auto registry = llama_safetensors_registry::load(argv[1]);
        const llama_safetensors_quant_adapters adapters(argv[1], registry);
        const llama_safetensors_quant_summary & summary = adapters.summary();
        std::cout << "shards=" << registry.shards().size() << " tensors=" << registry.tensors().size()
                  << " nvfp4=" << summary.nvfp4 << " fp8_channel=" << summary.fp8_channel
                  << " fp8_block=" << summary.fp8_block << '\n';
        return 0;
    }
    require(argc == 1, "usage: test-safetensors-registry [model-directory [control.gguf]]");

    temp_dir dir;

    // Quant adapters are architecture-independent contracts. Exercise them
    // directly so an importer cannot accidentally make a correctness-critical
    // scale optional while rearranging canonical names.
    {
        const auto path = dir.path / "fp8-channel";
        write_single_shard_model(path, {
            { "module.weight", "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "BF16", { 2 }, std::vector<uint8_t>(4) },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);

        llama_safetensors_quant_adapters incomplete(path, registry);
        const auto weight = incomplete.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3,
                "FP8 channel weight binding is wrong");
        incomplete.consume(*weight);
        require_rejected([&] { incomplete.validate_complete(); },
                         "FP8 channel weight was accepted without its scale");

        llama_safetensors_quant_adapters complete(path, registry);
        const auto complete_weight = complete.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale = complete.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(complete_weight.has_value() && scale.has_value() && scale->target_type == GGML_TYPE_BF16 &&
                    scale->target_shape == std::vector<int64_t>({ 2 }),
                "FP8 channel scale binding is wrong");
        complete.consume(*complete_weight);
        complete.consume(*scale);
        complete.validate_complete();
    }
    {
        const auto path = dir.path / "fp8-channel-missing-scale";
        write_single_shard_model(path, {
            { "module.weight", "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "FP8 channel contract accepted a missing scale");
    }
    {
        const auto path = dir.path / "fp8-channel-wrong-scale-type";
        write_single_shard_model(path, {
            { "module.weight", "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "F16", { 2 }, std::vector<uint8_t>(4) },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "FP8 channel contract accepted the wrong scale dtype");
    }
    {
        const auto path = dir.path / "fp8-channel-wrong-weight-type";
        write_single_shard_model(path, {
            { "module.weight", "BF16", { 2, 32 }, std::vector<uint8_t>(128) },
            { "module.weight_scale", "BF16", { 2 }, std::vector<uint8_t>(4) },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "FP8 channel contract accepted a BF16 weight");
    }
    {
        const auto path = dir.path / "fp8-block";
        write_single_shard_model(path, {
            { "module.weight", "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
            { "module.weight_scale_inv", "BF16", { 1, 1 }, std::vector<uint8_t>(2) },
        });
        write_text(path / "config.json", fp8_block_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(path, registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        require(weight.has_value() && scale.has_value() && scale->target_type == GGML_TYPE_F32 &&
                    scale->materialization == llama_safetensors_quant_materialization::FP8_BLOCK_SCALE,
                "FP8 block binding is wrong");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.validate_complete();
        require(adapters.finalize(*scale, adapters.read(*scale)).size() == sizeof(float),
                "FP8 block scale conversion has the wrong size");
    }
    {
        const auto path = dir.path / "fp8-block-wrong-grid";
        write_single_shard_model(path, {
            { "module.weight", "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
            { "module.weight_scale_inv", "BF16", { 1, 2 }, std::vector<uint8_t>(4) },
        });
        write_text(path / "config.json", fp8_block_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "FP8 block contract accepted the wrong scale grid");
    }
    {
        const auto path = dir.path / "nvfp4";
        write_single_shard_model(path, {
            { "module.weight_packed", "U8", { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "F8_E4M3", { 2, 4 }, std::vector<uint8_t>(8, 0x38) },
            { "module.weight_global_scale", "F32", {}, f32_bytes(1.0f) },
            { "module.input_global_scale", "F32", {}, f32_bytes(1.0f) },
        });
        write_text(path / "config.json", nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(path, registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_NVFP4 &&
                    weight->target_shape == std::vector<int64_t>({ 64, 2 }) &&
                    adapters.read(*weight).size() == 72,
                "NVFP4 packed-weight binding or repack is wrong");
        require(scale.has_value() && input.has_value(), "NVFP4 global scale binding is incomplete");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "nvfp4-zero-point";
        write_single_shard_model(path, {
            { "module.weight_packed", "U8", { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "F8_E4M3", { 2, 4 }, std::vector<uint8_t>(8, 0x38) },
            { "module.weight_global_scale", "F32", {}, f32_bytes(1.0f) },
            { "module.input_global_scale", "F32", {}, f32_bytes(1.0f) },
            { "module.weight_zero_point", "U8", { 1 }, std::vector<uint8_t>(1) },
        });
        write_text(path / "config.json", nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "NVFP4 contract accepted a zero point");
    }
    {
        const auto path = dir.path / "nvfp4-invalid-global-scale";
        write_single_shard_model(path, {
            { "module.weight_packed", "U8", { 2, 32 }, std::vector<uint8_t>(64) },
            { "module.weight_scale", "F8_E4M3", { 2, 4 }, std::vector<uint8_t>(8, 0x38) },
            { "module.weight_global_scale", "F32", {}, f32_bytes(0.0f) },
            { "module.input_global_scale", "F32", {}, f32_bytes(1.0f) },
        });
        write_text(path / "config.json", nvfp4_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "NVFP4 contract accepted a non-positive global scale");
    }

    const std::string shard_a_header =
        R"({"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"packed":{"dtype":"U8","shape":[2],"data_offsets":[4,6]}})";
    const std::string shard_b_header =
        R"({"fp8":{"dtype":"F8_E4M3","shape":[2,2],"data_offsets":[0,4]},"__metadata__":{"format":"pt"}})";
    write_shard(dir.path / "model-00001-of-00002.safetensors", shard_a_header, { 1, 2, 3, 4, 5, 6 });
    write_shard(dir.path / "model-00002-of-00002.safetensors", shard_b_header, { 7, 8, 9, 10 });
    write_text(
        dir.path / "model.safetensors.index.json",
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors","packed":"model-00001-of-00002.safetensors","fp8":"model-00002-of-00002.safetensors"}})");

    const auto registry = llama_safetensors_registry::load(dir.path);
    require(registry.shards().size() == 2, "unexpected safetensors shard count");
    require(registry.tensors().size() == 3, "unexpected safetensors tensor count");

    const auto * packed = registry.find("packed");
    require(packed != nullptr, "packed tensor is missing");
    require(packed->dtype == llama_safetensors_dtype::U8, "packed tensor has the wrong dtype");
    require(packed->shape == std::vector<uint64_t>({ 2 }), "packed tensor has the wrong shape");
    require(packed->size == 2, "packed tensor has the wrong size");
    require(packed->offset == 8 + shard_a_header.size() + 4, "packed tensor has the wrong file offset");

    const auto * fp8 = registry.find("fp8");
    require(fp8 != nullptr, "FP8 tensor is missing");
    require(fp8->dtype == llama_safetensors_dtype::F8_E4M3, "FP8 tensor has the wrong dtype");
    require(fp8->shape == std::vector<uint64_t>({ 2, 2 }), "FP8 tensor has the wrong shape");
    require(registry.find("missing") == nullptr, "missing tensor lookup unexpectedly succeeded");

    write_text(
        dir.path / "config.json",
        R"({"quantization_config":{"quant_method":"compressed-tensors","format":"mixed-precision","ignore":["re:^ignored\\."],"config_groups":{"z_specific":{"format":"float-quantized","targets":["re:.*layers\\.(56|57)\\.mlp\\.(gate|up|down)_proj$"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"token","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"scale_dtype":null,"strategy":"channel","symmetric":true,"type":"float","zp_dtype":null}},"a_generic":{"format":"nvfp4-pack-quantized","targets":["re:.*mlp\\.(gate|up|down)_proj$"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":"local","group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null},"weights":{"actorder":"static","block_structure":null,"dynamic":false,"group_size":16,"num_bits":4,"scale_dtype":"torch.float8_e4m3fn","strategy":"tensor_group","symmetric":true,"type":"float","zp_dtype":null}}}}})");
    const auto   quant = llama_safetensors_quant_config::load(dir.path);
    const auto * early = quant.match("model.layers.12.mlp.gate_proj");
    require(early != nullptr, "early-layer quantization group is missing");
    require(early->format == llama_safetensors_quant_format::NVFP4_PACK, "early layer has the wrong format");
    const auto * late = quant.match("model.layers.56.mlp.gate_proj");
    require(late != nullptr, "late-layer quantization group is missing");
    require(late->format == llama_safetensors_quant_format::FP8_CHANNEL, "late layer has the wrong format");
    require(quant.ignored("ignored.mlp.gate_proj"), "ignored tensor was not recognized");
    require(quant.match("ignored.mlp.gate_proj") == nullptr, "ignored tensor matched a quantization group");

    // The producer selects the first matching target in document order. The
    // group names intentionally sort in the opposite order.
    require(late->name == "z_specific", "overlapping targets were not resolved in declaration order");

    const std::string valid_config = read_text(dir.path / "config.json");
    for (const auto & [needle, replacement] : std::array<std::pair<const char *, const char *>, 3>{ {
             { "\"symmetric\":true", "\"symmetric\":false" },
             { "\"dynamic\":false", "\"dynamic\":true" },
             { "\"group_size\":16", "\"group_size\":32" },
         } }) {
        std::string invalid = valid_config;
        const auto  pos     = invalid.rfind(needle);
        require(pos != std::string::npos, "invalid quantization fixture mutation");
        invalid.replace(pos, std::strlen(needle), replacement);
        write_text(dir.path / "config.json", invalid);
        bool invalid_rejected = false;
        try {
            (void) llama_safetensors_quant_config::load(dir.path);
        } catch (const std::runtime_error &) {
            invalid_rejected = true;
        }
        require(invalid_rejected, "unsupported quantization contract was accepted");
    }
    write_text(dir.path / "config.json", valid_config);

    // The index is authoritative; a tensor assigned to another shard must be
    // rejected before any model allocation begins.
    write_text(
        dir.path / "model.safetensors.index.json",
        R"({"weight_map":{"a":"model-00002-of-00002.safetensors","packed":"model-00001-of-00002.safetensors","fp8":"model-00002-of-00002.safetensors"}})");
    bool rejected = false;
    try {
        (void) llama_safetensors_registry::load(dir.path);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "registry accepted a tensor from the wrong indexed shard");

    // A valid safetensors container is not sufficient: the architecture and
    // quantization contract must be one the direct importer actually supports.
    write_text(
        dir.path / "model.safetensors.index.json",
        R"({"weight_map":{"a":"model-00001-of-00002.safetensors","packed":"model-00001-of-00002.safetensors","fp8":"model-00002-of-00002.safetensors"}})");
    bool unsupported_rejected = false;
    try {
        (void) llama_safetensors_qwen35_importer(dir.path);
    } catch (const std::exception &) {
        unsupported_rejected = true;
    }
    require(unsupported_rejected, "Qwen3.5 importer accepted an unsupported model contract");

    return 0;
}
