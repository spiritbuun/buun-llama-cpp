#include "ggml-backend.h"
#include "gguf.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-quant.h"
#include "llama-safetensors-qwen3.h"
#include "llama-safetensors-qwen35.h"
#include "llama-safetensors-tensor.h"
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
#include <memory>
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
    std::string           name;
    std::string           dtype;
    std::vector<uint64_t> shape;
    std::vector<uint8_t>  data;
};

void write_single_shard_model(const std::filesystem::path & path, const std::vector<tensor_fixture> & tensors) {
    std::filesystem::create_directories(path);
    json                 header = json::object();
    std::vector<uint8_t> data;
    for (const tensor_fixture & tensor : tensors) {
        const size_t begin = data.size();
        data.insert(data.end(), tensor.data.begin(), tensor.data.end());
        header[tensor.name] = {
            { "dtype",        tensor.dtype           },
            { "shape",        tensor.shape           },
            { "data_offsets", { begin, data.size() } },
        };
    }
    write_shard(path / "model.safetensors", header.dump(), data);
}

template <typename Fn> void require_rejected(Fn && fn, const char * message) {
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

const char * w8a8_config =
    R"({"quantization_config":{"quant_method":"compressed-tensors","format":"int-quantized","config_groups":{"int8":{"targets":["module"],"input_activations":{"actorder":null,"block_structure":null,"dynamic":true,"group_size":null,"num_bits":8,"strategy":"token","symmetric":true,"type":"int"},"weights":{"actorder":null,"block_structure":null,"dynamic":false,"group_size":null,"num_bits":8,"strategy":"channel","symmetric":true,"type":"int"}}}}})";

const char * awq_config =
    R"({"quantization_config":{"bits":4,"group_size":128,"modules_to_not_convert":null,"quant_method":"awq","version":"gemm","zero_point":true}})";

const char * gptq_config =
    R"({"quantization_config":{"bits":4,"checkpoint_format":"gptq","desc_act":false,"group_size":128,"pack_dtype":"int32","quant_method":"gptq","sym":true}})";

const char * gptq_act_order_config =
    R"({"quantization_config":{"bits":4,"checkpoint_format":"gptq","desc_act":true,"group_size":128,"pack_dtype":"int32","quant_method":"gptq","sym":true}})";

std::vector<uint8_t> f32_bytes(float value) {
    std::vector<uint8_t> result(sizeof(value));
    std::memcpy(result.data(), &value, sizeof(value));
    return result;
}

std::vector<uint8_t> i32_bytes(const std::vector<uint32_t> & values) {
    std::vector<uint8_t> result(values.size() * sizeof(uint32_t));
    std::memcpy(result.data(), values.data(), result.size());
    return result;
}

void store_f16(std::vector<uint8_t> & data, size_t index, float value) {
    const ggml_fp16_t bits = ggml_fp32_to_fp16(value);
    std::memcpy(data.data() + index * sizeof(bits), &bits, sizeof(bits));
}

uint16_t load_u16(const uint8_t * data) {
    uint16_t result;
    std::memcpy(&result, data, sizeof(result));
    return result;
}

void require_q4_1_block(
        const uint8_t * block,
        float scale,
        uint8_t zero,
        const std::array<uint8_t, 32> & codes,
        const char * message) {
    require(load_u16(block) == ggml_fp32_to_fp16(scale), message);
    require(load_u16(block + sizeof(ggml_fp16_t)) == ggml_fp32_to_fp16(-scale * zero), message);
    for (size_t i = 0; i < codes.size() / 2; ++i) {
        require(block[2 * sizeof(ggml_fp16_t) + i] == (codes[i] | (codes[i + 16] << 4)), message);
    }
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

std::unique_ptr<llama_safetensors_importer> make_importer(const std::filesystem::path & model_dir) {
    llama_safetensors_json config = llama_safetensors_read_json(model_dir / "config.json");
    if (llama_safetensors_qwen3_importer::probe(config)) {
        return std::make_unique<llama_safetensors_qwen3_importer>(model_dir, std::move(config));
    }
    if (llama_safetensors_qwen35_importer::probe(config)) {
        return std::make_unique<llama_safetensors_qwen35_importer>(model_dir, std::move(config));
    }
    throw std::runtime_error("unsupported test importer architecture");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 5 && std::string(argv[1]) == "compare") {
        ggml_backend_load_all();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers       = 99;
        llama_model * native            = llama_model_load_from_safetensors_dir(argv[2], model_params);
        llama_model * control           = llama_model_load_from_file(argv[3], model_params);
        if (native == nullptr || control == nullptr) {
            throw std::runtime_error("failed to load a live-comparison model");
        }

        const std::string   text          = read_text(argv[4]);
        const llama_vocab * native_vocab  = llama_model_get_vocab(native);
        const llama_vocab * control_vocab = llama_model_get_vocab(control);
        int                 n_tokens = -llama_tokenize(native_vocab, text.data(), text.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n_tokens);
        std::vector<llama_token> control_tokens(n_tokens);
        require(llama_tokenize(native_vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true) ==
                        n_tokens &&
                    llama_tokenize(control_vocab, text.data(), text.size(), control_tokens.data(),
                                   control_tokens.size(), true, true) == n_tokens &&
                    tokens == control_tokens,
                "native and GGUF tokenization differ");
        constexpr int n_compare = 16384;
        require(n_tokens >= n_compare, "live-comparison corpus is too short");
        tokens.resize(n_compare);

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx                = n_compare;
        context_params.n_batch              = 512;
        context_params.n_ubatch             = 512;
        context_params.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * native_ctx          = llama_init_from_model(native, context_params);
        llama_context * control_ctx         = llama_init_from_model(control, context_params);
        if (native_ctx == nullptr || control_ctx == nullptr) {
            throw std::runtime_error("failed to create live-comparison contexts");
        }

        const int n_vocab  = llama_vocab_n_tokens(native_vocab);
        size_t    compared = 0;
        for (int begin = 0; begin < n_compare; begin += 512) {
            const int   count         = std::min(512, n_compare - begin);
            llama_batch native_batch  = llama_batch_init(count, 0, 1);
            llama_batch control_batch = llama_batch_init(count, 0, 1);
            for (int i = 0; i < count; ++i) {
                for (llama_batch * batch : { &native_batch, &control_batch }) {
                    batch->token[i]     = tokens[begin + i];
                    batch->pos[i]       = begin + i;
                    batch->n_seq_id[i]  = 1;
                    batch->seq_id[i][0] = 0;
                    batch->logits[i]    = true;
                    batch->n_tokens     = count;
                }
            }
            if (llama_decode(native_ctx, native_batch) != 0 || llama_decode(control_ctx, control_batch) != 0) {
                throw std::runtime_error("live-comparison decode failed");
            }
            for (int i = 0; i < count; ++i) {
                const float * actual   = llama_get_logits_ith(native_ctx, i);
                const float * expected = llama_get_logits_ith(control_ctx, i);
                if (std::memcmp(actual, expected, size_t(n_vocab) * sizeof(float)) != 0) {
                    throw std::runtime_error("native and GGUF logits differ at token " + std::to_string(begin + i));
                }
                ++compared;
            }
            llama_batch_free(native_batch);
            llama_batch_free(control_batch);
        }
        std::cout << "exact_logit_rows=" << compared << " vocab=" << n_vocab << '\n';
        llama_free(native_ctx);
        llama_free(control_ctx);
        llama_model_free(native);
        llama_model_free(control);
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "metadata") {
        std::unique_ptr<llama_safetensors_importer> importer            = make_importer(argv[2]);
        gguf_context *                              actual              = importer->build_metadata();
        ggml_context *                              expected_tensor_ctx = nullptr;
        gguf_init_params                            params              = {
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
                    case GGUF_TYPE_BOOL:
                        mismatch = gguf_get_val_bool(actual, i) != gguf_get_val_bool(expected, j);
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
            const char *                       name = gguf_get_tensor_name(expected, i);
            ggml_type                          type;
            std::array<int64_t, GGML_MAX_DIMS> ne;
            if (!importer->describe(name, type, ne)) {
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
                std::cerr << "tensor contract mismatch " << name << " actual_type=" << ggml_type_name(type)
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
        std::unique_ptr<llama_safetensors_importer> importer   = make_importer(argv[2]);
        ggml_context *                              tensor_ctx = nullptr;
        gguf_context *                              metadata   = nullptr;
        const bool                                  native     = std::string(argv[3]) == "native";
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
            llama_safetensors_importer * importer;
            size_t                       tensors = 0;
            size_t                       bytes   = 0;
        } state{ importer.get() };

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
        llama_model * model             = native ? llama_model_load_from_safetensors_dir(argv[2], model_params) :
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
        std::unique_ptr<llama_safetensors_importer> importer   = make_importer(argv[1]);
        ggml_context *                              tensor_ctx = nullptr;
        gguf_init_params                            params     = {
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
            std::vector<uint8_t> actual = importer->materialize(name, tensor->type, size);
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
        const auto                             registry = llama_safetensors_registry::load(argv[1]);
        const llama_safetensors_quant_adapters adapters(
            llama_safetensors_read_json(std::filesystem::path(argv[1]) / "config.json"), registry);
        const llama_safetensors_quant_summary & summary = adapters.summary();
        std::cout << "shards=" << registry.shards().size() << " tensors=" << registry.tensors().size()
                  << " nvfp4=" << summary.nvfp4 << " fp8_channel=" << summary.fp8_channel
                  << " fp8_block=" << summary.fp8_block << " w8a8=" << summary.w8a8
                  << " awq=" << summary.awq << " gptq=" << summary.gptq << '\n';
        return 0;
    }

    const auto q_proj = llama_safetensors_map_decoder_tensor("model.layers.7.", "attn_q.weight");
    require(q_proj.has_value(), "shared decoder mapper missed attention Q");
    require(q_proj->source == "model.layers.7.self_attn.q_proj.weight", "attention Q source name is wrong");
    require(q_proj->module == "model.layers.7.self_attn.q_proj", "attention Q module name is wrong");
    require(q_proj->quant_role == llama_safetensors_quant_role::WEIGHT, "attention Q quant role is wrong");

    const auto ffn_scale = llama_safetensors_map_decoder_tensor("model.layers.3.", "ffn_down.scale");
    require(ffn_scale.has_value(), "shared decoder mapper missed FFN scale");
    require(ffn_scale->source == "model.layers.3.mlp.down_proj.weight_scale", "FFN scale source name is wrong");
    require(ffn_scale->quant_role == llama_safetensors_quant_role::WEIGHT_SCALE, "FFN scale quant role is wrong");

    const auto norm = llama_safetensors_map_decoder_tensor("model.layers.2.", "post_attention_norm.weight");
    require(norm.has_value(), "shared decoder mapper missed post-attention norm");
    require(norm->source == "model.layers.2.post_attention_layernorm.weight" && !norm->quant_role,
            "post-attention norm mapping is wrong");
    require(!llama_safetensors_map_decoder_tensor("model.layers.2.", "ssm_a").has_value(),
            "shared decoder mapper accepted an architecture-specific tensor");
    require(argc == 1, "usage: test-safetensors-registry [model-directory [control.gguf]]");

    temp_dir dir;

    // A conventional Qwen3 adapter must reuse the shared block-FP8 contract
    // and decoder naming seam without architecture-specific materialization.
    {
        const std::filesystem::path qwen3_dir = dir.path / "qwen3";
        write_single_shard_model(
            qwen3_dir,
            {
                { "model.layers.0.self_attn.q_proj.weight",           "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
                { "model.layers.0.self_attn.q_proj.weight_scale_inv", "BF16",    { 1, 1 },     { 0x80, 0x3f }                  },
        });
        write_text(qwen3_dir / "generation_config.json", R"({"bos_token_id":0,"eos_token_id":1})");
        write_text(qwen3_dir / "tokenizer.json", "{}");
        write_text(qwen3_dir / "tokenizer_config.json", "{}");
        llama_safetensors_json config = {
            { "model_type",          "qwen3" },
            { "num_hidden_layers",   1       },
            { "hidden_size",         128     },
            { "num_attention_heads", 1       },
            { "num_key_value_heads", 1       },
            { "head_dim",            128     },
        };
        config["quantization_config"] = llama_safetensors_json::parse(fp8_block_config).at("quantization_config");
        llama_safetensors_qwen3_importer   importer(qwen3_dir, config);
        ggml_type                          type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(importer.describe("blk.0.attn_q.weight", type, ne), "Qwen3 FP8 weight was not described");
        require(type == GGML_TYPE_F8_E4M3 && ne[0] == 128 && ne[1] == 128, "Qwen3 FP8 weight contract is wrong");
        require(importer.describe("blk.0.attn_q.scale", type, ne), "Qwen3 FP8 scale was not described");
        require(type == GGML_TYPE_F32 && ne[0] == 1 && ne[1] == 1, "Qwen3 FP8 block-scale contract is wrong");
        require(importer.materialize("blk.0.attn_q.scale", GGML_TYPE_F32, sizeof(float)) == f32_bytes(1.0f),
                "Qwen3 FP8 scale materialization changed source bytes");
    }

    // Older compressed-tensors checkpoints use one top-level format and a
    // module-class selector instead of per-group formats and regexes.
    {
        const std::filesystem::path legacy_dir = dir.path / "legacy-channel-fp8";
        const std::string           module     = "model.layers.0.self_attn.q_proj";
        write_single_shard_model(legacy_dir,
                                 {
                                     { module + ".weight",       "F8_E4M3", { 2, 2 }, { 0, 0, 0, 0 }             },
                                     { module + ".weight_scale", "BF16",    { 2 },    { 0x80, 0x3f, 0x80, 0x3f } },
        });
        const llama_safetensors_json           config   = llama_safetensors_json::parse(R"({
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "ignore": ["lm_head"],
                "config_groups": { "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "type": "float", "num_bits": 8, "strategy": "channel",
                        "symmetric": true, "dynamic": false,
                        "group_size": null, "block_structure": null, "actorder": null
                    },
                    "input_activations": {
                        "type": "float", "num_bits": 8, "strategy": "token",
                        "symmetric": true, "dynamic": true,
                        "group_size": null, "block_structure": null, "actorder": null
                    }
                } }
            }
        })");
        const llama_safetensors_registry       registry = llama_safetensors_registry::load(legacy_dir);
        const llama_safetensors_quant_adapters adapters(config, registry);
        require(adapters.summary().fp8_channel == 1, "legacy channel-FP8 group was not recognized");
        require(adapters.bind(module, llama_safetensors_quant_role::WEIGHT).has_value(),
                "legacy Linear selector did not bind a projection");
        require(!adapters.bind("lm_head", llama_safetensors_quant_role::WEIGHT).has_value(),
                "legacy compressed-tensors ignore rule was not honored");
    }

    // Catch-all producer rules may leave selected modules plain without adding
    // them to an ignore list. Applicability follows the stored representation,
    // while an unexpected quant sidecar remains a malformed contract.
    for (const bool w8a8 : { false, true }) {
        const std::filesystem::path path = dir.path / (w8a8 ? "w8-plain-exception" : "fp8-plain-exception");
        const std::string quant_module = "quant";
        std::vector<tensor_fixture> tensors = {
            { "module.weight", "BF16", { 2, 32 }, std::vector<uint8_t>(128, 0x2a) },
        };
        llama_safetensors_json config = llama_safetensors_json::parse(w8a8 ? w8a8_config : fp8_channel_config);
        auto & group = config["quantization_config"]["config_groups"].begin().value();
        group["targets"][0] = "re:.*";
        if (w8a8) {
            tensors.push_back({ quant_module + ".weight", "I8", { 2, 32 }, std::vector<uint8_t>(64) });
            tensors.push_back({ quant_module + ".weight_scale", "BF16", { 2, 1 },
                                { 0x80, 0x3f, 0x80, 0x3f } });
        } else {
            tensors.push_back({ quant_module + ".weight", "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) });
            tensors.push_back({ quant_module + ".weight_scale", "BF16", { 2 },
                                { 0x80, 0x3f, 0x80, 0x3f } });
        }
        write_single_shard_model(path, tensors);
        write_text(path / "config.json", config.dump());
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(config, registry);
        require(!adapters.applies("module") &&
                    !adapters.bind("module", llama_safetensors_quant_role::WEIGHT).has_value() &&
                    adapters.applies(quant_module),
                "catch-all quantization did not preserve a plain module exception");
        const auto binding = llama_safetensors_bind_tensor(
            adapters, { "module.weight", "module", llama_safetensors_quant_role::WEIGHT });
        ggml_type type;
        std::array<int64_t, GGML_MAX_DIMS> ne;
        require(llama_safetensors_describe_tensor(registry, binding, type, ne) && type == GGML_TYPE_BF16 &&
                    ne[0] == 32 && ne[1] == 2 &&
                    llama_safetensors_materialize_tensor(registry, adapters, binding, type, 128) ==
                        std::vector<uint8_t>(128, 0x2a),
                "plain catch-all exception did not follow the ordinary BF16 path");
    }

    // The dense runtime supports the 24-, 32-, and 64-layer Qwen3.5 families.
    // The importer must derive recurrent head geometry from config rather than
    // silently assuming the 27B layout.
    {
        const std::filesystem::path geometry_dir = dir.path / "geometry";
        write_single_shard_model(geometry_dir, {
                                                   { "module.weight",       "F8_E4M3", { 1, 1 }, { 0 }    },
                                                   { "module.weight_scale", "BF16",    { 1 },    { 0, 0 } },
        });
        write_text(geometry_dir / "generation_config.json", "{}");
        write_text(geometry_dir / "tokenizer.json", "{}");

        llama_safetensors_json config = {
            { "model_type",  "qwen3_5" },
            { "text_config",
             {
                  { "model_type", "qwen3_5_text" },
                  { "num_hidden_layers", 24 },
                  { "mtp_num_hidden_layers", 1 },
                  { "linear_num_key_heads", 8 },
                  { "linear_num_value_heads", 16 },
                  { "linear_key_head_dim", 64 },
                  { "linear_value_head_dim", 64 },
              }                        },
        };
        config["quantization_config"] = llama_safetensors_json::parse(fp8_channel_config).at("quantization_config");
        (void) llama_safetensors_qwen35_importer(geometry_dir, config);

        config["text_config"]["num_hidden_layers"] = 48;
        require_rejected([&] { (void) llama_safetensors_qwen35_importer(geometry_dir, config); },
                         "Qwen importer accepted a layer count unsupported by its model implementation");
        config["text_config"]["num_hidden_layers"]      = 32;
        config["text_config"]["linear_num_value_heads"] = 15;
        require_rejected([&] { (void) llama_safetensors_qwen35_importer(geometry_dir, config); },
                         "Qwen importer accepted non-integral recurrent head groups");
        config["text_config"]["linear_num_value_heads"] = 16;
        config["text_config"]["linear_value_head_dim"]  = 32;
        require_rejected([&] { (void) llama_safetensors_qwen35_importer(geometry_dir, config); },
                         "Qwen importer accepted unequal recurrent key/value dimensions");

        const std::filesystem::path channel_dir    = dir.path / "channel-transform";
        const std::string           channel_module = "model.language_model.layers.0.linear_attn.out_proj";
        write_single_shard_model(channel_dir, {
                                                  { channel_module + ".weight",       "F8_E4M3", { 2, 2 }, { 0, 0, 0, 0 } },
                                                  { channel_module + ".weight_scale", "BF16",    { 2 },    { 1, 2, 3, 4 } },
        });
        write_text(channel_dir / "generation_config.json", "{}");
        write_text(channel_dir / "tokenizer.json", "{}");
        config["text_config"]["num_hidden_layers"]                          = 24;
        config["text_config"]["linear_num_key_heads"]                       = 1;
        config["text_config"]["linear_num_value_heads"]                     = 2;
        config["text_config"]["linear_key_head_dim"]                        = 1;
        config["text_config"]["linear_value_head_dim"]                      = 1;
        config["quantization_config"]["config_groups"]["fp8"]["targets"][0] = channel_module;
        llama_safetensors_qwen35_importer channel_importer(channel_dir, config);
        require(channel_importer.materialize("blk.0.ssm_out.scale", GGML_TYPE_BF16, 4) ==
                    std::vector<uint8_t>({ 1, 2, 3, 4 }),
                "Qwen output channel scale incorrectly followed the weight's column permutation");

        const std::filesystem::path w8_dir = dir.path / "w8-column-transform";
        write_single_shard_model(w8_dir, {
            { channel_module + ".weight",       "I8",   { 2, 32 }, std::vector<uint8_t>(64) },
            { channel_module + ".weight_scale", "BF16", { 2, 1 },  { 0x80, 0x3f, 0x80, 0x3f } },
        });
        write_text(w8_dir / "generation_config.json", "{}");
        write_text(w8_dir / "tokenizer.json", "{}");
        config["quantization_config"] = llama_safetensors_json::parse(w8a8_config).at("quantization_config");
        config["quantization_config"]["config_groups"]["int8"]["targets"][0] = channel_module;
        llama_safetensors_qwen35_importer w8_importer(w8_dir, config);
        require_rejected(
            [&] {
                ggml_type type;
                std::array<int64_t, GGML_MAX_DIMS> ne;
                (void) w8_importer.describe("blk.0.ssm_out.weight", type, ne);
            },
            "Qwen3.5 accepted a W8A8 column transform after repacking");

        const std::filesystem::path awq_dir = dir.path / "awq-row-transform";
        const std::string awq_module = "model.language_model.layers.0.linear_attn.in_proj_z";
        write_single_shard_model(awq_dir, {
            { awq_module + ".qweight", "I32", { 128, 1 }, std::vector<uint8_t>(128 * sizeof(uint32_t)) },
            { awq_module + ".qzeros",  "I32", { 1, 1 },   i32_bytes({ 1 })                              },
            { awq_module + ".scales",  "F16", { 1, 8 },   std::vector<uint8_t>(8 * sizeof(uint16_t), 1) },
        });
        write_text(awq_dir / "generation_config.json", "{}");
        write_text(awq_dir / "tokenizer.json", "{}");
        config["quantization_config"] = llama_safetensors_json::parse(awq_config).at("quantization_config");
        llama_safetensors_qwen35_importer awq_importer(awq_dir, config);
        require_rejected(
            [&] {
                ggml_type type;
                std::array<int64_t, GGML_MAX_DIMS> ne;
                (void) awq_importer.describe("blk.0.attn_gate.weight", type, ne);
            },
            "Qwen3.5 accepted an AWQ row transform after repacking");
    }

    // Metadata and tokenizer conversion are shared by architecture importers.
    // Keep their strict parsing and GGUF representation covered independently
    // of the full Qwen model fixture.
    {
        const llama_safetensors_json generation = {
            { "top_k",       7     },
            { "top_p",       0.75f },
            { "temperature", 0.5f  },
        };
        const llama_safetensors_json tokenizer = {
            { "model",
             {
                  { "vocab", { { "a", 0 }, { "b", 1 } } },
                  { "merges", { llama_safetensors_json::array({ "a", "b" }), "a b" } },
              } },
            { "added_tokens",
             {
                  { { "id", 2 }, { "content", "<|pad|>" }, { "special", true } },
                  { { "id", 3 }, { "content", "plain-added" }, { "special", false } },
              } },
        };

        llama_safetensors_metadata_sink sink;
        llama_safetensors_emit_sampling_defaults(sink, generation);
        const auto rope = llama_safetensors_parse_rope(
            llama_safetensors_json{
                { "rope_theta",            1000000.0f  },
                { "partial_rotary_factor", 0.5f        },
                { "mrope_section",         { 8, 8, 0 } },
        },
            { 1, 2, 3, 4 }, 0.25f);
        require(rope.theta == 1000000.0f && rope.partial_rotary_factor == 0.5f &&
                    rope.mrope_sections == std::array<int32_t, 4>{ 8, 8, 0, 4 },
                "generic RoPE parsing is wrong");
        llama_safetensors_emit_bpe_tokenizer(sink, tokenizer, { "fixture", 4, 0, 1, std::string("<|pad|>"), true, {} },
                                             std::string("{{ messages }}"));

        gguf_context * metadata = sink.release();
        const int64_t  top_k    = gguf_find_key(metadata, "general.sampling.top_k");
        const int64_t  top_p    = gguf_find_key(metadata, "general.sampling.top_p");
        const int64_t  temp     = gguf_find_key(metadata, "general.sampling.temp");
        const int64_t  tokens   = gguf_find_key(metadata, "tokenizer.ggml.tokens");
        const int64_t  types    = gguf_find_key(metadata, "tokenizer.ggml.token_type");
        const int64_t  pad      = gguf_find_key(metadata, "tokenizer.ggml.padding_token_id");
        const int64_t  chat     = gguf_find_key(metadata, "tokenizer.chat_template");
        require(top_k >= 0 && gguf_get_val_i32(metadata, top_k) == 7 && top_p >= 0 &&
                    gguf_get_val_f32(metadata, top_p) == 0.75f && temp >= 0 && gguf_get_val_f32(metadata, temp) == 0.5f,
                "generic sampling metadata is wrong");
        require(tokens >= 0 && gguf_get_arr_n(metadata, tokens) == 4 &&
                    std::string(gguf_get_arr_str(metadata, tokens, 2)) == "<|pad|>" && types >= 0 &&
                    static_cast<const int32_t *>(gguf_get_arr_data(metadata, types))[2] == 3 &&
                    static_cast<const int32_t *>(gguf_get_arr_data(metadata, types))[3] == 4 && pad >= 0 &&
                    gguf_get_val_u32(metadata, pad) == 2 && chat >= 0 &&
                    std::string(gguf_get_val_str(metadata, chat)) == "{{ messages }}",
                "generic BPE tokenizer metadata is wrong");
        gguf_free(metadata);

        require_rejected(
            [&] {
                (void) llama_safetensors_parse_rope(
                    llama_safetensors_json{
                        { "rope_theta",    1.0f              },
                        { "mrope_section", { 1, 2, 3, 4, 5 } },
                },
                    { 0, 0, 0, 0 }, 1.0f);
            },
            "oversized mrope_section was accepted");
        require_rejected(
            [&] { (void) llama_safetensors_first_token_id(llama_safetensors_json::array(), "eos_token_id"); },
            "empty token-id array was accepted");
    }

    // Quant adapters are architecture-independent contracts. Exercise them
    // directly so an importer cannot accidentally make a correctness-critical
    // scale optional while rearranging canonical names.
    {
        const auto path = dir.path / "fp8-channel";
        write_single_shard_model(path, {
                                           { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
                                           { "module.weight_scale", "BF16",    { 2 },     std::vector<uint8_t>(4)  },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);

        llama_safetensors_quant_adapters incomplete(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = incomplete.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_F8_E4M3, "FP8 channel weight binding is wrong");
        incomplete.consume(*weight);
        require_rejected([&] { incomplete.validate_complete(); }, "FP8 channel weight was accepted without its scale");

        llama_safetensors_quant_adapters complete(llama_safetensors_read_json(path / "config.json"), registry);
        const auto complete_weight = complete.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto scale           = complete.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
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
                                           { "module.weight",       "F8_E4M3", { 2, 32 }, std::vector<uint8_t>(64) },
                                           { "module.weight_scale", "F16",     { 2 },     std::vector<uint8_t>(4)  },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "FP8 channel contract accepted the wrong scale dtype");
    }
    {
        const auto path = dir.path / "fp8-channel-wrong-weight-type";
        write_single_shard_model(path, {
                                           { "module.weight",       "BF16", { 2, 32 }, std::vector<uint8_t>(128) },
                                           { "module.weight_scale", "BF16", { 2 },     std::vector<uint8_t>(4)   },
        });
        write_text(path / "config.json", fp8_channel_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "FP8 channel contract accepted a BF16 weight");
    }
    {
        const auto path = dir.path / "fp8-block";
        write_single_shard_model(path,
                                 {
                                     { "module.weight",           "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
                                     { "module.weight_scale_inv", "BF16",    { 1, 1 },     std::vector<uint8_t>(2)         },
        });
        write_text(path / "config.json", fp8_block_config);
        const auto                       registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto                       scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
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
        write_single_shard_model(path,
                                 {
                                     { "module.weight",           "F8_E4M3", { 128, 128 }, std::vector<uint8_t>(128 * 128) },
                                     { "module.weight_scale_inv", "BF16",    { 1, 2 },     std::vector<uint8_t>(4)         },
        });
        write_text(path / "config.json", fp8_block_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "FP8 block contract accepted the wrong scale grid");
    }
    {
        const auto path = dir.path / "w8a8";
        std::vector<uint8_t> weights(128);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(i % 64) - 32));
        }
        write_single_shard_model(path, {
                                           { "module.weight",       "I8",   { 2, 64 }, weights                    },
                                           { "module.weight_scale", "BF16", { 2, 1 },  { 0x80, 0x3f, 0x00, 0x40 } },
        });
        write_text(path / "config.json", w8a8_config);
        const auto                       registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q8_0 &&
                    weight->materialization == llama_safetensors_quant_materialization::W8A8_REPACK &&
                    !adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE).has_value() &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q8_0,
                "W8A8 binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = sizeof(ggml_fp16_t) + 32;
        require(repacked.size() == 4 * block_size, "W8A8 Q8_0 repack has the wrong size");
        for (size_t row = 0; row < 2; ++row) {
            const uint16_t expected_scale = ggml_fp32_to_fp16(row == 0 ? 1.0f : 2.0f);
            for (size_t block = 0; block < 2; ++block) {
                const uint8_t * actual = repacked.data() + (row * 2 + block) * block_size;
                require(load_u16(actual) == expected_scale &&
                            std::equal(weights.begin() + row * 64 + block * 32,
                                       weights.begin() + row * 64 + (block + 1) * 32,
                                       actual + sizeof(ggml_fp16_t)),
                        "W8A8 Q8_0 repack changed a row scale or code block");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "w8a8-wrong-scale-type";
        write_single_shard_model(path, {
                                           { "module.weight",       "I8",  { 2, 32 }, std::vector<uint8_t>(64) },
                                           { "module.weight_scale", "F16", { 2, 1 },  std::vector<uint8_t>(4)  },
        });
        write_text(path / "config.json", w8a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "W8A8 contract accepted the wrong scale dtype");
    }
    {
        const auto path = dir.path / "w8a8-wrong-weight-type";
        write_single_shard_model(path, {
                                           { "module.weight",       "BF16", { 2, 32 }, std::vector<uint8_t>(128) },
                                           { "module.weight_scale", "BF16", { 2, 1 },  std::vector<uint8_t>(4)   },
        });
        write_text(path / "config.json", w8a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "W8A8 contract accepted a BF16 weight with an INT8 scale sidecar");
    }
    {
        const auto path = dir.path / "w8a8-scale-overflow";
        write_single_shard_model(path, {
                                           { "module.weight",       "I8",   { 1, 32 }, std::vector<uint8_t>(32) },
                                           { "module.weight_scale", "BF16", { 1, 1 },  { 0x80, 0x47 }             },
        });
        write_text(path / "config.json", w8a8_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value(), "W8A8 overflow fixture did not bind");
        require_rejected([&] { (void) adapters.read(*weight); },
                         "W8A8 accepted a scale that overflows Q8_0");
    }
    {
        const auto path = dir.path / "awq";
        constexpr size_t cols = 256;
        constexpr size_t rows = 8;
        constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };
        std::vector<uint32_t> qweight(cols);
        std::vector<uint32_t> qzeros(2);
        std::vector<uint8_t> scales(2 * rows * sizeof(uint16_t));
        for (size_t col = 0; col < cols; ++col) {
            for (size_t row = 0; row < rows; ++row) {
                qweight[col] |= ((col + 3 * row) % 16) << shifts[row];
            }
        }
        for (size_t group = 0; group < 2; ++group) {
            for (size_t row = 0; row < rows; ++row) {
                qzeros[group] |= (1 + (group + row) % 8) << shifts[row];
                store_f16(scales, group * rows + row, 0.5f + 0.25f * ((group + row) % 8));
            }
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { cols, 1 }, i32_bytes(qweight) },
            { "module.qzeros",  "I32", { 2, 1 },    i32_bytes(qzeros)  },
            { "module.scales",  "F16", { 2, rows }, scales             },
        });
        write_text(path / "config.json", awq_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->materialization == llama_safetensors_quant_materialization::AWQ_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "AWQ binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        require(repacked.size() == rows * (cols / 32) * block_size, "AWQ Q4_1 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const size_t group = block / 4;
                std::array<uint8_t, 32> codes{};
                for (size_t col = 0; col < codes.size(); ++col) {
                    codes[col] = (block * 32 + col + 3 * row) % 16;
                }
                require_q4_1_block(
                    repacked.data() + (row * (cols / 32) + block) * block_size,
                    0.5f + 0.25f * ((group + row) % 8), 1 + (group + row) % 8, codes,
                    "AWQ Q4_1 repack changed a row/group code, scale, or zero point");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "awq-missing-zero";
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 128, 1 }, std::vector<uint8_t>(128 * 4) },
            { "module.scales",  "F16", { 1, 8 },   std::vector<uint8_t>(16)      },
        });
        write_text(path / "config.json", awq_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "AWQ contract accepted a missing zero-point sidecar");
    }
    {
        const auto path = dir.path / "awq-minimum-overflow";
        std::vector<uint8_t> scales(8 * sizeof(uint16_t));
        for (size_t row = 0; row < 8; ++row) {
            const uint16_t maximum = 0x7bff;
            std::memcpy(scales.data() + row * sizeof(maximum), &maximum, sizeof(maximum));
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 128, 1 }, std::vector<uint8_t>(128 * sizeof(uint32_t)) },
            { "module.qzeros",  "I32", { 1, 1 },   i32_bytes({ 15 })                              },
            { "module.scales",  "F16", { 1, 8 },   scales                                         },
        });
        write_text(path / "config.json", awq_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value(), "AWQ overflow fixture did not bind");
        require_rejected([&] { (void) adapters.read(*weight); },
                         "AWQ accepted a minimum that overflows Q4_1");
    }
    {
        const auto path = dir.path / "gptq";
        constexpr size_t cols = 256;
        constexpr size_t rows = 8;
        std::vector<uint32_t> qweight((cols / 8) * rows);
        std::vector<uint32_t> qzeros(2);
        std::vector<uint8_t> scales(2 * rows * sizeof(uint16_t));
        for (size_t packed_col = 0; packed_col < cols / 8; ++packed_col) {
            for (size_t row = 0; row < rows; ++row) {
                uint32_t word = 0;
                for (size_t lane = 0; lane < 8; ++lane) {
                    word |= ((packed_col * 8 + lane + 5 * row) % 16) << (4 * lane);
                }
                qweight[packed_col * rows + row] = word;
            }
        }
        for (size_t group = 0; group < 2; ++group) {
            for (size_t row = 0; row < rows; ++row) {
                const uint8_t zero = 1 + (2 * group + row) % 8;
                qzeros[group] |= uint32_t(zero - 1) << (4 * row);
                store_f16(scales, group * rows + row, 0.5f + 0.25f * ((2 * group + row) % 8));
            }
        }
        std::vector<uint32_t> groups(cols);
        for (size_t col = 0; col < cols; ++col) {
            groups[col] = col / 128;
        }
        write_single_shard_model(path, {
            { "module.qweight", "I32", { cols / 8, rows }, i32_bytes(qweight) },
            { "module.qzeros",  "I32", { 2, 1 },           i32_bytes(qzeros)  },
            { "module.scales",  "F16", { 2, rows },        scales             },
            { "module.g_idx",   "I32", { cols },           i32_bytes(groups)  },
        });
        write_text(path / "config.json", gptq_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value() && weight->target_type == GGML_TYPE_Q4_1 &&
                    weight->materialization == llama_safetensors_quant_materialization::GPTQ_REPACK &&
                    weight->target_shape == std::vector<int64_t>({ cols, rows }) &&
                    adapters.file_type() == LLAMA_FTYPE_MOSTLY_Q4_1,
                "GPTQ binding is wrong");
        const std::vector<uint8_t> repacked = adapters.read(*weight);
        constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + 16;
        require(repacked.size() == rows * (cols / 32) * block_size, "GPTQ Q4_1 repack has the wrong size");
        for (size_t row = 0; row < rows; ++row) {
            for (size_t block = 0; block < cols / 32; ++block) {
                const size_t group = block / 4;
                std::array<uint8_t, 32> codes{};
                for (size_t col = 0; col < codes.size(); ++col) {
                    codes[col] = (block * 32 + col + 5 * row) % 16;
                }
                require_q4_1_block(
                    repacked.data() + (row * (cols / 32) + block) * block_size,
                    0.5f + 0.25f * ((2 * group + row) % 8), 1 + (2 * group + row) % 8, codes,
                    "GPTQ Q4_1 repack changed a row/group code, scale, or zero point");
            }
        }
        adapters.consume(*weight);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "gptq-act-order";
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 16, 8 }, std::vector<uint8_t>(16 * 8 * 4) },
        });
        write_text(path / "config.json", gptq_act_order_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected(
            [&] { (void) llama_safetensors_quant_adapters(llama_safetensors_read_json(path / "config.json"), registry); },
            "act-order GPTQ was not rejected");
    }
    {
        const auto path = dir.path / "gptq-partial-packed-row";
        std::vector<uint32_t> groups(128);
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 16, 9 },  std::vector<uint8_t>(16 * 9 * sizeof(uint32_t)) },
            { "module.qzeros",  "I32", { 1, 1 },   std::vector<uint8_t>(sizeof(uint32_t))          },
            { "module.scales",  "F16", { 1, 9 },   std::vector<uint8_t>(9 * sizeof(uint16_t))      },
            { "module.g_idx",   "I32", { 128 },    i32_bytes(groups)                               },
        });
        write_text(path / "config.json", gptq_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "GPTQ accepted output rows not divisible by its eight-row packing");
    }
    {
        const auto path = dir.path / "gptq-minimum-overflow";
        std::vector<uint8_t> scales(8 * sizeof(uint16_t));
        for (size_t row = 0; row < 8; ++row) {
            const uint16_t maximum = 0x7bff;
            std::memcpy(scales.data() + row * sizeof(maximum), &maximum, sizeof(maximum));
        }
        std::vector<uint32_t> groups(128);
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 16, 8 }, std::vector<uint8_t>(16 * 8 * sizeof(uint32_t)) },
            { "module.qzeros",  "I32", { 1, 1 },  i32_bytes({ 14 })                               },
            { "module.scales",  "F16", { 1, 8 },  scales                                          },
            { "module.g_idx",   "I32", { 128 },   i32_bytes(groups)                               },
        });
        write_text(path / "config.json", gptq_config);
        const auto registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        require(weight.has_value(), "GPTQ overflow fixture did not bind");
        require_rejected([&] { (void) adapters.read(*weight); },
                         "GPTQ accepted a minimum that overflows Q4_1");
    }
    {
        const auto path = dir.path / "gptq-nonidentity-groups";
        std::vector<uint32_t> groups(128);
        groups.back() = 1;
        write_single_shard_model(path, {
            { "module.qweight", "I32", { 16, 8 }, std::vector<uint8_t>(16 * 8 * 4) },
            { "module.qzeros",  "I32", { 1, 1 },  std::vector<uint8_t>(4)          },
            { "module.scales",  "F16", { 1, 8 },  std::vector<uint8_t>(16)         },
            { "module.g_idx",   "I32", { 128 },   i32_bytes(groups)                 },
        });
        write_text(path / "config.json", gptq_config);
        const auto registry = llama_safetensors_registry::load(path);
        require_rejected([&] { (void) llama_safetensors_quant_adapters(path, registry); },
                         "GPTQ contract accepted a non-identity group map");
    }
    {
        const auto path = dir.path / "nvfp4";
        write_single_shard_model(path,
                                 {
                                     { "module.weight_packed", "U8", { 2, 32 }, std::vector<uint8_t>(64) },
                                     { "module.weight_scale", "F8_E4M3", { 2, 4 }, std::vector<uint8_t>(8, 0x38) },
                                     { "module.weight_global_scale", "F32", {}, f32_bytes(1.0f) },
                                     { "module.input_global_scale", "F32", {}, f32_bytes(1.0f) },
        });
        write_text(path / "config.json", nvfp4_config);
        const auto                       registry = llama_safetensors_registry::load(path);
        llama_safetensors_quant_adapters adapters(llama_safetensors_read_json(path / "config.json"), registry);
        const auto                       weight = adapters.bind("module", llama_safetensors_quant_role::WEIGHT);
        const auto                       scale  = adapters.bind("module", llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto                       input  = adapters.bind("module", llama_safetensors_quant_role::INPUT_SCALE);
        require(weight.has_value() && weight->target_type == GGML_TYPE_NVFP4 &&
                    weight->target_shape == std::vector<int64_t>({ 64, 2 }) && adapters.read(*weight).size() == 72,
                "NVFP4 packed-weight binding or repack is wrong");
        require(scale.has_value() && input.has_value(), "NVFP4 global scale binding is incomplete");
        adapters.consume(*weight);
        adapters.consume(*scale);
        adapters.consume(*input);
        adapters.validate_complete();
    }
    {
        const auto path = dir.path / "nvfp4-zero-point";
        write_single_shard_model(path,
                                 {
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
        write_single_shard_model(path,
                                 {
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

    {
        ggml_init_params params {
            /*.mem_size   =*/ 2 * ggml_tensor_overhead() + 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        require(ctx != nullptr, "failed to create direct-upload test context");
        ggml_tensor * destination = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 2, 2);
        ggml_backend_buffer_t buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        require(buffer != nullptr, "failed to allocate direct-upload test buffer");

        const llama_safetensors_tensor_binding binding { "fp8", std::nullopt };
        require(llama_safetensors_load_tensor_direct(registry, binding, destination, false),
                "mapped canonical tensor did not take the direct-upload path");
        std::array<uint8_t, 4> uploaded{};
        ggml_backend_tensor_get(destination, uploaded.data(), 0, uploaded.size());
        require(uploaded == std::array<uint8_t, 4>({ 7, 8, 9, 10 }),
                "direct upload changed canonical tensor bytes");
        const auto buffered = llama_safetensors_registry::load(
            dir.path, llama_safetensors_io_mode::BUFFERED);
        require(!llama_safetensors_load_tensor_direct(buffered, binding, destination, false),
                "buffered tensor incorrectly took the mapped direct-upload path");

        llama_safetensors_quant_binding transformed {
            llama_safetensors_quant_materialization::W8A8_REPACK,
            "fp8",
            {},
            GGML_TYPE_F8_E4M3,
            { 2, 2 },
        };
        require(!llama_safetensors_load_tensor_direct(
                    registry, { "fp8", transformed }, destination, false),
                "transformed tensor incorrectly took the direct-upload path");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }

    const auto no_mmap = llama_safetensors_registry::load(dir.path, llama_safetensors_io_mode::BUFFERED);
    const auto * no_mmap_packed = no_mmap.find("packed");
    require(no_mmap_packed != nullptr && no_mmap.data(*no_mmap_packed) == nullptr &&
                no_mmap.read(*no_mmap_packed) == std::vector<uint8_t>({ 5, 6 }),
            "non-mmap safetensors fallback did not perform an exact bounded read");
    const auto * no_mmap_a = no_mmap.find("a");
    require(no_mmap_a != nullptr && no_mmap.data(*no_mmap_a) == nullptr &&
                no_mmap.read(*no_mmap_a) == std::vector<uint8_t>({ 1, 2, 3, 4 }),
            "non-mmap canonical tensor did not preserve exact bytes");

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
    for (const auto & [needle, replacement] : std::array<std::pair<const char *, const char *>, 3>{
             {
              { "\"symmetric\":true", "\"symmetric\":false" },
              { "\"dynamic\":false", "\"dynamic\":true" },
              { "\"group_size\":16", "\"group_size\":32" },
              }
    }) {
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
        (void) llama_safetensors_qwen35_importer(dir.path, llama_safetensors_read_json(dir.path / "config.json"));
    } catch (const std::exception &) {
        unsupported_rejected = true;
    }
    require(unsupported_rejected, "Qwen3.5 importer accepted an unsupported model contract");

    return 0;
}
