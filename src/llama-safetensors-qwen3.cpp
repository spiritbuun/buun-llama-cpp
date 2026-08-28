#include "llama-safetensors-qwen3.h"

#include "llama-safetensors-metadata.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-tensor.h"
#include "llama.h"

#include <algorithm>
#include <regex>
#include <stdexcept>

namespace {

llama_safetensors_source_name plain_source(std::string name) {
    return { std::move(name), {}, std::nullopt };
}

llama_safetensors_source_name projection_source(std::string module, llama_safetensors_quant_role role) {
    std::string source = module;
    switch (role) {
        case llama_safetensors_quant_role::WEIGHT:
            source += ".weight";
            break;
        case llama_safetensors_quant_role::WEIGHT_SCALE:
            source += ".weight_scale";
            break;
        case llama_safetensors_quant_role::INPUT_SCALE:
            source += ".input_global_scale";
            break;
    }
    return { std::move(source), std::move(module), role };
}

std::optional<llama_safetensors_source_name> map_target_source(uint32_t n_layer, const std::string & target_name) {
    if (target_name == "token_embd.weight") {
        return plain_source("model.embed_tokens.weight");
    }
    if (target_name == "output_norm.weight") {
        return plain_source("model.norm.weight");
    }
    if (target_name == "output.weight") {
        return projection_source("lm_head", llama_safetensors_quant_role::WEIGHT);
    }
    if (target_name == "output.scale") {
        return projection_source("lm_head", llama_safetensors_quant_role::WEIGHT_SCALE);
    }
    if (target_name == "output.input_scale") {
        return projection_source("lm_head", llama_safetensors_quant_role::INPUT_SCALE);
    }

    static const std::regex layer_pattern(R"(^blk\.([0-9]+)\.(.+)$)");
    std::smatch             match;
    if (!std::regex_match(target_name, match, layer_pattern)) {
        return std::nullopt;
    }
    const uint32_t layer = std::stoul(match[1].str());
    if (layer >= n_layer) {
        throw std::runtime_error("native Qwen3 importer does not support layer " + std::to_string(layer));
    }
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    return llama_safetensors_map_decoder_tensor(prefix, match[2].str());
}

llama_safetensors_tensor_binding map_target(const llama_safetensors_quant_adapters & quant,
                                            uint32_t                                 n_layer,
                                            const std::string &                      target_name) {
    auto source = map_target_source(n_layer, target_name);
    if (!source) {
        throw std::runtime_error("unsupported Qwen3 target tensor '" + target_name + "'");
    }
    return llama_safetensors_bind_tensor(quant, std::move(*source));
}

void validate_model_contract(const llama_safetensors_json & config) {
    if (config.value("model_type", std::string()) != "qwen3") {
        throw std::runtime_error("native Qwen3 importer requires model_type 'qwen3'");
    }
    const uint32_t n_layer  = config.value("num_hidden_layers", 0U);
    const uint32_t n_embd   = config.value("hidden_size", 0U);
    const uint32_t n_head   = config.value("num_attention_heads", 0U);
    const uint32_t n_kv     = config.value("num_key_value_heads", 0U);
    const uint32_t head_dim = config.value("head_dim", 0U);
    if (n_layer == 0 || n_embd == 0 || n_head == 0 || n_kv == 0 || head_dim == 0 || n_head % n_kv != 0) {
        throw std::runtime_error("native Qwen3 importer does not support this tensor geometry");
    }
}

}  // namespace

llama_safetensors_qwen3_importer::llama_safetensors_qwen3_importer(const std::filesystem::path & model_dir,
                                                                   llama_safetensors_json        config) :
    model_dir_(model_dir),
    config_(std::move(config)) {
    validate_model_contract(config_);
    n_layer_                                      = config_.at("num_hidden_layers").get<uint32_t>();
    generation_                                   = llama_safetensors_read_json(model_dir_ / "generation_config.json");
    tokenizer_                                    = llama_safetensors_read_json(model_dir_ / "tokenizer.json");
    const llama_safetensors_json tokenizer_config = llama_safetensors_read_json(model_dir_ / "tokenizer_config.json");
    if (tokenizer_config.contains("chat_template") && tokenizer_config.at("chat_template").is_string()) {
        chat_template_ = tokenizer_config.at("chat_template").get<std::string>();
    }
    registry_ = llama_safetensors_registry::load(model_dir_);
    quant_    = std::make_unique<llama_safetensors_quant_adapters>(config_, registry_);
}

bool llama_safetensors_qwen3_importer::probe(const llama_safetensors_json & config) {
    return config.value("model_type", std::string()) == "qwen3";
}

gguf_context * llama_safetensors_qwen3_importer::build_metadata() const {
    llama_safetensors_metadata_sink sink;
    sink.set_string("general.architecture", "qwen3");
    sink.set_string("general.type", "model");
    sink.set_string("general.name",
                    model_dir_.filename().empty() ? "Qwen3 Safetensors" : model_dir_.filename().string());
    const llama_safetensors_quant_summary & summary = quant_->summary();
    sink.set_u32("general.file_type", summary.nvfp4 != 0 ? LLAMA_FTYPE_MOSTLY_NVFP4 : LLAMA_FTYPE_MOSTLY_F8_E4M3);
    sink.set_u32("general.quantization_version", 2);
    llama_safetensors_emit_sampling_defaults(sink, generation_);

    sink.set_u32("qwen3.block_count", config_.at("num_hidden_layers").get<uint32_t>());
    sink.set_u32("qwen3.context_length", config_.at("max_position_embeddings").get<uint32_t>());
    sink.set_u32("qwen3.embedding_length", config_.at("hidden_size").get<uint32_t>());
    sink.set_u32("qwen3.feed_forward_length", config_.at("intermediate_size").get<uint32_t>());
    sink.set_u32("qwen3.attention.head_count", config_.at("num_attention_heads").get<uint32_t>());
    sink.set_u32("qwen3.attention.head_count_kv", config_.at("num_key_value_heads").get<uint32_t>());
    sink.set_u32("qwen3.attention.key_length", config_.at("head_dim").get<uint32_t>());
    sink.set_u32("qwen3.attention.value_length", config_.at("head_dim").get<uint32_t>());
    sink.set_f32("qwen3.rope.freq_base", config_.at("rope_theta").get<float>());
    sink.set_f32("qwen3.attention.layer_norm_rms_epsilon", config_.at("rms_norm_eps").get<float>());

    llama_safetensors_bpe_policy tokenizer_policy{
        "qwen2",
        config_.at("vocab_size").get<uint32_t>(),
        llama_safetensors_first_token_id(generation_.at("bos_token_id"), "bos_token_id"),
        llama_safetensors_first_token_id(generation_.at("eos_token_id"), "eos_token_id"),
        std::string("<|endoftext|>"),
        true,
        {},
    };
    llama_safetensors_emit_bpe_tokenizer(sink, tokenizer_, tokenizer_policy, chat_template_);
    return sink.release();
}

bool llama_safetensors_qwen3_importer::describe(const std::string &                  target_name,
                                                ggml_type &                          type,
                                                std::array<int64_t, GGML_MAX_DIMS> & ne) const {
    auto source = map_target_source(n_layer_, target_name);
    if (!source) {
        return false;
    }
    llama_safetensors_tensor_binding binding = llama_safetensors_bind_tensor(*quant_, std::move(*source));
    return llama_safetensors_describe_tensor(registry_, binding, type, ne);
}

size_t llama_safetensors_qwen3_importer::tensor_capacity_hint() const {
    return std::max<size_t>(256, registry_.tensors().size() * 2);
}

void llama_safetensors_qwen3_importer::bind(const std::string & target_name) const {
    llama_safetensors_consume_tensor(*quant_, map_target(*quant_, n_layer_, target_name));
}

std::vector<uint8_t> llama_safetensors_qwen3_importer::materialize(const std::string & target_name,
                                                                   ggml_type           target_type,
                                                                   size_t              target_size) const {
    try {
        return llama_safetensors_materialize_tensor(registry_, *quant_, map_target(*quant_, n_layer_, target_name),
                                                    target_type, target_size);
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to materialize '" + target_name + "': " + error.what());
    }
}

void llama_safetensors_qwen3_importer::validate_complete() const {
    quant_->validate_complete();
}
