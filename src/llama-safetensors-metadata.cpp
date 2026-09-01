#include "llama-safetensors-metadata.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace {

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

uint64_t read_varint(std::string_view data, size_t & offset, std::string_view context) {
    uint64_t result = 0;
    for (unsigned shift = 0; shift < 64 && offset < data.size(); shift += 7) {
        const uint8_t byte = static_cast<uint8_t>(data[offset++]);
        result |= uint64_t(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) {
            return result;
        }
    }
    throw std::runtime_error("invalid protobuf varint in " + std::string(context));
}

std::string_view read_message(std::string_view data, size_t & offset, std::string_view context) {
    const uint64_t length = read_varint(data, offset, context);
    if (length > data.size() - offset) {
        throw std::runtime_error("truncated protobuf message in " + std::string(context));
    }
    const std::string_view result = data.substr(offset, static_cast<size_t>(length));
    offset += static_cast<size_t>(length);
    return result;
}

void skip_protobuf_field(std::string_view data, size_t & offset, uint32_t wire, std::string_view context) {
    switch (wire) {
        case 0: (void) read_varint(data, offset, context); return;
        case 1:
            if (data.size() - offset < 8) break;
            offset += 8;
            return;
        case 2: (void) read_message(data, offset, context); return;
        case 5:
            if (data.size() - offset < 4) break;
            offset += 4;
            return;
        default: break;
    }
    throw std::runtime_error("unsupported or truncated protobuf field in " + std::string(context));
}

struct spm_piece {
    std::string token;
    float       score = 0.0f;
    int32_t     type  = 1;
};

spm_piece parse_spm_piece(std::string_view data) {
    spm_piece result;
    size_t offset = 0;
    while (offset < data.size()) {
        const uint64_t tag   = read_varint(data, offset, "SentencePiece");
        const uint32_t field = tag >> 3;
        const uint32_t wire  = tag & 7;
        if (field == 1 && wire == 2) {
            result.token = std::string(read_message(data, offset, "SentencePiece.piece"));
        } else if (field == 2 && wire == 5) {
            if (data.size() - offset < sizeof(float)) {
                throw std::runtime_error("truncated SentencePiece score");
            }
            std::memcpy(&result.score, data.data() + offset, sizeof(float));
            offset += sizeof(float);
        } else if (field == 3 && wire == 0) {
            result.type = static_cast<int32_t>(read_varint(data, offset, "SentencePiece.type"));
        } else {
            skip_protobuf_field(data, offset, wire, "SentencePiece");
        }
    }
    if (result.token.empty()) {
        throw std::runtime_error("SentencePiece entry is missing its token");
    }
    if (result.type < 1 || result.type > 6) {
        throw std::runtime_error("SentencePiece entry has an unsupported type");
    }
    return result;
}

}  // namespace

llama_safetensors_json llama_safetensors_read_json(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open JSON file '" + path.string() + "'");
    }
    try {
        return llama_safetensors_json::parse(input);
    } catch (const llama_safetensors_json::exception & error) {
        throw std::runtime_error("invalid JSON in '" + path.string() + "': " + error.what());
    }
}

std::string llama_safetensors_read_text(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open text file '" + path.string() + "'");
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::optional<std::string> llama_safetensors_read_optional_text(const std::filesystem::path & path) {
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }
    return llama_safetensors_read_text(path);
}

llama_safetensors_metadata_sink::llama_safetensors_metadata_sink() : context_(gguf_init_empty()) {
    if (context_ == nullptr) {
        throw std::runtime_error("failed to allocate native safetensors metadata");
    }
}

llama_safetensors_metadata_sink::~llama_safetensors_metadata_sink() {
    gguf_free(context_);
}

void llama_safetensors_metadata_sink::set_string(std::string_view key, std::string_view value) {
    const std::string key_string(key);
    const std::string value_string(value);
    gguf_set_val_str(context_, key_string.c_str(), value_string.c_str());
}

void llama_safetensors_metadata_sink::set_u32(std::string_view key, uint32_t value) {
    const std::string key_string(key);
    gguf_set_val_u32(context_, key_string.c_str(), value);
}

void llama_safetensors_metadata_sink::set_i32(std::string_view key, int32_t value) {
    const std::string key_string(key);
    gguf_set_val_i32(context_, key_string.c_str(), value);
}

void llama_safetensors_metadata_sink::set_f32(std::string_view key, float value) {
    const std::string key_string(key);
    gguf_set_val_f32(context_, key_string.c_str(), value);
}

void llama_safetensors_metadata_sink::set_bool(std::string_view key, bool value) {
    const std::string key_string(key);
    gguf_set_val_bool(context_, key_string.c_str(), value);
}

void llama_safetensors_metadata_sink::set_f32_array(std::string_view key, const float * values, size_t count) {
    const std::string key_string(key);
    gguf_set_arr_data(context_, key_string.c_str(), GGUF_TYPE_FLOAT32, values, count);
}

void llama_safetensors_metadata_sink::set_i32_array(std::string_view key, const int32_t * values, size_t count) {
    const std::string key_string(key);
    gguf_set_arr_data(context_, key_string.c_str(), GGUF_TYPE_INT32, values, count);
}

void llama_safetensors_metadata_sink::set_u32_array(std::string_view key, const uint32_t * values, size_t count) {
    const std::string key_string(key);
    gguf_set_arr_data(context_, key_string.c_str(), GGUF_TYPE_UINT32, values, count);
}

void llama_safetensors_metadata_sink::set_u64_array(std::string_view key, const uint64_t * values, size_t count) {
    const std::string key_string(key);
    gguf_set_arr_data(context_, key_string.c_str(), GGUF_TYPE_UINT64, values, count);
}

void llama_safetensors_metadata_sink::set_string_array(std::string_view key, const std::vector<std::string> & values) {
    std::vector<const char *> pointers(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        pointers[i] = values[i].c_str();
    }
    const std::string key_string(key);
    gguf_set_arr_str(context_, key_string.c_str(), pointers.data(), pointers.size());
}

gguf_context * llama_safetensors_metadata_sink::release() {
    gguf_context * result = context_;
    context_              = nullptr;
    return result;
}

void llama_safetensors_emit_sampling_defaults(llama_safetensors_metadata_sink &           sink,
                                              const llama_safetensors_json &              generation,
                                              const llama_safetensors_sampling_defaults & defaults) {
    if (!generation.is_object()) {
        throw std::runtime_error("generation configuration must be a JSON object");
    }
    sink.set_i32("general.sampling.top_k", generation.value("top_k", defaults.top_k));
    sink.set_f32("general.sampling.top_p", generation.value("top_p", defaults.top_p));
    sink.set_f32("general.sampling.temp", generation.value("temperature", defaults.temperature));
}

llama_safetensors_rope_config llama_safetensors_parse_rope(const llama_safetensors_json & rope,
                                                           std::array<int32_t, 4>         default_sections,
                                                           float default_partial_rotary_factor) {
    if (!rope.is_object()) {
        throw std::runtime_error("RoPE configuration must be a JSON object");
    }
    llama_safetensors_rope_config result{
        rope.at("rope_theta").get<float>(),
        rope.value("partial_rotary_factor", default_partial_rotary_factor),
        default_sections,
    };
    if (rope.contains("mrope_section")) {
        const auto configured = rope.at("mrope_section").get<std::vector<int32_t>>();
        if (configured.size() > result.mrope_sections.size()) {
            throw std::runtime_error("RoPE mrope_section contains more than four dimensions");
        }
        std::copy(configured.begin(), configured.end(), result.mrope_sections.begin());
    }
    return result;
}

uint32_t llama_safetensors_first_token_id(const llama_safetensors_json & value, std::string_view context) {
    if (value.is_array()) {
        if (value.empty()) {
            throw std::runtime_error(std::string(context) + " must not be an empty array");
        }
        return value.at(0).get<uint32_t>();
    }
    return value.get<uint32_t>();
}

void llama_safetensors_emit_bpe_tokenizer(llama_safetensors_metadata_sink &    sink,
                                          const llama_safetensors_json &       tokenizer,
                                          const llama_safetensors_bpe_policy & policy,
                                          const std::optional<std::string> &   chat_template) {
    if (!tokenizer.is_object() || !tokenizer.contains("model") || !tokenizer.at("model").is_object()) {
        throw std::runtime_error("tokenizer.json is missing its model object");
    }
    const auto & model = tokenizer.at("model");
    if (!model.contains("vocab") || !model.at("vocab").is_object()) {
        throw std::runtime_error("tokenizer model is missing its vocabulary");
    }
    if (!model.contains("merges") || !model.at("merges").is_array()) {
        throw std::runtime_error("tokenizer model is missing its merge array");
    }

    std::vector<std::string> tokens(policy.vocab_size);
    std::vector<int32_t>     token_types(policy.vocab_size, 5);  // GGML unused token
    for (uint32_t id = 0; id < policy.vocab_size; ++id) {
        tokens[id] = "[PAD" + std::to_string(id) + "]";
    }
    for (const auto & [token, id_json] : model.at("vocab").items()) {
        const uint32_t id = id_json.get<uint32_t>();
        if (id >= policy.vocab_size) {
            throw std::runtime_error("tokenizer vocabulary id exceeds configured vocabulary size");
        }
        tokens[id]      = token;
        token_types[id] = 1;  // GGML normal token
    }

    const llama_safetensors_json empty_added = llama_safetensors_json::array();
    const auto & added_tokens = tokenizer.contains("added_tokens") ? tokenizer.at("added_tokens") : empty_added;
    if (!added_tokens.is_array()) {
        throw std::runtime_error("tokenizer added_tokens must be an array");
    }
    for (const auto & added : added_tokens) {
        const uint32_t id = added.at("id").get<uint32_t>();
        if (id >= policy.vocab_size) {
            throw std::runtime_error("added token id exceeds configured vocabulary size");
        }
        const std::string content = added.at("content").get<std::string>();
        tokens[id]                = content;
        bool looks_control        = added.value("special", false) || (policy.angle_pipe_tokens_are_control &&
                                                               starts_with(content, "<|") && ends_with(content, "|>"));
        for (const std::string & prefix : policy.control_token_prefixes) {
            looks_control |= starts_with(content, prefix);
        }
        token_types[id] = looks_control ? 3 : 4;  // control or user-defined
    }

    std::vector<std::string> merges;
    merges.reserve(model.at("merges").size());
    for (const auto & merge : model.at("merges")) {
        if (merge.is_array() && merge.size() == 2) {
            merges.push_back(merge[0].get<std::string>() + " " + merge[1].get<std::string>());
        } else if (merge.is_string()) {
            merges.push_back(merge.get<std::string>());
        } else {
            throw std::runtime_error("unsupported tokenizer merge entry");
        }
    }

    sink.set_string("tokenizer.ggml.model", "gpt2");
    sink.set_string("tokenizer.ggml.pre", policy.pre_tokenizer);
    sink.set_string_array("tokenizer.ggml.tokens", tokens);
    sink.set_i32_array("tokenizer.ggml.token_type", token_types.data(), token_types.size());
    sink.set_string_array("tokenizer.ggml.merges", merges);
    sink.set_u32("tokenizer.ggml.bos_token_id", policy.bos_token_id);
    sink.set_u32("tokenizer.ggml.eos_token_id", policy.eos_token_id);
    sink.set_bool("tokenizer.ggml.add_bos_token", policy.add_bos_token);

    if (policy.padding_token) {
        const auto it = std::find(tokens.begin(), tokens.end(), *policy.padding_token);
        if (it == tokens.end()) {
            throw std::runtime_error("token not found: '" + *policy.padding_token + "'");
        }
        sink.set_u32("tokenizer.ggml.padding_token_id", static_cast<uint32_t>(it - tokens.begin()));
    }
    if (chat_template) {
        sink.set_string("tokenizer.chat_template", *chat_template);
    }
}

void llama_safetensors_emit_spm_tokenizer(llama_safetensors_metadata_sink &  sink,
                                          const std::filesystem::path &       tokenizer_model,
                                          uint32_t                            vocab_size,
                                          uint32_t                            bos_token_id,
                                          uint32_t                            eos_token_id,
                                          std::optional<uint32_t>             padding_token_id,
                                          bool                                add_bos_token,
                                          const std::optional<std::string> & chat_template) {
    const std::string data = llama_safetensors_read_text(tokenizer_model);
    std::vector<spm_piece> pieces;
    size_t offset = 0;
    while (offset < data.size()) {
        const uint64_t tag   = read_varint(data, offset, "SentencePiece ModelProto");
        const uint32_t field = tag >> 3;
        const uint32_t wire  = tag & 7;
        if (field == 1 && wire == 2) {
            pieces.push_back(parse_spm_piece(read_message(data, offset, "ModelProto.pieces")));
        } else {
            skip_protobuf_field(data, offset, wire, "SentencePiece ModelProto");
        }
    }
    if (pieces.size() != vocab_size) {
        throw std::runtime_error("tokenizer.model vocabulary does not match model vocab_size");
    }

    std::vector<std::string> tokens;
    std::vector<float>       scores;
    std::vector<int32_t>     types;
    tokens.reserve(pieces.size());
    scores.reserve(pieces.size());
    types.reserve(pieces.size());
    for (spm_piece & piece : pieces) {
        tokens.push_back(std::move(piece.token));
        scores.push_back(piece.score);
        types.push_back(piece.type);
    }

    sink.set_string("tokenizer.ggml.model", "llama");
    sink.set_string("tokenizer.ggml.pre", "default");
    sink.set_string_array("tokenizer.ggml.tokens", tokens);
    sink.set_f32_array("tokenizer.ggml.scores", scores.data(), scores.size());
    sink.set_i32_array("tokenizer.ggml.token_type", types.data(), types.size());
    sink.set_u32("tokenizer.ggml.bos_token_id", bos_token_id);
    sink.set_u32("tokenizer.ggml.eos_token_id", eos_token_id);
    if (padding_token_id) {
        sink.set_u32("tokenizer.ggml.padding_token_id", *padding_token_id);
    }
    sink.set_bool("tokenizer.ggml.add_bos_token", add_bos_token);
    if (chat_template) {
        sink.set_string("tokenizer.chat_template", *chat_template);
    }
}
