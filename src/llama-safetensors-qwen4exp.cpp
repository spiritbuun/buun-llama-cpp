#include "llama-safetensors-qwen4exp.h"

#include "llama-safetensors-metadata.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-tensor.h"

#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {

const llama_safetensors_json & text_config(const llama_safetensors_json & config) {
    return config.contains("text_config") ? config.at("text_config") : config;
}

enum class transform_kind {
    OFFSET_NORM,
    A_LOG,
    QKV_ROWS,
    V_ROWS,
    HEAD_ROWS,
    CONV_ROWS,
    V_COLUMNS,
};

struct source_spec {
    std::string source;
    std::optional<llama_safetensors_quant_binding> quant;
    std::vector<transform_kind> transforms;
    size_t row_offset = 0;
    size_t row_count = 0;
    size_t scale_broadcast = 0;
    std::vector<std::string> concat_sources;
    std::vector<std::string> stack_sources;
    std::vector<llama_safetensors_quant_binding> stack_quant_weights;
    std::vector<llama_safetensors_quant_binding> stack_quant_scales;
    bool ple_table = false;
    bool ple_scale = false;
};

class unsupported_target : public std::runtime_error {
  public:
    explicit unsupported_target(const std::string & target) :
        std::runtime_error("unsupported Qwen4 target tensor '" + target + "'") {}
};

struct qwen4_geometry {
    std::string_view model_prefix;
    uint32_t n_layer;
    uint32_t n_mtp;
    uint32_t n_key_heads;
    uint32_t n_value_heads;
    uint32_t key_head_dim;
    uint32_t value_head_dim;
    uint32_t indexer_n_heads;
    uint32_t indexer_head_dim;
    uint32_t full_attention_interval;
    uint32_t n_expert;
    uint32_t n_ff_exp;
    uint32_t ple_layer;
    uint32_t ple_shards;
};

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

const llama_safetensors_tensor & require_tensor(
        const llama_safetensors_registry & registry, const std::string & name) {
    const auto * tensor = registry.find(name);
    if (tensor == nullptr) {
        throw std::runtime_error("safetensors tensor not found for Qwen4 import: '" + name + "'");
    }
    return *tensor;
}

std::string layer_prefix(uint32_t layer, const qwen4_geometry & geometry) {
    if (layer < geometry.n_layer) {
        return std::string(geometry.model_prefix) + ".layers." + std::to_string(layer) + ".";
    }
    if (layer < geometry.n_layer + geometry.n_mtp) {
        return "mtp.layers." + std::to_string(layer - geometry.n_layer) + ".";
    }
        throw unsupported_target("blk." + std::to_string(layer));
}

source_spec bind_projection(
        const llama_safetensors_quant_adapters & quant,
        const std::string & module,
        llama_safetensors_quant_role role,
        std::string plain = {}) {
    if (auto binding = quant.bind(module, role)) {
        return { binding->primary, std::move(binding) };
    }
    if (quant.applies(module)) {
        return {};
    }
    if (plain.empty()) {
        switch (role) {
            case llama_safetensors_quant_role::WEIGHT:       plain = module + ".weight"; break;
            case llama_safetensors_quant_role::WEIGHT_SCALE: plain = module + ".weight_scale"; break;
            case llama_safetensors_quant_role::INPUT_SCALE:  plain = module + ".input_scale"; break;
        }
    }
    return { std::move(plain) };
}

source_spec bind_decoder_source(
        const llama_safetensors_quant_adapters & quant,
        llama_safetensors_source_name source,
        std::vector<transform_kind> transforms = {}) {
    source_spec result;
    if (source.quant_role) {
        result = bind_projection(quant, source.module, *source.quant_role, source.source);
    } else {
        result.source = std::move(source.source);
    }
    result.transforms = std::move(transforms);
    return result;
}

bool recurrent_layer(uint32_t layer, const qwen4_geometry & geometry) {
    return layer < geometry.n_layer && (layer + 1) % geometry.full_attention_interval != 0;
}

source_spec map_target(
        const llama_safetensors_registry & registry,
        const llama_safetensors_quant_adapters & quant,
        const qwen4_geometry & geometry,
        const std::string & target) {
    if (target == "token_embd.weight") {
        return bind_projection(quant, std::string(geometry.model_prefix) + ".embed_tokens",
                               llama_safetensors_quant_role::WEIGHT);
    }
    if (target == "output.weight") {
        return bind_projection(quant, "lm_head", llama_safetensors_quant_role::WEIGHT);
    }
    if (target == "output.scale") {
        return bind_projection(quant, "lm_head", llama_safetensors_quant_role::WEIGHT_SCALE);
    }
    if (target == "output.input_scale") {
        return bind_projection(quant, "lm_head", llama_safetensors_quant_role::INPUT_SCALE);
    }
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 3> head = {{
        { "output_hc_norm.weight", "hyper_connection_mixer.hc_norm.weight" },
        { "output_hc_down.weight", "hyper_connection_mixer.input_mix_weight_down.weight" },
        { "output_hc_up.weight",   "hyper_connection_mixer.input_mix_weight_up.weight" },
    }};
    for (const auto & [canonical, source] : head) {
        if (target == canonical) {
            source_spec result { std::string(geometry.model_prefix) + "." + std::string(source) };
            if (canonical == std::string_view("output_hc_norm.weight")) {
                result.transforms.push_back(transform_kind::OFFSET_NORM);
            }
            return result;
        }
    }
    if (target == "per_layer_token_embd.weight") {
        source_spec result;
        result.ple_table = true;
        return result;
    }
    if (target == "per_layer_token_embd.scale") {
        source_spec result;
        result.ple_scale = true;
        return result;
    }

    static const std::regex layer_pattern(R"(^blk\.([0-9]+)\.(.+)$)");
    std::smatch match;
    if (!std::regex_match(target, match, layer_pattern)) {
        throw unsupported_target(target);
    }
    const uint32_t layer = std::stoul(match[1].str());
    const std::string suffix = match[2].str();
    const std::string prefix = layer_prefix(layer, geometry);

    struct name_pair { std::string_view target; std::string_view source; bool offset_norm; };
    static constexpr std::array<name_pair, 8> hc = {{
        { "hc_attn_norm.weight",   "attn_hyper_connection.hc_norm.weight", true },
        { "hc_attn_down.weight",   "attn_hyper_connection.input_mix_weight_down.weight", false },
        { "hc_attn_up.weight",     "attn_hyper_connection.input_mix_weight_up.weight", false },
        { "hc_attn_inject.weight", "attn_hyper_connection.block_inject_weight.weight", false },
        { "hc_ffn_norm.weight",    "mlp_hyper_connection.hc_norm.weight", true },
        { "hc_ffn_down.weight",    "mlp_hyper_connection.input_mix_weight_down.weight", false },
        { "hc_ffn_up.weight",      "mlp_hyper_connection.input_mix_weight_up.weight", false },
        { "hc_ffn_inject.weight",  "mlp_hyper_connection.block_inject_weight.weight", false },
    }};
    for (const name_pair & name : hc) {
        if (suffix == name.target) {
            source_spec result { prefix + std::string(name.source) };
            if (name.offset_norm) {
                result.transforms.push_back(transform_kind::OFFSET_NORM);
            }
            return result;
        }
    }

    if (suffix == "ffn_gate_inp.weight") {
        return { prefix + "mlp.gate.weight" };
    }
    if (suffix == "ffn_gate_inp_shexp.weight") {
        return { prefix + "mlp.shared_expert_gate.weight" };
    }
    if (suffix == "ffn_gate_shexp.weight") {
        return bind_projection(quant, prefix + "mlp.shared_expert.gate_proj",
                               llama_safetensors_quant_role::WEIGHT);
    }
    if (suffix == "ffn_up_shexp.weight") {
        return bind_projection(quant, prefix + "mlp.shared_expert.up_proj",
                               llama_safetensors_quant_role::WEIGHT);
    }
    if (suffix == "ffn_down_shexp.weight") {
        return bind_projection(quant, prefix + "mlp.shared_expert.down_proj",
                               llama_safetensors_quant_role::WEIGHT);
    }

    const std::string gate_up_module = prefix + "mlp.experts.gate_up_proj";
    const std::string down_module    = prefix + "mlp.experts.down_proj";
    const bool gate_up_quantized = quant.applies(gate_up_module);
    const bool split_experts = registry.find(prefix + "mlp.experts.0.gate_proj.weight") != nullptr;
    const auto bind_split_experts = [&](const char * projection) {
        source_spec result;
        const std::string first_module = prefix + "mlp.experts.0." + projection;
        const bool quantized = quant.applies(first_module);
        for (uint32_t expert = 0; expert < geometry.n_expert; ++expert) {
            const std::string module = prefix + "mlp.experts." + std::to_string(expert) + "." + projection;
            if (quantized) {
                auto weight = quant.bind(module, llama_safetensors_quant_role::WEIGHT);
                auto scale  = quant.bind(module, llama_safetensors_quant_role::WEIGHT_SCALE);
                if (!weight || !scale || weight->target_type != GGML_TYPE_F8_E4M3 ||
                        scale->target_type != GGML_TYPE_F32 ||
                        scale->materialization != llama_safetensors_quant_materialization::FP8_BLOCK_SCALE) {
                    throw std::runtime_error("unsupported split-expert FP8 contract for '" + module + "'");
                }
                result.stack_quant_weights.push_back(std::move(*weight));
                result.stack_quant_scales.push_back(std::move(*scale));
            } else {
                if (quant.applies(module)) {
                    throw std::runtime_error("mixed split-expert quantization in '" + module + "'");
                }
                result.stack_sources.push_back(module + ".weight");
            }
        }
        return result;
    };
    if (suffix == "ffn_gate_up_exps.weight") {
        if (gate_up_quantized) {
            return {};
        }
        if (split_experts) {
            return {};
        }
        return { gate_up_module };
    }
    if (suffix == "ffn_gate_exps.weight" || suffix == "ffn_up_exps.weight") {
        if (!gate_up_quantized && !split_experts) {
            return {};
        }
        if (split_experts) {
            return bind_split_experts(suffix == "ffn_gate_exps.weight" ? "gate_proj" : "up_proj");
        }
        source_spec result = bind_projection(
            quant, gate_up_module, llama_safetensors_quant_role::WEIGHT);
        result.row_offset = suffix == "ffn_gate_exps.weight" ? 0 : geometry.n_ff_exp;
        result.row_count  = geometry.n_ff_exp;
        return result;
    }
    if (suffix == "ffn_down_exps.weight") {
        if (quant.applies(down_module)) {
            return bind_projection(quant, down_module, llama_safetensors_quant_role::WEIGHT);
        }
        if (split_experts) {
            return bind_split_experts("down_proj");
        }
        return { down_module };
    }
    if (suffix == "ffn_gate_exps.scale" || suffix == "ffn_up_exps.scale" ||
        suffix == "ffn_down_exps.scale") {
        if (split_experts) {
            const char * projection = suffix == "ffn_down_exps.scale" ? "down_proj" :
                suffix == "ffn_gate_exps.scale" ? "gate_proj" : "up_proj";
            if (quant.applies(prefix + "mlp.experts.0." + projection)) {
                return {};
            }
        }
        const std::string & module = suffix == "ffn_down_exps.scale" ? down_module : gate_up_module;
        source_spec result = bind_projection(quant, module, llama_safetensors_quant_role::WEIGHT_SCALE);
        result.scale_broadcast = geometry.n_expert;
        return result;
    }
    if (ends_with(suffix, "_exps.input_scale")) {
        return {};
    }

    if (auto ordinary = llama_safetensors_map_decoder_tensor(prefix, suffix)) {
        std::vector<transform_kind> transforms;
        if (suffix == "attn_q_norm.weight" || suffix == "attn_k_norm.weight") {
            transforms.push_back(transform_kind::OFFSET_NORM);
        }
        return bind_decoder_source(quant, std::move(*ordinary), std::move(transforms));
    }

    if (suffix == "indexer.q_proj.weight" || suffix == "indexer.k_proj.weight") {
        source_spec result { prefix + "self_attn.indexer.index_qk_proj.weight" };
        const size_t n_q = static_cast<size_t>(geometry.indexer_n_heads) * geometry.indexer_head_dim;
        result.row_offset = suffix == "indexer.q_proj.weight" ? 0 : n_q;
        result.row_count  = suffix == "indexer.q_proj.weight" ? n_q : geometry.indexer_head_dim;
        return result;
    }
    if (suffix == "indexer.q_norm.weight" || suffix == "indexer.k_norm.weight") {
        source_spec result { prefix + "self_attn.indexer." +
            std::string(suffix == "indexer.q_norm.weight" ? "q_layernorm.weight" : "k_layernorm.weight") };
        result.transforms.push_back(transform_kind::OFFSET_NORM);
        return result;
    }

    static constexpr std::array<std::pair<std::string_view, std::string_view>, 6> ple = {{
        { "ple_key.weight",        "ple.key_proj.weight" },
        { "ple_value.weight",      "ple.value_proj.weight" },
        { "ple_norm_key.weight",   "ple.norm_key.weight" },
        { "ple_norm_query.weight", "ple.norm_query.weight" },
        { "ple_norm_conv.weight",  "ple.norm_conv.weight" },
        { "ple_conv1d.weight",     "ple.conv1d.weight" },
    }};
    for (const auto & [canonical, source] : ple) {
        if (suffix == canonical) {
            source_spec result { prefix + std::string(source) };
            if (suffix.rfind("ple_norm_", 0) == 0) {
                result.transforms.push_back(transform_kind::OFFSET_NORM);
            }
            return result;
        }
    }

    if (recurrent_layer(layer, geometry)) {
        struct recurrent_name { std::string_view target; std::string_view source; transform_kind transform; };
        static constexpr std::array<recurrent_name, 10> recurrent = {{
            { "attn_gate.weight", "linear_attn.in_proj_z.weight", transform_kind::V_ROWS },
            { "attn_qkv.weight",  "linear_attn.in_proj_qkv.weight", transform_kind::QKV_ROWS },
            { "ssm_alpha.weight", "linear_attn.in_proj_a.weight", transform_kind::HEAD_ROWS },
            { "ssm_beta.weight",  "linear_attn.in_proj_b.weight", transform_kind::HEAD_ROWS },
            { "ssm_norm.weight",  "linear_attn.norm.weight", transform_kind::V_ROWS },
            { "ssm_conv1d.weight", "linear_attn.conv1d.weight", transform_kind::CONV_ROWS },
            { "ssm_a",            "linear_attn.A_log", transform_kind::A_LOG },
            { "ssm_dt.bias",      "linear_attn.dt_bias", transform_kind::HEAD_ROWS },
            { "ssm_out.weight",   "linear_attn.out_proj.weight", transform_kind::V_COLUMNS },
            { "ssm_out.scale",    "linear_attn.out_proj.weight_scale", transform_kind::V_COLUMNS },
        }};
        for (const recurrent_name & name : recurrent) {
            if (suffix == name.target) {
                source_spec result;
                if (ends_with(name.target, ".weight")) {
                    const std::string module = prefix + std::string(name.source).substr(
                        0, std::string(name.source).size() - std::string_view(".weight").size());
                    result = bind_projection(quant, module, llama_safetensors_quant_role::WEIGHT);
                } else if (ends_with(name.target, ".scale")) {
                    const std::string source_name = prefix + std::string(name.source);
                    const std::string module = source_name.substr(
                        0, source_name.size() - std::string_view(".weight_scale").size());
                    result = bind_projection(quant, module, llama_safetensors_quant_role::WEIGHT_SCALE, source_name);
                } else {
                    result.source = prefix + std::string(name.source);
                }
                if (name.transform == transform_kind::A_LOG) {
                    // Preserve the source element width while permuting; A_LOG
                    // then converts the reordered values to F32.
                    result.transforms = { transform_kind::HEAD_ROWS, transform_kind::A_LOG };
                } else if (suffix != "ssm_norm.weight") {
                    result.transforms = { name.transform };
                }
                return result;
            }
        }
    }

    if (layer >= geometry.n_layer) {
        if (suffix == "nextn.enorm.weight") {
            return { "mtp.pre_fc_norm_embedding.weight", std::nullopt, { transform_kind::OFFSET_NORM } };
        }
        if (suffix == "nextn.hnorm.weight") {
            return { "mtp.pre_fc_norm_hidden.weight", std::nullopt, { transform_kind::OFFSET_NORM } };
        }
        if (suffix == "nextn.eh_proj.weight") {
            source_spec result;
            result.concat_sources = { "mtp.fc_embedding.weight", "mtp.fc_hidden.weight" };
            return result;
        }
        if (suffix == "nextn.hc_head_norm.weight" || suffix == "nextn.hc_head_down.weight" ||
            suffix == "nextn.hc_head_up.weight") {
            const std::string tail = suffix == "nextn.hc_head_norm.weight" ? "hc_norm.weight" :
                suffix == "nextn.hc_head_down.weight" ? "input_mix_weight_down.weight" :
                                                        "input_mix_weight_up.weight";
            source_spec result { "mtp.hyper_connection_mixer." + tail };
            if (suffix == "nextn.hc_head_norm.weight") {
                result.transforms.push_back(transform_kind::OFFSET_NORM);
            }
            return result;
        }
    }

    throw unsupported_target(target);
}

std::vector<int64_t> reverse_shape(const llama_safetensors_tensor & source) {
    std::vector<int64_t> result;
    result.reserve(source.shape.size());
    for (auto it = source.shape.rbegin(); it != source.shape.rend(); ++it) {
        if (*it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("Qwen4 source dimension exceeds runtime limits");
        }
        if (*it != 1 || source.shape.size() == 1) {
            result.push_back(static_cast<int64_t>(*it));
        }
    }
    return result.empty() ? std::vector<int64_t>{ 1 } : result;
}

ggml_type plain_type(const llama_safetensors_tensor & source, const source_spec & spec,
                     const std::string & target) {
    const bool numerical_transform = std::find(spec.transforms.begin(), spec.transforms.end(),
        transform_kind::OFFSET_NORM) != spec.transforms.end() ||
        std::find(spec.transforms.begin(), spec.transforms.end(),
        transform_kind::A_LOG) != spec.transforms.end();
    const bool backend_requires_f32 = ends_with(target, "ssm_conv1d.weight");
    switch (source.dtype) {
        case llama_safetensors_dtype::BF16:
            if (numerical_transform || backend_requires_f32 || source.shape.size() < 2) {
                return target == "token_embd.weight" ? GGML_TYPE_BF16 : GGML_TYPE_F32;
            }
            return GGML_TYPE_BF16;
        case llama_safetensors_dtype::F16:
            if (numerical_transform || backend_requires_f32 || source.shape.size() < 2) {
                return target == "token_embd.weight" ? GGML_TYPE_F16 : GGML_TYPE_F32;
            }
            return GGML_TYPE_F16;
        case llama_safetensors_dtype::F32: return GGML_TYPE_F32;
        case llama_safetensors_dtype::F8_E4M3: return GGML_TYPE_F8_E4M3;
        default: throw std::runtime_error("unsupported Qwen4 source dtype for '" + target + "'");
    }
}

std::vector<size_t> v_permutation(const qwen4_geometry & geometry, size_t head_dim) {
    const size_t ratio = geometry.n_value_heads / geometry.n_key_heads;
    std::vector<size_t> result(static_cast<size_t>(geometry.n_value_heads) * head_dim);
    for (size_t lane = 0; lane < ratio; ++lane) {
        for (size_t group = 0; group < geometry.n_key_heads; ++group) {
            const size_t destination = lane * geometry.n_key_heads + group;
            const size_t source = group * ratio + lane;
            for (size_t dim = 0; dim < head_dim; ++dim) {
                result[destination * head_dim + dim] = source * head_dim + dim;
            }
        }
    }
    return result;
}

std::vector<uint8_t> permute_rows(
        const std::vector<uint8_t> & source, size_t row_size, size_t prefix_rows,
        const std::vector<size_t> & permutation) {
    if (row_size == 0 || source.size() % row_size != 0 ||
        prefix_rows + permutation.size() > source.size() / row_size) {
        throw std::runtime_error("invalid Qwen4 row permutation");
    }
    std::vector<uint8_t> result = source;
    for (size_t row = 0; row < permutation.size(); ++row) {
        std::memcpy(result.data() + (prefix_rows + row) * row_size,
                    source.data() + (prefix_rows + permutation[row]) * row_size, row_size);
    }
    return result;
}

std::vector<uint8_t> transform_plain(
        transform_kind transform, const qwen4_geometry & geometry,
        const llama_safetensors_tensor & source_desc, std::vector<uint8_t> source) {
    if (transform == transform_kind::OFFSET_NORM) {
        std::vector<uint8_t> values;
        if (source_desc.dtype == llama_safetensors_dtype::BF16) {
            values = llama_safetensors_bf16_to_f32(source);
        } else if (source_desc.dtype == llama_safetensors_dtype::F16) {
            values = llama_safetensors_f16_to_f32(source);
        } else if (source_desc.dtype == llama_safetensors_dtype::F32) {
            values = std::move(source);
        } else {
            throw std::runtime_error("Qwen4 offset norm has an unsupported dtype");
        }
        for (size_t i = 0; i < values.size(); i += sizeof(float)) {
            float value;
            std::memcpy(&value, values.data() + i, sizeof(value));
            value += 1.0f;
            std::memcpy(values.data() + i, &value, sizeof(value));
        }
        return values;
    }
    if (transform == transform_kind::A_LOG) {
        std::vector<uint8_t> values;
        if (source_desc.dtype == llama_safetensors_dtype::BF16) {
            values = llama_safetensors_bf16_to_f32(source);
        } else if (source_desc.dtype == llama_safetensors_dtype::F16) {
            values = llama_safetensors_f16_to_f32(source);
        } else if (source_desc.dtype == llama_safetensors_dtype::F32) {
            values = std::move(source);
        } else {
            throw std::runtime_error("Qwen4 A_log has an unsupported dtype");
        }
        for (size_t i = 0; i < values.size(); i += sizeof(float)) {
            float value;
            std::memcpy(&value, values.data() + i, sizeof(value));
            value = -std::exp(value);
            std::memcpy(values.data() + i, &value, sizeof(value));
        }
        return values;
    }
    if (source_desc.shape.empty()) {
        return source;
    }
    const size_t element = llama_safetensors_dtype_size(source_desc.dtype);
    const size_t rows = source_desc.shape[0];
    if (rows == 0 || source.size() % rows != 0) {
        throw std::runtime_error("invalid Qwen4 transformed tensor shape");
    }
    const size_t row_size = source.size() / rows;
    if (transform == transform_kind::QKV_ROWS) {
        return permute_rows(source, row_size,
            2 * geometry.n_key_heads * geometry.key_head_dim,
            v_permutation(geometry, geometry.value_head_dim));
    }
    if (transform == transform_kind::V_ROWS || transform == transform_kind::HEAD_ROWS) {
        const size_t dim = transform == transform_kind::V_ROWS ? geometry.value_head_dim : 1;
        return permute_rows(source, row_size, 0, v_permutation(geometry, dim));
    }
    if (transform == transform_kind::CONV_ROWS) {
        return permute_rows(source, row_size,
            2 * geometry.n_key_heads * geometry.key_head_dim,
            v_permutation(geometry, geometry.value_head_dim));
    }
    if (transform == transform_kind::V_COLUMNS) {
        if (source_desc.shape.size() != 2) {
            throw std::runtime_error("Qwen4 output projection must be rank two");
        }
        const size_t output_rows = source_desc.shape[0];
        const size_t columns = source_desc.shape[1];
        const auto permutation = v_permutation(geometry, geometry.value_head_dim);
        if (permutation.size() != columns || source.size() != output_rows * columns * element) {
            throw std::runtime_error("invalid Qwen4 output projection permutation");
        }
        std::vector<uint8_t> result(source.size());
        for (size_t row = 0; row < output_rows; ++row) {
            for (size_t col = 0; col < columns; ++col) {
                std::memcpy(result.data() + (row * columns + col) * element,
                            source.data() + (row * columns + permutation[col]) * element, element);
            }
        }
        return result;
    }
    return source;
}

std::vector<uint8_t> slice_rows_2d(
        const llama_safetensors_tensor & source, size_t offset, size_t count,
        const std::vector<uint8_t> & data) {
    if (source.shape.size() != 2 || offset + count > source.shape[0] || source.shape[0] == 0) {
        throw std::runtime_error("invalid Qwen4 source-row slice");
    }
    const size_t row_size = data.size() / source.shape[0];
    return { data.begin() + offset * row_size, data.begin() + (offset + count) * row_size };
}

std::vector<uint8_t> slice_quant_rows_3d(
        ggml_type type, const std::vector<int64_t> & shape,
        size_t offset, size_t count, const std::vector<uint8_t> & data) {
    if (shape.size() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 ||
        offset + count > static_cast<size_t>(shape[1])) {
        throw std::runtime_error("invalid Qwen4 quantized expert-row slice");
    }
    const size_t row_size = ggml_row_size(type, shape[0]);
    const size_t rows = static_cast<size_t>(shape[1]);
    const size_t experts = static_cast<size_t>(shape[2]);
    if (data.size() != row_size * rows * experts) {
        throw std::runtime_error("Qwen4 quantized expert aggregate has the wrong size");
    }
    std::vector<uint8_t> result(row_size * count * experts);
    for (size_t expert = 0; expert < experts; ++expert) {
        std::memcpy(result.data() + expert * row_size * count,
                    data.data() + (expert * rows + offset) * row_size,
                    row_size * count);
    }
    return result;
}

std::vector<uint64_t> read_u64_vector(
        const llama_safetensors_registry & registry, const std::string & name) {
    const auto & tensor = require_tensor(registry, name);
    if (tensor.shape.size() != 1 ||
        (tensor.dtype != llama_safetensors_dtype::I64 && tensor.dtype != llama_safetensors_dtype::U64)) {
        throw std::runtime_error("Qwen4 PLE constant has an invalid contract: '" + name + "'");
    }
    const std::vector<uint8_t> bytes = registry.read(tensor);
    std::vector<uint64_t> result(tensor.shape[0]);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    if (tensor.dtype == llama_safetensors_dtype::I64) {
        for (uint64_t value : result) {
            if (static_cast<int64_t>(value) < 0) {
                throw std::runtime_error("Qwen4 PLE constant must be non-negative: '" + name + "'");
            }
        }
    }
    return result;
}

} // namespace

llama_safetensors_qwen4exp_importer::llama_safetensors_qwen4exp_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_json config,
        llama_safetensors_io_mode io_mode) :
    model_dir_(model_dir), config_(std::move(config)) {
    if (!probe(config_)) {
        throw std::runtime_error("native Qwen4 importer requires qwen4_exp text configuration");
    }
    const auto & text = text_config(config_);
    model_prefix_ = config_.contains("text_config") ? "model.language_model" : "model";
    n_layer_ = text.value("num_hidden_layers", 0U);
    n_mtp_ = text.value("mtp_num_hidden_layers", 0U);
    n_key_heads_ = text.value("linear_num_key_heads", 0U);
    n_value_heads_ = text.value("linear_num_value_heads", 0U);
    key_head_dim_ = text.value("linear_key_head_dim", 0U);
    value_head_dim_ = text.value("linear_value_head_dim", 0U);
    indexer_n_heads_ = text.value("indexer_n_heads", 0U);
    indexer_head_dim_ = text.value("indexer_head_dim", 0U);
    full_attention_interval_ = text.value("full_attention_interval", 4U);
    ple_shards_ = text.value("split_ngram_parts", 0U);
    const auto ple_layers = text.value("ple_layer_ids", std::vector<uint32_t>{});
    if (n_layer_ == 0 || n_mtp_ > 1 || n_key_heads_ == 0 || n_value_heads_ == 0 ||
        n_value_heads_ % n_key_heads_ != 0 || key_head_dim_ == 0 || value_head_dim_ == 0 ||
        indexer_n_heads_ == 0 || indexer_head_dim_ == 0 ||
        full_attention_interval_ == 0 || ple_layers.size() > 1 || (ple_layers.empty() != (ple_shards_ == 0))) {
        throw std::runtime_error("native Qwen4 importer does not support this tensor geometry");
    }
    if (!ple_layers.empty()) {
        if (ple_layers[0] == 0 || ple_layers[0] > n_layer_) {
            throw std::runtime_error("Qwen4 PLE layer id is out of range");
        }
        ple_layer_ = ple_layers[0] - 1;
    }
    generation_ = std::filesystem::is_regular_file(model_dir_ / "generation_config.json") ?
        llama_safetensors_read_json(model_dir_ / "generation_config.json") : llama_safetensors_json::object();
    tokenizer_ = llama_safetensors_read_json(model_dir_ / "tokenizer.json");
    chat_template_ = llama_safetensors_read_optional_text(model_dir_ / "chat_template.jinja");
    registry_ = llama_safetensors_registry::load(model_dir_, io_mode);
    quant_ = std::make_unique<llama_safetensors_quant_adapters>(config_, registry_);
    if (n_mtp_ != 0 && std::none_of(registry_.tensors().begin(), registry_.tensors().end(),
            [](const llama_safetensors_tensor & tensor) { return tensor.name.rfind("mtp.", 0) == 0; })) {
        n_mtp_ = 0;
    }
}

bool llama_safetensors_qwen4exp_importer::probe(const llama_safetensors_json & config) {
    const std::string model_type = config.value("model_type", std::string());
    return model_type == "qwen4_exp" || model_type == "qwen4_exp_text";
}

gguf_context * llama_safetensors_qwen4exp_importer::build_metadata() const {
    const auto & text = text_config(config_);
    const auto rope = llama_safetensors_parse_rope(text.at("rope_parameters"), { 11, 11, 10, 0 }, 0.25f);
    llama_safetensors_metadata_sink sink;
    const std::string arch = "qwen4exp";
    sink.set_string("general.architecture", arch);
    sink.set_string("general.type", "model");
    sink.set_string("general.name", model_dir_.filename().empty() ? "Qwen4 Safetensors" : model_dir_.filename().string());
    sink.set_u32("general.file_type", quant_->file_type());
    sink.set_u32("general.quantization_version", 2);
    llama_safetensors_emit_sampling_defaults(sink, generation_);
    sink.set_u32(arch + ".block_count", n_layer_ + n_mtp_);
    sink.set_u32(arch + ".context_length", text.at("max_position_embeddings").get<uint32_t>());
    sink.set_u32(arch + ".embedding_length", text.at("hidden_size").get<uint32_t>());
    sink.set_u32(arch + ".feed_forward_length", text.at("shared_expert_intermediate_size").get<uint32_t>());
    sink.set_u32(arch + ".expert_feed_forward_length", text.at("moe_intermediate_size").get<uint32_t>());
    sink.set_u32(arch + ".expert_shared_feed_forward_length", text.at("shared_expert_intermediate_size").get<uint32_t>());
    sink.set_u32(arch + ".expert_count", text.at("num_experts").get<uint32_t>());
    sink.set_u32(arch + ".expert_used_count", text.at("num_experts_per_tok").get<uint32_t>());
    sink.set_u32(arch + ".attention.head_count", text.at("num_attention_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.head_count_kv", text.at("num_key_value_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.key_length", text.at("head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.value_length", text.at("head_dim").get<uint32_t>());
    sink.set_f32(arch + ".attention.layer_norm_rms_epsilon", text.at("rms_norm_eps").get<float>());
    sink.set_f32(arch + ".rope.freq_base", rope.theta);
    sink.set_i32_array(arch + ".rope.dimension_sections", rope.mrope_sections.data(), rope.mrope_sections.size());
    sink.set_u32(arch + ".rope.dimension_count", static_cast<uint32_t>(text.at("head_dim").get<float>() * rope.partial_rotary_factor));
    sink.set_u32(arch + ".ssm.conv_kernel", text.at("linear_conv_kernel_dim").get<uint32_t>());
    sink.set_u32(arch + ".ssm.state_size", text.at("linear_key_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".ssm.group_count", text.at("linear_num_key_heads").get<uint32_t>());
    sink.set_u32(arch + ".ssm.time_step_rank", text.at("linear_num_value_heads").get<uint32_t>());
    sink.set_u32(arch + ".ssm.inner_size", text.at("linear_num_value_heads").get<uint32_t>() * text.at("linear_value_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".full_attention_interval", full_attention_interval_);
    sink.set_u32(arch + ".hyper_connection.count", text.at("hc_count").get<uint32_t>());
    sink.set_u32(arch + ".hyper_connection.low_rank", text.at("hc_lowrank").get<uint32_t>());
    if (n_mtp_ != 0) sink.set_u32(arch + ".nextn_predict_layers", n_mtp_);
    sink.set_u32(arch + ".attention.indexer.head_count", text.at("indexer_n_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.key_length", text.at("indexer_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.top_k", text.at("indexer_budget").get<uint32_t>());
    std::vector<uint32_t> ratios(n_layer_ + n_mtp_, 0);
    const auto & layer_types = text.at("layer_types");
    for (uint32_t i = 0; i < n_layer_; ++i) {
        const std::string layer_type = layer_types.at(i).get<std::string>();
        if (layer_type == "full_attention" || layer_type == "qwen_sparse_attention") {
            ratios[i] = text.at("indexer_compress_ratio").get<uint32_t>();
        } else if (layer_type != "linear_attention") {
            throw std::runtime_error("unsupported Qwen4 layer type '" + layer_type + "'");
        }
    }
    sink.set_u32_array(arch + ".attention.compress_ratios", ratios.data(), ratios.size());
    if (ple_layer_ != UINT32_MAX) {
        sink.set_u32_array(arch + ".ple.layers", &ple_layer_, 1);
        sink.set_u32(arch + ".ple.ngram_size", text.at("ngram_size").get<uint32_t>());
        sink.set_u32(arch + ".ple.heads_per_ngram", text.at("heads_per_ngram").get<uint32_t>());
        sink.set_u32(arch + ".ple.conv_kernel", text.at("ple_conv_kernel_size").get<uint32_t>());
        const uint32_t eos = llama_safetensors_first_token_id(
            generation_.contains("eos_token_id") ? generation_.at("eos_token_id") : text.at("eos_token_id"), "eos_token_id");
        sink.set_u32(arch + ".ple.eos_token_id", eos);
        if (config_.contains("image_token_id")) sink.set_u32(arch + ".ple.image_token_id", config_.at("image_token_id").get<uint32_t>());
        const std::string ple = model_prefix_ + ".layers." + std::to_string(ple_layer_) + ".ple.ple_embedding.";
        const std::string first_shard_name = ple + "ngram_embedding.shard_0.weight";
        const auto & first_shard = require_tensor(registry_, first_shard_name);
        if (first_shard.shape.size() != 2 || first_shard.shape[1] == 0 || first_shard.shape[1] > UINT32_MAX) {
            throw std::runtime_error("Qwen4 PLE shard has an invalid row dimension");
        }
        sink.set_u32(arch + ".embedding_length_per_layer_input", static_cast<uint32_t>(first_shard.shape[1]));
        const auto multipliers = read_u64_vector(registry_, ple + "layer_multipliers");
        const auto offsets = read_u64_vector(registry_, ple + "ngram_heads_offsets");
        const auto vocab_sizes = read_u64_vector(registry_, ple + "ngram_heads_vocab_sizes");
        sink.set_u64_array(arch + ".ple.layer_multipliers", multipliers.data(), multipliers.size());
        sink.set_u64_array(arch + ".ple.head_offsets", offsets.data(), offsets.size());
        sink.set_u64_array(arch + ".ple.head_vocab_sizes", vocab_sizes.data(), vocab_sizes.size());
    }
    const uint32_t eos = llama_safetensors_first_token_id(
        generation_.contains("eos_token_id") ? generation_.at("eos_token_id") : text.at("eos_token_id"), "eos_token_id");
    const uint32_t bos = generation_.contains("bos_token_id") && !generation_.at("bos_token_id").is_null() ?
        llama_safetensors_first_token_id(generation_.at("bos_token_id"), "bos_token_id") : eos;
    llama_safetensors_emit_bpe_tokenizer(sink, tokenizer_, {
        "qwen35", text.at("vocab_size").get<uint32_t>(), bos, eos,
        std::string("<|vision_pad|>"), true, { "<tts_" }, false,
    }, chat_template_);
    return sink.release();
}

bool llama_safetensors_qwen4exp_importer::describe(
        const std::string & target, ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const {
    const auto & text = text_config(config_);
    const qwen4_geometry geometry { model_prefix_, n_layer_, n_mtp_, n_key_heads_, n_value_heads_, key_head_dim_, value_head_dim_,
        indexer_n_heads_, indexer_head_dim_, full_attention_interval_, text.at("num_experts").get<uint32_t>(), text.at("moe_intermediate_size").get<uint32_t>(),
        ple_layer_, ple_shards_ };
    source_spec spec;
    try { spec = map_target(registry_, *quant_, geometry, target); }
    catch (const unsupported_target &) { return false; }
    std::vector<int64_t> shape;
    if (spec.ple_table) {
        if (ple_layer_ == UINT32_MAX || ple_shards_ == 0) return false;
        uint64_t rows = 0, dim = 0;
        llama_safetensors_dtype dtype = llama_safetensors_dtype::F32;
        const std::string prefix = model_prefix_ + ".layers." + std::to_string(ple_layer_) +
            ".ple.ple_embedding.ngram_embedding.shard_";
        for (uint32_t i = 0; i < ple_shards_; ++i) {
            const std::string shard_name = prefix + std::to_string(i) + ".weight";
            const auto & shard = require_tensor(registry_, shard_name);
            if (shard.shape.size() != 2 || (i != 0 && (shard.shape[1] != dim || shard.dtype != dtype)) ||
                (shard.dtype != llama_safetensors_dtype::BF16 && shard.dtype != llama_safetensors_dtype::F16 &&
                 shard.dtype != llama_safetensors_dtype::F32 && shard.dtype != llama_safetensors_dtype::F8_E4M3) ||
                rows > std::numeric_limits<uint64_t>::max() - shard.shape[0]) {
                throw std::runtime_error("invalid Qwen4 PLE shard contract");
            }
            dtype = shard.dtype; dim = shard.shape[1]; rows += shard.shape[0];
        }
        if (dim > INT64_MAX || rows > INT64_MAX) throw std::runtime_error("Qwen4 PLE table exceeds runtime limits");
        type = dtype == llama_safetensors_dtype::F8_E4M3 ? GGML_TYPE_F8_E4M3 : GGML_TYPE_F32;
        shape = { static_cast<int64_t>(dim), static_cast<int64_t>(rows) };
    } else if (spec.ple_scale) {
        if (ple_layer_ == UINT32_MAX || ple_shards_ == 0) return false;
        const std::string name = model_prefix_ + ".layers." + std::to_string(ple_layer_) +
            ".ple.ple_embedding.ngram_embedding.weight_scale";
        const auto & scale = require_tensor(registry_, name);
        if (scale.dtype != llama_safetensors_dtype::BF16 || scale.shape != std::vector<uint64_t>{1}) {
            throw std::runtime_error("invalid Qwen4 PLE scale contract");
        }
        type = GGML_TYPE_F32;
        shape = { 1 };
    } else if (!spec.stack_quant_weights.empty()) {
        const auto & first = spec.stack_quant_weights.front();
        if (first.target_shape.size() != 2 || first.target_shape[0] % 128 != 0 ||
                first.target_shape[1] % 128 != 0 ||
                spec.stack_quant_scales.size() != spec.stack_quant_weights.size()) {
            throw std::runtime_error("invalid split-expert FP8 target geometry");
        }
        for (size_t i = 0; i < spec.stack_quant_weights.size(); ++i) {
            if (spec.stack_quant_weights[i].target_shape != first.target_shape ||
                    spec.stack_quant_scales[i].target_shape !=
                        std::vector<int64_t>{ first.target_shape[1] / 128, first.target_shape[0] / 128 }) {
                throw std::runtime_error("incompatible split-expert FP8 source tensors");
            }
        }
        type = GGML_TYPE_Q8_0_G128;
        shape = first.target_shape;
        shape.push_back(static_cast<int64_t>(spec.stack_quant_weights.size()));
    } else if (!spec.stack_sources.empty()) {
        const auto & first = require_tensor(registry_, spec.stack_sources.front());
        if (first.shape.size() != 2) throw std::runtime_error("Qwen4 expert source must be rank two");
        for (const auto & name : spec.stack_sources) {
            const auto & item = require_tensor(registry_, name);
            if (item.dtype != first.dtype || item.shape != first.shape) {
                throw std::runtime_error("incompatible Qwen4 per-expert sources");
            }
        }
        type = plain_type(first, spec, target);
        shape = reverse_shape(first);
        shape.push_back(static_cast<int64_t>(spec.stack_sources.size()));
    } else if (!spec.concat_sources.empty()) {
        const auto & first = require_tensor(registry_, spec.concat_sources.front());
        if (first.shape.size() != 2) throw std::runtime_error("Qwen4 concatenated source must be rank two");
        uint64_t cols = 0;
        for (const auto & name : spec.concat_sources) {
            const auto & item = require_tensor(registry_, name);
            if (item.dtype != first.dtype || item.shape.size() != 2 || item.shape[0] != first.shape[0] ||
                cols > std::numeric_limits<uint64_t>::max() - item.shape[1]) throw std::runtime_error("incompatible Qwen4 concatenated sources");
            cols += item.shape[1];
        }
        type = plain_type(first, spec, target);
        shape = { static_cast<int64_t>(cols), static_cast<int64_t>(first.shape[0]) };
    } else {
        if (spec.source.empty() || registry_.find(spec.source) == nullptr) return false;
        const auto & source = require_tensor(registry_, spec.source);
        type = spec.quant ? spec.quant->target_type : plain_type(source, spec, target);
        shape = spec.quant ? spec.quant->target_shape : reverse_shape(source);
        if (spec.row_count != 0) {
            if (spec.quant) {
                if (shape.size() < 2) throw std::runtime_error("Qwen4 quantized slice has invalid rank");
                shape[1] = static_cast<int64_t>(spec.row_count);
            } else {
                if (shape.size() != 2) throw std::runtime_error("Qwen4 plain slice has invalid rank");
                shape[1] = static_cast<int64_t>(spec.row_count);
            }
        }
        if (spec.scale_broadcast != 0) shape = { static_cast<int64_t>(spec.scale_broadcast) };
    }
    if (shape.size() > GGML_MAX_DIMS) throw std::runtime_error("Qwen4 target rank exceeds GGML_MAX_DIMS");
    ne.fill(1); std::copy(shape.begin(), shape.end(), ne.begin()); return true;
}

size_t llama_safetensors_qwen4exp_importer::tensor_capacity_hint() const {
    return std::max<size_t>(512, registry_.tensors().size() * 2);
}

void llama_safetensors_qwen4exp_importer::bind(const std::string & target) const {
    const auto & text = text_config(config_);
    const qwen4_geometry geometry { model_prefix_, n_layer_, n_mtp_, n_key_heads_, n_value_heads_, key_head_dim_, value_head_dim_,
        indexer_n_heads_, indexer_head_dim_, full_attention_interval_, text.at("num_experts").get<uint32_t>(), text.at("moe_intermediate_size").get<uint32_t>(), ple_layer_, ple_shards_ };
    const source_spec spec = map_target(registry_, *quant_, geometry, target);
    if (spec.quant) quant_->consume(*spec.quant);
    for (const auto & binding : spec.stack_quant_weights) quant_->consume(binding);
    for (const auto & binding : spec.stack_quant_scales)  quant_->consume(binding);
}

bool llama_safetensors_qwen4exp_importer::load(
        const std::string & target, ggml_tensor * destination, bool check_tensor) const {
    const auto & text = text_config(config_);
    const qwen4_geometry geometry { model_prefix_, n_layer_, n_mtp_, n_key_heads_, n_value_heads_, key_head_dim_, value_head_dim_,
        indexer_n_heads_, indexer_head_dim_, full_attention_interval_, text.at("num_experts").get<uint32_t>(), text.at("moe_intermediate_size").get<uint32_t>(), ple_layer_, ple_shards_ };
    const source_spec spec = map_target(registry_, *quant_, geometry, target);
    if (spec.ple_table) {
        if (destination->type != GGML_TYPE_F32 && destination->type != GGML_TYPE_F8_E4M3) {
            throw std::runtime_error("Qwen4 PLE destination has an unsupported type");
        }
        size_t offset = 0;
        const std::string prefix = model_prefix_ + ".layers." + std::to_string(ple_layer_) +
            ".ple.ple_embedding.ngram_embedding.shard_";
        for (uint32_t i = 0; i < ple_shards_; ++i) {
            const std::string shard_name = prefix + std::to_string(i) + ".weight";
            const auto & desc = require_tensor(registry_, shard_name);
            std::vector<uint8_t> owned;
            const uint8_t * data = nullptr;
            size_t size = static_cast<size_t>(desc.size);
            if (destination->type == GGML_TYPE_F32) {
                owned = registry_.read(desc);
                if (desc.dtype == llama_safetensors_dtype::BF16) owned = llama_safetensors_bf16_to_f32(owned);
                else if (desc.dtype == llama_safetensors_dtype::F16) owned = llama_safetensors_f16_to_f32(owned);
                data = owned.data();
                size = owned.size();
            } else if (desc.dtype != llama_safetensors_dtype::F8_E4M3) {
                throw std::runtime_error("Qwen4 PLE shard type does not match its destination");
            } else if ((data = registry_.data(desc)) == nullptr) {
                owned = registry_.read(desc);
                data = owned.data();
            }
            if (check_tensor && !ggml_validate_row_data(destination->type, data, size))
                throw std::runtime_error("Qwen4 PLE shard contains invalid data");
            if (offset > ggml_nbytes(destination) || size > ggml_nbytes(destination) - offset)
                throw std::runtime_error("Qwen4 PLE shard exceeds destination");
            ggml_backend_tensor_set(destination, data, offset, size);
            offset += size;
        }
        if (offset != ggml_nbytes(destination)) throw std::runtime_error("Qwen4 PLE upload is incomplete");
        return true;
    }
    if (spec.ple_scale) {
        return false;
    }
    if (!spec.transforms.empty() || spec.row_count != 0 || spec.scale_broadcast != 0 ||
        !spec.stack_quant_weights.empty() ||
        !spec.concat_sources.empty() || !spec.stack_sources.empty()) return false;
    return llama_safetensors_load_tensor_direct(registry_, { spec.source, spec.quant }, destination, check_tensor);
}

std::vector<uint8_t> llama_safetensors_qwen4exp_importer::materialize(
        const std::string & target, ggml_type target_type, size_t target_size) const {
    const auto & text = text_config(config_);
    const qwen4_geometry geometry { model_prefix_, n_layer_, n_mtp_, n_key_heads_, n_value_heads_, key_head_dim_, value_head_dim_,
        indexer_n_heads_, indexer_head_dim_, full_attention_interval_, text.at("num_experts").get<uint32_t>(), text.at("moe_intermediate_size").get<uint32_t>(), ple_layer_, ple_shards_ };
    try {
        const source_spec spec = map_target(registry_, *quant_, geometry, target);
        if (spec.ple_scale) {
            const std::string name = model_prefix_ + ".layers." + std::to_string(ple_layer_) +
                ".ple.ple_embedding.ngram_embedding.weight_scale";
            std::vector<uint8_t> result = llama_safetensors_bf16_to_f32(registry_.read(require_tensor(registry_, name)));
            if (target_type != GGML_TYPE_F32 || result.size() != target_size) {
                throw std::runtime_error("Qwen4 PLE scale materialization has the wrong type or size");
            }
            return result;
        }
        if (spec.ple_table) throw std::runtime_error("PLE table must use bounded direct upload");
        std::vector<uint8_t> result;
        const llama_safetensors_tensor * source_desc = nullptr;
        if (!spec.stack_quant_weights.empty()) {
            if (target_type != GGML_TYPE_Q8_0_G128 || spec.stack_quant_weights.size() != spec.stack_quant_scales.size()) {
                throw std::runtime_error("split-expert FP8 repack has an invalid target");
            }
            const size_t n_expert = spec.stack_quant_weights.size();
            const size_t n_cols = static_cast<size_t>(spec.stack_quant_weights.front().target_shape[0]);
            const size_t n_rows = static_cast<size_t>(spec.stack_quant_weights.front().target_shape[1]);
            const size_t n_col_blocks = n_cols / 128;
            const size_t n_row_blocks = n_rows / 128;
            const size_t source_row_size = n_cols;
            const size_t target_row_size = ggml_row_size(target_type, n_cols);
            std::vector<uint8_t> weights(n_expert * n_rows * source_row_size);
            std::vector<float> scales(n_expert * n_row_blocks * n_col_blocks);
            for (size_t expert = 0; expert < n_expert; ++expert) {
                const auto weight = quant_->read(spec.stack_quant_weights[expert]);
                if (weight.size() != n_rows * source_row_size) {
                    throw std::runtime_error("split-expert FP8 weight has the wrong size");
                }
                std::memcpy(weights.data() + expert * n_rows * source_row_size,
                            weight.data(), weight.size());
                const auto scale = quant_->finalize(
                    spec.stack_quant_scales[expert], quant_->read(spec.stack_quant_scales[expert]));
                if (scale.size() != n_row_blocks * n_col_blocks * sizeof(float)) {
                    throw std::runtime_error("split-expert FP8 scale has the wrong size");
                }
                for (size_t i = 0; i < n_row_blocks * n_col_blocks; ++i) {
                    float value;
                    std::memcpy(&value, scale.data() + i * sizeof(value), sizeof(value));
                    if (!(value > 0.0f) || !std::isfinite(value)) {
                        throw std::runtime_error("split-expert FP8 scale must be finite and positive");
                    }
                    scales[expert * n_row_blocks * n_col_blocks + i] = value;
                }
            }
            result.resize(n_expert * n_rows * target_row_size);
            const auto * source_traits = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
            const auto * target_traits = ggml_get_type_traits(target_type);
            if (!source_traits->to_float || !target_traits->from_float_ref || result.size() != target_size) {
                throw std::runtime_error("split-expert FP8 repack is unavailable");
            }
            const int64_t total_rows = static_cast<int64_t>(n_expert * n_rows);
            const size_t n_threads = std::max<size_t>(1, std::min<size_t>(
                std::thread::hardware_concurrency(), static_cast<size_t>(total_rows)));
            std::vector<std::thread> workers;
            workers.reserve(n_threads);
            for (size_t thread = 0; thread < n_threads; ++thread) {
                const int64_t row_begin = total_rows * static_cast<int64_t>(thread) /
                    static_cast<int64_t>(n_threads);
                const int64_t row_end = total_rows * static_cast<int64_t>(thread + 1) /
                    static_cast<int64_t>(n_threads);
                workers.emplace_back([&, row_begin, row_end] {
                std::vector<float> row(n_cols);
                for (int64_t flat_row = row_begin; flat_row < row_end; ++flat_row) {
                    const size_t expert = static_cast<size_t>(flat_row) / n_rows;
                    const size_t output_row = static_cast<size_t>(flat_row) % n_rows;
                    source_traits->to_float(
                        weights.data() + static_cast<size_t>(flat_row) * source_row_size,
                        row.data(), static_cast<int64_t>(n_cols));
                    const float * expert_scales =
                        scales.data() + expert * n_row_blocks * n_col_blocks;
                    for (size_t block = 0; block < n_col_blocks; ++block) {
                        const float scale = expert_scales[block * n_row_blocks + output_row / 128];
                        for (size_t col = block * 128; col < (block + 1) * 128; ++col) {
                            row[col] *= scale;
                        }
                    }
                    target_traits->from_float_ref(
                        row.data(), result.data() + static_cast<size_t>(flat_row) * target_row_size,
                        static_cast<int64_t>(n_cols));
                }
                });
            }
            for (auto & worker : workers) worker.join();
            return result;
        } else if (!spec.stack_sources.empty()) {
            source_desc = &require_tensor(registry_, spec.stack_sources.front());
            result.reserve(target_size);
            for (const auto & name : spec.stack_sources) {
                const auto & desc = require_tensor(registry_, name);
                auto bytes = registry_.read(desc);
                result.insert(result.end(), bytes.begin(), bytes.end());
            }
        } else if (!spec.concat_sources.empty()) {
            const auto & first = require_tensor(registry_, spec.concat_sources.front());
            source_desc = &first;
            std::vector<std::vector<uint8_t>> sources;
            size_t rows = first.shape[0], total_row = 0;
            for (const auto & name : spec.concat_sources) {
                const auto & desc = require_tensor(registry_, name);
                auto bytes = registry_.read(desc);
                total_row += bytes.size() / rows;
                sources.push_back(std::move(bytes));
            }
            result.resize(rows * total_row);
            for (size_t row = 0; row < rows; ++row) {
                size_t cursor = row * total_row;
                for (const auto & bytes : sources) {
                    const size_t row_bytes = bytes.size() / rows;
                    std::memcpy(result.data() + cursor, bytes.data() + row * row_bytes, row_bytes);
                    cursor += row_bytes;
                }
            }
        } else {
            source_desc = &require_tensor(registry_, spec.source);
            result = spec.quant ? quant_->read(*spec.quant) : registry_.read(*source_desc);
            if (spec.quant) result = quant_->finalize(*spec.quant, std::move(result));
        }
        if (spec.row_count != 0) {
            result = spec.quant ? slice_quant_rows_3d(spec.quant->target_type, spec.quant->target_shape,
                spec.row_offset, spec.row_count, result) :
                slice_rows_2d(*source_desc, spec.row_offset, spec.row_count, result);
        }
        if (spec.scale_broadcast != 0) {
            if (result.size() != sizeof(float) || target_type != GGML_TYPE_F32) throw std::runtime_error("Qwen4 expert scale is not scalar F32");
            const auto scalar = result;
            result.resize(spec.scale_broadcast * sizeof(float));
            for (size_t i = 0; i < spec.scale_broadcast; ++i) std::memcpy(result.data() + i * sizeof(float), scalar.data(), sizeof(float));
        }
        for (transform_kind transform : spec.transforms) {
            if (spec.quant) throw std::runtime_error("Qwen4 transformed quantized projection is not supported");
            result = transform_plain(transform, geometry, *source_desc, std::move(result));
        }
        const bool numerically_transformed = std::find(spec.transforms.begin(), spec.transforms.end(),
            transform_kind::OFFSET_NORM) != spec.transforms.end() ||
            std::find(spec.transforms.begin(), spec.transforms.end(),
            transform_kind::A_LOG) != spec.transforms.end();
        if (!spec.quant && !numerically_transformed && spec.scale_broadcast == 0) {
            if (target_type == GGML_TYPE_F32 && source_desc->dtype == llama_safetensors_dtype::BF16)
                result = llama_safetensors_bf16_to_f32(result);
            else if (target_type == GGML_TYPE_F32 && source_desc->dtype == llama_safetensors_dtype::F16)
                result = llama_safetensors_f16_to_f32(result);
        }
        if (result.size() != target_size) throw std::runtime_error("produced " + std::to_string(result.size()) + " bytes, expected " + std::to_string(target_size));
        return result;
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to materialize Qwen4 tensor '" + target + "': " + error.what());
    }
}

void llama_safetensors_qwen4exp_importer::validate_complete() const {
    quant_->validate_complete();
}
