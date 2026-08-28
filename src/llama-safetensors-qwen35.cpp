#include "llama-safetensors-qwen35.h"
#include "llama.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string_view>

namespace {

using json = nlohmann::json;

enum class transform_kind {
    NONE,
    QWEN_LINEAR,
};

struct source_spec {
    std::string    name;
    transform_kind transform = transform_kind::NONE;
    std::optional<llama_safetensors_quant_binding> quant;
};

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

bool is_qwen_offset_norm(std::string_view name) {
    return (ends_with(name, "norm.weight") && !ends_with(name, "linear_attn.norm.weight")) ||
           name.find("pre_fc_norm_embedding.weight") != std::string_view::npos ||
           name.find("pre_fc_norm_hidden.weight") != std::string_view::npos;
}

json read_json(const std::filesystem::path & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open JSON file '" + path.string() + "'");
    }
    try {
        return json::parse(input);
    } catch (const json::exception & error) {
        throw std::runtime_error("invalid JSON in '" + path.string() + "': " + error.what());
    }
}

std::string read_text(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open text file '" + path.string() + "'");
    }
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

const llama_safetensors_tensor & require_tensor(const llama_safetensors_registry & registry, const std::string & name) {
    const llama_safetensors_tensor * tensor = registry.find(name);
    if (tensor == nullptr) {
        throw std::runtime_error("safetensors tensor not found for import: '" + name + "'");
    }
    return *tensor;
}

size_t dtype_size(llama_safetensors_dtype dtype) {
    switch (dtype) {
        case llama_safetensors_dtype::BOOL:
        case llama_safetensors_dtype::U8:
        case llama_safetensors_dtype::I8:
        case llama_safetensors_dtype::F8_E4M3:
        case llama_safetensors_dtype::F8_E5M2:
            return 1;
        case llama_safetensors_dtype::U16:
        case llama_safetensors_dtype::I16:
        case llama_safetensors_dtype::F16:
        case llama_safetensors_dtype::BF16:
            return 2;
        case llama_safetensors_dtype::U32:
        case llama_safetensors_dtype::I32:
        case llama_safetensors_dtype::F32:
            return 4;
        case llama_safetensors_dtype::U64:
        case llama_safetensors_dtype::I64:
        case llama_safetensors_dtype::F64:
            return 8;
    }
    throw std::runtime_error("unknown safetensors dtype");
}

std::string layer_source_prefix(int layer) {
    if (layer < 64) {
        return "model.language_model.layers." + std::to_string(layer) + ".";
    }
    if (layer == 64) {
        return "mtp.layers.0.";
    }
    throw std::runtime_error("native Qwen3.5 importer does not support layer " + std::to_string(layer));
}

source_spec quantized_or_plain(
                    const llama_safetensors_quant_adapters & quant,
                    const std::string & module,
                    llama_safetensors_quant_role role,
                    transform_kind transform,
                    const std::string & plain_name) {
    if (auto binding = quant.bind(module, role)) {
        return { binding->primary, transform, std::move(binding) };
    }
    return { plain_name, transform, std::nullopt };
}

source_spec map_mlp(const llama_safetensors_quant_adapters & quant,
                    const std::string &                prefix,
                    std::string_view                   target_suffix,
                    std::string_view                   target_proj,
                    std::string_view                   source_proj) {
    const std::string source_base = prefix + "mlp." + std::string(source_proj);
    const std::string target_base = "ffn_" + std::string(target_proj);
    const std::string rest(target_suffix.substr(target_base.size()));

    if (rest == ".weight") {
        return quantized_or_plain(
            quant, source_base, llama_safetensors_quant_role::WEIGHT,
            transform_kind::NONE, source_base + ".weight");
    }
    if (rest == ".scale") {
        return quantized_or_plain(
            quant, source_base, llama_safetensors_quant_role::WEIGHT_SCALE,
            transform_kind::QWEN_LINEAR, source_base + ".weight_scale");
    }
    if (rest == ".input_scale") {
        return quantized_or_plain(
            quant, source_base, llama_safetensors_quant_role::INPUT_SCALE,
            transform_kind::NONE, source_base + ".input_global_scale");
    }
    throw std::runtime_error("unsupported MLP target suffix '" + std::string(target_suffix) + "'");
}

source_spec map_target(
        const llama_safetensors_quant_adapters & quant,
        const std::string & target_name) {
    if (target_name == "token_embd.weight") {
        return { "model.language_model.embed_tokens.weight", transform_kind::NONE, std::nullopt };
    }
    if (target_name == "output_norm.weight") {
        return { "model.language_model.norm.weight", transform_kind::NONE, std::nullopt };
    }
    if (target_name == "output.weight") {
        return quantized_or_plain(
            quant, "lm_head", llama_safetensors_quant_role::WEIGHT,
            transform_kind::QWEN_LINEAR, "lm_head.weight");
    }
    if (target_name == "output.scale") {
        return quantized_or_plain(
            quant, "lm_head", llama_safetensors_quant_role::WEIGHT_SCALE,
            transform_kind::QWEN_LINEAR, "lm_head.weight_scale");
    }
    if (target_name == "output.input_scale") {
        return quantized_or_plain(
            quant, "lm_head", llama_safetensors_quant_role::INPUT_SCALE,
            transform_kind::NONE, "lm_head.input_global_scale");
    }

    static const std::regex layer_pattern(R"(^blk\.([0-9]+)\.(.+)$)");
    std::smatch             match;
    if (!std::regex_match(target_name, match, layer_pattern)) {
        throw std::runtime_error("unsupported Qwen3.5 target tensor '" + target_name + "'");
    }
    const int         layer  = std::stoi(match[1].str());
    const std::string suffix = match[2].str();
    const std::string prefix = layer_source_prefix(layer);

    if (suffix.rfind("ffn_down", 0) == 0) {
        return map_mlp(quant, prefix, suffix, "down", "down_proj");
    }
    if (suffix.rfind("ffn_gate", 0) == 0) {
        return map_mlp(quant, prefix, suffix, "gate", "gate_proj");
    }
    if (suffix.rfind("ffn_up", 0) == 0) {
        return map_mlp(quant, prefix, suffix, "up", "up_proj");
    }

    const std::array<std::pair<std::string_view, std::string_view>, 18> ordinary = {
        {
         { "attn_norm.weight", "input_layernorm.weight" },
         { "post_attention_norm.weight", "post_attention_layernorm.weight" },
         { "attn_q.weight", "self_attn.q_proj.weight" },
         { "attn_q.scale", "self_attn.q_proj.weight_scale" },
         { "attn_k.weight", "self_attn.k_proj.weight" },
         { "attn_k.scale", "self_attn.k_proj.weight_scale" },
         { "attn_v.weight", "self_attn.v_proj.weight" },
         { "attn_v.scale", "self_attn.v_proj.weight_scale" },
         { "attn_output.weight", "self_attn.o_proj.weight" },
         { "attn_output.scale", "self_attn.o_proj.weight_scale" },
         { "attn_q_norm.weight", "self_attn.q_norm.weight" },
         { "attn_k_norm.weight", "self_attn.k_norm.weight" },
         { "attn_gate.weight", "linear_attn.in_proj_z.weight" },
         { "attn_gate.scale", "linear_attn.in_proj_z.weight_scale" },
         { "attn_qkv.weight", "linear_attn.in_proj_qkv.weight" },
         { "attn_qkv.scale", "linear_attn.in_proj_qkv.weight_scale" },
         { "ssm_norm.weight", "linear_attn.norm.weight" },
         { "ssm_conv1d.weight", "linear_attn.conv1d.weight" },
         }
    };
    for (const auto & [target, source] : ordinary) {
        if (suffix == target) {
            const std::string source_name = prefix + std::string(source);
            const transform_kind transform = source_name.find("linear_attn.") != std::string::npos ?
                transform_kind::QWEN_LINEAR : transform_kind::NONE;
            if (ends_with(target, ".scale")) {
                const std::string module = source_name.substr(
                    0, source_name.size() - std::string_view(".weight_scale").size());
                return quantized_or_plain(
                    quant, module, llama_safetensors_quant_role::WEIGHT_SCALE,
                    transform, source_name);
            }
            if (ends_with(target, ".weight")) {
                const std::string module = source_name.substr(
                    0, source_name.size() - std::string_view(".weight").size());
                return quantized_or_plain(
                    quant, module, llama_safetensors_quant_role::WEIGHT,
                    transform, source_name);
            }
            return { source_name, transform, std::nullopt };
        }
    }

    const std::array<std::pair<std::string_view, std::string_view>, 6> linear = {
        {
         { "ssm_alpha.weight", "linear_attn.in_proj_a.weight" },
         { "ssm_beta.weight", "linear_attn.in_proj_b.weight" },
         { "ssm_a", "linear_attn.A_log" },
         { "ssm_dt.bias", "linear_attn.dt_bias" },
         { "ssm_out.weight", "linear_attn.out_proj.weight" },
         { "ssm_out.scale", "linear_attn.out_proj.weight_scale" },
         }
    };
    for (const auto & [target, source] : linear) {
        if (suffix == target) {
            const std::string source_name = prefix + std::string(source);
            if (ends_with(target, ".scale")) {
                const std::string module = source_name.substr(
                    0, source_name.size() - std::string_view(".weight_scale").size());
                return quantized_or_plain(
                    quant, module, llama_safetensors_quant_role::WEIGHT_SCALE,
                    transform_kind::QWEN_LINEAR, source_name);
            }
            if (ends_with(target, ".weight")) {
                const std::string module = source_name.substr(
                    0, source_name.size() - std::string_view(".weight").size());
                return quantized_or_plain(
                    quant, module, llama_safetensors_quant_role::WEIGHT,
                    transform_kind::QWEN_LINEAR, source_name);
            }
            return { source_name, transform_kind::QWEN_LINEAR, std::nullopt };
        }
    }

    if (layer == 64) {
        const std::array<std::pair<std::string_view, std::string_view>, 4> mtp = {
            {
             { "nextn.eh_proj.weight", "mtp.fc.weight" },
             { "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight" },
             { "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight" },
             { "nextn.shared_head_norm.weight", "mtp.norm.weight" },
             }
        };
        for (const auto & [target, source] : mtp) {
            if (suffix == target) {
                return { std::string(source), transform_kind::NONE, std::nullopt };
            }
        }
    }

    throw std::runtime_error("unsupported Qwen3.5 target tensor '" + target_name + "'");
}

ggml_type target_type_for(
        const llama_safetensors_registry & registry,
        const source_spec & spec,
        const std::string & target_name) {
    if (spec.quant) {
        return spec.quant->target_type;
    }
    const llama_safetensors_tensor & source = require_tensor(registry, spec.name);
    switch (source.dtype) {
        case llama_safetensors_dtype::F8_E4M3:
            return GGML_TYPE_F8_E4M3;
        case llama_safetensors_dtype::BF16:
            if (ends_with(source.name, ".weight_scale_inv")) {
                return GGML_TYPE_F32;
            }
            if (ends_with(target_name, ".scale") || target_name == "token_embd.weight" ||
                (source.shape.size() >= 2 && target_name.find("ssm_conv1d.weight") == std::string::npos &&
                 !is_qwen_offset_norm(source.name))) {
                return GGML_TYPE_BF16;
            }
            return GGML_TYPE_F32;
        case llama_safetensors_dtype::F32:
            return GGML_TYPE_F32;
        default:
            throw std::runtime_error("unsupported source dtype for target '" + target_name + "'");
    }
}

std::vector<int64_t> target_shape_for(
        const llama_safetensors_registry & registry,
        const source_spec & spec,
        const std::string & target_name) {
    const llama_safetensors_tensor & source = require_tensor(registry, spec.name);
    if (spec.quant) {
        return spec.quant->target_shape;
    }
    if (ends_with(target_name, ".scale")) {
        return { static_cast<int64_t>(source.shape[0]) };
    }
    std::vector<int64_t> result;
    for (auto it = source.shape.rbegin(); it != source.shape.rend(); ++it) {
        if (*it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("source dimension exceeds runtime limits for '" + target_name + "'");
        }
        if (*it != 1 || source.shape.size() == 1) {
            result.push_back(static_cast<int64_t>(*it));
        }
    }
    if (result.empty()) {
        result.push_back(1);
    }
    return result;
}

std::vector<size_t> v_head_row_permutation(size_t head_dim) {
    constexpr size_t    n_k       = 16;
    constexpr size_t    n_v_per_k = 3;
    std::vector<size_t> result(n_k * n_v_per_k * head_dim);
    for (size_t v = 0; v < n_v_per_k; ++v) {
        for (size_t k = 0; k < n_k; ++k) {
            for (size_t h = 0; h < head_dim; ++h) {
                const size_t dst = ((v * n_k + k) * head_dim + h);
                const size_t src = ((k * n_v_per_k + v) * head_dim + h);
                result[dst]      = src;
            }
        }
    }
    return result;
}

std::vector<uint8_t> permute_rows(const std::vector<uint8_t> & source,
                                  size_t                       row_size,
                                  size_t                       prefix_rows,
                                  const std::vector<size_t> &  permutation) {
    if (source.size() < (prefix_rows + permutation.size()) * row_size) {
        throw std::runtime_error("row permutation exceeds source tensor");
    }
    std::vector<uint8_t> result(source);
    for (size_t dst = 0; dst < permutation.size(); ++dst) {
        std::memcpy(result.data() + (prefix_rows + dst) * row_size,
                    source.data() + (prefix_rows + permutation[dst]) * row_size, row_size);
    }
    return result;
}

std::vector<uint8_t> permute_columns(const std::vector<uint8_t> & source,
                                     size_t                       rows,
                                     size_t                       cols,
                                     size_t                       element_size,
                                     const std::vector<size_t> &  permutation) {
    if (permutation.size() != cols || source.size() != rows * cols * element_size) {
        throw std::runtime_error("column permutation shape mismatch");
    }
    std::vector<uint8_t> result(source.size());
    for (size_t row = 0; row < rows; ++row) {
        for (size_t dst = 0; dst < cols; ++dst) {
            std::memcpy(result.data() + (row * cols + dst) * element_size,
                        source.data() + (row * cols + permutation[dst]) * element_size, element_size);
        }
    }
    return result;
}

// The GGUF converter casts BF16 A_log to F32 before torch.exp(). The
// MKL-backed vector implementation used there differs by one ULP from a
// correctly rounded exp for a small, finite subset of the BF16 domain. Recurrent
// state amplifies those differences, so pin the exceptional outputs and use a
// high-precision scalar exp everywhere else. This table is an exhaustive diff
// over finite BF16 inputs against PyTorch 2.11.0+cu126 CPU/MKL, not values
// selected from this checkpoint.
struct bf16_exp_correction {
    uint16_t input;
    uint32_t output;
};

constexpr std::array<bf16_exp_correction, 134> bf16_exp_corrections = {
    {
     { 0x3380, 0x3f800000 }, { 0x3d28, 0x3f855bf2 }, { 0x3d29, 0x3f856448 }, { 0x3d55, 0x3f86d516 },
     { 0x3d6a, 0x3f878682 }, { 0x3d76, 0x3f87ec4d }, { 0x3d93, 0x3f898678 }, { 0x3dc0, 0x3f8c949b },
     { 0x3dc8, 0x3f8d2176 }, { 0x3df2, 0x3f900e0c }, { 0x3df4, 0x3f903214 }, { 0x3df6, 0x3f905625 },
     { 0x3e76, 0x3fa2c20f }, { 0x3e85, 0x3fa5f7d8 }, { 0x3e91, 0x3fa9e76a }, { 0x3e99, 0x3fac945e },
     { 0x3e9e, 0x3fae45ee }, { 0x3ec1, 0x3fba9a5f }, { 0x3ece, 0x3fbf66d2 }, { 0x3ed7, 0x3fc2cbbe },
     { 0x3ee2, 0x3fc706b5 }, { 0x3ef1, 0x3fccf17c }, { 0x3efa, 0x3fd093e2 }, { 0x3f1e, 0x3fed4646 },
     { 0x3f4b, 0x400d6fc7 }, { 0x3f66, 0x401d2b39 }, { 0x3f70, 0x40236e03 }, { 0x3f7a, 0x4029f0a5 },
     { 0x3f94, 0x404b643e }, { 0x3f9f, 0x405da4c2 }, { 0x3fa0, 0x405f61c8 }, { 0x3fa9, 0x406fa762 },
     { 0x3fac, 0x4075564a }, { 0x3fb6, 0x4084a2e6 }, { 0x3fbb, 0x4089eb82 }, { 0x3fd9, 0x40ae58de },
     { 0x3fe8, 0x40c40616 }, { 0x3ffa, 0x40e19f3a }, { 0x4014, 0x41219823 }, { 0x4015, 0x41242397 },
     { 0x404c, 0x41c1d280 }, { 0x4070, 0x422a1596 }, { 0x4098, 0x42e72b28 }, { 0x40a4, 0x43282c94 },
     { 0x40ae, 0x4365dde7 }, { 0x40af, 0x436d29de }, { 0x40c3, 0x43dd8a39 }, { 0x40d0, 0x44264910 },
     { 0x40d7, 0x444ef20e }, { 0x40e5, 0x44a04303 }, { 0x40f0, 0x44e2015c }, { 0x40f1, 0x44e92df3 },
     { 0x4101, 0x45465369 }, { 0x4106, 0x45878a26 }, { 0x410f, 0x45ede124 }, { 0x413b, 0x47e890fa },
     { 0x4151, 0x48e5f454 }, { 0x415d, 0x4973681f }, { 0x416f, 0x4a3b6fb7 }, { 0x4173, 0x4a70ac4e },
     { 0x4187, 0x4ba2a220 }, { 0x418a, 0x4beca14a }, { 0x4195, 0x4ce9f8f6 }, { 0x41b5, 0x4fc799dc },
     { 0x41f2, 0x5547ad55 }, { 0x4202, 0x56eccf78 }, { 0x4206, 0x57a0edec }, { 0x4208, 0x5804a9f2 },
     { 0x4209, 0x582a57ff }, { 0x4244, 0x62cecb80 }, { 0x424c, 0x643f009f }, { 0x4255, 0x65e285c7 },
     { 0x4287, 0x7026cb04 }, { 0x4289, 0x70e2b1fc }, { 0x4298, 0x76482253 }, { 0x42aa, 0x7cc5f63a },
     { 0xbc42, 0x3f7cfc94 }, { 0xbcd0, 0x3f7994f2 }, { 0xbd0f, 0x3f77377a }, { 0xbd1d, 0x3f765f88 },
     { 0xbd1f, 0x3f7640be }, { 0xbd30, 0x3f753ba4 }, { 0xbd65, 0x3f72148a }, { 0xbd6d, 0x3f719b9e },
     { 0xbd78, 0x3f70f5bc }, { 0xbd88, 0x3f6f8d5a }, { 0xbd94, 0x3f6e2713 }, { 0xbd9f, 0x3f6ce07e },
     { 0xbdc2, 0x3f68dcf7 }, { 0xbdcd, 0x3f679da3 }, { 0xbdce, 0x3f6780b1 }, { 0xbdd5, 0x3f66b679 },
     { 0xbdd6, 0x3f6699a4 }, { 0xbdf1, 0x3f639479 }, { 0xbdf3, 0x3f635b9b }, { 0xbe0d, 0x3f5f11b9 },
     { 0xbe0f, 0x3f5ea24c }, { 0xbe7e, 0x3f47c345 }, { 0xbeec, 0x3f21750a }, { 0xbef0, 0x3f203362 },
     { 0xbf48, 0x3eea6923 }, { 0xbf4c, 0x3ee6c6c8 }, { 0xbf67, 0x3ecfad25 }, { 0xbf73, 0x3ec62a89 },
     { 0xbf77, 0x3ec31809 }, { 0xbf94, 0x3ea11ba3 }, { 0xbfd1, 0x3e481182 }, { 0xbfe6, 0x3e29cbbc },
     { 0xc00d, 0x3de23783 }, { 0xc013, 0x3dcdf907 }, { 0xc057, 0x3d0e5d54 }, { 0xc05a, 0x3d07d861 },
     { 0xc061, 0x3cf38aaf }, { 0xc064, 0x3ce863a0 }, { 0xc065, 0x3ce4c94b }, { 0xc094, 0x3c209f83 },
     { 0xc0a9, 0x3ba6a90a }, { 0xc0bb, 0x3b3deb9e }, { 0xc0eb, 0x3a298202 }, { 0xc0fc, 0x39c74c0b },
     { 0xc155, 0x35ddf470 }, { 0xc1bd, 0x2e71934c }, { 0xc1c3, 0x2de4396a }, { 0xc1cc, 0x2d142fe8 },
     { 0xc1d7, 0x2c15decd }, { 0xc1e7, 0x2aa2430e }, { 0xc1fa, 0x28f17bbd }, { 0xc206, 0x274b9e05 },
     { 0xc20f, 0x25abb056 }, { 0xc210, 0x2585b61e }, { 0xc24e, 0x1a501c44 }, { 0xc258, 0x1888a976 },
     { 0xc297, 0x0906f907 }, { 0xc2a8, 0x02e0f96e },
     }
};

constexpr bool bf16_exp_corrections_are_sorted() {
    for (size_t i = 1; i < bf16_exp_corrections.size(); ++i) {
        if (bf16_exp_corrections[i - 1].input >= bf16_exp_corrections[i].input) {
            return false;
        }
    }
    return true;
}

static_assert(bf16_exp_corrections_are_sorted(), "BF16 exp correction table must be strictly sorted");

float converter_exp_bf16(uint16_t bits16) {
    const auto it = std::lower_bound(
        bf16_exp_corrections.begin(), bf16_exp_corrections.end(), bits16,
        [](const bf16_exp_correction & correction, uint16_t value) { return correction.input < value; });
    if (it != bf16_exp_corrections.end() && it->input == bits16) {
        float result;
        std::memcpy(&result, &it->output, sizeof(result));
        return result;
    }

    const uint32_t bits32 = uint32_t(bits16) << 16;
    float          value;
    std::memcpy(&value, &bits32, sizeof(value));
    return static_cast<float>(std::exp(static_cast<double>(value)));
}

std::vector<uint8_t> negate_exp_f32(std::vector<uint8_t> source) {
    if (source.size() % 4 != 0) {
        throw std::runtime_error("invalid F32 byte count for Qwen A_log transform");
    }
    for (size_t i = 0; i < source.size() / 4; ++i) {
        float value;
        std::memcpy(&value, source.data() + 4 * i, sizeof(value));
        value = -static_cast<float>(std::exp(static_cast<double>(value)));
        std::memcpy(source.data() + 4 * i, &value, sizeof(value));
    }
    return source;
}

std::vector<uint8_t> negate_exp_bf16_to_f32(const std::vector<uint8_t> & source) {
    if (source.size() % 2 != 0) {
        throw std::runtime_error("invalid BF16 byte count for Qwen A_log transform");
    }
    std::vector<uint8_t> result(source.size() * 2);
    for (size_t i = 0; i < source.size() / 2; ++i) {
        uint16_t bits16;
        std::memcpy(&bits16, source.data() + 2 * i, sizeof(bits16));
        const float value = -converter_exp_bf16(bits16);
        std::memcpy(result.data() + 4 * i, &value, sizeof(value));
    }
    return result;
}

std::vector<uint8_t> apply_qwen_transform(const llama_safetensors_tensor & tensor, std::vector<uint8_t> source) {
    const std::string & name         = tensor.name;
    const size_t        element_size = dtype_size(tensor.dtype);
    const bool          block_scale  = ends_with(name, ".weight_scale_inv");
    if (tensor.shape.empty()) {
        return source;
    }

    if (name.find(".linear_attn.in_proj_qkv.") != std::string::npos) {
        const size_t qk_rows = block_scale ? 2 * 16 : 2 * 16 * 128;
        const auto permutation = v_head_row_permutation(block_scale ? 1 : 128);
        const size_t     rows        = tensor.shape[0];
        const size_t     row_size    = source.size() / rows;
        return permute_rows(source, row_size, qk_rows, permutation);
    }
    if (name.find(".linear_attn.in_proj_z.") != std::string::npos) {
        const auto permutation = v_head_row_permutation(block_scale ? 1 : 128);
        return permute_rows(source, source.size() / tensor.shape[0], 0, permutation);
    }
    if (name.find(".linear_attn.in_proj_a.") != std::string::npos ||
        name.find(".linear_attn.in_proj_b.") != std::string::npos || ends_with(name, ".linear_attn.A_log") ||
        ends_with(name, ".linear_attn.dt_bias")) {
        const auto permutation = v_head_row_permutation(1);
        return permute_rows(source, source.size() / tensor.shape[0], 0, permutation);
    }
    if (name.find(".linear_attn.conv1d.weight") != std::string::npos) {
        constexpr size_t qk_rows     = 2 * 16 * 128;
        const auto       permutation = v_head_row_permutation(128);
        return permute_rows(source, source.size() / tensor.shape[0], qk_rows, permutation);
    }
    if (ends_with(name, ".linear_attn.out_proj.weight") ||
        ends_with(name, ".linear_attn.out_proj.weight_scale_inv")) {
        if (tensor.shape.size() != 2) {
            throw std::runtime_error("unexpected linear-attention output projection rank");
        }
        return permute_columns(
            source, tensor.shape[0], tensor.shape[1], element_size,
            v_head_row_permutation(block_scale ? 1 : 128));
    }
    return source;
}

std::vector<uint8_t> bf16_to_f32(const std::vector<uint8_t> & source) {
    if (source.size() % 2 != 0) {
        throw std::runtime_error("invalid BF16 byte count");
    }
    std::vector<uint8_t> result(source.size() * 2);
    for (size_t i = 0; i < source.size() / 2; ++i) {
        uint16_t bits16;
        std::memcpy(&bits16, source.data() + 2 * i, sizeof(bits16));
        const uint32_t bits32 = uint32_t(bits16) << 16;
        std::memcpy(result.data() + 4 * i, &bits32, sizeof(bits32));
    }
    return result;
}

std::vector<uint8_t> bf16_add_one_to_f32(const std::vector<uint8_t> & source) {
    std::vector<uint8_t> result = bf16_to_f32(source);
    for (size_t i = 0; i < result.size() / sizeof(float); ++i) {
        float value;
        std::memcpy(&value, result.data() + i * sizeof(float), sizeof(value));
        value += 1.0f;
        std::memcpy(result.data() + i * sizeof(float), &value, sizeof(value));
    }
    return result;
}

void validate_model_contract(const std::filesystem::path & model_dir) {
    const json root = read_json(model_dir / "config.json");
    if (root.value("model_type", std::string()) != "qwen3_5") {
        throw std::runtime_error("native Qwen3.5 importer requires model_type 'qwen3_5'");
    }
    const json & text = root.at("text_config");
    if (text.value("model_type", std::string()) != "qwen3_5_text" || text.value("num_hidden_layers", 0U) != 64 ||
        text.value("linear_num_key_heads", 0U) != 16 || text.value("linear_num_value_heads", 0U) != 48 ||
        text.value("linear_value_head_dim", 0U) != 128) {
        throw std::runtime_error("native Qwen3.5 importer does not support this tensor geometry");
    }

}

}  // namespace

llama_safetensors_qwen35_importer::llama_safetensors_qwen35_importer(const std::filesystem::path & model_dir) :
    model_dir_(model_dir),
    registry_(llama_safetensors_registry::load(model_dir)) {
    validate_model_contract(model_dir_);
    quant_ = std::make_unique<llama_safetensors_quant_adapters>(model_dir_, registry_);
}

bool llama_safetensors_qwen35_importer::probe(const std::filesystem::path & model_dir) {
    const json root = read_json(model_dir / "config.json");
    return root.value("model_type", std::string()) == "qwen3_5";
}

gguf_context * llama_safetensors_qwen35_importer::build_metadata() const {
    const json root       = read_json(model_dir_ / "config.json");
    const json text       = root.at("text_config");
    const json rope       = text.at("rope_parameters");
    const json generation = read_json(model_dir_ / "generation_config.json");
    const json tokenizer  = read_json(model_dir_ / "tokenizer.json");

    gguf_context * metadata = gguf_init_empty();
    if (metadata == nullptr) {
        throw std::runtime_error("failed to allocate native safetensors metadata");
    }
    try {
        gguf_set_val_str(metadata, "general.architecture", "qwen35");
        gguf_set_val_str(metadata, "general.type", "model");
        const std::string source_name = model_dir_.filename().empty() ?
            "Qwen3.5 Safetensors" : model_dir_.filename().string();
        gguf_set_val_str(metadata, "general.name", source_name.c_str());
        const llama_safetensors_quant_summary & quant_summary = quant_->summary();
        gguf_set_val_u32(
            metadata, "general.file_type",
            quant_summary.nvfp4 != 0 ? LLAMA_FTYPE_MOSTLY_NVFP4 : LLAMA_FTYPE_MOSTLY_F8_E4M3);
        gguf_set_val_u32(metadata, "general.quantization_version", 2);
        gguf_set_val_i32(metadata, "general.sampling.top_k", generation.value("top_k", 20));
        gguf_set_val_f32(metadata, "general.sampling.top_p", generation.value("top_p", 0.95f));
        gguf_set_val_f32(metadata, "general.sampling.temp", generation.value("temperature", 1.0f));

        const uint32_t n_layer = text.at("num_hidden_layers").get<uint32_t>();
        const uint32_t n_mtp   = text.value("mtp_num_hidden_layers", root.value("mtp_num_hidden_layers", 0U));
        gguf_set_val_u32(metadata, "qwen35.block_count", n_layer + n_mtp);
        gguf_set_val_u32(metadata, "qwen35.context_length", text.at("max_position_embeddings").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.embedding_length", text.at("hidden_size").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.feed_forward_length", text.at("intermediate_size").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.attention.head_count", text.at("num_attention_heads").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.attention.head_count_kv", text.at("num_key_value_heads").get<uint32_t>());
        std::array<int32_t, 4> sections = { 11, 11, 10, 0 };
        if (rope.contains("mrope_section")) {
            const auto configured = rope.at("mrope_section").get<std::vector<int32_t>>();
            for (size_t i = 0; i < std::min(configured.size(), sections.size()); ++i) {
                sections[i] = configured[i];
            }
        }
        gguf_set_arr_data(metadata, "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, sections.data(),
                          sections.size());
        gguf_set_val_f32(metadata, "qwen35.rope.freq_base", rope.at("rope_theta").get<float>());
        gguf_set_val_f32(metadata, "qwen35.attention.layer_norm_rms_epsilon", text.at("rms_norm_eps").get<float>());
        gguf_set_val_u32(metadata, "qwen35.attention.key_length", text.at("head_dim").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.attention.value_length", text.at("head_dim").get<uint32_t>());
        if (n_mtp != 0) {
            gguf_set_val_u32(metadata, "qwen35.nextn_predict_layers", n_mtp);
        }
        gguf_set_val_u32(metadata, "qwen35.ssm.conv_kernel", text.at("linear_conv_kernel_dim").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.ssm.state_size", text.at("linear_key_head_dim").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.ssm.group_count", text.at("linear_num_key_heads").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.ssm.time_step_rank", text.at("linear_num_value_heads").get<uint32_t>());
        gguf_set_val_u32(
            metadata, "qwen35.ssm.inner_size",
            text.at("linear_num_value_heads").get<uint32_t>() * text.at("linear_value_head_dim").get<uint32_t>());
        gguf_set_val_u32(metadata, "qwen35.full_attention_interval", text.value("full_attention_interval", 4U));
        gguf_set_val_u32(
            metadata, "qwen35.rope.dimension_count",
            static_cast<uint32_t>(text.at("head_dim").get<float>() * rope.value("partial_rotary_factor", 0.25f)));

        const uint32_t           vocab_size = text.at("vocab_size").get<uint32_t>();
        std::vector<std::string> tokens(vocab_size);
        std::vector<int32_t>     token_types(vocab_size, 5);  // GGML unused token
        for (uint32_t id = 0; id < vocab_size; ++id) {
            tokens[id] = "[PAD" + std::to_string(id) + "]";
        }
        for (const auto & [token, id_json] : tokenizer.at("model").at("vocab").items()) {
            const uint32_t id = id_json.get<uint32_t>();
            if (id >= vocab_size) {
                throw std::runtime_error("tokenizer vocabulary id exceeds configured vocabulary size");
            }
            tokens[id]      = token;
            token_types[id] = 1;  // GGML normal token
        }
        for (const json & added : tokenizer.at("added_tokens")) {
            const uint32_t id = added.at("id").get<uint32_t>();
            if (id >= vocab_size) {
                throw std::runtime_error("added token id exceeds configured vocabulary size");
            }
            const std::string content = added.at("content").get<std::string>();
            tokens[id]                = content;
            const bool looks_control  = added.value("special", false) ||
                                       (content.rfind("<|", 0) == 0 && ends_with(content, "|>")) ||
                                       content.rfind("<tts_", 0) == 0;
            token_types[id] = looks_control ? 3 : 4;  // control or user-defined
        }
        std::vector<const char *> token_ptrs(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            token_ptrs[i] = tokens[i].c_str();
        }
        std::vector<std::string> merges;
        merges.reserve(tokenizer.at("model").at("merges").size());
        for (const json & merge : tokenizer.at("model").at("merges")) {
            if (merge.is_array() && merge.size() == 2) {
                merges.push_back(merge[0].get<std::string>() + " " + merge[1].get<std::string>());
            } else if (merge.is_string()) {
                merges.push_back(merge.get<std::string>());
            } else {
                throw std::runtime_error("unsupported tokenizer merge entry");
            }
        }
        std::vector<const char *> merge_ptrs(merges.size());
        for (size_t i = 0; i < merges.size(); ++i) {
            merge_ptrs[i] = merges[i].c_str();
        }
        gguf_set_val_str(metadata, "tokenizer.ggml.model", "gpt2");
        gguf_set_val_str(metadata, "tokenizer.ggml.pre", "qwen35");
        gguf_set_arr_str(metadata, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types.data(),
                          token_types.size());
        gguf_set_arr_str(metadata, "tokenizer.ggml.merges", merge_ptrs.data(), merge_ptrs.size());
        gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", generation.at("bos_token_id").get<uint32_t>());
        const json eos = generation.at("eos_token_id");
        gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id",
                         eos.is_array() ? eos[0].get<uint32_t>() : eos.get<uint32_t>());
        const auto token_id = [&](const std::string & content) -> uint32_t {
            const auto it = std::find(tokens.begin(), tokens.end(), content);
            if (it == tokens.end()) {
                throw std::runtime_error("token not found: '" + content + "'");
            }
            return static_cast<uint32_t>(it - tokens.begin());
        };
        gguf_set_val_u32(metadata, "tokenizer.ggml.padding_token_id", token_id("<|vision_pad|>"));
        const auto chat_template = model_dir_ / "chat_template.jinja";
        if (std::filesystem::exists(chat_template)) {
            gguf_set_val_str(metadata, "tokenizer.chat_template", read_text(chat_template).c_str());
        }

        return metadata;
    } catch (...) {
        gguf_free(metadata);
        throw;
    }
}

bool llama_safetensors_qwen35_importer::describe(
        const std::string & target_name,
        ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const {
    source_spec spec;
    try {
        spec = map_target(*quant_, target_name);
    } catch (const std::runtime_error &) {
        return false;
    }
    if (registry_.find(spec.name) == nullptr) {
        return false;
    }

    type = target_type_for(registry_, spec, target_name);
    const std::vector<int64_t> shape = target_shape_for(registry_, spec, target_name);
    if (shape.size() > GGML_MAX_DIMS) {
        throw std::runtime_error("target tensor rank exceeds GGML_MAX_DIMS for '" + target_name + "'");
    }
    ne.fill(1);
    std::copy(shape.begin(), shape.end(), ne.begin());
    return true;
}

size_t llama_safetensors_qwen35_importer::tensor_capacity_hint() const {
    // Transforms can expose more canonical tensors than physical source
    // tensors (for example weight, weight scale, and input scale). Reserving
    // twice the registry size keeps this a conservative allocation hint.
    return std::max<size_t>(256, registry_.tensors().size() * 2);
}

void llama_safetensors_qwen35_importer::bind(const std::string & target_name) const {
    const source_spec spec = map_target(*quant_, target_name);
    if (spec.quant) {
        quant_->consume(*spec.quant);
    }
}

void llama_safetensors_qwen35_importer::validate_complete() const {
    quant_->validate_complete();
}

std::vector<uint8_t> llama_safetensors_qwen35_importer::materialize(const std::string & target_name,
                                                                    ggml_type           target_type,
                                                                    size_t              target_size) const {
    try {
        const source_spec                spec        = map_target(*quant_, target_name);
        const llama_safetensors_tensor & source_desc = require_tensor(registry_, spec.name);
        std::vector<uint8_t> result = spec.quant ? quant_->read(*spec.quant) : registry_.read(source_desc);

        if (is_qwen_offset_norm(source_desc.name) && source_desc.dtype == llama_safetensors_dtype::BF16 &&
            target_type == GGML_TYPE_F32) {
            result = bf16_add_one_to_f32(result);
        } else if (ends_with(source_desc.name, ".linear_attn.A_log")) {
            if (source_desc.dtype == llama_safetensors_dtype::BF16) {
                result = negate_exp_bf16_to_f32(result);
            } else if (source_desc.dtype == llama_safetensors_dtype::F32) {
                result = negate_exp_f32(std::move(result));
            } else {
                throw std::runtime_error("Qwen A_log has unsupported dtype");
            }
            result = permute_rows(result, sizeof(float), 0, v_head_row_permutation(1));
        } else {
            result = apply_qwen_transform(source_desc, std::move(result));
            if (spec.quant) {
                result = quant_->finalize(*spec.quant, std::move(result));
            } else if (target_type == GGML_TYPE_F32 && source_desc.dtype == llama_safetensors_dtype::BF16) {
                result = bf16_to_f32(result);
            }
        }

        if (result.size() != target_size) {
            throw std::runtime_error("produced " + std::to_string(result.size()) + " bytes, expected " +
                                     std::to_string(target_size));
        }
        return result;
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to materialize '" + target_name + "': " + error.what());
    }
}
