#include "llama-safetensors.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>
#include <string_view>

namespace {

// compressed-tensors resolves overlapping targets in declaration order. Use
// ordered_json for every parsed object so config_groups retains file order.
using json = llama_safetensors_json;

constexpr uint64_t MAX_HEADER_SIZE = 100ULL * 1024 * 1024;

uint64_t checked_add(uint64_t a, uint64_t b, const std::string & what) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        throw std::runtime_error(what + " overflows uint64");
    }
    return a + b;
}

uint64_t checked_mul(uint64_t a, uint64_t b, const std::string & what) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        throw std::runtime_error(what + " overflows uint64");
    }
    return a * b;
}

uint64_t checked_file_size(const std::filesystem::path & path) {
    std::error_code ec;
    const uint64_t  result = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("failed to stat safetensors shard '" + path.string() + "': " + ec.message());
    }
    return result;
}

uint64_t read_header_size(std::ifstream & in, const std::filesystem::path & path) {
    std::array<unsigned char, 8> raw{};
    in.read(reinterpret_cast<char *>(raw.data()), raw.size());
    if (!in) {
        throw std::runtime_error("safetensors shard '" + path.string() + "' is shorter than its 8-byte header prefix");
    }

    uint64_t result = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
        result |= uint64_t(raw[i]) << (8 * i);
    }
    return result;
}

std::pair<llama_safetensors_dtype, uint32_t> parse_dtype(const std::string & name) {
    static const std::map<std::string, std::pair<llama_safetensors_dtype, uint32_t>> types = {
        { "BOOL",    { llama_safetensors_dtype::BOOL, 8 }    },
        { "U8",      { llama_safetensors_dtype::U8, 8 }      },
        { "I8",      { llama_safetensors_dtype::I8, 8 }      },
        { "U16",     { llama_safetensors_dtype::U16, 16 }    },
        { "I16",     { llama_safetensors_dtype::I16, 16 }    },
        { "U32",     { llama_safetensors_dtype::U32, 32 }    },
        { "I32",     { llama_safetensors_dtype::I32, 32 }    },
        { "U64",     { llama_safetensors_dtype::U64, 64 }    },
        { "I64",     { llama_safetensors_dtype::I64, 64 }    },
        { "F8_E4M3", { llama_safetensors_dtype::F8_E4M3, 8 } },
        { "F8_E5M2", { llama_safetensors_dtype::F8_E5M2, 8 } },
        { "F16",     { llama_safetensors_dtype::F16, 16 }    },
        { "BF16",    { llama_safetensors_dtype::BF16, 16 }   },
        { "F32",     { llama_safetensors_dtype::F32, 32 }    },
        { "F64",     { llama_safetensors_dtype::F64, 64 }    },
    };

    const auto it = types.find(name);
    if (it == types.end()) {
        throw std::runtime_error("unsupported safetensors dtype '" + name + "'");
    }
    return it->second;
}

std::filesystem::path checked_shard_path(const std::filesystem::path & model_dir, const std::string & shard_name) {
    const std::filesystem::path relative(shard_name);
    if (relative.empty() || relative.is_absolute()) {
        throw std::runtime_error("invalid safetensors shard path '" + shard_name + "'");
    }
    for (const auto & component : relative) {
        if (component == "..") {
            throw std::runtime_error("safetensors shard path escapes the model directory: '" + shard_name + "'");
        }
    }
    return model_dir / relative;
}

struct parsed_shard {
    llama_safetensors_shard               shard;
    std::vector<llama_safetensors_tensor> tensors;
    std::unordered_map<std::string, std::string> metadata;
};

parsed_shard parse_shard(const std::filesystem::path & path, uint32_t shard_index) {
    parsed_shard result;
    result.shard.path      = path;
    result.shard.file_size = checked_file_size(path);

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open safetensors shard '" + path.string() + "'");
    }

    const uint64_t header_size = read_header_size(in, path);
    if (header_size == 0 || header_size > MAX_HEADER_SIZE) {
        throw std::runtime_error("invalid safetensors header size in '" + path.string() + "'");
    }
    result.shard.data_begin = checked_add(8, header_size, "safetensors data offset");
    if (result.shard.data_begin > result.shard.file_size) {
        throw std::runtime_error("safetensors header extends beyond shard '" + path.string() + "'");
    }

    std::string header(header_size, '\0');
    in.read(header.data(), header.size());
    if (!in) {
        throw std::runtime_error("failed to read safetensors header from '" + path.string() + "'");
    }

    json parsed;
    try {
        parsed = json::parse(header);
    } catch (const json::exception & e) {
        throw std::runtime_error("invalid safetensors JSON header in '" + path.string() + "': " + e.what());
    }
    if (!parsed.is_object()) {
        throw std::runtime_error("safetensors header in '" + path.string() + "' is not an object");
    }

    struct range {
        uint64_t    begin;
        uint64_t    end;
        std::string name;
    };

    std::vector<range> ranges;

    for (const auto & [name, desc] : parsed.items()) {
        if (name == "__metadata__") {
            if (!desc.is_object()) {
                throw std::runtime_error("safetensors __metadata__ in '" + path.string() + "' is not an object");
            }
            for (const auto & [key, value] : desc.items()) {
                if (!value.is_string()) {
                    throw std::runtime_error("non-string safetensors metadata value for '" + key + "'");
                }
                result.metadata.emplace(key, value.get<std::string>());
            }
            continue;
        }
        if (!desc.is_object() || !desc.contains("dtype") || !desc.contains("shape") || !desc.contains("data_offsets")) {
            throw std::runtime_error("invalid descriptor for safetensors tensor '" + name + "'");
        }

        const auto [dtype, bits]  = parse_dtype(desc.at("dtype").get<std::string>());
        const auto & shape_json   = desc.at("shape");
        const auto & offsets_json = desc.at("data_offsets");
        if (!shape_json.is_array() || !offsets_json.is_array() || offsets_json.size() != 2) {
            throw std::runtime_error("invalid shape or data_offsets for safetensors tensor '" + name + "'");
        }

        llama_safetensors_tensor tensor;
        tensor.name  = name;
        tensor.dtype = dtype;
        tensor.shard = shard_index;

        uint64_t elements = 1;
        for (const auto & dim_json : shape_json) {
            if (!dim_json.is_number_unsigned()) {
                throw std::runtime_error("non-unsigned shape dimension for safetensors tensor '" + name + "'");
            }
            const uint64_t dim = dim_json.get<uint64_t>();
            tensor.shape.push_back(dim);
            elements = checked_mul(elements, dim, "element count for tensor '" + name + "'");
        }

        if (!offsets_json[0].is_number_unsigned() || !offsets_json[1].is_number_unsigned()) {
            throw std::runtime_error("non-unsigned data_offsets for safetensors tensor '" + name + "'");
        }
        const uint64_t relative_begin = offsets_json[0].get<uint64_t>();
        const uint64_t relative_end   = offsets_json[1].get<uint64_t>();
        if (relative_end < relative_begin) {
            throw std::runtime_error("reversed data_offsets for safetensors tensor '" + name + "'");
        }

        const uint64_t expected_bits = checked_mul(elements, bits, "stored size for tensor '" + name + "'");
        if (expected_bits % 8 != 0 || (relative_end - relative_begin) != expected_bits / 8) {
            throw std::runtime_error("stored size does not match dtype and shape for safetensors tensor '" + name +
                                     "'");
        }

        tensor.offset = checked_add(result.shard.data_begin, relative_begin, "file offset for tensor '" + name + "'");
        const uint64_t absolute_end =
            checked_add(result.shard.data_begin, relative_end, "file end for tensor '" + name + "'");
        if (absolute_end > result.shard.file_size) {
            throw std::runtime_error("safetensors tensor '" + name + "' extends beyond shard bounds");
        }
        tensor.size = relative_end - relative_begin;
        ranges.push_back({ relative_begin, relative_end, name });
        result.tensors.push_back(std::move(tensor));
    }

    std::sort(ranges.begin(), ranges.end(), [](const range & a, const range & b) { return a.begin < b.begin; });
    uint64_t next = 0;
    for (const range & current : ranges) {
        if (current.begin != next) {
            throw std::runtime_error("safetensors data ranges overlap or contain a gap before tensor '" + current.name +
                                     "'");
        }
        next = current.end;
    }
    if (next != result.shard.file_size - result.shard.data_begin) {
        throw std::runtime_error("safetensors shard '" + path.string() + "' contains unindexed trailing data");
    }

    return result;
}

const json & require_json_value(const json & object, const char * key, const std::string & context) {
    if (!object.is_object() || !object.contains(key)) {
        throw std::runtime_error(context + " is missing '" + key + "'");
    }
    return object.at(key);
}

}  // namespace

size_t llama_safetensors_dtype_size(llama_safetensors_dtype dtype) {
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

llama_safetensors_quant_config::rule llama_safetensors_quant_config::make_rule(const std::string & target,
                                                                               uint32_t            group) {
    rule result;
    result.target   = target;
    result.is_regex = target.rfind("re:", 0) == 0;
    result.group    = group;
    if (result.is_regex) {
        try {
            result.pattern = std::regex(target.substr(3));
        } catch (const std::regex_error & e) {
            throw std::runtime_error("invalid compressed-tensors target regex '" + target + "': " + e.what());
        }
    }
    return result;
}

bool llama_safetensors_quant_config::rule_matches(const rule & candidate, const std::string & module_name) {
    if (!candidate.is_regex) {
        return candidate.target == module_name;
    }
    std::match_results<std::string::const_iterator> match;
    return std::regex_search(module_name.begin(), module_name.end(), match, candidate.pattern,
                             std::regex_constants::match_continuous);
}

static std::vector<int64_t> standard_projection_shape(
        const llama_safetensors_json & root, const std::string & module) {
    const auto & config = root.contains("text_config") ? root.at("text_config") : root;
    const uint64_t hidden = config.value("hidden_size", uint64_t(0));
    const uint64_t intermediate = config.value("intermediate_size", uint64_t(0));
    const uint64_t heads = config.value("num_attention_heads", uint64_t(0));
    const uint64_t kv_heads = config.value("num_key_value_heads", uint64_t(0));
    const uint64_t head_dim = config.value("head_dim", heads == 0 ? uint64_t(0) : hidden / heads);
    const uint64_t vocab = config.value("vocab_size", uint64_t(0));
    const auto suffix = [&](const char * value) {
        const std::string_view tail(value);
        return module.size() >= tail.size() &&
            std::string_view(module).substr(module.size() - tail.size()) == tail;
    };
    uint64_t cols = 0;
    uint64_t rows = 0;
    if (module == "lm_head") {
        cols = hidden;
        rows = vocab;
    } else if (suffix(".q_proj")) {
        cols = hidden;
        rows = heads * head_dim;
    } else if (suffix(".k_proj") || suffix(".v_proj")) {
        cols = hidden;
        rows = kv_heads * head_dim;
    } else if (suffix(".o_proj")) {
        cols = heads * head_dim;
        rows = hidden;
    } else if (suffix(".gate_proj") || suffix(".up_proj")) {
        cols = hidden;
        rows = intermediate;
    } else if (suffix(".down_proj")) {
        cols = intermediate;
        rows = hidden;
    }
    if (cols == 0 || rows == 0 || cols > uint64_t(std::numeric_limits<int64_t>::max()) ||
        rows > uint64_t(std::numeric_limits<int64_t>::max())) {
        throw std::runtime_error("cannot infer the dense projection shape for Quanto module '" + module + "'");
    }
    return { static_cast<int64_t>(cols), static_cast<int64_t>(rows) };
}

llama_safetensors_quant_config llama_safetensors_quant_config::load(const std::filesystem::path & model_dir) {
    return from_json(llama_safetensors_read_model_config(model_dir));
}

llama_safetensors_quant_config llama_safetensors_quant_config::from_json(const llama_safetensors_json & root) {
    llama_safetensors_quant_config result;
    if (!root.contains("quantization_config")) {
        return result;
    }
    const json                     quant = require_json_value(root, "quantization_config", "config.json");
    if (!quant.is_object()) {
        throw std::runtime_error("config.json quantization_config must be an object");
    }
    const bool modelopt_producer = quant.contains("producer") && quant.at("producer").is_object() &&
        quant.at("producer").value("name", std::string()) == "modelopt";
    const std::string quant_method = quant.value(
        "quant_method", modelopt_producer ? std::string("modelopt") : std::string());
    if (quant_method.empty()) {
        throw std::runtime_error("quantization_config is missing 'quant_method'");
    }

    if (quant_method == "mxfp8") {
        llama_safetensors_quant_group group;
        group.name            = "modelopt-MXFP8";
        group.format          = llama_safetensors_quant_format::MXFP8;
        group.num_bits        = 8;
        group.group_size      = 32;
        group.input_quantized = true;
        group.input_dynamic   = true;
        group.modelopt        = true;
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));
        if (quant.contains("ignored_layers")) {
            const json & ignored = quant.at("ignored_layers");
            if (!ignored.is_array()) {
                throw std::runtime_error("quantization_config.ignored_layers must be an array");
            }
            for (const auto & target : ignored) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string target in quantization_config.ignored_layers");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    // Intel Neural Compressor / AutoRound uses the same group-32 MXFP
    // tensors as compressed-tensors, but records the contract in a compact
    // flat schema (frequently in quantization_config.json).
    if (quant_method == "auto-round" &&
        quant.value("packing_format", std::string()) == "auto_round:llm_compressor" &&
        quant.value("data_type", std::string()).find("mx_fp") != std::string::npos) {
        const uint32_t bits       = require_json_value(quant, "bits", "AutoRound MXFP config").get<uint32_t>();
        const uint32_t group_size = require_json_value(quant, "group_size", "AutoRound MXFP config").get<uint32_t>();
        const bool symmetric      = require_json_value(quant, "sym", "AutoRound MXFP config").get<bool>();
        const uint32_t act_bits   = require_json_value(quant, "act_bits", "AutoRound MXFP config").get<uint32_t>();
        const uint32_t act_group  = require_json_value(quant, "act_group_size", "AutoRound MXFP config").get<uint32_t>();
        const bool act_symmetric  = require_json_value(quant, "act_sym", "AutoRound MXFP config").get<bool>();
        const bool act_dynamic    = require_json_value(quant, "act_dynamic", "AutoRound MXFP config").get<bool>();
        const std::string act_type =
            require_json_value(quant, "act_data_type", "AutoRound MXFP config").get<std::string>();
        if ((bits != 4 && bits != 8) || group_size != 32 || !symmetric ||
            act_bits != bits || act_group != 32 || !act_symmetric || !act_dynamic ||
            act_type.find("mx_fp") == std::string::npos ||
            (quant.contains("block_name_to_quantize") && !quant.at("block_name_to_quantize").is_null()) ||
            (quant.contains("to_quant_block_names") && !quant.at("to_quant_block_names").is_null())) {
            throw std::runtime_error("unsupported native AutoRound MXFP quantization contract");
        }

        llama_safetensors_quant_group group;
        group.name            = bits == 4 ? "autoround-mxfp4" : "autoround-mxfp8";
        group.format          = bits == 4 ? llama_safetensors_quant_format::MXFP4_PACK :
                                            llama_safetensors_quant_format::MXFP8;
        group.num_bits        = bits;
        group.group_size      = group_size;
        group.input_quantized = true;
        group.input_dynamic   = true;
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));

        if (quant.contains("extra_config")) {
            const json & extra = quant.at("extra_config");
            if (!extra.is_object()) {
                throw std::runtime_error("AutoRound MXFP extra_config must be an object");
            }
            for (const auto & [module, override_config] : extra.items()) {
                if (!override_config.is_object()) {
                    throw std::runtime_error("AutoRound MXFP module override must be an object");
                }
                const uint32_t override_bits = override_config.value("bits", bits);
                const std::string override_type = override_config.value(
                    "data_type", override_bits >= 16 ? std::string("float") : quant.at("data_type").get<std::string>());
                if (override_bits >= 16 || override_type == "float") {
                    result.ignore_.push_back(make_rule(module, 0));
                    continue;
                }
                if (override_bits != bits || override_config.value("group_size", group_size) != group_size ||
                    !override_config.value("sym", symmetric) ||
                    override_type.find("mx_fp") == std::string::npos ||
                    override_config.value("act_bits", act_bits) != act_bits ||
                    override_config.value("act_group_size", act_group) != act_group ||
                    !override_config.value("act_sym", act_symmetric) ||
                    !override_config.value("act_dynamic", act_dynamic) ||
                    override_config.value("act_data_type", act_type).find("mx_fp") == std::string::npos) {
                    throw std::runtime_error("unsupported per-module AutoRound MXFP override");
                }
            }
        }
        return result;
    }

    if (quant_method == "fp8") {
        llama_safetensors_quant_group group;
        const std::string activation_scheme =
            require_json_value(quant, "activation_scheme", "quantization_config").get<std::string>();
        if (quant.contains("weight_block_size")) {
            const json block = quant.at("weight_block_size");
            if (require_json_value(quant, "fmt", "quantization_config").get<std::string>() != "e4m3" ||
                activation_scheme != "dynamic" || !block.is_array() || block.size() != 2 ||
                block[0].get<uint32_t>() != 128 || block[1].get<uint32_t>() != 128) {
                throw std::runtime_error("unsupported native block-FP8 quantization contract");
            }
            group.name            = "fp8-block-128x128";
            group.format          = llama_safetensors_quant_format::FP8_BLOCK;
            group.group_size      = 128;
            group.block_structure = { 128, 128 };
        } else {
            if (activation_scheme != "dynamic" && activation_scheme != "static") {
                throw std::runtime_error("unsupported native legacy FP8 activation scheme");
            }
            group.name                   = "legacy-fp8-tensor";
            group.format                 = llama_safetensors_quant_format::FP8_TENSOR;
            group.input_quantized        = true;
            group.input_dynamic          = activation_scheme == "dynamic";
            group.modelopt               = true;
            group.legacy_fp8_i8_storage  = true;
        }
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));

        if (quant.contains("modules_to_not_convert")) {
            const json & ignored = quant.at("modules_to_not_convert");
            if (!ignored.is_array()) {
                throw std::runtime_error("quantization_config.modules_to_not_convert must be an array");
            }
            for (const auto & target : ignored) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string module in quantization_config.modules_to_not_convert");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    if (quant_method == "fbgemm_fp8") {
        const float activation_scale_ub =
            require_json_value(quant, "activation_scale_ub", "FBGEMM FP8 quantization_config").get<float>();
        if (!(activation_scale_ub > 0.0f) || !std::isfinite(activation_scale_ub)) {
            throw std::runtime_error("FBGEMM FP8 activation_scale_ub must be finite and positive");
        }

        llama_safetensors_quant_group group;
        group.name            = "fbgemm-fp8-channel";
        group.format          = llama_safetensors_quant_format::FP8_CHANNEL;
        group.num_bits        = 8;
        group.input_quantized = true;
        group.input_dynamic   = true;
        group.modelopt        = true; // FBGEMM serializes channel scales as F32.
        group.input_scale_ub  = activation_scale_ub;
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));

        if (quant.contains("modules_to_not_convert") && !quant.at("modules_to_not_convert").is_null()) {
            const json & ignored = quant.at("modules_to_not_convert");
            if (!ignored.is_array()) {
                throw std::runtime_error("FBGEMM FP8 modules_to_not_convert must be null or an array");
            }
            for (const auto & target : ignored) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string module in FBGEMM FP8 modules_to_not_convert");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    if (quant_method == "quark") {
        const json global = require_json_value(quant, "global_quant_config", "Quark quantization_config");
        const json weight = require_json_value(global, "weight", "Quark global_quant_config");
        const json export_config = require_json_value(quant, "export", "Quark quantization_config");
        const bool has_layer_overrides =
            (quant.contains("layer_quant_config") && !quant.at("layer_quant_config").empty()) ||
            (quant.contains("layer_type_quant_config") && !quant.at("layer_type_quant_config").empty());
        if (!global.value("output_tensors", json()).is_null() ||
            !global.value("bias", json()).is_null() ||
            require_json_value(export_config, "pack_method", "Quark export config").get<std::string>() != "reorder" ||
            require_json_value(export_config, "weight_format", "Quark export config").get<std::string>() !=
                "real_quantized" || has_layer_overrides) {
            throw std::runtime_error("unsupported native Quark quantization contract");
        }

        llama_safetensors_quant_group group;
        const std::string dtype = weight.is_object() ?
            require_json_value(weight, "dtype", "Quark weight config").get<std::string>() : std::string();
        if (weight.is_array()) {
            const json input = require_json_value(global, "input_tensors", "Quark global_quant_config");
            if (weight.size() != 2 || !weight[0].is_object() || !weight[1].is_object() ||
                !input.is_array() || input.size() != 2 || !input[0].is_object() || !input[1].is_object()) {
                throw std::runtime_error("unsupported native Quark sequential quantization contract");
            }
            const auto valid_fp4_stage = [](const json & stage, bool dynamic) {
                return stage.value("dtype", std::string()) == "fp4" &&
                    stage.value("qscheme", std::string()) == "per_group" &&
                    stage.value("group_size", 0U) == 16U && stage.value("ch_axis", 0) == -1 &&
                    stage.value("is_dynamic", !dynamic) == dynamic && !stage.value("is_scale_quant", true) &&
                    stage.value("scale_type", std::string()) == "float32";
            };
            const auto valid_fp8_scale_stage = [](const json & stage) {
                return stage.value("dtype", std::string()) == "fp8_e4m3" &&
                    stage.value("qscheme", std::string()) == "per_tensor" &&
                    !stage.value("is_dynamic", true) && stage.value("is_scale_quant", false) &&
                    stage.value("scale_type", std::string()) == "float32";
            };
            if (!valid_fp4_stage(weight[0], false) || !valid_fp8_scale_stage(weight[1]) ||
                !valid_fp4_stage(input[0], true) || !valid_fp8_scale_stage(input[1])) {
                throw std::runtime_error("unsupported native Quark sequential quantization contract");
            }
            group.name            = "quark-nvfp4";
            group.format          = llama_safetensors_quant_format::NVFP4_PACK;
            group.num_bits        = 4;
            group.group_size      = 16;
            group.input_quantized = true;
            group.input_dynamic   = true;
            group.modelopt        = true;
        } else if (dtype == "fp4") {
            const json input = require_json_value(global, "input_tensors", "Quark global_quant_config");
            const auto valid_mxfp4 = [](const json & spec, bool dynamic) {
                return spec.is_object() && spec.value("dtype", std::string()) == "fp4" &&
                    spec.value("qscheme", std::string()) == "per_group" &&
                    spec.value("group_size", 0U) == 32U && spec.value("ch_axis", 0) == -1 &&
                    spec.value("is_dynamic", !dynamic) == dynamic && !spec.value("is_scale_quant", true) &&
                    spec.value("scale_format", std::string()) == "e8m0" &&
                    spec.value("scale_type", std::string()) == "float";
            };
            if (!valid_mxfp4(weight, false) || !valid_mxfp4(input, true)) {
                throw std::runtime_error("unsupported native Quark MXFP4 quantization contract");
            }
            group.name            = "quark-mxfp4";
            group.format          = llama_safetensors_quant_format::MXFP4_PACK;
            group.num_bits        = 4;
            group.group_size      = 32;
            group.input_quantized = true;
            group.input_dynamic   = true;
            group.modelopt        = true; // Quark uses .weight rather than .weight_packed.
        } else if (dtype == "fp8_e4m3") {
            if (weight.value("is_dynamic", false)) {
                throw std::runtime_error("unsupported native Quark dynamic weight quantization contract");
            }
            const json input = require_json_value(global, "input_tensors", "Quark global_quant_config");
            const std::string weight_scheme =
                require_json_value(weight, "qscheme", "Quark weight config").get<std::string>();
            const std::string input_scheme = input.is_object() ?
                require_json_value(input, "qscheme", "Quark input config").get<std::string>() : std::string();
            const bool per_tensor = weight_scheme == "per_tensor" && input_scheme == "per_tensor";
            const bool per_token_channel = weight_scheme == "per_channel" && weight.value("ch_axis", -1) == 0 &&
                input_scheme == "per_channel" && input.value("ch_axis", -1) == 1 &&
                input.value("is_dynamic", false);
            if (!input.is_object() ||
                require_json_value(input, "dtype", "Quark input config").get<std::string>() != "fp8_e4m3" ||
                (!per_tensor && !per_token_channel)) {
                throw std::runtime_error("unsupported native Quark FP8 W8A8 quantization contract");
            }
            group.name            = per_token_channel ? "quark-fp8-ptpc" : "quark-fp8-w8a8";
            group.format          = per_token_channel ? llama_safetensors_quant_format::FP8_CHANNEL :
                                                        llama_safetensors_quant_format::FP8_TENSOR;
            group.num_bits        = 8;
            group.input_quantized = true;
            group.input_dynamic   = input.value("is_dynamic", false);
            group.modelopt        = !per_token_channel;
        } else if (dtype == "int8") {
            if (weight.value("is_dynamic", false)) {
                throw std::runtime_error("unsupported native Quark dynamic weight quantization contract");
            }
            const json input = require_json_value(global, "input_tensors", "Quark global_quant_config");
            const bool input_dynamic = input.value("is_dynamic", false);
            const std::string input_scheme =
                require_json_value(input, "qscheme", "Quark input config").get<std::string>();
            if (require_json_value(weight, "qscheme", "Quark weight config").get<std::string>() !=
                    "per_channel" || !weight.value("symmetric", false) || !input.is_object() ||
                require_json_value(input, "dtype", "Quark input config").get<std::string>() != "int8" ||
                input_scheme != (input_dynamic ? "per_channel" : "per_tensor")) {
                throw std::runtime_error("unsupported native Quark INT8 W8A8 quantization contract");
            }
            group.name            = "quark-int8-w8a8";
            group.format          = llama_safetensors_quant_format::INT8_CHANNEL;
            group.num_bits        = 8;
            group.input_quantized = true;
            group.input_dynamic   = input_dynamic;
            group.input_symmetric = input.value("symmetric", true);
            group.modelopt        = true;
        } else if (dtype == "uint4" || dtype == "int4") {
            const bool symmetric = require_json_value(weight, "symmetric", "Quark weight config").get<bool>();
            const uint32_t group_size =
                require_json_value(weight, "group_size", "Quark weight config").get<uint32_t>();
            if (symmetric != (dtype == "int4") || group_size < 32 || group_size % 32 != 0 ||
                require_json_value(weight, "qscheme", "Quark weight config").get<std::string>() != "per_group" ||
                weight.value("is_scale_quant", false) ||
                weight.value("scale_type", std::string("float")) != "float" ||
                !global.value("input_tensors", json()).is_null()) {
                throw std::runtime_error("unsupported native Quark W4A16 quantization contract");
            }
            group.name       = symmetric ? "quark-int4-w4a16" : "quark-uint4-w4a16";
            group.format     = llama_safetensors_quant_format::QUARK_W4A16;
            group.num_bits   = 4;
            group.group_size = group_size;
            group.symmetric  = symmetric;
        } else {
            throw std::runtime_error("unsupported native Quark quantization dtype '" + dtype + "'");
        }
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));

        if (quant.contains("exclude")) {
            const json & ignored = quant.at("exclude");
            if (!ignored.is_array()) {
                throw std::runtime_error("Quark quantization_config.exclude must be an array");
            }
            const auto glob_rule = [&](const std::string & target) {
                std::string regex = "re:";
                for (char c : target) {
                    if (c == '*') {
                        regex += ".*";
                    } else {
                        if (std::string_view(".^$|()[]{}+?\\").find(c) != std::string_view::npos) {
                            regex.push_back('\\');
                        }
                        regex.push_back(c);
                    }
                }
                regex.push_back('$');
                return make_rule(regex, 0);
            };
            for (const auto & target : ignored) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string module in Quark quantization_config.exclude");
                }
                result.ignore_.push_back(glob_rule(target.get<std::string>()));
            }
        }
        return result;
    }

    if (quant_method == "awq") {
        const uint32_t bits = quant.contains("bits") ? quant.at("bits").get<uint32_t>() :
                                                       require_json_value(quant, "w_bit", "quantization_config").get<uint32_t>();
        const int64_t group_size = quant.contains("group_size") ? quant.at("group_size").get<int64_t>() :
                                                                 require_json_value(quant, "q_group_size", "quantization_config").get<int64_t>();
        const std::string version = quant.value("version", std::string("gemm"));
        if (bits != 4 ||
            (group_size != -1 && (group_size < 32 || group_size % 32 != 0)) ||
            require_json_value(quant, "zero_point", "quantization_config").get<bool>() != true ||
            (version != "gemm" && version != "GEMM")) {
            throw std::runtime_error("unsupported native AWQ quantization contract");
        }

        llama_safetensors_quant_group group;
        group.name       = "awq-w4a16";
        group.format     = llama_safetensors_quant_format::AWQ_GROUP;
        group.group_size = group_size == -1 ? 0 : static_cast<uint32_t>(group_size);
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));

        if (quant.contains("modules_to_not_convert") && !quant.at("modules_to_not_convert").is_null()) {
            const json & ignored = quant.at("modules_to_not_convert");
            if (!ignored.is_array()) {
                throw std::runtime_error("quantization_config.modules_to_not_convert must be null or an array");
            }
            for (const auto & target : ignored) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string module in quantization_config.modules_to_not_convert");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    if (quant_method == "gptq" || quant_method == "auto-round") {
        const bool autoround = quant_method == "auto-round";
        const int64_t group_size = require_json_value(quant, "group_size", "quantization_config").get<int64_t>();
        const bool desc_act = autoround ? quant.value("desc_act", false) :
            require_json_value(quant, "desc_act", "quantization_config").get<bool>();
        const uint32_t bits = require_json_value(quant, "bits", "quantization_config").get<uint32_t>();
        if ((bits != 4 && bits != 8) || (bits == 8 && desc_act) ||
            (group_size != -1 && (group_size < 32 || group_size % 32 != 0)) ||
            (!autoround && quant.value("checkpoint_format", std::string("gptq")) != "gptq") ||
            (autoround && (desc_act || !quant.value("sym", true) ||
                           quant.value("data_type", std::string("int")) != "int" ||
                           quant.value("packing_format", std::string()) != "auto_round:auto_gptq")) ||
            quant.value("pack_dtype", std::string("int32")) != "int32") {
            throw std::runtime_error("unsupported native " +
                std::string(autoround ? "AutoRound" : "GPTQ") + " quantization contract");
        }

        llama_safetensors_quant_group group;
        group.name       = bits == 4 ? "gptq-w4a16" : "gptq-w8a16";
        group.format     = llama_safetensors_quant_format::GPTQ_GROUP;
        group.num_bits   = bits;
        group.group_size = group_size == -1 ? 0 : static_cast<uint32_t>(group_size);
        group.act_order  = desc_act && group_size != -1;
        result.groups_.push_back(std::move(group));
        if (autoround && quant.contains("extra_config")) {
            const json & extra = quant.at("extra_config");
            if (!extra.is_object()) {
                throw std::runtime_error("AutoRound extra_config must be an object");
            }
            const auto override_rule = [](const std::string & target) {
                // INC treats keys containing these characters as Python
                // search regexes; prefixing .* preserves that behavior under
                // this parser's deliberately anchored regex matcher.
                return target.find_first_of("*+?^$()[]{}|\\") == std::string::npos ?
                    target : "re:.*(?:" + target + ")";
            };
            for (const auto & [module, override_config] : extra.items()) {
                if (!override_config.is_object()) {
                    throw std::runtime_error("AutoRound module override must be an object");
                }
                const uint32_t override_bits = override_config.value("bits", bits);
                const int64_t override_group = override_config.value("group_size", group_size);
                const bool override_sym = override_config.value("sym", true);
                const std::string override_type = override_config.value(
                    "data_type", override_bits >= 16 ? std::string("float") : std::string("int"));
                const std::string target = override_rule(module);
                if (override_bits >= 16 || override_type == "float") {
                    result.ignore_.push_back(make_rule(target, 0));
                    continue;
                }
                if ((override_bits != 4 && override_bits != 8) || !override_sym ||
                    override_type != "int" ||
                    (override_group != -1 && (override_group < 32 || override_group % 32 != 0))) {
                    throw std::runtime_error("unsupported per-module AutoRound GPTQ override");
                }
                llama_safetensors_quant_group override_group_desc;
                override_group_desc.name       = "autoround-gptq:" + module;
                override_group_desc.format     = llama_safetensors_quant_format::GPTQ_GROUP;
                override_group_desc.num_bits   = override_bits;
                override_group_desc.group_size = override_group == -1 ? 0 : static_cast<uint32_t>(override_group);
                const uint32_t group_index = static_cast<uint32_t>(result.groups_.size());
                result.groups_.push_back(std::move(override_group_desc));
                result.rules_.push_back(make_rule(target, group_index));
            }
        }
        result.rules_.push_back(make_rule("re:.*", 0));
        return result;
    }

    if (quant_method == "eetq") {
        if (quant.value("bits", 8U) != 8U || quant.value("weights", std::string("int8")) != "int8" ||
            quant.value("zero_point", false)) {
            throw std::runtime_error("unsupported native EETQ quantization contract");
        }
        llama_safetensors_quant_group group;
        group.name       = "eetq-w8a16";
        group.format     = llama_safetensors_quant_format::EETQ_INT8;
        group.num_bits   = 8;
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));

        if (quant.contains("modules_to_not_convert") && !quant.at("modules_to_not_convert").is_null()) {
            const json & ignored = quant.at("modules_to_not_convert");
            if (!ignored.is_array()) {
                throw std::runtime_error("EETQ modules_to_not_convert must be null or an array");
            }
            for (const auto & target : ignored) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string EETQ module exclusion");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    if (quant_method == "hqq") {
        const json hqq = require_json_value(quant, "quant_config", "HQQ quantization_config");
        const json weights = require_json_value(hqq, "weight_quant_params", "HQQ quant_config");
        const uint32_t bits = require_json_value(weights, "nbits", "HQQ weight config").get<uint32_t>();
        const uint32_t group_size = require_json_value(weights, "group_size", "HQQ weight config").get<uint32_t>();
        const uint32_t axis = require_json_value(weights, "axis", "HQQ weight config").get<uint32_t>();
        if (bits != 4 || group_size < 32 || group_size % 32 != 0 || axis != 1 ||
            !require_json_value(weights, "channel_wise", "HQQ weight config").get<bool>() ||
            !require_json_value(weights, "round_zero", "HQQ weight config").get<bool>() ||
            weights.value("view_as_float", false) || weights.value("offload_meta", false) ||
            hqq.value("offload_meta", false) ||
            (hqq.contains("scale_quant_params") && !hqq.at("scale_quant_params").is_null()) ||
            (hqq.contains("zero_quant_params") && !hqq.at("zero_quant_params").is_null())) {
            throw std::runtime_error("unsupported native HQQ quantization contract");
        }
        llama_safetensors_quant_group group;
        group.name       = "hqq-int4";
        group.format     = llama_safetensors_quant_format::HQQ_INT4;
        group.num_bits   = bits;
        group.group_size = group_size;
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));
        if (quant.contains("skip_modules")) {
            const json & skipped = quant.at("skip_modules");
            if (!skipped.is_array()) {
                throw std::runtime_error("HQQ skip_modules must be an array");
            }
            for (const auto & target : skipped) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string HQQ skip module");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    if (quant_method == "quanto") {
        const json qmap = require_json_value(quant, "quantization_map", "Quanto quantization_config");
        if (!qmap.is_object() || qmap.empty()) {
            throw std::runtime_error("Quanto quantization_map must be a non-empty object");
        }
        for (const auto & [module, entry] : qmap.items()) {
            const std::string weights = entry.is_object() ?
                entry.value("weights", std::string()) : std::string();
            const bool quanto_fp8 = weights == "qfloat8" || weights == "qfloat8_e4m3fn";
            if ((weights != "qint4" && weights != "qint8" && !quanto_fp8) ||
                entry.value("activations", std::string("none")) != "none") {
                throw std::runtime_error("unsupported Quanto quantization contract for module '" + module + "'");
            }
            llama_safetensors_quant_group group;
            group.name         = "quanto-" + weights + ":" + module;
            group.format       = weights == "qint4" ? llama_safetensors_quant_format::QUANTO_INT4 :
                (weights == "qint8" ? llama_safetensors_quant_format::QUANTO_INT8 :
                                      llama_safetensors_quant_format::QUANTO_FP8);
            group.num_bits     = weights == "qint4" ? 4 : 8;
            group.target_shape = standard_projection_shape(root, module);
            if (weights == "qint4") {
                const uint32_t cols = static_cast<uint32_t>(group.target_shape[0]);
                group.group_size = cols <= 128 ? cols : 128;
                while (cols % group.group_size != 0 && group.group_size > 32) {
                    group.group_size -= 32;
                }
                if (group.group_size < 32 || cols % group.group_size != 0 || cols % 32 != 0) {
                    throw std::runtime_error("unsupported Quanto group geometry for module '" + module + "'");
                }
            }
            const uint32_t group_index = static_cast<uint32_t>(result.groups_.size());
            result.groups_.push_back(std::move(group));
            result.rules_.push_back(make_rule(module, group_index));
        }
        return result;
    }

    if (quant_method == "torchao") {
        const json qmap = require_json_value(quant, "quantization_map", "TorchAO quantization_config");
        if (!qmap.is_object() || qmap.empty()) {
            throw std::runtime_error("TorchAO quantization metadata must be a non-empty object");
        }
        for (const auto & [tensor_name, entry] : qmap.items()) {
            constexpr std::string_view suffix = ".weight";
            if (!entry.is_object() || tensor_name.size() <= suffix.size() ||
                tensor_name.compare(tensor_name.size() - suffix.size(), suffix.size(), suffix) != 0) {
                throw std::runtime_error("unsupported TorchAO tensor metadata for '" + tensor_name + "'");
            }
            const std::string type = entry.value("_type", std::string());
            const json block = entry.value("block_size", json());
            if (!block.is_array() || block.size() != 2 || !block[0].is_number_unsigned() ||
                !block[1].is_number_unsigned() || block[0].get<uint32_t>() != 1) {
                throw std::runtime_error("unsupported TorchAO block geometry for '" + tensor_name + "'");
            }
            llama_safetensors_quant_group group;
            group.name       = "torchao:" + tensor_name;
            group.group_size = block[1].get<uint32_t>();
            const std::string module = tensor_name.substr(0, tensor_name.size() - suffix.size());
            if (type == "Int4TilePackedTo4dTensor") {
                const json shape = entry.value("shape", json());
                if (!shape.is_array() || shape.size() != 2 || !shape[0].is_number_unsigned() ||
                    !shape[1].is_number_unsigned() || shape[0].get<uint64_t>() == 0 ||
                    shape[1].get<uint64_t>() == 0 || group.group_size < 32 || group.group_size % 32 != 0) {
                    throw std::runtime_error("unsupported TorchAO tiled-int4 geometry for '" + tensor_name + "'");
                }
                group.format       = llama_safetensors_quant_format::TORCHAO_INT4;
                group.num_bits     = 4;
                group.target_shape = {
                    static_cast<int64_t>(shape[1].get<uint64_t>()),
                    static_cast<int64_t>(shape[0].get<uint64_t>()),
                };
            } else if (type == "IntxUnpackedToInt8Tensor") {
                const json activation = entry.value("activation_quantization", json());
                if (entry.value("target_dtype", std::string()) != "torch.int8" ||
                    (entry.value("dtype", std::string()) != "torch.bfloat16" &&
                     entry.value("dtype", std::string()) != "torch.float16") ||
                    !activation.is_null() || group.group_size == 0 || group.group_size % 32 != 0) {
                    throw std::runtime_error("unsupported TorchAO unpacked-intx contract for '" + tensor_name + "'");
                }
                group.format   = llama_safetensors_quant_format::TORCHAO_INTX;
                group.num_bits = 8;
            } else {
                throw std::runtime_error("unsupported TorchAO tensor subclass '" + type + "' for '" + tensor_name + "'");
            }
            const uint32_t group_index = static_cast<uint32_t>(result.groups_.size());
            result.groups_.push_back(std::move(group));
            result.rules_.push_back(make_rule(module, group_index));
        }
        return result;
    }

    if (quant_method == "bitsandbytes") {
        const bool load_in_4bit = quant.value("load_in_4bit", quant.value("_load_in_4bit", false));
        const bool load_in_8bit = quant.value("load_in_8bit", quant.value("_load_in_8bit", false));
        if (load_in_4bit == load_in_8bit ||
            (load_in_4bit && quant.value("bnb_4bit_quant_storage", std::string("uint8")) != "uint8")) {
            throw std::runtime_error("unsupported native BitsAndBytes quantization contract");
        }
        if (load_in_8bit) {
            const float outlier_threshold = quant.value("llm_int8_threshold", 6.0f);
            if (!std::isfinite(outlier_threshold) || outlier_threshold < 0.0f) {
                throw std::runtime_error("BitsAndBytes llm_int8_threshold must be finite and non-negative");
            }
            llama_safetensors_quant_group group;
            group.name              = "bitsandbytes-int8";
            group.format            = llama_safetensors_quant_format::BNB_INT8;
            group.num_bits          = 8;
            group.input_quantized   = true;
            group.input_dynamic     = true;
            group.input_symmetric   = true;
            group.outlier_threshold = outlier_threshold;
            result.groups_.push_back(std::move(group));
            result.rules_.push_back(make_rule("re:.*", 0));

            if (quant.contains("llm_int8_skip_modules")) {
                const json & ignored = quant.at("llm_int8_skip_modules");
                if (!ignored.is_array() && !ignored.is_null()) {
                    throw std::runtime_error("quantization_config.llm_int8_skip_modules must be null or an array");
                }
                if (ignored.is_array()) {
                    for (const auto & target : ignored) {
                        if (!target.is_string()) {
                            throw std::runtime_error("non-string BitsAndBytes skip module");
                        }
                        result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
                    }
                }
            }
            return result;
        }
        const std::string quant_type =
            require_json_value(quant, "bnb_4bit_quant_type", "quantization_config").get<std::string>();
        if (quant_type != "nf4" && quant_type != "fp4") {
            throw std::runtime_error("unsupported BitsAndBytes 4-bit quantization type '" + quant_type + "'");
        }

        llama_safetensors_quant_group group;
        group.name       = "bitsandbytes-" + quant_type;
        group.format     = quant_type == "nf4" ? llama_safetensors_quant_format::BNB_NF4 :
                                                  llama_safetensors_quant_format::BNB_FP4;
        group.num_bits   = 4;
        group.group_size = 64;
        result.groups_.push_back(std::move(group));
        result.rules_.push_back(make_rule("re:.*", 0));

        if (quant.contains("llm_int8_skip_modules")) {
            const json & ignored = quant.at("llm_int8_skip_modules");
            if (!ignored.is_array()) {
                throw std::runtime_error("quantization_config.llm_int8_skip_modules must be an array");
            }
            for (const auto & target : ignored) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string BitsAndBytes skip module");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    if (quant_method == "modelopt") {
        const json nested = quant.contains("quantization") && quant.at("quantization").is_object() ?
            quant.at("quantization") : json::object();
        std::string quant_algo = quant.value("quant_algo", nested.value("quant_algo", std::string()));
        std::transform(quant_algo.begin(), quant_algo.end(), quant_algo.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (quant_algo == "NVFP4") {
            throw std::runtime_error(
                "native safetensors does not yet support ModelOpt NVFP4 W4A4 activations; "
                "use a W4A16_NVFP4 checkpoint");
        }
        if (quant_algo != "W4A16_NVFP4" && quant_algo != "MIXED_PRECISION" &&
            quant_algo != "FP8" && quant_algo != "FP8_PER_CHANNEL_PER_TOKEN" &&
            quant_algo != "FP8_PB_WO" && quant_algo != "MXFP8") {
            throw std::runtime_error("unsupported native ModelOpt quantization contract '" + quant_algo + "'");
        }

        const auto add_ignored = [&](const char * key) {
            if (!quant.contains(key)) {
                return;
            }
            const json & ignored = quant.at(key);
            if (!ignored.is_array()) {
                throw std::runtime_error(std::string("quantization_config.") + key + " must be an array");
            }
            for (const auto & target_json : ignored) {
                if (!target_json.is_string()) {
                    throw std::runtime_error(std::string("non-string target in quantization_config.") + key);
                }
                const std::string target = target_json.get<std::string>();
                std::string pattern = "re:";
                for (char c : target) {
                    if (c == '*') {
                        pattern += ".*";
                    } else {
                        if (std::string_view(".^$|()[]{}+?\\").find(c) != std::string_view::npos) {
                            pattern += '\\';
                        }
                        pattern += c;
                    }
                }
                result.ignore_.push_back(make_rule(pattern, 0));
            }
        };

        const auto add_targets = [&](const json & desc, uint32_t group_index, const std::string & context) {
            const json & targets = require_json_value(desc, "targets", context);
            if (!targets.is_array() || targets.empty()) {
                throw std::runtime_error(context + " has no targets");
            }
            for (const auto & target : targets) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string target in " + context);
                }
                result.rules_.push_back(make_rule(target.get<std::string>(), group_index));
            }
        };

        if (quant_algo == "W4A16_NVFP4") {
            if (quant.value("group_size", 16U) != 16U) {
                throw std::runtime_error("unsupported native ModelOpt W4A16_NVFP4 group size");
            }
            llama_safetensors_quant_group group;
            group.name            = "modelopt-w4a16-nvfp4";
            group.format          = llama_safetensors_quant_format::NVFP4_PACK;
            group.group_size      = 16;
            group.input_quantized = false;
            group.modelopt        = true;
            result.groups_.push_back(std::move(group));
            result.rules_.push_back(make_rule("re:.*", 0));
        } else if (quant_algo == "FP8" || quant_algo == "FP8_PER_CHANNEL_PER_TOKEN" ||
                   quant_algo == "FP8_PB_WO" || quant_algo == "MXFP8") {
            llama_safetensors_quant_group group;
            group.name     = "modelopt-" + quant_algo;
            group.modelopt = true;
            if (quant_algo == "FP8") {
                group.format          = llama_safetensors_quant_format::FP8_TENSOR;
                group.input_quantized = true;
                group.input_dynamic   = false;
            } else if (quant_algo == "FP8_PER_CHANNEL_PER_TOKEN") {
                group.format          = llama_safetensors_quant_format::FP8_CHANNEL;
                group.input_quantized = true;
                group.input_dynamic   = true;
            } else if (quant_algo == "FP8_PB_WO") {
                group.format          = llama_safetensors_quant_format::FP8_BLOCK;
                group.group_size      = 128;
                group.block_structure = { 128, 128 };
            } else {
                group.format          = llama_safetensors_quant_format::MXFP8;
                group.group_size      = 32;
                group.input_quantized = true;
                group.input_dynamic   = true;
            }
            result.groups_.push_back(std::move(group));
            result.rules_.push_back(make_rule("re:.*", 0));
        } else {
            const json groups = require_json_value(quant, "config_groups", "ModelOpt MIXED_PRECISION config");
            if (!groups.is_object() || groups.empty()) {
                throw std::runtime_error("ModelOpt MIXED_PRECISION config_groups must be a non-empty object");
            }
            for (const auto & [name, desc] : groups.items()) {
                const std::string context = "ModelOpt quantization group '" + name + "'";
                const json & weights = require_json_value(desc, "weights", context);
                const json & input = desc.contains("input_activations") ? desc.at("input_activations") : json(nullptr);
                const std::string type = require_json_value(weights, "type", context).get<std::string>();
                const uint32_t bits = require_json_value(weights, "num_bits", context).get<uint32_t>();
                const bool dynamic = require_json_value(weights, "dynamic", context).get<bool>();

                llama_safetensors_quant_group group;
                group.name     = name;
                group.modelopt = true;
                if (type == "float" && bits == 4 && !dynamic && weights.value("group_size", 0U) == 16U &&
                    input.is_null()) {
                    group.format          = llama_safetensors_quant_format::NVFP4_PACK;
                    group.group_size      = 16;
                    group.input_quantized = false;
                } else if (type == "float" && bits == 8 && !dynamic && input.is_object() &&
                           input.value("type", std::string()) == "float" &&
                           input.value("num_bits", 0U) == 8U && !input.value("dynamic", true)) {
                    group.format          = llama_safetensors_quant_format::FP8_TENSOR;
                    group.input_quantized = true;
                    group.input_dynamic   = false;
                } else {
                    throw std::runtime_error("unsupported " + context + " tensor contract");
                }
                const uint32_t group_index = result.groups_.size();
                result.groups_.push_back(std::move(group));
                add_targets(desc, group_index, context);
            }
        }
        add_ignored("ignore");
        add_ignored("exclude_modules");
        if (nested.contains("exclude_modules")) {
            const json & excluded = nested.at("exclude_modules");
            if (!excluded.is_array()) {
                throw std::runtime_error("quantization_config.quantization.exclude_modules must be an array");
            }
            for (const auto & target : excluded) {
                if (!target.is_string()) {
                    throw std::runtime_error("non-string target in quantization_config.quantization.exclude_modules");
                }
                result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
            }
        }
        return result;
    }

    if (quant_method != "compressed-tensors") {
        throw std::runtime_error("native safetensors does not support quant_method '" + quant_method + "'");
    }
    const std::string container_format =
        require_json_value(quant, "format", "quantization_config").get<std::string>();
    if (container_format != "mixed-precision" && container_format != "float-quantized" &&
        container_format != "int-quantized" && container_format != "pack-quantized" &&
        container_format != "naive-quantized" && container_format != "mxfp4-pack-quantized" &&
        container_format != "mxfp8-quantized") {
        throw std::runtime_error(
            "native safetensors does not support compressed-tensors format '" + container_format + "'");
    }

    const json config_groups = require_json_value(quant, "config_groups", "quantization_config");
    if (!config_groups.is_object() || config_groups.empty()) {
        throw std::runtime_error("quantization_config.config_groups must be a non-empty object");
    }

    for (const auto & [name, desc] : config_groups.items()) {
        const json        weights = require_json_value(desc, "weights", "quantization group '" + name + "'");
        const std::string type =
            require_json_value(weights, "type", "quantization group '" + name + "'").get<std::string>();
        std::string format = desc.value("format", container_format);
        if (format == "naive-quantized") {
            // The naive compressor is the storage-equivalent fallback behind
            // the older float/int aliases: it writes the quantized tensor and
            // its ordinary scale/zero-point sidecars without further packing.
            format = type == "float" ? "float-quantized" :
                     type == "int"   ? "int-quantized" : format;
        }
        const auto require_null = [&](const json & object, const char * key, const std::string & context) {
            if (!object.contains(key)) {
                return;
            }
            if (!object.at(key).is_null()) {
                throw std::runtime_error(context + " requires null '" + key + "'");
            }
        };
        const bool symmetric =
            require_json_value(weights, "symmetric", "quantization group '" + name + "'").get<bool>();
        const bool dynamic =
            require_json_value(weights, "dynamic", "quantization group '" + name + "'").get<bool>();
        const uint32_t                group_index = result.groups_.size();
        llama_safetensors_quant_group group;
        group.name = name;

        if (format == "pack-quantized") {
            const uint32_t num_bits =
                require_json_value(weights, "num_bits", "quantization group '" + name + "'").get<uint32_t>();
            const std::string strategy =
                require_json_value(weights, "strategy", "quantization group '" + name + "'").get<std::string>();
            const uint32_t group_size =
                require_json_value(weights, "group_size", "quantization group '" + name + "'").get<uint32_t>();
            const json actorder = require_json_value(weights, "actorder", "quantization group '" + name + "'");
            const std::string group_context = "quantization group '" + name + "'";
            const json & zp_dtype = weights.contains("zp_dtype") ? weights.at("zp_dtype") : json(nullptr);
            const json input = require_json_value(desc, "input_activations", "quantization group '" + name + "'");
            const bool supported_int4 = num_bits == 4 && group_size >= 32 && group_size % 32 == 0 &&
                ((symmetric && zp_dtype.is_null()) ||
                 (!symmetric && zp_dtype.is_string() && zp_dtype.get<std::string>() == "torch.int8"));
            const bool supported_int8 = num_bits == 8 && group_size == 128 && symmetric && zp_dtype.is_null();
            const bool supported_w4a8_fp8 = num_bits == 4 && group_size == 128 && symmetric &&
                zp_dtype.is_null() && actorder.is_null() && input.is_object();
            if (type != "int" || strategy != "group" || dynamic ||
                (!supported_int4 && !supported_int8 && !supported_w4a8_fp8) ||
                ((supported_int4 || supported_int8) &&
                 (!actorder.is_null() && (!actorder.is_string() || actorder.get<std::string>() != "static")))) {
                throw std::runtime_error("unsupported packed integer quantization in group '" + name + "'");
            }
            if (weights.contains("block_structure") && !weights.at("block_structure").is_null()) {
                throw std::runtime_error("quantization group '" + name + "' requires null 'block_structure'");
            }
            if (weights.contains("scale_dtype") && !weights.at("scale_dtype").is_null()) {
                throw std::runtime_error("quantization group '" + name + "' requires null 'scale_dtype'");
            }
            if (supported_w4a8_fp8) {
                if (!input.is_object() ||
                    require_json_value(input, "type", "input activations for group '" + name + "'").get<std::string>() != "float" ||
                    require_json_value(input, "num_bits", "input activations for group '" + name + "'").get<uint32_t>() != 8 ||
                    require_json_value(input, "strategy", "input activations for group '" + name + "'").get<std::string>() != "token" ||
                    require_json_value(input, "dynamic", "input activations for group '" + name + "'").get<bool>() != true ||
                    require_json_value(input, "symmetric", "input activations for group '" + name + "'").get<bool>() != true) {
                    throw std::runtime_error("unsupported W4A8-FP8 activation contract in group '" + name + "'");
                }
            } else if (!input.is_null()) {
                throw std::runtime_error("packed WNA16 group '" + name + "' requires null input_activations");
            }
            require_null(desc, "output_activations", "quantization group '" + name + "'");
            group.format     = supported_w4a8_fp8 ? llama_safetensors_quant_format::PACKED_INT4_FP8 :
                                                   llama_safetensors_quant_format::PACKED_INT;
            group.num_bits   = num_bits;
            group.group_size = group_size;
            group.symmetric  = symmetric;
            group.input_quantized = supported_w4a8_fp8;
            group.input_dynamic   = supported_w4a8_fp8;
        } else if (format == "mxfp4-pack-quantized" || format == "mxfp8-quantized") {
            const bool mxfp8 = format == "mxfp8-quantized";
            const uint32_t num_bits =
                require_json_value(weights, "num_bits", "quantization group '" + name + "'").get<uint32_t>();
            const std::string strategy =
                require_json_value(weights, "strategy", "quantization group '" + name + "'").get<std::string>();
            const uint32_t group_size =
                require_json_value(weights, "group_size", "quantization group '" + name + "'").get<uint32_t>();
            const std::string scale_dtype =
                require_json_value(weights, "scale_dtype", "quantization group '" + name + "'").get<std::string>();
            require_null(weights, "block_structure", "quantization group '" + name + "'");
            require_null(weights, "zp_dtype", "quantization group '" + name + "'");
            require_null(weights, "actorder", "quantization group '" + name + "'");
            if (type != "float" || num_bits != (mxfp8 ? 8u : 4u) || strategy != "group" || group_size != 32 ||
                !symmetric || dynamic || scale_dtype != "torch.uint8") {
                throw std::runtime_error("unsupported MXFP weight contract in group '" + name + "'");
            }

            const json input = require_json_value(desc, "input_activations", "quantization group '" + name + "'");
            if (require_json_value(input, "type", "input activations for group '" + name + "'").get<std::string>() != "float" ||
                require_json_value(input, "num_bits", "input activations for group '" + name + "'").get<uint32_t>() != (mxfp8 ? 8u : 4u) ||
                require_json_value(input, "strategy", "input activations for group '" + name + "'").get<std::string>() != "group" ||
                require_json_value(input, "group_size", "input activations for group '" + name + "'").get<uint32_t>() != 32 ||
                require_json_value(input, "symmetric", "input activations for group '" + name + "'").get<bool>() != true ||
                require_json_value(input, "dynamic", "input activations for group '" + name + "'").get<bool>() != true ||
                require_json_value(input, "scale_dtype", "input activations for group '" + name + "'").get<std::string>() != "torch.uint8" ||
                (input.contains("block_structure") && !input.at("block_structure").is_null()) ||
                (input.contains("zp_dtype") && !input.at("zp_dtype").is_null()) ||
                (input.contains("actorder") && !input.at("actorder").is_null())) {
                throw std::runtime_error("unsupported MXFP input activation contract in group '" + name + "'");
            }
            group.format          = mxfp8 ? llama_safetensors_quant_format::MXFP8 :
                                            llama_safetensors_quant_format::MXFP4_PACK;
            group.num_bits        = num_bits;
            group.group_size      = group_size;
            group.input_quantized = true;
            group.input_dynamic   = true;
        } else if (format == "nvfp4-pack-quantized") {
            const uint32_t num_bits =
                require_json_value(weights, "num_bits", "quantization group '" + name + "'").get<uint32_t>();
            const std::string strategy =
                require_json_value(weights, "strategy", "quantization group '" + name + "'").get<std::string>();
            const uint32_t group_size =
                require_json_value(weights, "group_size", "quantization group '" + name + "'").get<uint32_t>();
            const std::string scale_dtype =
                require_json_value(weights, "scale_dtype", "quantization group '" + name + "'").get<std::string>();
            const std::string actorder =
                require_json_value(weights, "actorder", "quantization group '" + name + "'").get<std::string>();
            require_null(weights, "block_structure", "quantization group '" + name + "'");
            require_null(weights, "zp_dtype", "quantization group '" + name + "'");
            if (type != "float" || num_bits != 4 || strategy != "tensor_group" || group_size != 16 ||
                !symmetric || dynamic || scale_dtype != "torch.float8_e4m3fn" || actorder != "static") {
                throw std::runtime_error("unsupported packed float quantization in group '" + name + "'");
            }
            group.format = llama_safetensors_quant_format::NVFP4_PACK;
            group.group_size = group_size;

            const json input = require_json_value(desc, "input_activations", "quantization group '" + name + "'");
            if (require_json_value(input, "type", "input activations for group '" + name + "'").get<std::string>() != "float" ||
                require_json_value(input, "num_bits", "input activations for group '" + name + "'").get<uint32_t>() != 4 ||
                require_json_value(input, "strategy", "input activations for group '" + name + "'").get<std::string>() != "tensor_group" ||
                require_json_value(input, "group_size", "input activations for group '" + name + "'").get<uint32_t>() != 16 ||
                require_json_value(input, "symmetric", "input activations for group '" + name + "'").get<bool>() != true ||
                require_json_value(input, "dynamic", "input activations for group '" + name + "'").get<std::string>() != "local" ||
                require_json_value(input, "scale_dtype", "input activations for group '" + name + "'").get<std::string>() != "torch.float8_e4m3fn" ||
                !require_json_value(input, "block_structure", "input activations for group '" + name + "'").is_null() ||
                !require_json_value(input, "zp_dtype", "input activations for group '" + name + "'").is_null() ||
                !require_json_value(input, "actorder", "input activations for group '" + name + "'").is_null()) {
                throw std::runtime_error("unsupported NVFP4 input activation contract in group '" + name + "'");
            }
        } else if (format == "float-quantized") {
            const uint32_t num_bits =
                require_json_value(weights, "num_bits", "quantization group '" + name + "'").get<uint32_t>();
            const std::string strategy =
                require_json_value(weights, "strategy", "quantization group '" + name + "'").get<std::string>();
            const bool grouped = strategy == "group";
            require_null(weights, "block_structure", "quantization group '" + name + "'");
            require_null(weights, "zp_dtype", "quantization group '" + name + "'");
            require_null(weights, "actorder", "quantization group '" + name + "'");
            if (type != "float" || num_bits != 8 ||
                (strategy != "channel" && strategy != "tensor" && !grouped) || !symmetric || dynamic ||
                (grouped &&
                 (require_json_value(weights, "group_size", "quantization group '" + name + "'").get<uint32_t>() != 32 ||
                  require_json_value(weights, "scale_dtype", "quantization group '" + name + "'").get<std::string>() !=
                      "torch.bfloat16")) ||
                (!grouped && weights.contains("group_size") && !weights.at("group_size").is_null()) ||
                (!grouped && weights.contains("scale_dtype") && !weights.at("scale_dtype").is_null())) {
                throw std::runtime_error("unsupported FP8 quantization in group '" + name + "'");
            }
            group.format = grouped ? llama_safetensors_quant_format::FP8_GROUP :
                strategy == "channel" ? llama_safetensors_quant_format::FP8_CHANNEL :
                                         llama_safetensors_quant_format::FP8_TENSOR;
            group.group_size = grouped ? 32 : 0;

            const json input = require_json_value(desc, "input_activations", "quantization group '" + name + "'");
            if (input.is_null() && !grouped) {
                group.input_quantized = false;
                group.input_dynamic = true;
            } else {
            if (!input.is_object()) {
                throw std::runtime_error("grouped FP8 requires dynamic group input activations in group '" + name + "'");
            }
            const std::string input_strategy =
                require_json_value(input, "strategy", "input activations for group '" + name + "'").get<std::string>();
            const bool input_dynamic =
                require_json_value(input, "dynamic", "input activations for group '" + name + "'").get<bool>();
            if (require_json_value(input, "type", "input activations for group '" + name + "'").get<std::string>() != "float" ||
                require_json_value(input, "num_bits", "input activations for group '" + name + "'").get<uint32_t>() != 8 ||
                require_json_value(input, "symmetric", "input activations for group '" + name + "'").get<bool>() != true ||
                !(grouped ?
                    (input_dynamic && input_strategy == "group" &&
                     require_json_value(input, "group_size", "input activations for group '" + name + "'").get<uint32_t>() == 32 &&
                     require_json_value(input, "scale_dtype", "input activations for group '" + name + "'").get<std::string>() ==
                         "torch.bfloat16") :
                    ((input_dynamic && input_strategy == "token") || (!input_dynamic && input_strategy == "tensor"))) ||
                (!grouped && input.contains("group_size") && !input.at("group_size").is_null()) ||
                (input.contains("block_structure") && !input.at("block_structure").is_null()) ||
                (!grouped && input.contains("scale_dtype") && !input.at("scale_dtype").is_null()) ||
                (input.contains("zp_dtype") && !input.at("zp_dtype").is_null()) ||
                (input.contains("actorder") && !input.at("actorder").is_null())) {
                throw std::runtime_error("unsupported FP8 input activation contract in group '" + name + "'");
            }
            group.input_quantized = true;
            group.input_dynamic = input_dynamic;
            }
        } else if (format == "int-quantized") {
            const uint32_t num_bits =
                require_json_value(weights, "num_bits", "quantization group '" + name + "'").get<uint32_t>();
            const std::string strategy =
                require_json_value(weights, "strategy", "quantization group '" + name + "'").get<std::string>();
            require_null(weights, "block_structure", "quantization group '" + name + "'");
            const json actorder = weights.contains("actorder") ? weights.at("actorder") : json(nullptr);
            const bool calibration_only_actorder = actorder.is_string() &&
                (actorder.get<std::string>() == "static" || actorder.get<std::string>() == "weight");
            if (!actorder.is_null() && !calibration_only_actorder) {
                throw std::runtime_error(
                    "unsupported activation-quantized weight ordering in group '" + name + "'");
            }
            const json group_size_json = require_json_value(
                weights, "group_size", "quantization group '" + name + "'");
            const bool w8a8 = num_bits == 8 && strategy == "channel" && group_size_json.is_null();
            const bool w4a8 = num_bits == 4 && strategy == "group" &&
                group_size_json.is_number_unsigned() && group_size_json.get<uint32_t>() == 128;
            if (type != "int" || !symmetric || dynamic || (!w8a8 && !w4a8)) {
                throw std::runtime_error("unsupported integer activation-quantized weight contract in group '" + name + "'");
            }
            group.format     = w8a8 ? llama_safetensors_quant_format::INT8_CHANNEL :
                                     llama_safetensors_quant_format::INT4_GROUP;
            group.num_bits   = num_bits;
            group.group_size = w4a8 ? 128 : 0;

            const json input = require_json_value(desc, "input_activations", "quantization group '" + name + "'");
            const std::string input_strategy =
                require_json_value(input, "strategy", "input activations for group '" + name + "'").get<std::string>();
            const bool input_dynamic =
                require_json_value(input, "dynamic", "input activations for group '" + name + "'").get<bool>();
            const bool input_symmetric =
                require_json_value(input, "symmetric", "input activations for group '" + name + "'").get<bool>();
            const json & input_zp_dtype = input.contains("zp_dtype") ? input.at("zp_dtype") : json(nullptr);
            const bool supported_symmetric_input = input_symmetric && input_zp_dtype.is_null();
            const bool supported_asymmetric_input = !input_dynamic && input_strategy == "tensor" &&
                !input_symmetric && input_zp_dtype.is_string() &&
                input_zp_dtype.get<std::string>() == "torch.int8";
            if (require_json_value(input, "type", "input activations for group '" + name + "'").get<std::string>() != "int" ||
                require_json_value(input, "num_bits", "input activations for group '" + name + "'").get<uint32_t>() != 8 ||
                !((input_dynamic && input_strategy == "token") || (!input_dynamic && input_strategy == "tensor")) ||
                (!supported_symmetric_input && !supported_asymmetric_input) ||
                (input.contains("group_size") && !input.at("group_size").is_null()) ||
                (input.contains("block_structure") && !input.at("block_structure").is_null()) ||
                (input.contains("actorder") && !input.at("actorder").is_null())) {
                throw std::runtime_error("unsupported W8A8 input activation contract in group '" + name + "'");
            }
            group.input_dynamic = input_dynamic;
            group.input_quantized = true;
            group.input_symmetric = input_symmetric;
        } else {
            throw std::runtime_error("unsupported compressed-tensors weight type '" + type + "'");
        }

        const json targets = require_json_value(desc, "targets", "quantization group '" + name + "'");
        if (!targets.is_array() || targets.empty()) {
            throw std::runtime_error("quantization group '" + name + "' has no targets");
        }
        result.groups_.push_back(std::move(group));
        for (const auto & target : targets) {
            if (!target.is_string()) {
                throw std::runtime_error("non-string target in quantization group '" + name + "'");
            }
            const std::string declared_target = target.get<std::string>();
            // compressed-tensors uses module class names as selectors in
            // older single-format checkpoints. Importers ask this registry
            // only about projection modules, so Linear is their catch-all.
            const std::string target_name = declared_target == "Linear" ? "re:.*" : declared_target;
            const auto existing = std::find_if(result.rules_.begin(), result.rules_.end(), [&](const rule & item) {
                return item.target == target_name;
            });
            if (existing == result.rules_.end()) {
                result.rules_.push_back(make_rule(target_name, group_index));
            } else {
                // Python's target->scheme dictionary keeps the target's first
                // insertion position while a later duplicate replaces its value.
                existing->group = group_index;
            }
        }
    }

    if (quant.contains("ignore")) {
        const json & ignored = quant.at("ignore");
        if (!ignored.is_array()) {
            throw std::runtime_error("quantization_config.ignore must be an array");
        }
        for (const auto & target : ignored) {
            if (!target.is_string()) {
                throw std::runtime_error("non-string target in quantization_config.ignore");
            }
            result.ignore_.push_back(make_rule(target.get<std::string>(), 0));
        }
    }

    return result;
}

bool llama_safetensors_quant_config::ignored(const std::string & module_name) const {
    return std::any_of(ignore_.begin(), ignore_.end(),
                       [&](const rule & candidate) { return rule_matches(candidate, module_name); });
}

const llama_safetensors_quant_group * llama_safetensors_quant_config::match(const std::string & module_name) const {
    if (ignored(module_name)) {
        return nullptr;
    }
    for (const rule & candidate : rules_) {
        if (rule_matches(candidate, module_name)) {
            return &groups_.at(candidate.group);
        }
    }
    return nullptr;
}

llama_safetensors_registry llama_safetensors_registry::load(
        const std::filesystem::path & model_dir, llama_safetensors_io_mode io_mode) {
    llama_safetensors_registry result;
    const auto                 index_path = model_dir / "model.safetensors.index.json";

    std::map<std::string, std::string> expected_shards;
    std::set<std::string>              shard_names;
    if (std::filesystem::is_regular_file(index_path)) {
        const json index = llama_safetensors_read_json(index_path);
        if (!index.is_object() || !index.contains("weight_map") || !index.at("weight_map").is_object()) {
            throw std::runtime_error("safetensors index is missing an object-valued weight_map");
        }
        for (const auto & [tensor_name, shard_json] : index.at("weight_map").items()) {
            if (!shard_json.is_string()) {
                throw std::runtime_error("non-string shard name for indexed tensor '" + tensor_name + "'");
            }
            const std::string shard_name = shard_json.get<std::string>();
            expected_shards.emplace(tensor_name, shard_name);
            shard_names.insert(shard_name);
        }
    } else {
        const auto single = model_dir / "model.safetensors";
        if (!std::filesystem::is_regular_file(single)) {
            throw std::runtime_error("model directory has neither model.safetensors nor model.safetensors.index.json");
        }
        shard_names.insert("model.safetensors");
    }

    for (const std::string & shard_name : shard_names) {
        const uint32_t shard_index = result.shards_.size();
        parsed_shard   parsed      = parse_shard(checked_shard_path(model_dir, shard_name), shard_index);
        result.shards_.push_back(std::move(parsed.shard));

        for (const auto & [key, value] : parsed.metadata) {
            const auto [it, inserted] = result.metadata_.emplace(key, value);
            if (!inserted && it->second != value) {
                throw std::runtime_error("conflicting safetensors metadata value for '" + key + "'");
            }
        }

        for (auto & tensor : parsed.tensors) {
            if (!expected_shards.empty()) {
                const auto expected = expected_shards.find(tensor.name);
                if (expected == expected_shards.end()) {
                    throw std::runtime_error("tensor '" + tensor.name +
                                             "' is present in a shard but absent from weight_map");
                }
                if (expected->second != shard_name) {
                    throw std::runtime_error("weight_map assigns tensor '" + tensor.name + "' to the wrong shard");
                }
            }
            if (!result.tensor_index_.emplace(tensor.name, result.tensors_.size()).second) {
                throw std::runtime_error("duplicate safetensors tensor '" + tensor.name + "'");
            }
            result.tensors_.push_back(std::move(tensor));
        }
    }

    if (!expected_shards.empty() && result.tensor_index_.size() != expected_shards.size()) {
        for (const auto & [name, shard] : expected_shards) {
            (void) shard;
            if (result.tensor_index_.find(name) == result.tensor_index_.end()) {
                throw std::runtime_error("weight_map tensor '" + name + "' is missing from its shard");
            }
        }
    }

    result.files_.reserve(result.shards_.size());
    for (const llama_safetensors_shard & shard : result.shards_) {
        result.files_.push_back(std::make_unique<llama_file>(
            shard.path.string().c_str(), "rb", io_mode == llama_safetensors_io_mode::DIRECT));
    }
    if (io_mode == llama_safetensors_io_mode::MMAP && llama_mmap::SUPPORTED) {
        result.mappings_.reserve(result.shards_.size());
        for (const auto & file : result.files_) {
            result.mappings_.push_back(std::make_unique<llama_mmap>(file.get(), 0, false));
        }
    }

    return result;
}

const llama_safetensors_tensor * llama_safetensors_registry::find(const std::string & name) const {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

const std::string * llama_safetensors_registry::metadata(const std::string & key) const {
    const auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}

const uint8_t * llama_safetensors_registry::data(const llama_safetensors_tensor & tensor) const {
    if (tensor.shard >= shards_.size() || tensor.shard >= mappings_.size()) {
        return nullptr;
    }
    const llama_safetensors_shard & shard = shards_[tensor.shard];
    if (tensor.offset > shard.file_size || tensor.size > shard.file_size - tensor.offset) {
        throw std::runtime_error("safetensors tensor '" + tensor.name + "' is outside its shard");
    }
    return static_cast<const uint8_t *>(mappings_[tensor.shard]->addr()) + tensor.offset;
}

void llama_safetensors_registry::read_into(
        const llama_safetensors_tensor & tensor,
        uint64_t offset,
        void * destination,
        size_t size) const {
    if (tensor.shard >= shards_.size() || offset > tensor.size || size > tensor.size - offset) {
        throw std::runtime_error("invalid read range for safetensors tensor '" + tensor.name + "'");
    }
    if (const uint8_t * mapped = data(tensor)) {
        std::memcpy(destination, mapped + offset, size);
        return;
    }

    llama_file & file = *files_.at(tensor.shard);
    file.seek(tensor.offset + offset, SEEK_SET);
    file.read_raw(destination, size);
}

std::vector<uint8_t> llama_safetensors_registry::read(const llama_safetensors_tensor & tensor) const {
    if (tensor.shard >= shards_.size()) {
        throw std::runtime_error("invalid shard index for safetensors tensor '" + tensor.name + "'");
    }
    if (tensor.size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("safetensors tensor '" + tensor.name + "' is too large for this host");
    }
    std::vector<uint8_t> result(static_cast<size_t>(tensor.size));
    read_into(tensor, 0, result.data(), result.size());
    return result;
}

const std::vector<llama_safetensors_shard> & llama_safetensors_registry::shards() const {
    return shards_;
}

const std::vector<llama_safetensors_tensor> & llama_safetensors_registry::tensors() const {
    return tensors_;
}
