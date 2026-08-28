#include "llama-safetensors.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <stdexcept>

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

llama_safetensors_quant_config llama_safetensors_quant_config::load(const std::filesystem::path & model_dir) {
    return from_json(llama_safetensors_read_json(model_dir / "config.json"));
}

llama_safetensors_quant_config llama_safetensors_quant_config::from_json(const llama_safetensors_json & root) {
    llama_safetensors_quant_config result;
    const json                     quant = require_json_value(root, "quantization_config", "config.json");
    const std::string quant_method =
        require_json_value(quant, "quant_method", "quantization_config").get<std::string>();

    if (quant_method == "fp8") {
        const json block = require_json_value(quant, "weight_block_size", "quantization_config");
        if (require_json_value(quant, "fmt", "quantization_config").get<std::string>() != "e4m3" ||
            require_json_value(quant, "activation_scheme", "quantization_config").get<std::string>() != "dynamic" ||
            !block.is_array() || block.size() != 2 || block[0].get<uint32_t>() != 128 ||
            block[1].get<uint32_t>() != 128) {
            throw std::runtime_error("unsupported native FP8 quantization contract");
        }

        llama_safetensors_quant_group group;
        group.name            = "fp8-block-128x128";
        group.format          = llama_safetensors_quant_format::FP8_BLOCK;
        group.group_size      = 128;
        group.block_structure = { 128, 128 };
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

    if (quant_method != "compressed-tensors") {
        throw std::runtime_error("native safetensors does not support quant_method '" + quant_method + "'");
    }
    if (require_json_value(quant, "format", "quantization_config").get<std::string>() != "mixed-precision") {
        throw std::runtime_error("native safetensors requires compressed-tensors format 'mixed-precision'");
    }

    const json config_groups = require_json_value(quant, "config_groups", "quantization_config");
    if (!config_groups.is_object() || config_groups.empty()) {
        throw std::runtime_error("quantization_config.config_groups must be a non-empty object");
    }

    for (const auto & [name, desc] : config_groups.items()) {
        const json        weights = require_json_value(desc, "weights", "quantization group '" + name + "'");
        const std::string format =
            require_json_value(desc, "format", "quantization group '" + name + "'").get<std::string>();
        const std::string type =
            require_json_value(weights, "type", "quantization group '" + name + "'").get<std::string>();
        const auto require_null = [&](const char * key) {
            if (!require_json_value(weights, key, "quantization group '" + name + "'").is_null()) {
                throw std::runtime_error("quantization group '" + name + "' requires null '" + key + "'");
            }
        };
        const bool symmetric =
            require_json_value(weights, "symmetric", "quantization group '" + name + "'").get<bool>();
        const bool dynamic =
            require_json_value(weights, "dynamic", "quantization group '" + name + "'").get<bool>();
        const uint32_t                group_index = result.groups_.size();
        llama_safetensors_quant_group group;
        group.name = name;

        if (format == "nvfp4-pack-quantized") {
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
            require_null("block_structure");
            require_null("zp_dtype");
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
            require_null("group_size");
            require_null("block_structure");
            require_null("scale_dtype");
            require_null("zp_dtype");
            require_null("actorder");
            if (type != "float" || num_bits != 8 || strategy != "channel" || !symmetric || dynamic) {
                throw std::runtime_error("unsupported FP8 quantization in group '" + name + "'");
            }
            group.format = llama_safetensors_quant_format::FP8_CHANNEL;

            const json input = require_json_value(desc, "input_activations", "quantization group '" + name + "'");
            if (require_json_value(input, "type", "input activations for group '" + name + "'").get<std::string>() != "float" ||
                require_json_value(input, "num_bits", "input activations for group '" + name + "'").get<uint32_t>() != 8 ||
                require_json_value(input, "strategy", "input activations for group '" + name + "'").get<std::string>() != "token" ||
                require_json_value(input, "symmetric", "input activations for group '" + name + "'").get<bool>() != true ||
                require_json_value(input, "dynamic", "input activations for group '" + name + "'").get<bool>() != true ||
                !require_json_value(input, "group_size", "input activations for group '" + name + "'").is_null() ||
                !require_json_value(input, "block_structure", "input activations for group '" + name + "'").is_null() ||
                !require_json_value(input, "scale_dtype", "input activations for group '" + name + "'").is_null() ||
                !require_json_value(input, "zp_dtype", "input activations for group '" + name + "'").is_null() ||
                !require_json_value(input, "actorder", "input activations for group '" + name + "'").is_null()) {
                throw std::runtime_error("unsupported FP8 input activation contract in group '" + name + "'");
            }
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
            const std::string target_name = target.get<std::string>();
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

llama_safetensors_registry llama_safetensors_registry::load(const std::filesystem::path & model_dir) {
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

    return result;
}

const llama_safetensors_tensor * llama_safetensors_registry::find(const std::string & name) const {
    const auto it = tensor_index_.find(name);
    return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

std::vector<uint8_t> llama_safetensors_registry::read(const llama_safetensors_tensor & tensor) const {
    if (tensor.shard >= shards_.size()) {
        throw std::runtime_error("invalid shard index for safetensors tensor '" + tensor.name + "'");
    }
    const llama_safetensors_shard & shard = shards_[tensor.shard];
    if (tensor.offset > shard.file_size || tensor.size > shard.file_size - tensor.offset) {
        throw std::runtime_error("safetensors tensor '" + tensor.name + "' is outside its shard");
    }
    if (tensor.size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("safetensors tensor '" + tensor.name + "' is too large for this host");
    }

    std::ifstream in(shard.path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open safetensors shard '" + shard.path.string() + "'");
    }
    in.seekg(static_cast<std::streamoff>(tensor.offset));
    if (!in) {
        throw std::runtime_error("failed to seek to safetensors tensor '" + tensor.name + "'");
    }

    std::vector<uint8_t> result(static_cast<size_t>(tensor.size));
    in.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(result.size()));
    if (!in) {
        throw std::runtime_error("failed to read safetensors tensor '" + tensor.name + "'");
    }
    return result;
}

const std::vector<llama_safetensors_shard> & llama_safetensors_registry::shards() const {
    return shards_;
}

const std::vector<llama_safetensors_tensor> & llama_safetensors_registry::tensors() const {
    return tensors_;
}
