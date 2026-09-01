#include "llama-safetensors-quant.h"
#include "llama-safetensors-tensor.h"
#include "llama.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

const llama_safetensors_tensor & require_tensor(
        const llama_safetensors_registry & registry, const std::string & name) {
    const llama_safetensors_tensor * tensor = registry.find(name);
    if (tensor == nullptr) {
        throw std::runtime_error("required quantization tensor is missing: '" + name + "'");
    }
    return *tensor;
}

void require_positive_f32_scalar(
        const llama_safetensors_registry & registry, const std::string & name) {
    const llama_safetensors_tensor & desc = require_tensor(registry, name);
    if (desc.dtype != llama_safetensors_dtype::F32 || desc.size != sizeof(float)) {
        throw std::runtime_error("quantization scalar must be one F32 value: '" + name + "'");
    }
    const std::vector<uint8_t> raw = registry.read(desc);
    float value;
    std::memcpy(&value, raw.data(), sizeof(value));
    if (!(value > 0.0f) || !std::isfinite(value)) {
        throw std::runtime_error("quantization scalar must be finite and positive: '" + name + "'");
    }
}

std::array<uint64_t, 2> read_weight_shape(
        const llama_safetensors_registry & registry, const std::string & module) {
    const std::string shape_name = module + ".weight_shape";
    const auto & desc = require_tensor(registry, shape_name);
    if (desc.dtype != llama_safetensors_dtype::I64 || desc.shape != std::vector<uint64_t>({ 2 }) ||
        desc.size != 2 * sizeof(int64_t)) {
        throw std::runtime_error("invalid packed integer weight_shape for '" + module + "'");
    }
    const std::vector<uint8_t> raw = registry.read(desc);
    std::array<uint64_t, 2> result;
    for (size_t i = 0; i < result.size(); ++i) {
        int64_t value;
        std::memcpy(&value, raw.data() + i * sizeof(value), sizeof(value));
        if (value <= 0) {
            throw std::runtime_error("packed integer weight_shape must be positive for '" + module + "'");
        }
        result[i] = static_cast<uint64_t>(value);
    }
    return result;
}

std::array<uint64_t, 2> read_i64_shape(
        const llama_safetensors_registry & registry,
        const std::string & name,
        const char * context) {
    const auto & desc = require_tensor(registry, name);
    if (desc.dtype != llama_safetensors_dtype::I64 || desc.shape != std::vector<uint64_t>({ 2 }) ||
        desc.size != 2 * sizeof(int64_t)) {
        throw std::runtime_error(std::string("invalid ") + context + " shape tensor '" + name + "'");
    }
    const std::vector<uint8_t> raw = registry.read(desc);
    std::array<uint64_t, 2> result;
    for (size_t i = 0; i < result.size(); ++i) {
        int64_t value;
        std::memcpy(&value, raw.data() + i * sizeof(value), sizeof(value));
        if (value <= 0) {
            throw std::runtime_error(std::string(context) + " shape must be positive: '" + name + "'");
        }
        result[i] = static_cast<uint64_t>(value);
    }
    return result;
}

int32_t read_i32_scalar(
        const llama_safetensors_registry & registry,
        const std::string & name,
        const char * context) {
    const auto & desc = require_tensor(registry, name);
    if (desc.dtype != llama_safetensors_dtype::I32 || !desc.shape.empty() || desc.size != sizeof(int32_t)) {
        throw std::runtime_error(std::string("invalid ") + context + " scalar '" + name + "'");
    }
    const std::vector<uint8_t> raw = registry.read(desc);
    int32_t result;
    std::memcpy(&result, raw.data(), sizeof(result));
    return result;
}

uint8_t read_u8_scalar(
        const llama_safetensors_registry & registry,
        const std::string & name,
        const char * context) {
    const auto & desc = require_tensor(registry, name);
    if (desc.dtype != llama_safetensors_dtype::U8 || !desc.shape.empty() || desc.size != 1) {
        throw std::runtime_error(std::string("invalid ") + context + " scalar '" + name + "'");
    }
    return registry.read(desc).front();
}

float load_bf16(const uint8_t * source) {
    uint16_t bits16;
    std::memcpy(&bits16, source, sizeof(bits16));
    const uint32_t bits32 = uint32_t(bits16) << 16;
    float value;
    std::memcpy(&value, &bits32, sizeof(value));
    return value;
}

std::vector<int64_t> reversed_shape(const llama_safetensors_tensor & tensor) {
    std::vector<int64_t> result;
    result.reserve(tensor.shape.size());
    for (auto it = tensor.shape.rbegin(); it != tensor.shape.rend(); ++it) {
        if (*it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("source tensor dimension exceeds runtime limits: '" + tensor.name + "'");
        }
        if (*it != 1 || tensor.shape.size() == 1) {
            result.push_back(static_cast<int64_t>(*it));
        }
    }
    if (result.empty()) {
        result.push_back(1);
    }
    return result;
}

std::vector<uint8_t> transpose_2d(
        const std::vector<uint8_t> & source, size_t rows, size_t cols, size_t element_size) {
    if (source.size() != rows * cols * element_size) {
        throw std::runtime_error("quantization scale transpose shape mismatch");
    }
    std::vector<uint8_t> result(source.size());
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            std::memcpy(result.data() + (col * rows + row) * element_size,
                        source.data() + (row * cols + col) * element_size, element_size);
        }
    }
    return result;
}

class source_bytes {
  public:
    source_bytes(
            const llama_safetensors_registry & registry,
            const llama_safetensors_tensor & tensor) :
        data_(registry.data(tensor)) {
        if (data_ == nullptr) {
            owned_ = registry.read(tensor);
            data_  = owned_.data();
        }
    }

    const uint8_t * data() const {
        return data_;
    }

  private:
    const uint8_t *      data_ = nullptr;
    std::vector<uint8_t> owned_;
};

struct bnb_state {
    std::string          quant_type;
    uint32_t             block_size = 0;
    std::vector<int64_t> shape;
    uint32_t             nested_block_size = 0;
    float                nested_offset = 0.0f;
};

bnb_state read_bnb_state(
        const llama_safetensors_registry & registry,
        const std::string & module,
        const std::string & quant_type) {
    const std::string state_name = module + ".weight.quant_state.bitsandbytes__" + quant_type;
    const auto & desc = require_tensor(registry, state_name);
    if (desc.dtype != llama_safetensors_dtype::U8 || desc.shape.size() != 1) {
        throw std::runtime_error("invalid BitsAndBytes quantization-state tensor for '" + module + "'");
    }
    const std::vector<uint8_t> bytes = registry.read(desc);
    llama_safetensors_json json;
    try {
        json = llama_safetensors_json::parse(bytes.begin(), bytes.end());
    } catch (const llama_safetensors_json::exception & e) {
        throw std::runtime_error("invalid BitsAndBytes quantization state for '" + module + "': " + e.what());
    }

    bnb_state result;
    result.quant_type = json.value("quant_type", std::string());
    result.block_size = json.value("blocksize", 0U);
    result.nested_block_size = json.value("nested_blocksize", 0U);
    result.nested_offset = json.value("nested_offset", 0.0f);
    if (!json.contains("shape") || !json.at("shape").is_array()) {
        throw std::runtime_error("BitsAndBytes quantization state has no logical shape for '" + module + "'");
    }
    for (const auto & dim : json.at("shape")) {
        if (!dim.is_number_unsigned() || dim.get<uint64_t>() == 0 ||
            dim.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("invalid BitsAndBytes logical shape for '" + module + "'");
        }
        result.shape.push_back(static_cast<int64_t>(dim.get<uint64_t>()));
    }
    if (result.quant_type != quant_type || result.block_size == 0 || result.shape.size() != 2 ||
        !std::isfinite(result.nested_offset)) {
        throw std::runtime_error("unsupported BitsAndBytes quantization state for '" + module + "'");
    }
    return result;
}

llama_safetensors_json effective_quant_config(
        const llama_safetensors_json & config,
        const llama_safetensors_registry & registry) {
    if (config.contains("quantization_config")) {
        return config;
    }
    const std::string * metadata = registry.metadata("quantization");
    if (metadata == nullptr) {
        return config;
    }
    llama_safetensors_json quantization_map;
    try {
        quantization_map = llama_safetensors_json::parse(*metadata);
    } catch (const llama_safetensors_json::exception & error) {
        throw std::runtime_error(std::string("invalid TorchAO safetensors quantization metadata: ") + error.what());
    }
    if (!quantization_map.is_object() || quantization_map.empty()) {
        throw std::runtime_error("TorchAO safetensors quantization metadata must be a non-empty object");
    }
    llama_safetensors_json result = config;
    result["quantization_config"] = {
        { "quant_method", "torchao" },
        { "quantization_map", std::move(quantization_map) },
    };
    return result;
}

}  // namespace

llama_safetensors_quant_adapters::llama_safetensors_quant_adapters(
        const llama_safetensors_json & config,
        const llama_safetensors_registry & registry) :
    registry_(registry),
    config_(llama_safetensors_quant_config::from_json(effective_quant_config(config, registry))),
    has_quantization_config_(config.contains("quantization_config") || registry.metadata("quantization") != nullptr) {
    validate();
}

const llama_safetensors_quant_group * llama_safetensors_quant_adapters::match(
        const std::string & module) const {
    return config_.match(module);
}

bool llama_safetensors_quant_adapters::applies(const std::string & module) const {
    const llama_safetensors_quant_group * group = match(module);
    return group != nullptr && format_applies(module, *group);
}

bool llama_safetensors_quant_adapters::format_applies(
        const std::string & module,
        const llama_safetensors_quant_group & group) const {
    switch (group.format) {
        case llama_safetensors_quant_format::NVFP4_PACK:
            if (group.modelopt) {
                const auto * weight = registry_.find(module + ".weight");
                return weight != nullptr && weight->dtype == llama_safetensors_dtype::U8;
            }
            [[fallthrough]];
        case llama_safetensors_quant_format::MXFP4_PACK:
            if (group.hf_mxfp4) {
                const auto * weight = registry_.find(module + ".weight");
                return weight != nullptr && weight->dtype == llama_safetensors_dtype::I8;
            }
            return registry_.find(module + (group.modelopt ? ".weight" : ".weight_packed")) != nullptr;
        case llama_safetensors_quant_format::MXFP8:
        case llama_safetensors_quant_format::FP8_GROUP: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::F8_E4M3;
        }
        case llama_safetensors_quant_format::AWQ_GROUP:
        case llama_safetensors_quant_format::GPTQ_GROUP:
            return registry_.find(module + ".qweight") != nullptr;
        case llama_safetensors_quant_format::QUARK_W4A16: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::I32;
        }
        case llama_safetensors_quant_format::EETQ_INT8: {
            const auto * weight = registry_.find(module + ".qweight");
            if (weight == nullptr) {
                weight = registry_.find(module + ".weight");
            }
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::I8 &&
                registry_.find(module + ".weight_scales") != nullptr;
        }
        case llama_safetensors_quant_format::QUANTO_INT4:
            return registry_.find(module + ".weight._data._data") != nullptr;
        case llama_safetensors_quant_format::QUANTO_INT8:
            return registry_.find(module + ".weight._data") != nullptr;
        case llama_safetensors_quant_format::QUANTO_FP8:
            return registry_.find(module + ".weight._data") != nullptr;
        case llama_safetensors_quant_format::TORCHAO_INT4:
        case llama_safetensors_quant_format::TORCHAO_INTX:
            return registry_.find(module + ".weight.__qdata") != nullptr;
        case llama_safetensors_quant_format::HQQ_INT4:
            return registry_.find(module + ".W_q") != nullptr;
        case llama_safetensors_quant_format::BNB_INT8: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::I8 &&
                registry_.find(module + ".SCB") != nullptr;
        }
        case llama_safetensors_quant_format::BNB_NF4:
        case llama_safetensors_quant_format::BNB_FP4: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::U8 &&
                registry_.find(module + ".weight.absmax") != nullptr;
        }
        case llama_safetensors_quant_format::PACKED_INT:
        case llama_safetensors_quant_format::PACKED_INT4_FP8:
            return registry_.find(module + ".weight_packed") != nullptr;
        case llama_safetensors_quant_format::INT8_CHANNEL: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::I8;
        }
        case llama_safetensors_quant_format::INT4_GROUP: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::I8;
        }
        case llama_safetensors_quant_format::FP8_TENSOR:
        case llama_safetensors_quant_format::FP8_CHANNEL:
        case llama_safetensors_quant_format::FP8_BLOCK: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && (weight->dtype == llama_safetensors_dtype::F8_E4M3 ||
                (group.legacy_fp8_i8_storage && weight->dtype == llama_safetensors_dtype::I8));
        }
    }
    return false;
}

std::optional<llama_safetensors_quant_binding> llama_safetensors_quant_adapters::bind(
        const std::string & module, llama_safetensors_quant_role role) const {
    const llama_safetensors_quant_group * group = match(module);
    if (group == nullptr) {
        return std::nullopt;
    }
    if (!format_applies(module, *group)) {
        return std::nullopt;
    }

    llama_safetensors_quant_binding result {
        llama_safetensors_quant_materialization::RAW,
        {},
        {},
        GGML_TYPE_COUNT,
        {},
    };

    if (group->format == llama_safetensors_quant_format::HQQ_INT4) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        const auto shape = read_i64_shape(registry_, module + ".shape", "HQQ");
        if (shape[0] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            shape[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("HQQ logical shape exceeds runtime limits for '" + module + "'");
        }
        result.primary = module + ".W_q";
        result.auxiliaries = {
            module + ".scale", module + ".zero", module + ".shape",
            module + ".axis", module + ".group_size", module + ".nbits",
            module + ".packing", module + ".channel_wise",
            module + ".quant_scale", module + ".quant_zero",
            module + ".round_zero", module + ".view_as_float",
        };
        result.target_type     = GGML_TYPE_Q4_1;
        result.target_shape    = { static_cast<int64_t>(shape[1]), static_cast<int64_t>(shape[0]) };
        result.materialization = llama_safetensors_quant_materialization::HQQ_INT4_REPACK;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::EETQ_INT8) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        const std::string weight_name = registry_.find(module + ".qweight") != nullptr ?
            module + ".qweight" : module + ".weight";
        const auto & weight = require_tensor(registry_, weight_name);
        if (weight.shape.size() != 2) {
            throw std::runtime_error("invalid EETQ weight rank for '" + module + "'");
        }
        result.primary         = weight_name;
        result.auxiliaries     = { module + ".weight_scales" };
        result.target_type     = GGML_TYPE_Q8_0;
        result.target_shape    = {
            static_cast<int64_t>(weight.shape[0]),
            static_cast<int64_t>(weight.shape[1]),
        };
        result.materialization = llama_safetensors_quant_materialization::EETQ_REPACK;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::QUANTO_INT4) {
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            return std::nullopt;
        }
        if (role == llama_safetensors_quant_role::WEIGHT_SCALE) {
            result.primary         = module + ".weight._scale";
            result.target_type     = GGML_TYPE_I8;
            result.target_shape    = { 1 };
            result.materialization = llama_safetensors_quant_materialization::QUANTO_W4A16_MARKER;
            return result;
        }
        result.primary         = module + ".weight._data._data";
        result.auxiliaries     = { module + ".weight._scale", module + ".weight._shift" };
        result.target_type     = GGML_TYPE_Q4_1;
        result.target_shape    = group->target_shape;
        result.materialization = llama_safetensors_quant_materialization::QUANTO_INT4_REPACK;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::QUANTO_INT8 ||
            group->format == llama_safetensors_quant_format::QUANTO_FP8) {
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            return std::nullopt;
        }
        if (role == llama_safetensors_quant_role::WEIGHT_SCALE) {
            result.primary         = module + ".weight._scale";
            result.target_type     = GGML_TYPE_I8;
            result.target_shape    = {
                static_cast<int64_t>(sizeof(ggml_w8a16_scale_header) +
                    group->target_shape.at(1) * sizeof(uint16_t)) };
            result.materialization = llama_safetensors_quant_materialization::QUANTO_W8A16_SCALE;
            return result;
        }
        result.primary         = module + ".weight._data";
        result.auxiliaries     = { module + ".weight._scale" };
        result.target_type     = group->format == llama_safetensors_quant_format::QUANTO_INT8 ?
            GGML_TYPE_I8 : GGML_TYPE_F8_E4M3;
        result.target_shape    = group->target_shape;
        result.materialization = llama_safetensors_quant_materialization::RAW;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::TORCHAO_INT4) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        result.primary         = module + ".weight.__qdata";
        result.auxiliaries     = { module + ".weight.__scale_and_zero" };
        result.target_type     = GGML_TYPE_Q4_1;
        result.target_shape    = group->target_shape;
        result.materialization = llama_safetensors_quant_materialization::TORCHAO_INT4_REPACK;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::TORCHAO_INTX) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        result.primary         = module + ".weight.__qdata";
        result.auxiliaries     = { module + ".weight.__scale", module + ".weight.__zero_point" };
        result.target_type     = GGML_TYPE_Q8_0;
        result.target_shape    = reversed_shape(require_tensor(registry_, result.primary));
        result.materialization = llama_safetensors_quant_materialization::TORCHAO_INTX_REPACK;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::BNB_INT8) {
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            result.primary         = module + ".weight";
            result.target_type     = GGML_TYPE_I32;
            result.target_shape    = { 1 };
            result.materialization = llama_safetensors_quant_materialization::DYNAMIC_INT8_MARKER;
            return result;
        }
        if (role == llama_safetensors_quant_role::WEIGHT) {
            result.primary      = module + ".weight";
            result.auxiliaries  = { module + ".SCB" };
            if (registry_.find(module + ".weight_format") != nullptr) {
                result.auxiliaries.push_back(module + ".weight_format");
            }
            result.target_type  = GGML_TYPE_I8;
            result.target_shape = reversed_shape(require_tensor(registry_, result.primary));
            return result;
        }
        result.primary         = module + ".SCB";
        result.target_type     = GGML_TYPE_F32;
        result.target_shape    = reversed_shape(require_tensor(registry_, result.primary));
        result.materialization = llama_safetensors_quant_materialization::BNB_INT8_SCALE;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::BNB_NF4 ||
        group->format == llama_safetensors_quant_format::BNB_FP4) {
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            return std::nullopt;
        }
        const std::string quant_type = group->format == llama_safetensors_quant_format::BNB_NF4 ? "nf4" : "fp4";
        const bnb_state state = read_bnb_state(registry_, module, quant_type);
        if (role == llama_safetensors_quant_role::WEIGHT) {
            result.primary      = module + ".weight";
            result.target_type  = group->format == llama_safetensors_quant_format::BNB_NF4 ?
                GGML_TYPE_BNB_NF4 : GGML_TYPE_BNB_FP4;
            result.target_shape = { state.shape[1], state.shape[0] };
            return result;
        }

        result.primary = module + ".weight.absmax";
        result.auxiliaries = {
            module + ".weight.quant_map",
            module + ".weight.quant_state.bitsandbytes__" + quant_type,
        };
        const auto align4 = [](size_t size) { return (size + 3) & ~size_t(3); };
        size_t bundle_size = sizeof(ggml_bnb_scale_header);
        bundle_size = align4(bundle_size + require_tensor(registry_, result.primary).size);
        bundle_size = align4(bundle_size + require_tensor(registry_, result.auxiliaries[0]).size);
        if (state.nested_block_size != 0) {
            result.auxiliaries.push_back(module + ".weight.nested_absmax");
            result.auxiliaries.push_back(module + ".weight.nested_quant_map");
            bundle_size = align4(bundle_size + require_tensor(registry_, result.auxiliaries[2]).size);
            bundle_size = align4(bundle_size + require_tensor(registry_, result.auxiliaries[3]).size);
        }
        if (bundle_size > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("BitsAndBytes scale bundle is too large for '" + module + "'");
        }
        result.target_type     = GGML_TYPE_I8;
        result.target_shape    = { static_cast<int64_t>(bundle_size) };
        result.materialization = llama_safetensors_quant_materialization::BNB_SCALE_BUNDLE;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::NVFP4_PACK) {
        if (role == llama_safetensors_quant_role::WEIGHT) {
            result.primary        = module + (group->modelopt ? ".weight" : ".weight_packed");
            result.auxiliaries    = { module + ".weight_scale" };
            result.target_type    = GGML_TYPE_NVFP4;
            result.materialization = llama_safetensors_quant_materialization::NVFP4_REPACK;
            const auto & source = require_tensor(registry_, result.primary);
            if ((source.shape.size() != 2 && source.shape.size() != 3) ||
                source.shape.back() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2) {
                throw std::runtime_error("invalid packed NVFP4 dimensions for '" + result.primary + "'");
            }
            result.target_shape.reserve(source.shape.size());
            result.target_shape.push_back(static_cast<int64_t>(source.shape.back() * 2));
            for (auto it = source.shape.rbegin() + 1; it != source.shape.rend(); ++it) {
                if (*it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw std::runtime_error("packed NVFP4 dimension exceeds runtime limits for '" + result.primary + "'");
                }
                result.target_shape.push_back(static_cast<int64_t>(*it));
            }
        } else if (role == llama_safetensors_quant_role::WEIGHT_SCALE) {
            result.primary = module + (group->modelopt ? ".weight_scale_2" : ".weight_global_scale");
            result.target_type     = GGML_TYPE_F32;
            result.target_shape    = { 1 };
            result.materialization = group->modelopt ?
                llama_safetensors_quant_materialization::POSITIVE_F32 :
                llama_safetensors_quant_materialization::RECIPROCAL_F32;
        } else {
            if (group->modelopt && !group->input_quantized) {
                return std::nullopt;
            }
            result.primary         = module + (group->modelopt ? ".input_scale" : ".input_global_scale");
            result.target_type     = GGML_TYPE_F32;
            result.target_shape    = { 1 };
            result.materialization = group->modelopt ?
                llama_safetensors_quant_materialization::POSITIVE_F32 :
                llama_safetensors_quant_materialization::RECIPROCAL_F32;
        }
        return result;
    }

    if (group->format == llama_safetensors_quant_format::MXFP4_PACK) {
        if (role == llama_safetensors_quant_role::WEIGHT_SCALE) {
            return std::nullopt;
        }
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            result.primary         = module + (group->modelopt || group->hf_mxfp4 ? ".weight" : ".weight_packed");
            result.target_type     = GGML_TYPE_I32;
            result.target_shape    = { 1 };
            result.materialization = llama_safetensors_quant_materialization::DYNAMIC_MXFP4_MARKER;
            return result;
        }
        result.primary         = module + (group->modelopt || group->hf_mxfp4 ? ".weight" : ".weight_packed");
        result.auxiliaries     = { module + (group->hf_mxfp4 ? ".scale" : ".weight_scale") };
        result.target_type     = GGML_TYPE_MXFP4;
        result.materialization = llama_safetensors_quant_materialization::MXFP4_REPACK;
        const auto & source = require_tensor(registry_, result.primary);
        if (source.shape.size() != 2 || source.shape[1] >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2) {
            throw std::runtime_error("invalid packed MXFP4 dimensions for '" + result.primary + "'");
        }
        result.target_shape = {
            static_cast<int64_t>(source.shape[1] * 2),
            static_cast<int64_t>(source.shape[0]),
        };
        return result;
    }

    if (group->format == llama_safetensors_quant_format::MXFP8 ||
            group->format == llama_safetensors_quant_format::FP8_GROUP) {
        const bool mxfp8 = group->format == llama_safetensors_quant_format::MXFP8;
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            result.primary         = module + ".weight";
            result.target_type     = mxfp8 ? GGML_TYPE_I32 : GGML_TYPE_I16;
            result.target_shape    = { 1 };
            result.materialization = mxfp8 ?
                llama_safetensors_quant_materialization::DYNAMIC_MXFP8_MARKER :
                llama_safetensors_quant_materialization::DYNAMIC_FP8_GROUP_MARKER;
        } else {
            result.primary      = module + (role == llama_safetensors_quant_role::WEIGHT ? ".weight" : ".weight_scale");
            // GGML has no unsigned byte tensor type. Preserve the E8M0 bit
            // pattern in I8 storage; consumers interpret each byte as uint8_t.
            result.target_type  = role == llama_safetensors_quant_role::WEIGHT ? GGML_TYPE_F8_E4M3 :
                                  mxfp8 ? GGML_TYPE_I8 : GGML_TYPE_BF16;
            const auto & source = require_tensor(registry_, result.primary);
            if (role == llama_safetensors_quant_role::WEIGHT_SCALE) {
                // Preserve both axes even when there is one E8M0 group. The
                // executor indexes the scale grid as [group, output row].
                result.target_shape = {
                    static_cast<int64_t>(source.shape[1]),
                    static_cast<int64_t>(source.shape[0]),
                };
            } else {
                result.target_shape = reversed_shape(source);
            }
        }
        return result;
    }

    if (group->format == llama_safetensors_quant_format::AWQ_GROUP) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        result.primary         = module + ".qweight";
        result.auxiliaries     = { module + ".qzeros", module + ".scales" };
        result.target_type     = GGML_TYPE_Q4_1;
        result.materialization = llama_safetensors_quant_materialization::AWQ_REPACK;
        const auto & source = require_tensor(registry_, result.primary);
        if (source.shape.size() != 2 || source.shape[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 8) {
            throw std::runtime_error("invalid packed AWQ dimensions for '" + result.primary + "'");
        }
        result.target_shape = {
            static_cast<int64_t>(source.shape[0]),
            static_cast<int64_t>(source.shape[1] * 8),
        };
        return result;
    }

    if (group->format == llama_safetensors_quant_format::QUARK_W4A16) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        result.primary     = module + ".weight";
        result.auxiliaries = { module + ".weight_scale" };
        if (!group->symmetric) {
            result.auxiliaries.push_back(module + ".weight_zero_point");
        }
        result.target_type     = GGML_TYPE_Q4_1;
        result.materialization = llama_safetensors_quant_materialization::QUARK_W4A16_REPACK;
        const auto & source = require_tensor(registry_, result.primary);
        if (source.shape.size() != 2 || source.shape[1] >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 8) {
            throw std::runtime_error("invalid packed Quark W4A16 dimensions for '" + result.primary + "'");
        }
        result.target_shape = {
            static_cast<int64_t>(source.shape[0]),
            static_cast<int64_t>(source.shape[1] * 8),
        };
        return result;
    }

    if (group->format == llama_safetensors_quant_format::GPTQ_GROUP) {
        if (role == llama_safetensors_quant_role::WEIGHT_SCALE && group->act_order) {
            if (group->num_bits != 4) {
                throw std::runtime_error("GPTQ act-order currently requires 4-bit weights");
            }
            const std::string qweight_name = module + ".qweight";
            const auto & source = require_tensor(registry_, qweight_name);
            if (source.shape.size() != 2 || source.shape[0] >
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 8) {
                throw std::runtime_error("invalid packed GPTQ dimensions for '" + source.name + "'");
            }
            const uint64_t cols = source.shape[0] * 8;
            const uint64_t rows = source.shape[1];
            const uint64_t groups = cols / group->group_size;
            if (cols > std::numeric_limits<uint32_t>::max() || rows > std::numeric_limits<uint32_t>::max() ||
                    groups > std::numeric_limits<uint16_t>::max()) {
                throw std::runtime_error("GPTQ act-order dimensions exceed the native auxiliary format");
            }
            uint64_t bundle_size = sizeof(ggml_gptq_ao_header);
            const auto append_size = [&](uint64_t size) {
                if (size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) - bundle_size - 3) {
                    throw std::runtime_error("GPTQ act-order auxiliary bundle is too large");
                }
                bundle_size = (bundle_size + size + 3) & ~uint64_t(3);
            };
            append_size(groups * rows);                    // unpacked zero codes
            append_size(groups * rows * sizeof(uint16_t)); // F16/BF16 scales
            append_size(cols * sizeof(uint16_t));          // per-column group map
            result.primary         = module + ".qzeros";
            result.auxiliaries     = { module + ".scales", module + ".g_idx" };
            result.target_type     = GGML_TYPE_I8;
            result.target_shape    = { static_cast<int64_t>(bundle_size) };
            result.materialization = llama_safetensors_quant_materialization::GPTQ_SCALE_BUNDLE;
            return result;
        }
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        result.primary         = module + ".qweight";
        result.auxiliaries     = { module + ".qzeros", module + ".scales" };
        if (registry_.find(module + ".g_idx") != nullptr) {
            result.auxiliaries.push_back(module + ".g_idx");
        }
        const std::string scales_name = module + ".scales";
        const auto & scales = require_tensor(registry_, scales_name);
        if (group->act_order) {
            result.target_type     = GGML_TYPE_GPTQ_AO;
            result.materialization = llama_safetensors_quant_materialization::RAW;
        } else if (group->num_bits == 4) {
            result.target_type     = GGML_TYPE_Q4_1;
            result.materialization = llama_safetensors_quant_materialization::GPTQ_REPACK;
        } else {
            result.target_type = scales.dtype == llama_safetensors_dtype::F16 ?
                GGML_TYPE_Q8_0 : GGML_TYPE_Q8_0_G128;
            result.materialization = llama_safetensors_quant_materialization::GPTQ8_REPACK;
        }
        const auto & source = require_tensor(registry_, result.primary);
        const uint64_t pack_factor = 32 / group->num_bits;
        if (source.shape.size() != 2 || source.shape[0] >
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / pack_factor) {
            throw std::runtime_error("invalid packed GPTQ dimensions for '" + result.primary + "'");
        }
        result.target_shape = {
            static_cast<int64_t>(source.shape[0] * pack_factor),
            static_cast<int64_t>(source.shape[1]),
        };
        return result;
    }

    if (group->format == llama_safetensors_quant_format::PACKED_INT) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        const auto shape = read_weight_shape(registry_, module);
        result.primary      = module + ".weight_packed";
        result.auxiliaries  = { module + ".weight_scale", module + ".weight_shape" };
        result.target_shape = {
            static_cast<int64_t>(shape[1]),
            static_cast<int64_t>(shape[0]),
        };
        if (group->num_bits == 4) {
            if (!group->symmetric) {
                result.auxiliaries.push_back(module + ".weight_zero_point");
            }
            result.target_type     = GGML_TYPE_Q4_A32;
            result.materialization = llama_safetensors_quant_materialization::PACKED_INT4_REPACK;
        } else {
            result.target_type     = GGML_TYPE_Q8_0_G128;
            result.materialization = llama_safetensors_quant_materialization::PACKED_INT8_REPACK;
        }
        return result;
    }

    if (group->format == llama_safetensors_quant_format::PACKED_INT4_FP8) {
        if (role == llama_safetensors_quant_role::WEIGHT_SCALE) {
            return std::nullopt;
        }
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            result.primary         = module + ".weight_packed";
            result.target_type     = GGML_TYPE_I16;
            result.target_shape    = { 1 };
            result.materialization = llama_safetensors_quant_materialization::DYNAMIC_W4A8_FP8_MARKER;
            return result;
        }
        const auto shape = read_weight_shape(registry_, module);
        result.primary         = module + ".weight_packed";
        result.auxiliaries     = {
            module + ".weight_scale", module + ".weight_chan_scale", module + ".weight_shape",
        };
        result.target_type     = GGML_TYPE_Q4_A32;
        result.target_shape    = {
            static_cast<int64_t>(shape[1]),
            static_cast<int64_t>(shape[0]),
        };
        result.materialization = llama_safetensors_quant_materialization::PACKED_INT4_FP8_REPACK;
        return result;
    }

    if (group->format == llama_safetensors_quant_format::INT8_CHANNEL) {
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            if (group->input_dynamic) {
                return std::nullopt;
            }
            result.primary      = module + ".input_scale";
            if (!group->input_symmetric) {
                result.auxiliaries     = { module + ".input_zero_point" };
                result.target_type     = GGML_TYPE_I64;
                result.materialization = llama_safetensors_quant_materialization::STATIC_INT8_ASYM_PARAMS;
            } else {
                result.target_type = GGML_TYPE_F32;
            }
            result.target_shape = { 1 };
            return result;
        }
        result.primary = module + (role == llama_safetensors_quant_role::WEIGHT ?
            ".weight" : ".weight_scale");
        if (role == llama_safetensors_quant_role::WEIGHT) {
            result.auxiliaries = { module + ".weight_scale" };
            result.target_type = GGML_TYPE_I8;
        } else {
            result.target_type = GGML_TYPE_F32;
        }
        result.target_shape    = reversed_shape(require_tensor(registry_, result.primary));
        return result;
    }

    if (group->format == llama_safetensors_quant_format::INT4_GROUP) {
        if (role == llama_safetensors_quant_role::WEIGHT_SCALE) {
            return std::nullopt;
        }
        if (role == llama_safetensors_quant_role::INPUT_SCALE) {
            if (group->input_dynamic) {
                result.primary         = module + ".weight";
                result.target_type     = GGML_TYPE_I32;
                result.target_shape    = { 1 };
                result.materialization = llama_safetensors_quant_materialization::DYNAMIC_INT8_MARKER;
            } else {
                result.primary      = module + ".input_scale";
                result.target_type  = GGML_TYPE_F32;
                result.target_shape = { 1 };
            }
            return result;
        }
        result.primary         = module + ".weight";
        result.auxiliaries     = { module + ".weight_scale" };
        result.target_type     = GGML_TYPE_Q4_A32;
        result.target_shape    = reversed_shape(require_tensor(registry_, result.primary));
        result.materialization = llama_safetensors_quant_materialization::INT4_GROUP_REPACK;
        return result;
    }

    if (role == llama_safetensors_quant_role::INPUT_SCALE) {
        if (!group->input_quantized) {
            return std::nullopt;
        }
        if (group->input_dynamic) {
            result.primary         = module + ".weight";
            result.target_type     = GGML_TYPE_I32;
            result.target_shape    = { 1 };
            result.materialization = llama_safetensors_quant_materialization::DYNAMIC_FP8_MARKER;
            return result;
        }
        result.primary      = module + (group->legacy_fp8_i8_storage ? ".in_scale" : ".input_scale");
        result.target_type  = GGML_TYPE_F32;
        result.target_shape = { 1 };
        return result;
    }
    if (role == llama_safetensors_quant_role::WEIGHT) {
        result.primary      = module + ".weight";
        result.target_type  = GGML_TYPE_F8_E4M3;
        result.target_shape = reversed_shape(require_tensor(registry_, result.primary));
        return result;
    }

    result.primary = weight_scale_name(module);
    const auto & scale = require_tensor(registry_, result.primary);
    if (group->format == llama_safetensors_quant_format::FP8_TENSOR) {
        result.target_type  = GGML_TYPE_BF16;
        const std::string weight_name = module + ".weight";
        const auto & weight = require_tensor(registry_, weight_name);
        result.target_shape = { static_cast<int64_t>(weight.shape[0]) };
        result.materialization = llama_safetensors_quant_materialization::BROADCAST_BF16_SCALAR;
    } else if (group->format == llama_safetensors_quant_format::FP8_CHANNEL) {
        result.target_type  = GGML_TYPE_BF16;
        result.target_shape = { static_cast<int64_t>(scale.shape[0]) };
        if (group->modelopt) {
            result.materialization = llama_safetensors_quant_materialization::POSITIVE_F32_TO_BF16;
        }
    } else if (group->format == llama_safetensors_quant_format::FP8_BLOCK && group->modelopt) {
        result.target_type     = GGML_TYPE_F32;
        result.target_shape    = {
            static_cast<int64_t>(scale.shape[0]),
            static_cast<int64_t>(scale.shape[2]),
        };
        result.materialization = llama_safetensors_quant_materialization::FP8_BLOCK_SCALE_MODELOPT;
    } else {
        result.target_type     = GGML_TYPE_F32;
        result.target_shape    = {
            static_cast<int64_t>(scale.shape[0]),
            static_cast<int64_t>(scale.shape[1]),
        };
        result.materialization = group->e8m0_block_scale ?
            llama_safetensors_quant_materialization::FP8_BLOCK_SCALE_E8M0 :
            llama_safetensors_quant_materialization::FP8_BLOCK_SCALE;
    }
    return result;
}

std::string llama_safetensors_quant_adapters::weight_scale_name(const std::string & module) const {
    const llama_safetensors_quant_group * group = match(module);
    if (group != nullptr && group->format == llama_safetensors_quant_format::FP8_BLOCK && !group->modelopt) {
        return module + (group->e8m0_block_scale ? ".scale" : ".weight_scale_inv");
    }
    return module + ".weight_scale";
}

const llama_safetensors_quant_summary & llama_safetensors_quant_adapters::summary() const {
    return summary_;
}

uint32_t llama_safetensors_quant_adapters::file_type() const {
    if (!has_quantization_config_) {
        for (const llama_safetensors_tensor & tensor : registry_.tensors()) {
            if (tensor.shape.size() >= 2 && tensor.dtype == llama_safetensors_dtype::BF16) {
                return LLAMA_FTYPE_MOSTLY_BF16;
            }
            if (tensor.shape.size() >= 2 && tensor.dtype == llama_safetensors_dtype::F16) {
                return LLAMA_FTYPE_MOSTLY_F16;
            }
        }
        return LLAMA_FTYPE_ALL_F32;
    }
    if (summary_.awq + summary_.quark_w4a16 + summary_.gptq + summary_.quanto_int4 + summary_.torchao_int4 +
            summary_.hqq_int4 + summary_.bnb_nf4 + summary_.bnb_fp4 +
            summary_.packed_int4 + summary_.w4a8 + summary_.w4a8_fp8 != 0) {
        return LLAMA_FTYPE_MOSTLY_Q4_1;
    }
    if (summary_.w8a8 + summary_.eetq + summary_.quanto_int8 + summary_.torchao_intx + summary_.bnb_int8 +
            summary_.gptq_int8 + summary_.packed_int8 != 0) {
        return LLAMA_FTYPE_MOSTLY_Q8_0;
    }
    if (summary_.nvfp4 != 0) {
        return LLAMA_FTYPE_MOSTLY_NVFP4;
    }
    if (summary_.mxfp4 != 0) {
        return LLAMA_FTYPE_MOSTLY_MXFP4;
    }
    return LLAMA_FTYPE_MOSTLY_F8_E4M3;
}

void llama_safetensors_quant_adapters::validate() {
    for (const llama_safetensors_tensor & tensor : registry_.tensors()) {
        for (uint64_t dim : tensor.shape) {
            if (dim == 0 || dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("unsupported dimension in source tensor '" + tensor.name + "'");
            }
        }
        if (ends_with(tensor.name, ".weight_zero_point")) {
            const std::string module =
                tensor.name.substr(0, tensor.name.size() - std::string_view(".weight_zero_point").size());
            const auto * group = match(module);
            if (group != nullptr && group->format == llama_safetensors_quant_format::PACKED_INT && !group->symmetric) {
                continue;
            }
            if (group != nullptr && group->format == llama_safetensors_quant_format::QUARK_W4A16) {
                continue;
            }
            throw std::runtime_error("unsupported zero-point tensor '" + tensor.name + "'");
        }
        if (ends_with(tensor.name, ".zero_point")) {
            throw std::runtime_error("unsupported zero-point tensor '" + tensor.name + "'");
        }

        std::string module;
        llama_safetensors_quant_format expected;
        if (ends_with(tensor.name, ".weight.__qdata")) {
            module = tensor.name.substr(
                0, tensor.name.size() - std::string_view(".weight.__qdata").size());
            const auto * group = match(module);
            if (group == nullptr ||
                (group->format != llama_safetensors_quant_format::TORCHAO_INT4 &&
                 group->format != llama_safetensors_quant_format::TORCHAO_INTX)) {
                throw std::runtime_error("TorchAO quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            if (expected == llama_safetensors_quant_format::TORCHAO_INT4) {
                ++summary_.torchao_int4;
            } else {
                ++summary_.torchao_intx;
            }
        } else if (ends_with(tensor.name, ".weight._data._data")) {
            module = tensor.name.substr(
                0, tensor.name.size() - std::string_view(".weight._data._data").size());
            const auto * group = match(module);
            if (group == nullptr || group->format != llama_safetensors_quant_format::QUANTO_INT4) {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            ++summary_.quanto_int4;
        } else if (ends_with(tensor.name, ".weight._data")) {
            module = tensor.name.substr(
                0, tensor.name.size() - std::string_view(".weight._data").size());
            const auto * group = match(module);
            if (group == nullptr ||
                    (group->format != llama_safetensors_quant_format::QUANTO_INT8 &&
                     group->format != llama_safetensors_quant_format::QUANTO_FP8)) {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            if (expected == llama_safetensors_quant_format::QUANTO_INT8) {
                ++summary_.quanto_int8;
            } else {
                ++summary_.quanto_fp8;
            }
        } else if (ends_with(tensor.name, ".W_q")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".W_q").size());
            const auto * group = match(module);
            if (group == nullptr || group->format != llama_safetensors_quant_format::HQQ_INT4) {
                throw std::runtime_error("HQQ quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            ++summary_.hqq_int4;
        } else if (ends_with(tensor.name, ".weight_packed")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".weight_packed").size());
            const auto * group = match(module);
            if (group == nullptr) {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            if (expected == llama_safetensors_quant_format::NVFP4_PACK) {
                ++summary_.nvfp4;
            } else if (expected == llama_safetensors_quant_format::MXFP4_PACK) {
                ++summary_.mxfp4;
            } else if (expected == llama_safetensors_quant_format::MXFP8) {
                ++summary_.mxfp8;
            } else if (expected == llama_safetensors_quant_format::PACKED_INT && group->num_bits == 4) {
                ++summary_.packed_int4;
            } else if (expected == llama_safetensors_quant_format::PACKED_INT && group->num_bits == 8) {
                ++summary_.packed_int8;
            } else if (expected == llama_safetensors_quant_format::PACKED_INT4_FP8) {
                ++summary_.w4a8_fp8;
            } else {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
        } else if (ends_with(tensor.name, ".qweight")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".qweight").size());
            const llama_safetensors_quant_group * group = match(module);
            if (group == nullptr ||
                (group->format != llama_safetensors_quant_format::AWQ_GROUP &&
                 group->format != llama_safetensors_quant_format::GPTQ_GROUP &&
                 group->format != llama_safetensors_quant_format::EETQ_INT8)) {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            if (expected == llama_safetensors_quant_format::EETQ_INT8) {
                if (tensor.dtype != llama_safetensors_dtype::I8) {
                    throw std::runtime_error("invalid EETQ qweight type for source tensor '" + tensor.name + "'");
                }
                ++summary_.eetq;
            } else if (expected == llama_safetensors_quant_format::AWQ_GROUP) {
                ++summary_.awq;
            } else if (group->num_bits == 8) {
                ++summary_.gptq_int8;
            } else {
                ++summary_.gptq;
            }
        } else if (ends_with(tensor.name, ".weight")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".weight").size());
            const llama_safetensors_quant_group * group = match(module);
            if (group != nullptr && group->format == llama_safetensors_quant_format::QUARK_W4A16 &&
                    tensor.dtype == llama_safetensors_dtype::I32) {
                expected = group->format;
                ++summary_.quark_w4a16;
            } else if (group != nullptr && group->format == llama_safetensors_quant_format::EETQ_INT8 &&
                    tensor.dtype == llama_safetensors_dtype::I8) {
                expected = group->format;
                ++summary_.eetq;
            } else if (group != nullptr && group->format == llama_safetensors_quant_format::BNB_INT8 &&
                    tensor.dtype == llama_safetensors_dtype::I8) {
                expected = group->format;
                ++summary_.bnb_int8;
            } else if (group != nullptr &&
                    (group->format == llama_safetensors_quant_format::BNB_NF4 ||
                     group->format == llama_safetensors_quant_format::BNB_FP4) &&
                    tensor.dtype == llama_safetensors_dtype::U8) {
                expected = group->format;
                if (expected == llama_safetensors_quant_format::BNB_NF4) {
                    ++summary_.bnb_nf4;
                } else {
                    ++summary_.bnb_fp4;
                }
            } else if (group != nullptr && group->format == llama_safetensors_quant_format::NVFP4_PACK &&
                    group->modelopt && tensor.dtype == llama_safetensors_dtype::U8) {
                expected = group->format;
                ++summary_.nvfp4;
            } else if (group != nullptr && group->format == llama_safetensors_quant_format::MXFP4_PACK &&
                    ((group->modelopt && tensor.dtype == llama_safetensors_dtype::U8) ||
                     (group->hf_mxfp4 && tensor.dtype == llama_safetensors_dtype::I8))) {
                expected = group->format;
                ++summary_.mxfp4;
            } else if (group != nullptr &&
                    (group->format == llama_safetensors_quant_format::INT8_CHANNEL ||
                     group->format == llama_safetensors_quant_format::INT4_GROUP)) {
                if (tensor.dtype != llama_safetensors_dtype::I8) {
                    const std::string scale_name = module + ".weight_scale";
                    if (registry_.find(scale_name) != nullptr) {
                        throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
                    }
                    continue;
                }
                expected = group->format;
                if (expected == llama_safetensors_quant_format::INT8_CHANNEL) {
                    ++summary_.w8a8;
                } else {
                    ++summary_.w4a8;
                }
            } else {
                const bool fp8_group = group != nullptr &&
                    (group->format == llama_safetensors_quant_format::FP8_TENSOR ||
                     group->format == llama_safetensors_quant_format::FP8_CHANNEL ||
                     group->format == llama_safetensors_quant_format::FP8_BLOCK ||
                     group->format == llama_safetensors_quant_format::FP8_GROUP ||
                     group->format == llama_safetensors_quant_format::MXFP8);
                if (!fp8_group) {
                    if (tensor.dtype == llama_safetensors_dtype::F8_E4M3) {
                        throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
                    }
                    continue;
                }
                if (tensor.dtype != llama_safetensors_dtype::F8_E4M3 &&
                        !(group->legacy_fp8_i8_storage && tensor.dtype == llama_safetensors_dtype::I8)) {
                    const std::string scale_name = module +
                        (group->format == llama_safetensors_quant_format::FP8_BLOCK ?
                            ".weight_scale_inv" : ".weight_scale");
                    // Some producers leave embeddings and LM heads in BF16 while
                    // using a catch-all FP8 target. A scale sidecar distinguishes
                    // a malformed FP8 module from that intentional exception.
                    if (registry_.find(scale_name) != nullptr) {
                        throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
                    }
                    continue;
                }
                expected = group->format;
                if (expected == llama_safetensors_quant_format::FP8_TENSOR) {
                    ++summary_.fp8_tensor;
                } else if (expected == llama_safetensors_quant_format::FP8_CHANNEL) {
                    ++summary_.fp8_channel;
                } else if (expected == llama_safetensors_quant_format::MXFP8) {
                    ++summary_.mxfp8;
                } else if (expected == llama_safetensors_quant_format::FP8_GROUP) {
                    ++summary_.fp8_group;
                } else {
                    ++summary_.fp8_block;
                }
            }
        } else {
            continue;
        }

        const llama_safetensors_quant_group * group = match(module);
        if (group == nullptr || group->format != expected) {
            throw std::runtime_error("compressed-tensors contract does not match source tensor '" + tensor.name + "'");
        }

        if (expected == llama_safetensors_quant_format::HQQ_INT4) {
            const auto & group_ref = *group;
            const auto shape = read_i64_shape(registry_, module + ".shape", "HQQ");
            if (shape[0] > std::numeric_limits<uint64_t>::max() / shape[1] ||
                shape[1] % 32 != 0) {
                throw std::runtime_error("invalid HQQ logical geometry for source tensor '" + tensor.name + "'");
            }
            const uint64_t values = shape[0] * shape[1];
            if (group_ref.group_size == 0 || values % group_ref.group_size != 0) {
                throw std::runtime_error("invalid HQQ group geometry for source tensor '" + tensor.name + "'");
            }
            const uint64_t groups = values / group_ref.group_size;
            const std::string scale_name   = module + ".scale";
            const std::string zero_name    = module + ".zero";
            const std::string packing_name = module + ".packing";
            const auto & scale   = require_tensor(registry_, scale_name);
            const auto & zero    = require_tensor(registry_, zero_name);
            const auto & packing = require_tensor(registry_, packing_name);
            dependencies_[tensor.name] = {
                scale.name, zero.name, module + ".shape", module + ".axis",
                module + ".group_size", module + ".nbits", packing.name,
                module + ".channel_wise", module + ".quant_scale",
                module + ".quant_zero", module + ".round_zero",
                module + ".view_as_float",
            };
            if (tensor.dtype != llama_safetensors_dtype::U8 || tensor.shape.size() != 2 ||
                groups % 2 != 0 || tensor.shape != std::vector<uint64_t>({ groups / 2, group_ref.group_size }) ||
                scale.dtype != llama_safetensors_dtype::F16 ||
                scale.shape != std::vector<uint64_t>({ groups, 1 }) ||
                zero.dtype != llama_safetensors_dtype::F16 ||
                zero.shape != std::vector<uint64_t>({ groups, 1 }) ||
                read_i32_scalar(registry_, module + ".axis", "HQQ axis") != 1 ||
                read_i32_scalar(registry_, module + ".group_size", "HQQ group size") !=
                    static_cast<int32_t>(group_ref.group_size) ||
                read_i32_scalar(registry_, module + ".nbits", "HQQ bit width") != 4 ||
                packing.dtype != llama_safetensors_dtype::U8 || packing.shape != std::vector<uint64_t>({ 7 }) ||
                registry_.read(packing) != std::vector<uint8_t>({ '4','b','i','t','_','u','8' }) ||
                read_u8_scalar(registry_, module + ".channel_wise", "HQQ channel-wise") != 1 ||
                read_u8_scalar(registry_, module + ".quant_scale", "HQQ scale quantization") != 0 ||
                read_u8_scalar(registry_, module + ".quant_zero", "HQQ zero quantization") != 0 ||
                read_u8_scalar(registry_, module + ".round_zero", "HQQ rounded zero") != 1 ||
                read_u8_scalar(registry_, module + ".view_as_float", "HQQ float view") != 0) {
                throw std::runtime_error("invalid HQQ INT4 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::TORCHAO_INT4) {
            const auto & group_ref = *match(module);
            const std::string scale_name = module + ".weight.__scale_and_zero";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (group_ref.target_shape.size() != 2 || group_ref.target_shape[0] <= 0 ||
                group_ref.target_shape[1] <= 0 || tensor.dtype != llama_safetensors_dtype::I32 ||
                tensor.shape.size() != 4 || tensor.shape[2] != 32 || tensor.shape[3] != 4 ||
                scale.dtype != llama_safetensors_dtype::BF16 || scale.shape.size() != 3 ||
                scale.shape[2] != 2 || scale.shape[1] < static_cast<uint64_t>(group_ref.target_shape[1]) ||
                scale.shape[0] * group_ref.group_size < static_cast<uint64_t>(group_ref.target_shape[0])) {
                throw std::runtime_error("invalid TorchAO tiled-int4 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::TORCHAO_INTX) {
            const auto & group_ref = *match(module);
            const std::string scale_name = module + ".weight.__scale";
            const std::string zero_name  = module + ".weight.__zero_point";
            const auto & scale = require_tensor(registry_, scale_name);
            const auto & zero  = require_tensor(registry_, zero_name);
            dependencies_[tensor.name] = { scale_name, zero_name };
            if (tensor.dtype != llama_safetensors_dtype::I8 || tensor.shape.size() != 2 ||
                tensor.shape[0] == 0 || tensor.shape[1] == 0 || tensor.shape[1] % group_ref.group_size != 0 ||
                (scale.dtype != llama_safetensors_dtype::BF16 && scale.dtype != llama_safetensors_dtype::F16) ||
                scale.shape != std::vector<uint64_t>({ tensor.shape[0], tensor.shape[1] / group_ref.group_size }) ||
                (zero.dtype != llama_safetensors_dtype::I8 && zero.dtype != scale.dtype) ||
                zero.shape != scale.shape) {
                throw std::runtime_error("invalid TorchAO unpacked-intx contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::QUANTO_INT4) {
            const auto & quanto_group = *match(module);
            const std::string scale_name = module + ".weight._scale";
            const std::string shift_name = module + ".weight._shift";
            const auto & scale = require_tensor(registry_, scale_name);
            const auto & shift = require_tensor(registry_, shift_name);
            dependencies_[tensor.name] = { scale_name, shift_name };
            if (quanto_group.target_shape.size() != 2 || quanto_group.group_size == 0) {
                throw std::runtime_error("invalid Quanto target geometry for source tensor '" + tensor.name + "'");
            }
            const uint64_t cols = static_cast<uint64_t>(quanto_group.target_shape[0]);
            const uint64_t rows = static_cast<uint64_t>(quanto_group.target_shape[1]);
            const uint64_t groups = rows * (cols / quanto_group.group_size);
            const bool scale_dtype = scale.dtype == llama_safetensors_dtype::F16 ||
                scale.dtype == llama_safetensors_dtype::BF16;
            const bool shift_dtype = shift.dtype == llama_safetensors_dtype::F16 ||
                shift.dtype == llama_safetensors_dtype::BF16;
            if (tensor.dtype != llama_safetensors_dtype::U8 || groups % 2 != 0 ||
                tensor.shape != std::vector<uint64_t>({ groups / 2, quanto_group.group_size }) ||
                !scale_dtype || scale.shape != std::vector<uint64_t>({ groups, 1 }) ||
                !shift_dtype || shift.shape != std::vector<uint64_t>({ groups, 1 })) {
                throw std::runtime_error("invalid Quanto qint4 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::QUANTO_INT8 ||
                expected == llama_safetensors_quant_format::QUANTO_FP8) {
            const auto & quanto_group = *match(module);
            const std::string scale_name = module + ".weight._scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            const bool scale_dtype = scale.dtype == llama_safetensors_dtype::F16 ||
                scale.dtype == llama_safetensors_dtype::BF16;
            if (quanto_group.target_shape.size() != 2 ||
                tensor.dtype != (expected == llama_safetensors_quant_format::QUANTO_INT8 ?
                    llama_safetensors_dtype::I8 : llama_safetensors_dtype::F8_E4M3) ||
                tensor.shape != std::vector<uint64_t>({
                    static_cast<uint64_t>(quanto_group.target_shape[1]),
                    static_cast<uint64_t>(quanto_group.target_shape[0]) }) ||
                !scale_dtype || scale.shape != std::vector<uint64_t>({
                    static_cast<uint64_t>(quanto_group.target_shape[1]), 1 })) {
                throw std::runtime_error("invalid Quanto 8-bit contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::EETQ_INT8) {
            const std::string scale_name = module + ".weight_scales";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            const uint64_t cols = tensor.shape.size() == 2 ? tensor.shape[0] : 0;
            const uint64_t rows = tensor.shape.size() == 2 ? tensor.shape[1] : 0;
            if (tensor.dtype != llama_safetensors_dtype::I8 || tensor.shape.size() != 2 ||
                cols == 0 || cols % 32 != 0 || rows == 0 ||
                scale.dtype != llama_safetensors_dtype::F16 ||
                (scale.shape != std::vector<uint64_t>({ rows }) &&
                 scale.shape != std::vector<uint64_t>({ 1, rows }))) {
                throw std::runtime_error(
                    "invalid EETQ W8A16 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::QUARK_W4A16) {
            const std::string scale_name = module + ".weight_scale";
            const std::string zero_name  = module + ".weight_zero_point";
            const auto & scale = require_tensor(registry_, scale_name);
            const auto * zero  = registry_.find(zero_name);
            dependencies_[tensor.name] = { scale_name };
            if (!group->symmetric) {
                dependencies_[tensor.name].push_back(zero_name);
            }
            const uint64_t cols = tensor.shape.empty() ? 0 : tensor.shape[0];
            const uint64_t rows = tensor.shape.size() == 2 ? tensor.shape[1] * 8 : 0;
            const uint64_t groups = group->group_size == 0 ? 0 : cols / group->group_size;
            const bool zero_valid = group->symmetric ?
                (zero == nullptr || (zero->dtype == llama_safetensors_dtype::I32 &&
                                     zero->shape == std::vector<uint64_t>({ groups, rows / 8 }))) :
                (zero != nullptr && zero->dtype == llama_safetensors_dtype::I32 &&
                                  zero->shape == std::vector<uint64_t>({ groups, rows / 8 }));
            if (tensor.dtype != llama_safetensors_dtype::I32 || tensor.shape.size() != 2 ||
                cols % 32 != 0 || cols % group->group_size != 0 || rows == 0 ||
                (scale.dtype != llama_safetensors_dtype::F16 &&
                 scale.dtype != llama_safetensors_dtype::BF16 &&
                 scale.dtype != llama_safetensors_dtype::F32) ||
                scale.shape != std::vector<uint64_t>({ groups, rows }) || !zero_valid) {
                throw std::runtime_error(
                    "invalid Quark W4A16 group contract for source tensor '" + tensor.name + "'");
            }
            if (group->symmetric && zero != nullptr) {
                const std::vector<uint8_t> zero_bytes = registry_.read(*zero);
                if (std::any_of(zero_bytes.begin(), zero_bytes.end(), [](uint8_t value) { return value != 0; })) {
                    throw std::runtime_error("symmetric Quark W4A16 tensor has a nonzero zero point");
                }
            }
        } else if (expected == llama_safetensors_quant_format::BNB_INT8) {
            const std::string scale_name = module + ".SCB";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            const std::string format_name = module + ".weight_format";
            if (const auto * format = registry_.find(format_name)) {
                dependencies_[tensor.name].push_back(format_name);
                if (format->dtype != llama_safetensors_dtype::U8 || !format->shape.empty() || format->size != 1 ||
                        registry_.read(*format).front() != 0) {
                    throw std::runtime_error(
                        "unsupported BitsAndBytes INT8 weight format for source tensor '" + tensor.name + "'");
                }
            }
            if (tensor.dtype != llama_safetensors_dtype::I8 || tensor.shape.size() != 2 ||
                tensor.shape[1] % 32 != 0 ||
                (scale.dtype != llama_safetensors_dtype::F32 &&
                 scale.dtype != llama_safetensors_dtype::F16 &&
                 scale.dtype != llama_safetensors_dtype::BF16) ||
                (scale.shape != std::vector<uint64_t>({ tensor.shape[0] }) &&
                 scale.shape != std::vector<uint64_t>({ tensor.shape[0], 1 }))) {
                throw std::runtime_error(
                    "invalid BitsAndBytes INT8 row-scale contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::BNB_NF4 ||
                expected == llama_safetensors_quant_format::BNB_FP4) {
            const std::string quant_type = expected == llama_safetensors_quant_format::BNB_NF4 ? "nf4" : "fp4";
            const bnb_state state = read_bnb_state(registry_, module, quant_type);
            const uint64_t rows = static_cast<uint64_t>(state.shape[0]);
            const uint64_t cols = static_cast<uint64_t>(state.shape[1]);
            if (rows > std::numeric_limits<uint64_t>::max() / cols || rows * cols % 2 != 0 ||
                state.block_size != 64 || rows * cols % state.block_size != 0 ||
                tensor.dtype != llama_safetensors_dtype::U8 || tensor.size != rows * cols / 2) {
                throw std::runtime_error("invalid BitsAndBytes packed weight contract for source tensor '" + tensor.name + "'");
            }
            const uint64_t n_blocks = rows * cols / state.block_size;
            const std::string absmax_name = module + ".weight.absmax";
            const std::string quant_map_name = module + ".weight.quant_map";
            const std::string state_name = module + ".weight.quant_state.bitsandbytes__" + quant_type;
            const auto & absmax = require_tensor(registry_, absmax_name);
            const auto & quant_map = require_tensor(registry_, quant_map_name);
            dependencies_[tensor.name] = { absmax_name, quant_map_name, state_name };
            if (quant_map.dtype != llama_safetensors_dtype::F32 ||
                quant_map.shape != std::vector<uint64_t>({ 16 })) {
                throw std::runtime_error("invalid BitsAndBytes codebook for source tensor '" + tensor.name + "'");
            }
            if (state.nested_block_size == 0) {
                if (absmax.dtype != llama_safetensors_dtype::F32 ||
                    absmax.shape != std::vector<uint64_t>({ n_blocks })) {
                    throw std::runtime_error("invalid BitsAndBytes absmax tensor for source tensor '" + tensor.name + "'");
                }
            } else {
                const uint64_t nested_blocks = (n_blocks + state.nested_block_size - 1) / state.nested_block_size;
                const std::string nested_absmax_name = module + ".weight.nested_absmax";
                const std::string nested_map_name = module + ".weight.nested_quant_map";
                const auto & nested_absmax = require_tensor(registry_, nested_absmax_name);
                const auto & nested_map = require_tensor(registry_, nested_map_name);
                dependencies_[tensor.name].push_back(nested_absmax_name);
                dependencies_[tensor.name].push_back(nested_map_name);
                if (state.nested_block_size != 256 || absmax.dtype != llama_safetensors_dtype::U8 ||
                    absmax.shape != std::vector<uint64_t>({ n_blocks }) ||
                    nested_absmax.dtype != llama_safetensors_dtype::F32 ||
                    nested_absmax.shape != std::vector<uint64_t>({ nested_blocks }) ||
                    nested_map.dtype != llama_safetensors_dtype::F32 ||
                    nested_map.shape != std::vector<uint64_t>({ 256 })) {
                    throw std::runtime_error("invalid nested BitsAndBytes absmax contract for source tensor '" + tensor.name + "'");
                }
            }
        } else if (expected == llama_safetensors_quant_format::PACKED_INT) {
            const auto & group = *match(module);
            const std::string scale_name = module + ".weight_scale";
            const std::string shape_name = module + ".weight_shape";
            const auto & scale = require_tensor(registry_, scale_name);
            const auto & shape_desc = require_tensor(registry_, shape_name);
            const auto shape = read_weight_shape(registry_, module);
            const uint64_t rows = shape[0];
            const uint64_t cols = shape[1];
            const uint64_t pack_factor = 32 / group.num_bits;
            const uint64_t groups = cols / group.group_size;
            dependencies_[tensor.name] = { scale.name, shape_desc.name };
            const bool common_valid = tensor.dtype == llama_safetensors_dtype::I32 && tensor.shape.size() == 2 &&
                cols % group.group_size == 0 && cols % 128 == 0 && rows > 0 &&
                tensor.shape == std::vector<uint64_t>({ rows, cols / pack_factor }) &&
                scale.dtype == llama_safetensors_dtype::BF16 &&
                scale.shape == std::vector<uint64_t>({ rows, groups });
            if (!common_valid) {
                throw std::runtime_error("invalid packed integer WNA16 contract for source tensor '" + tensor.name + "'");
            }
            if (group.num_bits == 4) {
                const std::string zero_name = module + ".weight_zero_point";
                if (group.symmetric) {
                    if (registry_.find(zero_name) != nullptr) {
                        throw std::runtime_error(
                            "symmetric packed integer tensor has an unexpected zero point: '" + tensor.name + "'");
                    }
                } else {
                    const auto & zero = require_tensor(registry_, zero_name);
                    dependencies_[tensor.name].push_back(zero.name);
                    if (zero.dtype != llama_safetensors_dtype::I32 ||
                        zero.shape != std::vector<uint64_t>({ (rows + 7) / 8, groups })) {
                        throw std::runtime_error(
                            "invalid packed integer zero-point contract for source tensor '" + tensor.name + "'");
                    }
                }
            } else if (registry_.find(module + ".weight_zero_point") != nullptr) {
                throw std::runtime_error(
                    "symmetric packed integer tensor has an unexpected zero point: '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::PACKED_INT4_FP8) {
            const auto shape = read_weight_shape(registry_, module);
            const uint64_t rows = shape[0];
            const uint64_t cols = shape[1];
            const std::string scale_name = module + ".weight_scale";
            const std::string channel_scale_name = module + ".weight_chan_scale";
            const std::string shape_name = module + ".weight_shape";
            const auto & scale = require_tensor(registry_, scale_name);
            const auto & channel_scale = require_tensor(registry_, channel_scale_name);
            const auto & shape_desc = require_tensor(registry_, shape_name);
            dependencies_[tensor.name] = { scale.name, channel_scale.name, shape_desc.name };
            if (tensor.dtype != llama_safetensors_dtype::I32 ||
                tensor.shape != std::vector<uint64_t>({ rows, cols / 8 }) ||
                cols % 128 != 0 || rows == 0 ||
                scale.dtype != llama_safetensors_dtype::F8_E4M3 ||
                scale.shape != std::vector<uint64_t>({ rows, cols / 128 }) ||
                channel_scale.dtype != llama_safetensors_dtype::F32 ||
                channel_scale.shape != std::vector<uint64_t>({ rows, 1 })) {
                throw std::runtime_error("invalid packed W4A8-FP8 group-128 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::AWQ_GROUP) {
            const std::string qzeros_name = module + ".qzeros";
            const std::string scales_name = module + ".scales";
            const auto & qzeros = require_tensor(registry_, qzeros_name);
            const auto & scales = require_tensor(registry_, scales_name);
            dependencies_[tensor.name] = { qzeros_name, scales_name };
            const uint64_t cols = tensor.shape.empty() ? 0 : tensor.shape[0];
            const uint64_t group_size = group->group_size == 0 ? cols : group->group_size;
            const uint64_t groups = group_size == 0 ? 0 : cols / group_size;
            if (tensor.dtype != llama_safetensors_dtype::I32 || tensor.shape.size() != 2 ||
                cols % 32 != 0 || group_size % 32 != 0 || cols % group_size != 0 || tensor.shape[1] == 0 ||
                qzeros.dtype != llama_safetensors_dtype::I32 || qzeros.shape.size() != 2 ||
                qzeros.shape[0] != groups || qzeros.shape[1] != tensor.shape[1] ||
                (scales.dtype != llama_safetensors_dtype::F16 && scales.dtype != llama_safetensors_dtype::BF16) ||
                scales.shape.size() != 2 || scales.shape[0] != groups || scales.shape[1] != tensor.shape[1] * 8) {
                throw std::runtime_error("invalid AWQ W4A16 group contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::GPTQ_GROUP) {
            const std::string qzeros_name = module + ".qzeros";
            const std::string scales_name = module + ".scales";
            const std::string g_idx_name  = module + ".g_idx";
            const auto & qzeros = require_tensor(registry_, qzeros_name);
            const auto & scales = require_tensor(registry_, scales_name);
            const auto * g_idx = registry_.find(g_idx_name);
            if (group->act_order && g_idx == nullptr) {
                throw std::runtime_error("act-order GPTQ tensor is missing '" + g_idx_name + "'");
            }
            dependencies_[tensor.name] = { qzeros_name, scales_name };
            if (g_idx != nullptr) {
                dependencies_[tensor.name].push_back(g_idx_name);
            }
            const uint64_t pack_factor = 32 / group->num_bits;
            if ((group->num_bits != 4 && group->num_bits != 8) || tensor.shape.size() != 2 ||
                tensor.shape[0] > std::numeric_limits<uint64_t>::max() / pack_factor ||
                tensor.shape[1] % pack_factor != 0) {
                throw std::runtime_error("invalid packed GPTQ dimensions for source tensor '" + tensor.name + "'");
            }
            const uint64_t cols = tensor.shape[0] * pack_factor;
            const uint64_t rows = tensor.shape[1];
            const uint64_t group_size = group->group_size == 0 ? cols : group->group_size;
            const uint64_t groups = group_size == 0 ? 0 : cols / group_size;
            if (tensor.dtype != llama_safetensors_dtype::I32 ||
                cols % 32 != 0 || group_size % 32 != 0 || cols % group_size != 0 || rows == 0 ||
                qzeros.dtype != llama_safetensors_dtype::I32 ||
                qzeros.shape != std::vector<uint64_t>({ groups, rows / pack_factor }) ||
                (scales.dtype != llama_safetensors_dtype::F16 && scales.dtype != llama_safetensors_dtype::BF16) ||
                scales.shape != std::vector<uint64_t>({ groups, rows }) ||
                (g_idx != nullptr && (g_idx->dtype != llama_safetensors_dtype::I32 ||
                                      g_idx->shape != std::vector<uint64_t>({ cols })))) {
                throw std::runtime_error("invalid GPTQ packed group contract for source tensor '" + tensor.name + "'");
            }
            if (group->num_bits == 8 && scales.dtype == llama_safetensors_dtype::BF16 && group_size % 128 != 0) {
                throw std::runtime_error(
                    "BF16-scaled GPTQ INT8 requires groups divisible by 128 for exact Q8_0_G128 execution");
            }
            if (g_idx != nullptr) {
                const std::vector<uint8_t> indices = registry_.read(*g_idx);
                for (size_t col = 0; col < cols; ++col) {
                    int32_t group_index;
                    std::memcpy(&group_index, indices.data() + col * sizeof(group_index), sizeof(group_index));
                    if (group_index < 0 || static_cast<uint64_t>(group_index) >= groups) {
                        throw std::runtime_error("GPTQ g_idx contains an out-of-range group index");
                    }
                    if (!group->act_order && static_cast<uint64_t>(group_index) != col / group_size) {
                        throw std::runtime_error("GPTQ g_idx is not the identity group map");
                    }
                }
            }
        } else if (expected == llama_safetensors_quant_format::INT8_CHANNEL) {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (!group->input_dynamic) {
                const std::string input_scale_name = module +
                    (group->legacy_fp8_i8_storage ? ".in_scale" : ".input_scale");
                const auto & input_scale = require_tensor(registry_, input_scale_name);
                dependencies_[tensor.name].push_back(input_scale_name);
                if ((input_scale.dtype != llama_safetensors_dtype::BF16 &&
                     input_scale.dtype != llama_safetensors_dtype::F16 &&
                     input_scale.dtype != llama_safetensors_dtype::F32) ||
                    (input_scale.shape != std::vector<uint64_t>({ 1 }) && !input_scale.shape.empty())) {
                    throw std::runtime_error(
                        "invalid static W8A8 input scale for source tensor '" + tensor.name + "'");
                }
                if (!group->input_symmetric) {
                    const std::string input_zero_name = module + ".input_zero_point";
                    const auto & input_zero = require_tensor(registry_, input_zero_name);
                    dependencies_[tensor.name].push_back(input_zero_name);
                    if (input_zero.dtype != llama_safetensors_dtype::I8 ||
                        (input_zero.shape != std::vector<uint64_t>({ 1 }) && !input_zero.shape.empty())) {
                        throw std::runtime_error(
                            "invalid static W8A8 input zero point for source tensor '" + tensor.name + "'");
                    }
                }
            }
            if (tensor.shape.size() != 2 || tensor.shape[1] % 32 != 0 ||
                (scale.dtype != llama_safetensors_dtype::BF16 &&
                 scale.dtype != llama_safetensors_dtype::F16 &&
                 (scale.dtype != llama_safetensors_dtype::F32 || !group->modelopt)) ||
                scale.shape.empty() || scale.shape.size() > 2 ||
                scale.shape[0] != tensor.shape[0] || (scale.shape.size() == 2 && scale.shape[1] != 1)) {
                throw std::runtime_error("invalid W8A8 channel-scale contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::INT4_GROUP) {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (!group->input_dynamic) {
                const std::string input_scale_name = module + ".input_scale";
                const auto & input_scale = require_tensor(registry_, input_scale_name);
                dependencies_[tensor.name].push_back(input_scale_name);
                if ((input_scale.dtype != llama_safetensors_dtype::BF16 &&
                     input_scale.dtype != llama_safetensors_dtype::F16 &&
                     input_scale.dtype != llama_safetensors_dtype::F32) ||
                    (input_scale.shape != std::vector<uint64_t>({ 1 }) && !input_scale.shape.empty())) {
                    throw std::runtime_error(
                        "invalid static W4A8 input scale for source tensor '" + tensor.name + "'");
                }
            }
            if (tensor.shape.size() != 2 || tensor.shape[1] % 128 != 0 ||
                scale.dtype != llama_safetensors_dtype::BF16 ||
                scale.shape != std::vector<uint64_t>({ tensor.shape[0], tensor.shape[1] / 128 })) {
                throw std::runtime_error("invalid W4A8 group-128 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::MXFP4_PACK) {
            const std::string scale_name = module + (group->hf_mxfp4 ? ".scale" : ".weight_scale");
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            const bool bytes_ok = group->hf_mxfp4 ? tensor.dtype == llama_safetensors_dtype::I8 :
                                                    tensor.dtype == llama_safetensors_dtype::U8;
            const bool scale_ok = group->hf_mxfp4 ? scale.dtype == llama_safetensors_dtype::F8_E8M0 :
                                                    scale.dtype == llama_safetensors_dtype::U8;
            if (!bytes_ok || tensor.shape.size() != 2 ||
                tensor.shape[1] % 16 != 0 || !scale_ok ||
                scale.shape != std::vector<uint64_t>({ tensor.shape[0], tensor.shape[1] / 16 })) {
                throw std::runtime_error("invalid packed MXFP4 group-32 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::MXFP8) {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (tensor.dtype != llama_safetensors_dtype::F8_E4M3 || tensor.shape.size() != 2 ||
                tensor.shape[1] % 32 != 0 || scale.dtype != llama_safetensors_dtype::U8 ||
                scale.shape != std::vector<uint64_t>({ tensor.shape[0], tensor.shape[1] / 32 })) {
                throw std::runtime_error("invalid MXFP8 group-32 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::FP8_GROUP) {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (tensor.dtype != llama_safetensors_dtype::F8_E4M3 || tensor.shape.size() != 2 ||
                tensor.shape[1] % 32 != 0 || scale.dtype != llama_safetensors_dtype::BF16 ||
                scale.shape != std::vector<uint64_t>({ tensor.shape[0], tensor.shape[1] / 32 })) {
                throw std::runtime_error("invalid grouped FP8 group-32 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::FP8_TENSOR) {
            const std::string weight_scale_name = module + ".weight_scale";
            const auto & weight_scale = require_tensor(registry_, weight_scale_name);
            dependencies_[tensor.name] = { weight_scale_name };
            const bool valid_scale_dtype = group->modelopt ?
                weight_scale.dtype == llama_safetensors_dtype::F32 :
                weight_scale.dtype == llama_safetensors_dtype::BF16;
            const bool valid_scale_shape = group->modelopt ?
                weight_scale.shape.empty() || weight_scale.shape == std::vector<uint64_t>({ 1 }) :
                weight_scale.shape == std::vector<uint64_t>({ 1 });
            if (tensor.shape.size() != 2 || !valid_scale_dtype || !valid_scale_shape) {
                throw std::runtime_error("invalid static tensor-scale FP8 contract for source tensor '" + tensor.name + "'");
            }
            if (group->input_quantized && !group->input_dynamic) {
                const std::string input_scale_name = module +
                    (group->legacy_fp8_i8_storage ? ".in_scale" : ".input_scale");
                const auto & input_scale = require_tensor(registry_, input_scale_name);
                dependencies_[tensor.name].push_back(input_scale_name);
                const bool valid_input_dtype = group->modelopt ?
                    input_scale.dtype == llama_safetensors_dtype::F32 :
                    input_scale.dtype == llama_safetensors_dtype::BF16;
                const bool valid_input_shape = group->modelopt ?
                    input_scale.shape.empty() || input_scale.shape == std::vector<uint64_t>({ 1 }) :
                    input_scale.shape == std::vector<uint64_t>({ 1 });
                if (!valid_input_dtype || !valid_input_shape) {
                    throw std::runtime_error(
                        "invalid static tensor-scale FP8 input for source tensor '" + tensor.name + "'");
                }
            }
        } else if (expected == llama_safetensors_quant_format::FP8_CHANNEL) {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            const llama_safetensors_dtype expected_scale_dtype = group->modelopt ?
                llama_safetensors_dtype::F32 : llama_safetensors_dtype::BF16;
            if (tensor.shape.size() != 2 || scale.dtype != expected_scale_dtype || scale.shape.empty() ||
                scale.shape.size() > 2 || scale.shape[0] != tensor.shape[0] ||
                (scale.shape.size() == 2 && scale.shape[1] != 1)) {
                throw std::runtime_error("invalid channel-scale contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::FP8_BLOCK) {
            const std::string scale_name = module + (group->modelopt ? ".weight_scale" :
                group->e8m0_block_scale ? ".scale" : ".weight_scale_inv");
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (tensor.shape.size() != 2 || tensor.shape[0] % 128 != 0 || tensor.shape[1] % 128 != 0) {
                throw std::runtime_error("invalid 128x128 block-scale contract for source tensor '" + tensor.name + "'");
            }
            const bool source_scale_valid = group->modelopt ?
                scale.dtype == llama_safetensors_dtype::F32 && scale.shape == std::vector<uint64_t>({
                    tensor.shape[0] / 128, 1, tensor.shape[1] / 128, 1 }) :
                group->e8m0_block_scale ?
                scale.dtype == llama_safetensors_dtype::F8_E8M0 && scale.shape == std::vector<uint64_t>({
                    tensor.shape[0] / 128, tensor.shape[1] / 128 }) :
                scale.dtype == llama_safetensors_dtype::BF16 && scale.shape == std::vector<uint64_t>({
                    tensor.shape[0] / 128, tensor.shape[1] / 128 });
            if (!source_scale_valid) {
                throw std::runtime_error("invalid 128x128 block-scale contract for source tensor '" + tensor.name + "'");
            }
        } else {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            const std::string weight_global = module + (group->modelopt ? ".weight_scale_2" : ".weight_global_scale");
            dependencies_[tensor.name].push_back(weight_global);
            if (!group->modelopt || group->input_quantized) {
                dependencies_[tensor.name].push_back(
                    module + (group->modelopt ? ".input_scale" : ".input_global_scale"));
            }
            const bool ranks_valid = (tensor.shape.size() == 2 || tensor.shape.size() == 3) &&
                scale.shape.size() == tensor.shape.size();
            const bool leading_dims_match = ranks_valid && std::equal(
                tensor.shape.begin(), tensor.shape.end() - 1, scale.shape.begin());
            if (!ranks_valid || tensor.dtype != llama_safetensors_dtype::U8 ||
                scale.dtype != llama_safetensors_dtype::F8_E4M3 || !leading_dims_match ||
                tensor.shape.back() % 8 != 0 || tensor.shape.back() / 8 != scale.shape.back() ||
                scale.shape.back() % 4 != 0) {
                throw std::runtime_error("invalid packed NVFP4 scale contract for source tensor '" + tensor.name + "'");
            }
            require_positive_f32_scalar(registry_, weight_global);
            if (!group->modelopt || group->input_quantized) {
                require_positive_f32_scalar(
                    registry_, module + (group->modelopt ? ".input_scale" : ".input_global_scale"));
            }
        }
    }
    if (has_quantization_config_ &&
        summary_.nvfp4 + summary_.mxfp4 + summary_.mxfp8 + summary_.fp8_group + summary_.fp8_tensor + summary_.fp8_channel + summary_.fp8_block + summary_.w8a8 + summary_.w4a8 + summary_.w4a8_fp8 + summary_.awq + summary_.quark_w4a16 + summary_.gptq + summary_.gptq_int8 + summary_.eetq + summary_.quanto_int4 + summary_.quanto_int8 + summary_.quanto_fp8 + summary_.torchao_int4 + summary_.torchao_intx + summary_.hqq_int4 + summary_.bnb_int8 + summary_.bnb_nf4 + summary_.bnb_fp4 +
                summary_.packed_int4 + summary_.packed_int8 == 0) {
        throw std::runtime_error("native safetensors model contains no supported quantized weights");
    }
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_nvfp4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        size_t weight_size,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        size_t scale_size) const {
    if (weight_desc.dtype != llama_safetensors_dtype::U8 || scale_desc.dtype != llama_safetensors_dtype::F8_E4M3 ||
        (weight_desc.shape.size() != 2 && weight_desc.shape.size() != 3) ||
        scale_desc.shape.size() != weight_desc.shape.size() ||
        !std::equal(weight_desc.shape.begin(), weight_desc.shape.end() - 1, scale_desc.shape.begin())) {
        throw std::runtime_error("invalid NVFP4 source tensor contract");
    }
    size_t rows = 1;
    for (size_t i = 0; i + 1 < weight_desc.shape.size(); ++i) {
        if (weight_desc.shape[i] > std::numeric_limits<size_t>::max() / rows) {
            throw std::runtime_error("NVFP4 source tensor is too large");
        }
        rows *= static_cast<size_t>(weight_desc.shape[i]);
    }
    const size_t packed_cols = weight_desc.shape.back();
    const size_t blocks      = scale_desc.shape.back();
    if (packed_cols != blocks * 8 || blocks % 4 != 0 ||
        weight_size != rows * packed_cols || scale_size != rows * blocks) {
        throw std::runtime_error("inconsistent NVFP4 source tensor shapes");
    }
    for (size_t i = 0; i < scale_size; ++i) {
        if ((scale[i] & 0x80) != 0 || scale[i] == 0x7f) {
            throw std::runtime_error("NVFP4 block scale is not a non-negative finite E4M3 value");
        }
    }

    std::vector<uint8_t> result(rows * (blocks / 4) * 36);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t super = 0; super < blocks / 4; ++super) {
            uint8_t * out = result.data() + (row * (blocks / 4) + super) * 36;
            for (size_t block = 0; block < 4; ++block) {
                const size_t  block_index = 4 * super + block;
                const uint8_t scale_byte = scale[row * blocks + block_index];
                out[block] = scale_byte;
                const uint8_t * in = weight + row * packed_cols + block_index * 8;
                uint8_t values[16];
                for (size_t i = 0; i < 8; ++i) {
                    values[2 * i + 0] = in[i] & 0x0f;
                    values[2 * i + 1] = in[i] >> 4;
                }
                for (size_t i = 0; i < 8; ++i) {
                    out[4 + block * 8 + i] = values[i] | (values[i + 8] << 4);
                }
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_eetq(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = sizeof(ggml_fp16_t) + qk;
    if (weight_desc.dtype != llama_safetensors_dtype::I8 || weight_desc.shape.size() != 2 ||
        weight_desc.shape[0] == 0 || weight_desc.shape[0] % qk != 0 || weight_desc.shape[1] == 0 ||
        scale_desc.dtype != llama_safetensors_dtype::F16 ||
        (scale_desc.shape != std::vector<uint64_t>({ weight_desc.shape[1] }) &&
         scale_desc.shape != std::vector<uint64_t>({ 1, weight_desc.shape[1] }))) {
        throw std::runtime_error("inconsistent EETQ W8A16 source tensors");
    }

    const size_t cols = weight_desc.shape[0];
    const size_t rows = weight_desc.shape[1];
    for (size_t row = 0; row < rows; ++row) {
        ggml_fp16_t scale_bits;
        std::memcpy(&scale_bits, scale + row * sizeof(scale_bits), sizeof(scale_bits));
        const float scale_f32 = ggml_fp16_to_fp32(scale_bits);
        if (!(scale_f32 > 0.0f) || !std::isfinite(scale_f32)) {
            throw std::runtime_error("EETQ W8A16 scale must be finite and positive");
        }
    }
    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
#ifdef _OPENMP
#pragma omp parallel for if(rows >= 128)
#endif
    for (int64_t row_i = 0; row_i < static_cast<int64_t>(rows); ++row_i) {
        const size_t row = static_cast<size_t>(row_i);
        ggml_fp16_t scale_bits;
        std::memcpy(&scale_bits, scale + row * sizeof(scale_bits), sizeof(scale_bits));
        for (size_t block = 0; block < cols / qk; ++block) {
            uint8_t * out = result.data() + (row * (cols / qk) + block) * block_size;
            std::memcpy(out, &scale_bits, sizeof(scale_bits));
            for (size_t local = 0; local < qk; ++local) {
                const size_t col = block * qk + local;
                out[sizeof(scale_bits) + local] = weight[col * rows + row];
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_quanto_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & shift_desc,
        const uint8_t * shift,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + qk / 2;
    if (target_shape.size() != 2 || target_shape[0] <= 0 || target_shape[1] <= 0 ||
        group_size == 0 || group_size % qk != 0) {
        throw std::runtime_error("invalid Quanto qint4 target geometry");
    }
    const size_t cols = static_cast<size_t>(target_shape[0]);
    const size_t rows = static_cast<size_t>(target_shape[1]);
    if (cols % group_size != 0) {
        throw std::runtime_error("invalid Quanto qint4 group size");
    }
    const size_t groups_per_row = cols / group_size;
    const size_t groups = rows * groups_per_row;
    const bool scale_dtype = scale_desc.dtype == llama_safetensors_dtype::F16 ||
        scale_desc.dtype == llama_safetensors_dtype::BF16;
    const bool shift_dtype = shift_desc.dtype == llama_safetensors_dtype::F16 ||
        shift_desc.dtype == llama_safetensors_dtype::BF16;
    if (weight_desc.dtype != llama_safetensors_dtype::U8 || groups % 2 != 0 ||
        weight_desc.shape != std::vector<uint64_t>({ groups / 2, group_size }) ||
        weight_desc.size != groups * group_size / 2 || !scale_dtype || !shift_dtype ||
        scale_desc.shape != std::vector<uint64_t>({ groups, 1 }) ||
        shift_desc.shape != std::vector<uint64_t>({ groups, 1 })) {
        throw std::runtime_error("inconsistent Quanto qint4 source tensors");
    }
    const auto read_aux = [](const llama_safetensors_tensor & desc, const uint8_t * data, size_t index) {
        if (desc.dtype == llama_safetensors_dtype::BF16) {
            return load_bf16(data + index * sizeof(uint16_t));
        }
        ggml_fp16_t bits;
        std::memcpy(&bits, data + index * sizeof(bits), sizeof(bits));
        return ggml_fp16_to_fp32(bits);
    };
    std::vector<ggml_fp16_t> scale_bits(groups);
    std::vector<ggml_fp16_t> minimum_bits(groups);
    for (size_t group = 0; group < groups; ++group) {
        const float d = read_aux(scale_desc, scale, group);
        const float m = -read_aux(shift_desc, shift, group);
        scale_bits[group] = ggml_fp32_to_fp16(d);
        minimum_bits[group] = ggml_fp32_to_fp16(m);
        if (!(d > 0.0f) || !std::isfinite(d) || !std::isfinite(m) ||
            !std::isfinite(ggml_fp16_to_fp32(scale_bits[group])) ||
            !std::isfinite(ggml_fp16_to_fp32(minimum_bits[group]))) {
            throw std::runtime_error("Quanto qint4 scale or shift is not representable in Q4_1");
        }
    }

    const auto code = [&](size_t group, size_t local) {
        const size_t packed_group = group % (groups / 2);
        const uint8_t packed = weight[packed_group * group_size + local];
        return static_cast<uint8_t>(group < groups / 2 ? packed & 0x0f : packed >> 4);
    };
    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
#ifdef _OPENMP
#pragma omp parallel for if(rows >= 128)
#endif
    for (int64_t row_i = 0; row_i < static_cast<int64_t>(rows); ++row_i) {
        const size_t row = static_cast<size_t>(row_i);
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t col = block * qk;
            const size_t group = row * groups_per_row + col / group_size;
            const size_t group_col = col % group_size;
            uint8_t * out = result.data() + (row * (cols / qk) + block) * block_size;
            std::memcpy(out, &scale_bits[group], sizeof(ggml_fp16_t));
            std::memcpy(out + sizeof(ggml_fp16_t), &minimum_bits[group], sizeof(ggml_fp16_t));
            for (size_t local = 0; local < qk / 2; ++local) {
                out[2 * sizeof(ggml_fp16_t) + local] =
                    code(group, group_col + local) | (code(group, group_col + local + qk / 2) << 4);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_torchao_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_zero_desc,
        const uint8_t * scale_zero,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const {
    constexpr size_t qk = 32;
    constexpr size_t inner_k_tiles = 8;
    constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + qk / 2;
    if (target_shape.size() != 2 || target_shape[0] <= 0 || target_shape[1] <= 0 ||
        group_size < qk || group_size % qk != 0 || weight_desc.dtype != llama_safetensors_dtype::I32 ||
        weight_desc.shape.size() != 4 || weight_desc.shape[2] != 32 ||
        weight_desc.shape[3] != inner_k_tiles / 2 || scale_zero_desc.dtype != llama_safetensors_dtype::BF16 ||
        scale_zero_desc.shape.size() != 3 || scale_zero_desc.shape[2] != 2) {
        throw std::runtime_error("inconsistent TorchAO tiled-int4 source tensors");
    }
    const size_t cols = static_cast<size_t>(target_shape[0]);
    const size_t rows = static_cast<size_t>(target_shape[1]);
    const size_t padded_rows = static_cast<size_t>(weight_desc.shape[0]) * 8;
    const size_t padded_cols = static_cast<size_t>(weight_desc.shape[1]) * inner_k_tiles * 16;
    const size_t scale_rows  = static_cast<size_t>(scale_zero_desc.shape[1]);
    const size_t scale_groups = static_cast<size_t>(scale_zero_desc.shape[0]);
    if (cols % group_size != 0 || rows > padded_rows || cols > padded_cols ||
        rows > scale_rows || cols / group_size > scale_groups) {
        throw std::runtime_error("TorchAO tiled-int4 source geometry does not cover its logical shape");
    }

    std::vector<uint8_t> packed_rows(padded_rows * (padded_cols / 2), 0);
    for (size_t n_tile = 0; n_tile < weight_desc.shape[0]; ++n_tile) {
        for (size_t k_outer = 0; k_outer < weight_desc.shape[1]; ++k_outer) {
            for (size_t lane = 0; lane < 32; ++lane) {
                const size_t row = n_tile * 8 + lane / 4;
                for (size_t inner = 0; inner < inner_k_tiles; inner += 2) {
                    const size_t index = (((n_tile * weight_desc.shape[1] + k_outer) * 32 + lane) *
                                          (inner_k_tiles / 2)) + inner / 2;
                    uint32_t word;
                    std::memcpy(&word, weight + index * sizeof(word), sizeof(word));
                    uint8_t values[4];
                    for (size_t i = 0; i < 4; ++i) {
                        values[i] = static_cast<uint8_t>(((word >> (16 + 4 * i)) & 0x0f) |
                                                        (((word >> (4 * i)) & 0x0f) << 4));
                    }
                    const size_t base = (k_outer * inner_k_tiles + inner) * 8;
                    const size_t offset = lane % 4;
                    const size_t positions[4] = { base + offset, base + offset + 4,
                                                  base + 8 + offset, base + 12 + offset };
                    for (size_t i = 0; i < 4; ++i) {
                        packed_rows[row * (padded_cols / 2) + positions[i]] = values[i];
                    }
                }
            }
        }
    }

    const auto scale_value = [&](size_t group, size_t row, size_t component) {
        return load_bf16(scale_zero + ((group * scale_rows + row) * 2 + component) * sizeof(uint16_t));
    };
    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
#ifdef _OPENMP
#pragma omp parallel for if(rows >= 128)
#endif
    for (int64_t row_i = 0; row_i < static_cast<int64_t>(rows); ++row_i) {
        const size_t row = static_cast<size_t>(row_i);
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t col = block * qk;
            const size_t group = col / group_size;
            const float d = scale_value(group, row, 0);
            const float m = scale_value(group, row, 1) - 8.0f * d;
            const ggml_fp16_t d_bits = ggml_fp32_to_fp16(d);
            const ggml_fp16_t m_bits = ggml_fp32_to_fp16(m);
            if (!(d > 0.0f) || !std::isfinite(d) || !std::isfinite(m) ||
                !std::isfinite(ggml_fp16_to_fp32(d_bits)) || !std::isfinite(ggml_fp16_to_fp32(m_bits))) {
                throw std::runtime_error("TorchAO tiled-int4 scale or zero is not representable in Q4_1");
            }
            uint8_t * out = result.data() + (row * (cols / qk) + block) * block_size;
            std::memcpy(out, &d_bits, sizeof(d_bits));
            std::memcpy(out + sizeof(d_bits), &m_bits, sizeof(m_bits));
            for (size_t i = 0; i < qk / 2; ++i) {
                const auto code = [&](size_t logical_col) {
                    const uint8_t packed = packed_rows[row * (padded_cols / 2) + logical_col / 2];
                    return static_cast<uint8_t>(logical_col % 2 == 0 ? packed >> 4 : packed & 0x0f);
                };
                out[2 * sizeof(ggml_fp16_t) + i] = code(col + i) | (code(col + i + qk / 2) << 4);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_torchao_intx(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & zero_desc,
        const uint8_t * zero,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = sizeof(ggml_fp16_t) + qk;
    if (target_shape.size() != 2 || target_shape[0] <= 0 || target_shape[1] <= 0 ||
        group_size < qk || group_size % qk != 0 || weight_desc.dtype != llama_safetensors_dtype::I8 ||
        weight_desc.shape.size() != 2 ||
        (scale_desc.dtype != llama_safetensors_dtype::BF16 && scale_desc.dtype != llama_safetensors_dtype::F16) ||
        (zero_desc.dtype != llama_safetensors_dtype::I8 && zero_desc.dtype != scale_desc.dtype)) {
        throw std::runtime_error("inconsistent TorchAO unpacked-intx source tensors");
    }
    const size_t cols = static_cast<size_t>(target_shape[0]);
    const size_t rows = static_cast<size_t>(target_shape[1]);
    const size_t groups = cols / group_size;
    if (cols % group_size != 0 || weight_desc.shape != std::vector<uint64_t>({ rows, cols }) ||
        scale_desc.shape != std::vector<uint64_t>({ rows, groups }) || zero_desc.shape != scale_desc.shape) {
        throw std::runtime_error("TorchAO unpacked-intx source geometry is inconsistent");
    }
    const auto read_float = [](const llama_safetensors_tensor & desc, const uint8_t * data, size_t index) {
        if (desc.dtype == llama_safetensors_dtype::BF16) {
            return load_bf16(data + index * sizeof(uint16_t));
        }
        ggml_fp16_t bits;
        std::memcpy(&bits, data + index * sizeof(bits), sizeof(bits));
        return ggml_fp16_to_fp32(bits);
    };
    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
#ifdef _OPENMP
#pragma omp parallel for if(rows >= 128)
#endif
    for (int64_t row_i = 0; row_i < static_cast<int64_t>(rows); ++row_i) {
        const size_t row = static_cast<size_t>(row_i);
        for (size_t group = 0; group < groups; ++group) {
            const size_t qparam = row * groups + group;
            const float zero_value = zero_desc.dtype == llama_safetensors_dtype::I8 ?
                static_cast<float>(static_cast<int8_t>(zero[qparam])) : read_float(zero_desc, zero, qparam);
            const float d = read_float(scale_desc, scale, qparam);
            const ggml_fp16_t d_bits = ggml_fp32_to_fp16(d);
            if (zero_value != 0.0f || !(d > 0.0f) || !std::isfinite(d) ||
                !std::isfinite(ggml_fp16_to_fp32(d_bits))) {
                throw std::runtime_error("TorchAO unpacked-intx requires a finite positive scale and zero zero-point");
            }
            for (size_t local_block = 0; local_block < group_size / qk; ++local_block) {
                const size_t block = group * (group_size / qk) + local_block;
                uint8_t * out = result.data() + (row * (cols / qk) + block) * block_size;
                std::memcpy(out, &d_bits, sizeof(d_bits));
                std::memcpy(out + sizeof(d_bits), weight + row * cols + block * qk, qk);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_hqq_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & zero_desc,
        const uint8_t * zero,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const {
    constexpr size_t qk = 32;
    constexpr size_t block_bytes = 2 * sizeof(ggml_fp16_t) + qk / 2;
    if (target_shape.size() != 2 || target_shape[0] <= 0 || target_shape[1] <= 0 ||
        group_size < qk || group_size % qk != 0) {
        throw std::runtime_error("invalid HQQ INT4 target geometry");
    }
    const size_t cols = static_cast<size_t>(target_shape[0]);
    const size_t rows = static_cast<size_t>(target_shape[1]);
    if (cols % qk != 0 || rows > std::numeric_limits<size_t>::max() / cols) {
        throw std::runtime_error("invalid HQQ INT4 target size");
    }
    const size_t values = rows * cols;
    if (values % group_size != 0) {
        throw std::runtime_error("invalid HQQ INT4 group geometry");
    }
    const size_t groups = values / group_size;
    if (groups % 2 != 0 || weight_desc.dtype != llama_safetensors_dtype::U8 ||
        weight_desc.shape != std::vector<uint64_t>({ groups / 2, group_size }) ||
        weight_desc.size != values / 2 || scale_desc.dtype != llama_safetensors_dtype::F16 ||
        scale_desc.shape != std::vector<uint64_t>({ groups, 1 }) ||
        zero_desc.dtype != llama_safetensors_dtype::F16 ||
        zero_desc.shape != std::vector<uint64_t>({ groups, 1 })) {
        throw std::runtime_error("inconsistent HQQ INT4 source tensors");
    }

    std::vector<uint8_t> result(rows * (cols / qk) * block_bytes);
    const size_t packed_groups = groups / 2;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t first = row * cols + block * qk;
            const size_t group = first / group_size;
            ggml_fp16_t scale_bits;
            ggml_fp16_t zero_bits;
            std::memcpy(&scale_bits, scale + group * sizeof(scale_bits), sizeof(scale_bits));
            std::memcpy(&zero_bits, zero + group * sizeof(zero_bits), sizeof(zero_bits));
            const float d = ggml_fp16_to_fp32(scale_bits);
            const float z = ggml_fp16_to_fp32(zero_bits);
            const ggml_fp16_t minimum = ggml_fp32_to_fp16(-d * z);
            if (!(d > 0.0f) || !std::isfinite(d) || !std::isfinite(z) ||
                !std::isfinite(ggml_fp16_to_fp32(minimum))) {
                throw std::runtime_error("HQQ INT4 scale or zero is not representable in Q4_1");
            }

            uint8_t * out = result.data() + (row * (cols / qk) + block) * block_bytes;
            std::memcpy(out, &scale_bits, sizeof(scale_bits));
            std::memcpy(out + sizeof(scale_bits), &minimum, sizeof(minimum));
            const auto code = [&](size_t col) {
                const size_t flat = row * cols + col;
                const size_t source_group = flat / group_size;
                const size_t source_col = flat % group_size;
                const size_t packed_group = source_group % packed_groups;
                const uint8_t packed = weight[packed_group * group_size + source_col];
                return uint8_t(source_group < packed_groups ? packed >> 4 : packed & 0x0f);
            };
            for (size_t lane = 0; lane < qk / 2; ++lane) {
                out[2 * sizeof(ggml_fp16_t) + lane] =
                    code(block * qk + lane) | (code(block * qk + lane + qk / 2) << 4);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_mxfp4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale) const {
    constexpr size_t values_per_block = 32;
    constexpr size_t packed_per_block = values_per_block / 2;
    constexpr size_t block_size = 1 + packed_per_block;
    if ((weight_desc.dtype != llama_safetensors_dtype::U8 && weight_desc.dtype != llama_safetensors_dtype::I8) ||
        weight_desc.shape.size() != 2 ||
        weight_desc.shape[1] % packed_per_block != 0 ||
        (scale_desc.dtype != llama_safetensors_dtype::U8 &&
         scale_desc.dtype != llama_safetensors_dtype::F8_E8M0) ||
        scale_desc.shape != std::vector<uint64_t>({
            weight_desc.shape[0], weight_desc.shape[1] / packed_per_block })) {
        throw std::runtime_error("inconsistent compressed-tensors MXFP4 source tensors");
    }

    const size_t rows = weight_desc.shape[0];
    const size_t packed_cols = weight_desc.shape[1];
    const size_t blocks_per_row = packed_cols / packed_per_block;
    std::vector<uint8_t> result(rows * blocks_per_row * block_size);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t block_index = row * blocks_per_row + block;
            if (scale[block_index] == 0xff) {
                throw std::runtime_error("MXFP4 E8M0 scale uses the reserved NaN encoding");
            }
            uint8_t * out = result.data() + block_index * block_size;
            out[0] = scale[block_index];
            const uint8_t * in = weight + row * packed_cols + block * packed_per_block;
            // The source packs adjacent values in each byte. GGML packs the
            // first and second 16-value halves together. Move two output
            // bytes per iteration without expanding the block to nibbles.
            for (size_t i = 0; i < packed_per_block / 2; ++i) {
                const uint8_t lo = in[i];
                const uint8_t hi = in[i + packed_per_block / 2];
                out[1 + 2 * i + 0] = (lo & 0x0f) | (hi << 4);
                out[1 + 2 * i + 1] = (lo >> 4) | (hi & 0xf0);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_packed_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor * zero_desc,
        const uint8_t * zero,
        const std::array<uint64_t, 2> & weight_shape,
        uint32_t group_size,
        bool symmetric) const {
    constexpr size_t pack_factor = 8;
    const size_t rows = weight_shape[0];
    const size_t cols = weight_shape[1];
    if (group_size < 32 || group_size % 32 != 0 || cols % group_size != 0) {
        throw std::runtime_error("inconsistent compressed-tensors INT4 group size");
    }
    const size_t groups = cols / group_size;
    const size_t packed_cols = cols / pack_factor;
    const size_t packed_rows = (rows + pack_factor - 1) / pack_factor;
    if (weight_desc.dtype != llama_safetensors_dtype::I32 ||
        weight_desc.shape != std::vector<uint64_t>({ rows, packed_cols }) ||
        scale_desc.dtype != llama_safetensors_dtype::BF16 ||
        scale_desc.shape != std::vector<uint64_t>({ rows, groups }) ||
        (symmetric ? zero_desc != nullptr || zero != nullptr :
            zero_desc == nullptr || zero == nullptr || zero_desc->dtype != llama_safetensors_dtype::I32 ||
            zero_desc->shape != std::vector<uint64_t>({ packed_rows, groups }))) {
        throw std::runtime_error("inconsistent compressed-tensors INT4 group source tensors");
    }

    constexpr size_t block_values = 128;
    constexpr size_t block_scales = 4 * sizeof(uint16_t);
    constexpr size_t block_zeros  = 2;
    constexpr size_t block_size   = block_scales + block_zeros + block_values / 2;
    if (cols % block_values != 0) {
        throw std::runtime_error("compressed-tensors INT4 rows must be divisible by 128");
    }

    std::vector<uint8_t> result(rows * (cols / block_values) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t ib = 0; ib < cols / block_values; ++ib) {
            uint8_t * out = result.data() + (row * (cols / block_values) + ib) * block_size;
            for (size_t local_group = 0; local_group < block_values / 32; ++local_group) {
                const size_t col = ib * block_values + local_group * 32;
                const size_t group = col / group_size;
                uint16_t scale_bits;
                std::memcpy(&scale_bits, scale + (row * groups + group) * sizeof(scale_bits), sizeof(scale_bits));
                const float scale_f32 = load_bf16(reinterpret_cast<const uint8_t *>(&scale_bits));
                if (!(scale_f32 > 0.0f) || !std::isfinite(scale_f32)) {
                    throw std::runtime_error("packed INT4 scale must be finite and positive");
                }
                std::memcpy(out + local_group * sizeof(scale_bits), &scale_bits, sizeof(scale_bits));

                uint8_t zero_code = 8;
                if (!symmetric) {
                    uint32_t packed_zero;
                    std::memcpy(&packed_zero,
                                zero + ((row / pack_factor) * groups + group) * sizeof(packed_zero),
                                sizeof(packed_zero));
                    zero_code = (packed_zero >> (4 * (row % pack_factor))) & 0x0f;
                }
                out[block_scales + local_group / 2] |= zero_code << (4 * (local_group % 2));
            }
            // compressed-tensors packs consecutive low-to-high nibbles, which
            // is already the canonical adjacent-pair byte order.
            std::memcpy(
                out + block_scales + block_zeros,
                weight + (row * packed_cols + ib * (block_values / pack_factor)) * sizeof(uint32_t),
                block_values / 2);
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_packed_int8(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const std::array<uint64_t, 2> & weight_shape) const {
    constexpr size_t pack_factor = 4;
    constexpr size_t group_size = 128;
    const size_t rows = weight_shape[0];
    const size_t cols = weight_shape[1];
    const size_t groups = cols / group_size;
    const size_t packed_cols = cols / pack_factor;
    if (weight_desc.dtype != llama_safetensors_dtype::I32 ||
        weight_desc.shape != std::vector<uint64_t>({ rows, packed_cols }) ||
        scale_desc.dtype != llama_safetensors_dtype::BF16 ||
        scale_desc.shape != std::vector<uint64_t>({ rows, groups })) {
        throw std::runtime_error("inconsistent compressed-tensors INT8 group-128 source tensors");
    }

    constexpr size_t block_values = 128;
    constexpr size_t block_size   = sizeof(uint16_t) + block_values;
    if (cols % block_values != 0) {
        throw std::runtime_error("compressed-tensors INT8 rows must be divisible by 128");
    }

    const size_t blocks_per_row = cols / block_values;
    std::vector<uint8_t> result(rows * blocks_per_row * block_size);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            const size_t group = block;
            uint8_t * out = result.data() + (row * blocks_per_row + block) * block_size;
            std::memcpy(out, scale + (row * groups + group) * sizeof(uint16_t), sizeof(uint16_t));
            const float scale_f32 = load_bf16(out);
            if (!(scale_f32 > 0.0f) || !std::isfinite(scale_f32)) {
                throw std::runtime_error("packed INT8 scale must be finite and positive");
            }
            const uint8_t * packed = weight +
                (row * packed_cols + block * (block_values / pack_factor)) * sizeof(uint32_t);
            for (size_t i = 0; i < block_values; ++i) {
                // Offset-binary to signed INT8 is exactly a sign-bit flip.
                out[sizeof(uint16_t) + i] = packed[i] ^ 0x80;
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_packed_int4_fp8(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & channel_scale_desc,
        const uint8_t * channel_scale,
        const std::array<uint64_t, 2> & weight_shape) const {
    constexpr size_t pack_factor = 8;
    constexpr size_t block_values = 128;
    constexpr size_t block_scales = 4 * sizeof(uint16_t);
    constexpr size_t block_zeros = 2;
    constexpr size_t block_size = block_scales + block_zeros + block_values / 2;
    const size_t rows = weight_shape[0];
    const size_t cols = weight_shape[1];
    const size_t groups = cols / block_values;
    if (weight_desc.dtype != llama_safetensors_dtype::I32 ||
        weight_desc.shape != std::vector<uint64_t>({ rows, cols / pack_factor }) ||
        scale_desc.dtype != llama_safetensors_dtype::F8_E4M3 ||
        scale_desc.shape != std::vector<uint64_t>({ rows, groups }) ||
        channel_scale_desc.dtype != llama_safetensors_dtype::F32 ||
        channel_scale_desc.shape != std::vector<uint64_t>({ rows, 1 }) ||
        cols % block_values != 0) {
        throw std::runtime_error("inconsistent compressed-tensors W4A8-FP8 source tensors");
    }

    const ggml_type_traits * f8 = ggml_get_type_traits(GGML_TYPE_F8_E4M3);
    std::vector<uint8_t> result(rows * groups * block_size, 0);
    for (size_t row = 0; row < rows; ++row) {
        float channel;
        std::memcpy(&channel, channel_scale + row * sizeof(channel), sizeof(channel));
        if (!(channel > 0.0f) || !std::isfinite(channel)) {
            throw std::runtime_error("W4A8-FP8 channel scale must be finite and positive");
        }
        for (size_t group = 0; group < groups; ++group) {
            uint8_t * out = result.data() + (row * groups + group) * block_size;
            float group_scale;
            f8->to_float(scale + row * groups + group, &group_scale, 1);
            const float effective = group_scale * channel;
            if (!(effective > 0.0f) || !std::isfinite(effective)) {
                throw std::runtime_error("W4A8-FP8 effective scale must be finite and positive");
            }
            const ggml_bf16_t bf16 = ggml_fp32_to_bf16(effective);
            if (!(ggml_bf16_to_fp32(bf16) > 0.0f) || !std::isfinite(ggml_bf16_to_fp32(bf16))) {
                throw std::runtime_error("W4A8-FP8 effective scale is not representable as BF16");
            }
            for (size_t local = 0; local < block_values / 32; ++local) {
                std::memcpy(out + local * sizeof(bf16.bits), &bf16.bits, sizeof(bf16.bits));
                out[block_scales + local / 2] |= uint8_t(8u << (4 * (local % 2)));
            }
            const uint8_t * packed = weight +
                (row * (cols / pack_factor) + group * (block_values / pack_factor)) * sizeof(uint32_t);
            uint8_t * codes = out + block_scales + block_zeros;
            for (size_t i = 0; i < block_values / 2; ++i) {
                // E2E CUTLASS W4A8 checkpoints have already converted the
                // compressed-tensors uint4b8 codes to signed int4. Q4_A32
                // stores unsigned codes with an explicit zero point, so move
                // both packed nibbles back to offset-binary.
                codes[i] = packed[i] ^ 0x88;
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_int4_group(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale) const {
    constexpr size_t block_values = 128;
    constexpr size_t groups_per_block = 4;
    constexpr size_t block_size = groups_per_block * sizeof(uint16_t) +
                                  groups_per_block / 2 + block_values / 2;
    if (weight_desc.dtype != llama_safetensors_dtype::I8 || weight_desc.shape.size() != 2 ||
        weight_desc.shape[1] % block_values != 0 ||
        scale_desc.dtype != llama_safetensors_dtype::BF16 ||
        scale_desc.shape != std::vector<uint64_t>({
            weight_desc.shape[0], weight_desc.shape[1] / block_values })) {
        throw std::runtime_error("inconsistent W4A8 group-128 source tensors");
    }

    const size_t rows = weight_desc.shape[0];
    const size_t cols = weight_desc.shape[1];
    const size_t blocks_per_row = cols / block_values;
    std::vector<uint8_t> result(rows * blocks_per_row * block_size);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t block = 0; block < blocks_per_row; ++block) {
            uint8_t * out = result.data() + (row * blocks_per_row + block) * block_size;
            uint16_t scale_bits;
            std::memcpy(&scale_bits,
                scale + (row * blocks_per_row + block) * sizeof(scale_bits), sizeof(scale_bits));
            const float scale_f32 = load_bf16(reinterpret_cast<const uint8_t *>(&scale_bits));
            if (!(scale_f32 > 0.0f) || !std::isfinite(scale_f32)) {
                throw std::runtime_error("W4A8 group scale must be finite and positive");
            }
            for (size_t group = 0; group < groups_per_block; ++group) {
                std::memcpy(out + group * sizeof(scale_bits), &scale_bits, sizeof(scale_bits));
            }
            out[groups_per_block * sizeof(scale_bits) + 0] = 0x88;
            out[groups_per_block * sizeof(scale_bits) + 1] = 0x88;

            const int8_t * source = reinterpret_cast<const int8_t *>(weight) +
                row * cols + block * block_values;
            uint8_t * codes = out + groups_per_block * sizeof(scale_bits) + groups_per_block / 2;
            for (size_t i = 0; i < block_values; i += 2) {
                if (source[i] < -8 || source[i] > 7 || source[i + 1] < -8 || source[i + 1] > 7) {
                    throw std::runtime_error("W4A8 weight value is outside signed INT4 range");
                }
                codes[i / 2] = uint8_t(source[i] + 8) | (uint8_t(source[i + 1] + 8) << 4);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_awq(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales,
        uint32_t group_size_config) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + qk / 2;
    constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };

    if (qweight_desc.shape.size() != 2) {
        throw std::runtime_error("invalid packed AWQ dimensions");
    }
    const size_t cols = qweight_desc.shape[0];
    const size_t rows = qweight_desc.shape[1] * 8;
    const size_t group_size = group_size_config == 0 ? cols : group_size_config;
    if (group_size == 0 || group_size % qk != 0 || cols % group_size != 0) {
        throw std::runtime_error("invalid AWQ W4A16 group size");
    }
    const size_t groups = cols / group_size;
    if (qweight_desc.dtype != llama_safetensors_dtype::I32 ||
        qzeros_desc.dtype != llama_safetensors_dtype::I32 || qzeros_desc.shape != std::vector<uint64_t>({ groups, rows / 8 }) ||
        (scales_desc.dtype != llama_safetensors_dtype::F16 && scales_desc.dtype != llama_safetensors_dtype::BF16) ||
        scales_desc.shape != std::vector<uint64_t>({ groups, rows }) ||
        qweight_desc.size != cols * rows / 2 || qzeros_desc.size != groups * rows / 2 ||
        scales_desc.size != groups * rows * sizeof(uint16_t)) {
        throw std::runtime_error("inconsistent AWQ W4A16 group source tensors");
    }

    const auto read_scale = [&](size_t index) {
        const uint8_t * source = scales + index * sizeof(uint16_t);
        if (scales_desc.dtype == llama_safetensors_dtype::BF16) {
            return load_bf16(source);
        }
        uint16_t bits;
        std::memcpy(&bits, source, sizeof(bits));
        return ggml_fp16_to_fp32(bits);
    };
    for (size_t i = 0; i < groups * rows; ++i) {
        const float value = read_scale(i);
        const ggml_fp16_t encoded = ggml_fp32_to_fp16(value);
        if (!(value > 0.0f) || !std::isfinite(value) || !std::isfinite(ggml_fp16_to_fp32(encoded))) {
            throw std::runtime_error("AWQ scale must be finite and positive");
        }
    }

    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        const size_t packed_row = row / 8;
        const uint32_t shift = shifts[row % 8];
        uint8_t * row_out = result.data() + row * (cols / qk) * block_size;
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t group = block / (group_size / qk);
            const float scale = read_scale(group * rows + row);
            const ggml_fp16_t scale_bits = ggml_fp32_to_fp16(scale);
            uint32_t packed_zero;
            std::memcpy(&packed_zero,
                        qzeros + (group * (rows / 8) + packed_row) * sizeof(packed_zero),
                        sizeof(packed_zero));
            const uint8_t zero = (packed_zero >> shift) & 0x0f;
            const ggml_fp16_t minimum = ggml_fp32_to_fp16(-scale * zero);
            if (!std::isfinite(ggml_fp16_to_fp32(minimum))) {
                throw std::runtime_error("AWQ minimum is not representable in Q4_1");
            }

            uint8_t * out = row_out + block * block_size;
            std::memcpy(out, &scale_bits, sizeof(scale_bits));
            std::memcpy(out + sizeof(scale_bits), &minimum, sizeof(minimum));
            for (size_t j = 0; j < qk / 2; ++j) {
                uint32_t packed_lo;
                uint32_t packed_hi;
                std::memcpy(&packed_lo,
                            qweight + ((block * qk + j) * (rows / 8) + packed_row) * sizeof(packed_lo),
                            sizeof(packed_lo));
                std::memcpy(&packed_hi,
                            qweight + ((block * qk + j + qk / 2) * (rows / 8) + packed_row) * sizeof(packed_hi),
                            sizeof(packed_hi));
                out[2 * sizeof(ggml_fp16_t) + j] =
                    ((packed_lo >> shift) & 0x0f) | (((packed_hi >> shift) & 0x0f) << 4);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_quark_w4a16(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor * zero_desc,
        const uint8_t * zero,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        uint32_t group_size,
        bool symmetric) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + qk / 2;
    constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };

    if (weight_desc.dtype != llama_safetensors_dtype::I32 || weight_desc.shape.size() != 2 ||
        group_size < qk || group_size % qk != 0 || weight_desc.shape[0] % group_size != 0) {
        throw std::runtime_error("invalid packed Quark W4A16 dimensions");
    }
    const size_t cols   = weight_desc.shape[0];
    const size_t rows   = weight_desc.shape[1] * 8;
    const size_t groups = cols / group_size;
    const bool zero_valid = symmetric ?
        (zero_desc == nullptr || (zero != nullptr && zero_desc->dtype == llama_safetensors_dtype::I32 &&
                                  zero_desc->shape == std::vector<uint64_t>({ groups, rows / 8 }))) :
        (zero_desc != nullptr && zero != nullptr && zero_desc->dtype == llama_safetensors_dtype::I32 &&
         zero_desc->shape == std::vector<uint64_t>({ groups, rows / 8 }));
    const size_t scale_element_size = scale_desc.dtype == llama_safetensors_dtype::F32 ? sizeof(float) :
        (scale_desc.dtype == llama_safetensors_dtype::F16 ||
         scale_desc.dtype == llama_safetensors_dtype::BF16) ? sizeof(uint16_t) : 0;
    if (scale_element_size == 0 ||
        scale_desc.shape != std::vector<uint64_t>({ groups, rows }) ||
        weight_desc.size != cols * rows / 2 || scale_desc.size != groups * rows * scale_element_size ||
        !zero_valid) {
        throw std::runtime_error("inconsistent Quark W4A16 source tensors");
    }

    const auto read_scale = [&](size_t index) {
        float value;
        if (scale_desc.dtype == llama_safetensors_dtype::F32) {
            std::memcpy(&value, scale + index * sizeof(value), sizeof(value));
        } else {
            uint16_t bits;
            std::memcpy(&bits, scale + index * sizeof(bits), sizeof(bits));
            value = scale_desc.dtype == llama_safetensors_dtype::BF16 ?
                load_bf16(reinterpret_cast<const uint8_t *>(&bits)) : ggml_fp16_to_fp32(bits);
        }
        const ggml_fp16_t encoded = ggml_fp32_to_fp16(value);
        if (!(value > 0.0f) || !std::isfinite(value) || !std::isfinite(ggml_fp16_to_fp32(encoded))) {
            throw std::runtime_error("Quark W4A16 scale must be finite, positive, and representable in FP16");
        }
        return value;
    };

    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        const size_t packed_row = row / 8;
        const uint32_t shift = shifts[row % 8];
        uint8_t * row_out = result.data() + row * (cols / qk) * block_size;
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t group = block / (group_size / qk);
            const float scale_value = read_scale(group * rows + row);
            const ggml_fp16_t scale_bits = ggml_fp32_to_fp16(scale_value);
            uint8_t zero_code = 8;
            if (!symmetric) {
                uint32_t packed_zero;
                std::memcpy(&packed_zero,
                            zero + (group * (rows / 8) + packed_row) * sizeof(packed_zero),
                            sizeof(packed_zero));
                zero_code = (packed_zero >> shift) & 0x0f;
            }
            const ggml_fp16_t minimum = ggml_fp32_to_fp16(-scale_value * zero_code);
            if (!std::isfinite(ggml_fp16_to_fp32(minimum))) {
                throw std::runtime_error("Quark W4A16 minimum is not representable in Q4_1");
            }

            uint8_t * out = row_out + block * block_size;
            std::memcpy(out, &scale_bits, sizeof(scale_bits));
            std::memcpy(out + sizeof(scale_bits), &minimum, sizeof(minimum));
            for (size_t j = 0; j < qk / 2; ++j) {
                uint32_t packed_lo;
                uint32_t packed_hi;
                std::memcpy(&packed_lo,
                            weight + ((block * qk + j) * (rows / 8) + packed_row) * sizeof(packed_lo),
                            sizeof(packed_lo));
                std::memcpy(&packed_hi,
                            weight + ((block * qk + j + qk / 2) * (rows / 8) + packed_row) * sizeof(packed_hi),
                            sizeof(packed_hi));
                uint8_t lo = (packed_lo >> shift) & 0x0f;
                uint8_t hi = (packed_hi >> shift) & 0x0f;
                if (symmetric) {
                    lo ^= 0x08;
                    hi ^= 0x08;
                }
                out[2 * sizeof(ggml_fp16_t) + j] = lo | (hi << 4);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_gptq(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales,
        uint32_t group_size_config) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + qk / 2;

    if (qweight_desc.shape.size() != 2) {
        throw std::runtime_error("invalid packed GPTQ dimensions");
    }
    const size_t cols = qweight_desc.shape[0] * 8;
    const size_t rows = qweight_desc.shape[1];
    const size_t group_size = group_size_config == 0 ? cols : group_size_config;
    if (group_size == 0 || group_size % qk != 0 || cols % group_size != 0) {
        throw std::runtime_error("invalid GPTQ W4A16 group size");
    }
    const size_t groups = cols / group_size;
    if (qweight_desc.dtype != llama_safetensors_dtype::I32 || rows % 8 != 0 ||
        qzeros_desc.dtype != llama_safetensors_dtype::I32 || qzeros_desc.shape != std::vector<uint64_t>({ groups, rows / 8 }) ||
        (scales_desc.dtype != llama_safetensors_dtype::F16 && scales_desc.dtype != llama_safetensors_dtype::BF16) ||
        scales_desc.shape != std::vector<uint64_t>({ groups, rows }) ||
        qweight_desc.size != cols * rows / 2 || qzeros_desc.size != groups * rows / 2 ||
        scales_desc.size != groups * rows * sizeof(uint16_t)) {
        throw std::runtime_error("inconsistent non-act-order GPTQ W4A16 group source tensors");
    }

    const auto read_scale = [&](size_t index) {
        const uint8_t * source = scales + index * sizeof(uint16_t);
        if (scales_desc.dtype == llama_safetensors_dtype::BF16) {
            return load_bf16(source);
        }
        uint16_t bits;
        std::memcpy(&bits, source, sizeof(bits));
        return ggml_fp16_to_fp32(bits);
    };
    for (size_t i = 0; i < groups * rows; ++i) {
        const float value = read_scale(i);
        const ggml_fp16_t encoded = ggml_fp32_to_fp16(value);
        if (value == 0.0f || !std::isfinite(value) || !std::isfinite(ggml_fp16_to_fp32(encoded))) {
            throw std::runtime_error("GPTQ scale must be finite and non-zero");
        }
    }

    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        const size_t packed_row = row / 8;
        const uint32_t row_shift = 4 * (row % 8);
        uint8_t * row_out = result.data() + row * (cols / qk) * block_size;
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t group = block / (group_size / qk);
            const float scale = read_scale(group * rows + row);
            const ggml_fp16_t scale_bits = ggml_fp32_to_fp16(scale);
            uint32_t packed_zero;
            std::memcpy(&packed_zero,
                        qzeros + (group * (rows / 8) + packed_row) * sizeof(packed_zero),
                        sizeof(packed_zero));
            const uint8_t zero = ((packed_zero >> row_shift) & 0x0f) + 1;
            const ggml_fp16_t minimum = ggml_fp32_to_fp16(-scale * zero);
            if (!std::isfinite(ggml_fp16_to_fp32(minimum))) {
                throw std::runtime_error("GPTQ minimum is not representable in Q4_1");
            }

            uint8_t * out = row_out + block * block_size;
            std::memcpy(out, &scale_bits, sizeof(scale_bits));
            std::memcpy(out + sizeof(scale_bits), &minimum, sizeof(minimum));
            for (size_t j = 0; j < qk / 2; ++j) {
                const size_t col_lo = block * qk + j;
                const size_t col_hi = col_lo + qk / 2;
                uint32_t packed_lo;
                uint32_t packed_hi;
                std::memcpy(&packed_lo,
                            qweight + ((col_lo / 8) * rows + row) * sizeof(packed_lo),
                            sizeof(packed_lo));
                std::memcpy(&packed_hi,
                            qweight + ((col_hi / 8) * rows + row) * sizeof(packed_hi),
                            sizeof(packed_hi));
                const uint8_t lo = (packed_lo >> (4 * (col_lo % 8))) & 0x0f;
                const uint8_t hi = (packed_hi >> (4 * (col_hi % 8))) & 0x0f;
                out[2 * sizeof(ggml_fp16_t) + j] = lo | (hi << 4);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_gptq8(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales,
        uint32_t group_size_config,
        ggml_type target_type) const {
    constexpr size_t pack_factor = 4;
    if (qweight_desc.shape.size() != 2) {
        throw std::runtime_error("invalid packed GPTQ INT8 dimensions");
    }
    const size_t cols = qweight_desc.shape[0] * pack_factor;
    const size_t rows = qweight_desc.shape[1];
    const size_t group_size = group_size_config == 0 ? cols : group_size_config;
    const size_t qk = ggml_blck_size(target_type);
    const size_t block_size = ggml_type_size(target_type);
    if ((target_type != GGML_TYPE_Q8_0 && target_type != GGML_TYPE_Q8_0_G128) ||
        qk == 0 || group_size == 0 || group_size % qk != 0 || cols % group_size != 0 || rows % pack_factor != 0) {
        throw std::runtime_error("invalid GPTQ W8A16 group size or target type");
    }
    const size_t groups = cols / group_size;
    if (qweight_desc.dtype != llama_safetensors_dtype::I32 ||
        qzeros_desc.dtype != llama_safetensors_dtype::I32 ||
        qzeros_desc.shape != std::vector<uint64_t>({ groups, rows / pack_factor }) ||
        ((target_type == GGML_TYPE_Q8_0 && scales_desc.dtype != llama_safetensors_dtype::F16) ||
         (target_type == GGML_TYPE_Q8_0_G128 && scales_desc.dtype != llama_safetensors_dtype::BF16)) ||
        scales_desc.shape != std::vector<uint64_t>({ groups, rows }) ||
        qweight_desc.size != cols * rows || qzeros_desc.size != groups * rows ||
        scales_desc.size != groups * rows * sizeof(uint16_t)) {
        throw std::runtime_error("inconsistent non-act-order GPTQ W8A16 source tensors");
    }

    for (size_t i = 0; i < groups * rows; ++i) {
        const uint8_t * source = scales + i * sizeof(uint16_t);
        float value;
        if (scales_desc.dtype == llama_safetensors_dtype::BF16) {
            value = load_bf16(source);
        } else {
            uint16_t bits;
            std::memcpy(&bits, source, sizeof(bits));
            value = ggml_fp16_to_fp32(bits);
        }
        if (value == 0.0f || !std::isfinite(value)) {
            throw std::runtime_error("GPTQ INT8 scale must be finite and non-zero");
        }
    }

    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        const size_t packed_row = row / pack_factor;
        const uint32_t row_shift = 8 * (row % pack_factor);
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t group = block / (group_size / qk);
            uint32_t packed_zero;
            std::memcpy(&packed_zero,
                        qzeros + (group * (rows / pack_factor) + packed_row) * sizeof(packed_zero),
                        sizeof(packed_zero));
            const uint8_t zero = uint8_t(((packed_zero >> row_shift) & 0xffu) + 1u);
            if (zero != 128) {
                throw std::runtime_error("symmetric GPTQ INT8 tensor has a non-midpoint zero code");
            }

            uint8_t * out = result.data() + (row * (cols / qk) + block) * block_size;
            std::memcpy(out, scales + (group * rows + row) * sizeof(uint16_t), sizeof(uint16_t));
            for (size_t local = 0; local < qk; ++local) {
                const size_t col = block * qk + local;
                uint32_t packed;
                std::memcpy(&packed,
                            qweight + ((col / pack_factor) * rows + row) * sizeof(packed),
                            sizeof(packed));
                const uint8_t code = (packed >> (8 * (col % pack_factor))) & 0xffu;
                out[sizeof(uint16_t) + local] = uint8_t(code - zero);
            }
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::read(
        const llama_safetensors_quant_binding & binding) const {
    if (binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_MARKER ||
            binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_INT8_MARKER ||
            binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_MXFP4_MARKER ||
            binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_MXFP8_MARKER) {
        int32_t marker = 0;
        if (binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_MARKER &&
                ends_with(binding.primary, ".weight")) {
            const std::string module = binding.primary.substr(0, binding.primary.size() - strlen(".weight"));
            const auto * group = match(module);
            if (group != nullptr && group->input_scale_ub > 0.0f) {
                static_assert(sizeof(marker) == sizeof(group->input_scale_ub));
                std::memcpy(&marker, &group->input_scale_ub, sizeof(marker));
            }
        }
        if (binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_INT8_MARKER &&
                ends_with(binding.primary, ".weight")) {
            const std::string module = binding.primary.substr(0, binding.primary.size() - strlen(".weight"));
            const auto * group = match(module);
            if (group != nullptr && group->format == llama_safetensors_quant_format::BNB_INT8) {
                static_assert(sizeof(marker) == sizeof(group->outlier_threshold));
                std::memcpy(&marker, &group->outlier_threshold, sizeof(marker));
            }
        }
        std::vector<uint8_t> result(sizeof(marker));
        std::memcpy(result.data(), &marker, sizeof(marker));
        return result;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::QUANTO_W4A16_MARKER) {
        return std::vector<uint8_t>(1, 0);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::QUANTO_W8A16_SCALE) {
        const auto & scale_desc = require_tensor(registry_, binding.primary);
        std::vector<uint8_t> scale = registry_.read(scale_desc);
        for (size_t offset = 0; offset < scale.size(); offset += sizeof(uint16_t)) {
            uint16_t bits;
            std::memcpy(&bits, scale.data() + offset, sizeof(bits));
            const float value = scale_desc.dtype == llama_safetensors_dtype::BF16 ?
                load_bf16(scale.data() + offset) : ggml_fp16_to_fp32(bits);
            if (!(value > 0.0f) || !std::isfinite(value)) {
                throw std::runtime_error("Quanto qint8 scale must be finite and positive");
            }
        }
        if (scale_desc.dtype == llama_safetensors_dtype::F16) {
            const std::vector<uint8_t> f32 = llama_safetensors_f16_to_f32(scale);
            scale.resize(f32.size() / sizeof(float) * sizeof(ggml_bf16_t));
            for (size_t i = 0; i < scale.size() / sizeof(ggml_bf16_t); ++i) {
                float value;
                std::memcpy(&value, f32.data() + i * sizeof(value), sizeof(value));
                const ggml_bf16_t converted = ggml_fp32_to_bf16(value);
                std::memcpy(scale.data() + i * sizeof(converted), &converted, sizeof(converted));
            }
        }
        const uint32_t n_channels = static_cast<uint32_t>(scale.size() / sizeof(uint16_t));
        ggml_w8a16_scale_header header {};
        header.magic         = GGML_W8A16_SCALE_MAGIC;
        header.version       = 1;
        header.n_channels    = n_channels;
        header.values_offset = sizeof(header);
        header.total_size    = sizeof(header) + scale.size();
        std::vector<uint8_t> result(header.total_size);
        std::memcpy(result.data(), &header, sizeof(header));
        std::memcpy(result.data() + header.values_offset, scale.data(), scale.size());
        return result;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_W4A8_FP8_MARKER ||
            binding.materialization == llama_safetensors_quant_materialization::DYNAMIC_FP8_GROUP_MARKER) {
        const int16_t marker = 0;
        std::vector<uint8_t> result(sizeof(marker));
        std::memcpy(result.data(), &marker, sizeof(marker));
        return result;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::STATIC_INT8_ASYM_PARAMS) {
        const auto & scale_desc = require_tensor(registry_, binding.primary);
        std::vector<uint8_t> scale = registry_.read(scale_desc);
        if (scale_desc.dtype == llama_safetensors_dtype::BF16) {
            scale = llama_safetensors_bf16_to_f32(scale);
        } else if (scale_desc.dtype == llama_safetensors_dtype::F16) {
            scale = llama_safetensors_f16_to_f32(scale);
        }
        float scale_value;
        std::memcpy(&scale_value, scale.data(), sizeof(scale_value));
        if (!(scale_value > 0.0f) || !std::isfinite(scale_value)) {
            throw std::runtime_error("static W8A8 input scale must be finite and positive");
        }
        const auto & zero_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const std::vector<uint8_t> zero = registry_.read(zero_desc);
        std::vector<uint8_t> result(sizeof(int64_t), 0);
        std::memcpy(result.data(), &scale_value, sizeof(scale_value));
        std::memcpy(result.data() + sizeof(scale_value), zero.data(), sizeof(int8_t));
        return result;
    }
    const llama_safetensors_tensor & primary = require_tensor(registry_, binding.primary);
    if (binding.materialization == llama_safetensors_quant_materialization::GPTQ_SCALE_BUNDLE) {
        const std::string suffix = ".qzeros";
        if (!ends_with(binding.primary, suffix) || binding.auxiliaries.size() != 2) {
            throw std::runtime_error("invalid GPTQ act-order scale-bundle binding");
        }
        const std::string module = binding.primary.substr(0, binding.primary.size() - suffix.size());
        const auto * group = match(module);
        if (group == nullptr || group->format != llama_safetensors_quant_format::GPTQ_GROUP ||
                !group->act_order || group->group_size == 0) {
            throw std::runtime_error("GPTQ act-order binding no longer matches its quantization group");
        }
        const auto & scales_desc = require_tensor(registry_, binding.auxiliaries[0]);
        const auto & g_idx_desc  = require_tensor(registry_, binding.auxiliaries[1]);
        const size_t groups = primary.shape.at(0);
        const size_t rows = primary.shape.at(1) * 8;
        const size_t cols = g_idx_desc.shape.at(0);
        if (groups == 0 || cols != groups * group->group_size || rows == 0 || rows % 8 != 0 ||
                primary.dtype != llama_safetensors_dtype::I32 ||
                primary.shape != std::vector<uint64_t>({ groups, rows / 8 }) ||
                (scales_desc.dtype != llama_safetensors_dtype::F16 &&
                 scales_desc.dtype != llama_safetensors_dtype::BF16) ||
                scales_desc.shape != std::vector<uint64_t>({ groups, rows }) ||
                g_idx_desc.dtype != llama_safetensors_dtype::I32 ||
                g_idx_desc.shape != std::vector<uint64_t>({ cols })) {
            throw std::runtime_error("inconsistent GPTQ act-order auxiliary tensors");
        }

        const source_bytes qzeros(registry_, primary);
        const source_bytes scales(registry_, scales_desc);
        const source_bytes g_idx(registry_, g_idx_desc);
        std::vector<uint8_t> result(size_t(binding.target_shape.at(0)), 0);
        ggml_gptq_ao_header header{};
        header.magic      = GGML_GPTQ_AO_MAGIC;
        header.version    = 1;
        header.cols       = static_cast<uint32_t>(cols);
        header.rows       = static_cast<uint32_t>(rows);
        header.groups     = static_cast<uint32_t>(groups);
        header.group_size = group->group_size;
        header.scale_type = scales_desc.dtype == llama_safetensors_dtype::BF16 ? 1 : 0;
        size_t cursor = sizeof(header);
        const auto reserve = [&](size_t size, uint32_t & offset) {
            offset = static_cast<uint32_t>(cursor);
            const size_t begin = cursor;
            cursor = (cursor + size + 3) & ~size_t(3);
            return begin;
        };
        const size_t zeros_begin = reserve(groups * rows, header.zeros_offset);
        const size_t scales_begin = reserve(groups * rows * sizeof(uint16_t), header.scales_offset);
        const size_t g_idx_begin = reserve(cols * sizeof(uint16_t), header.g_idx_offset);
        if (cursor != result.size()) {
            throw std::runtime_error("GPTQ act-order scale-bundle size mismatch");
        }
        std::memcpy(result.data() + scales_begin, scales.data(), groups * rows * sizeof(uint16_t));
        for (size_t group_index = 0; group_index < groups; ++group_index) {
            for (size_t row = 0; row < rows; ++row) {
                uint32_t packed;
                std::memcpy(&packed,
                    qzeros.data() + (group_index * (rows / 8) + row / 8) * sizeof(packed), sizeof(packed));
                result[zeros_begin + group_index * rows + row] =
                    static_cast<uint8_t>(((packed >> (4 * (row % 8))) & 0x0f) + 1);
            }
        }
        for (size_t col = 0; col < cols; ++col) {
            int32_t group_index;
            std::memcpy(&group_index, g_idx.data() + col * sizeof(group_index), sizeof(group_index));
            if (group_index < 0 || static_cast<size_t>(group_index) >= groups) {
                throw std::runtime_error("GPTQ g_idx contains an out-of-range group index");
            }
            const uint16_t compact = static_cast<uint16_t>(group_index);
            std::memcpy(result.data() + g_idx_begin + col * sizeof(compact), &compact, sizeof(compact));
        }
        header.total_size = static_cast<uint32_t>(result.size());
        std::memcpy(result.data(), &header, sizeof(header));
        return result;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::BNB_SCALE_BUNDLE) {
        const std::string suffix = ".weight.absmax";
        if (!ends_with(binding.primary, suffix) || binding.auxiliaries.size() < 2) {
            throw std::runtime_error("invalid BitsAndBytes scale-bundle binding");
        }
        const std::string module = binding.primary.substr(0, binding.primary.size() - suffix.size());
        const std::string & state_name = binding.auxiliaries[1];
        const std::string nf4_suffix = ".weight.quant_state.bitsandbytes__nf4";
        const std::string fp4_suffix = ".weight.quant_state.bitsandbytes__fp4";
        const std::string quant_type = ends_with(state_name, nf4_suffix) ? "nf4" :
                                       ends_with(state_name, fp4_suffix) ? "fp4" : std::string();
        if (quant_type.empty()) {
            throw std::runtime_error("invalid BitsAndBytes quantization-state binding");
        }
        const bnb_state state = read_bnb_state(registry_, module, quant_type);
        const uint64_t elements = uint64_t(state.shape[0]) * uint64_t(state.shape[1]);
        const uint64_t n_blocks = elements / state.block_size;
        if (n_blocks > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("BitsAndBytes tensor has too many quantization blocks");
        }

        const auto & quant_map_desc = require_tensor(registry_, binding.auxiliaries[0]);
        const std::vector<uint8_t> absmax = registry_.read(primary);
        const std::vector<uint8_t> quant_map = registry_.read(quant_map_desc);
        const auto validate_f32 = [&](const std::vector<uint8_t> & values, const char * what, bool positive) {
            for (size_t offset = 0; offset < values.size(); offset += sizeof(float)) {
                float value;
                std::memcpy(&value, values.data() + offset, sizeof(value));
                if (!std::isfinite(value) || (positive && !(value > 0.0f))) {
                    throw std::runtime_error(std::string("invalid BitsAndBytes ") + what + " for '" + module + "'");
                }
            }
        };
        validate_f32(quant_map, "codebook", false);

        std::vector<uint8_t> nested_absmax;
        std::vector<uint8_t> nested_map;
        if (state.nested_block_size == 0) {
            validate_f32(absmax, "absmax", true);
        } else {
            nested_absmax = registry_.read(require_tensor(registry_, binding.auxiliaries.at(2)));
            nested_map = registry_.read(require_tensor(registry_, binding.auxiliaries.at(3)));
            validate_f32(nested_absmax, "nested absmax", true);
            validate_f32(nested_map, "nested codebook", false);
        }

        std::vector<uint8_t> result(size_t(binding.target_shape.at(0)), 0);
        ggml_bnb_scale_header header{};
        header.magic             = GGML_BNB_SCALE_MAGIC;
        header.version           = 1;
        header.n_blocks          = static_cast<uint32_t>(n_blocks);
        header.block_size        = state.block_size;
        header.nested_block_size = state.nested_block_size;
        header.nested_offset     = state.nested_offset;
        size_t cursor = sizeof(header);
        const auto append = [&](const std::vector<uint8_t> & bytes, uint32_t & offset) {
            offset = static_cast<uint32_t>(cursor);
            std::memcpy(result.data() + cursor, bytes.data(), bytes.size());
            cursor = (cursor + bytes.size() + 3) & ~size_t(3);
        };
        append(absmax, header.absmax_offset);
        append(quant_map, header.quant_map_offset);
        if (state.nested_block_size != 0) {
            append(nested_absmax, header.nested_absmax_offset);
            append(nested_map, header.nested_quant_map_offset);
        }
        if (cursor != result.size()) {
            throw std::runtime_error("BitsAndBytes scale-bundle size mismatch");
        }
        header.total_size = static_cast<uint32_t>(result.size());
        std::memcpy(result.data(), &header, sizeof(header));
        return result;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::NVFP4_REPACK) {
        const llama_safetensors_tensor & auxiliary = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, auxiliary);
        return repack_nvfp4(primary, weight.data(), primary.size, auxiliary, scale.data(), auxiliary.size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::EETQ_REPACK) {
        const llama_safetensors_tensor & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        return repack_eetq(primary, weight.data(), scale_desc, scale.data());
    }
    if (binding.materialization == llama_safetensors_quant_materialization::QUANTO_INT4_REPACK) {
        const llama_safetensors_tensor & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor & shift_desc = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        const source_bytes shift(registry_, shift_desc);
        const std::string suffix = ".weight._data._data";
        const auto * group = match(primary.name.substr(0, primary.name.size() - suffix.size()));
        if (group == nullptr || group->format != llama_safetensors_quant_format::QUANTO_INT4) {
            throw std::runtime_error("Quanto binding no longer matches its quantization group");
        }
        return repack_quanto_int4(
            primary, weight.data(), scale_desc, scale.data(), shift_desc, shift.data(),
            binding.target_shape, group->group_size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::TORCHAO_INT4_REPACK) {
        const llama_safetensors_tensor & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        constexpr std::string_view suffix = ".weight.__qdata";
        const auto * group = match(primary.name.substr(0, primary.name.size() - suffix.size()));
        if (group == nullptr || group->format != llama_safetensors_quant_format::TORCHAO_INT4) {
            throw std::runtime_error("TorchAO tiled-int4 binding no longer matches its quantization group");
        }
        return repack_torchao_int4(
            primary, weight.data(), scale_desc, scale.data(), binding.target_shape, group->group_size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::TORCHAO_INTX_REPACK) {
        const llama_safetensors_tensor & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor & zero_desc  = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        const source_bytes zero(registry_, zero_desc);
        constexpr std::string_view suffix = ".weight.__qdata";
        const auto * group = match(primary.name.substr(0, primary.name.size() - suffix.size()));
        if (group == nullptr || group->format != llama_safetensors_quant_format::TORCHAO_INTX) {
            throw std::runtime_error("TorchAO unpacked-intx binding no longer matches its quantization group");
        }
        return repack_torchao_intx(
            primary, weight.data(), scale_desc, scale.data(), zero_desc, zero.data(),
            binding.target_shape, group->group_size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::HQQ_INT4_REPACK) {
        const std::string suffix = ".W_q";
        const std::string module = binding.primary.substr(0, binding.primary.size() - suffix.size());
        const auto * group = match(module);
        if (group == nullptr || group->format != llama_safetensors_quant_format::HQQ_INT4 ||
            group->num_bits != 4) {
            throw std::runtime_error("HQQ binding no longer matches its quantization group");
        }
        const auto & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const auto & zero_desc  = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        const source_bytes zero(registry_, zero_desc);
        return repack_hqq_int4(
            primary, weight.data(), scale_desc, scale.data(), zero_desc, zero.data(),
            binding.target_shape, group->group_size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::MXFP4_REPACK) {
        const llama_safetensors_tensor & auxiliary = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, auxiliary);
        return repack_mxfp4(primary, weight.data(), auxiliary, scale.data());
    }
    if (binding.materialization == llama_safetensors_quant_materialization::AWQ_REPACK) {
        const llama_safetensors_tensor & qzeros_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor & scales_desc = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes qweight(registry_, primary);
        const source_bytes qzeros(registry_, qzeros_desc);
        const source_bytes scales(registry_, scales_desc);
        const std::string suffix = ".qweight";
        const auto * group = match(primary.name.substr(0, primary.name.size() - suffix.size()));
        if (group == nullptr || group->format != llama_safetensors_quant_format::AWQ_GROUP) {
            throw std::runtime_error("AWQ binding no longer matches its quantization group");
        }
        return repack_awq(
            primary, qweight.data(), qzeros_desc, qzeros.data(), scales_desc, scales.data(), group->group_size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::QUARK_W4A16_REPACK) {
        const llama_safetensors_tensor & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor * zero_desc = binding.auxiliaries.size() == 2 ?
            &require_tensor(registry_, binding.auxiliaries.at(1)) : nullptr;
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        const std::optional<source_bytes> zero = zero_desc == nullptr ?
            std::nullopt : std::optional<source_bytes>(std::in_place, registry_, *zero_desc);
        const std::string suffix = ".weight";
        const auto * group = match(primary.name.substr(0, primary.name.size() - suffix.size()));
        if (group == nullptr || group->format != llama_safetensors_quant_format::QUARK_W4A16) {
            throw std::runtime_error("Quark W4A16 binding no longer matches its quantization group");
        }
        return repack_quark_w4a16(
            primary, weight.data(), zero_desc, zero.has_value() ? zero->data() : nullptr,
            scale_desc, scale.data(), group->group_size, group->symmetric);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::GPTQ_REPACK) {
        const llama_safetensors_tensor & qzeros_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor & scales_desc = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes qweight(registry_, primary);
        const source_bytes qzeros(registry_, qzeros_desc);
        const source_bytes scales(registry_, scales_desc);
        const std::string suffix = ".qweight";
        const auto * group = match(primary.name.substr(0, primary.name.size() - suffix.size()));
        if (group == nullptr || group->format != llama_safetensors_quant_format::GPTQ_GROUP) {
            throw std::runtime_error("GPTQ binding no longer matches its quantization group");
        }
        return repack_gptq(
            primary, qweight.data(), qzeros_desc, qzeros.data(), scales_desc, scales.data(), group->group_size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::GPTQ8_REPACK) {
        const llama_safetensors_tensor & qzeros_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor & scales_desc = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes qweight(registry_, primary);
        const source_bytes qzeros(registry_, qzeros_desc);
        const source_bytes scales(registry_, scales_desc);
        const std::string suffix = ".qweight";
        const auto * group = match(primary.name.substr(0, primary.name.size() - suffix.size()));
        if (group == nullptr || group->format != llama_safetensors_quant_format::GPTQ_GROUP ||
                group->num_bits != 8 || group->act_order) {
            throw std::runtime_error("GPTQ INT8 binding no longer matches its quantization group");
        }
        return repack_gptq8(primary, qweight.data(), qzeros_desc, qzeros.data(), scales_desc, scales.data(),
                           group->group_size, binding.target_type);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::PACKED_INT4_REPACK ||
        binding.materialization == llama_safetensors_quant_materialization::PACKED_INT8_REPACK) {
        const std::string suffix = ".weight_packed";
        const std::string module = binding.primary.substr(0, binding.primary.size() - suffix.size());
        const auto weight_shape = read_weight_shape(registry_, module);
        const auto & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes packed_weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        if (binding.materialization == llama_safetensors_quant_materialization::PACKED_INT4_REPACK) {
            const std::string suffix = ".weight_packed";
            const std::string module = binding.primary.substr(0, binding.primary.size() - suffix.size());
            const auto * group = match(module);
            if (group == nullptr || group->format != llama_safetensors_quant_format::PACKED_INT ||
                    group->num_bits != 4) {
                throw std::runtime_error("packed INT4 binding no longer matches its quantization group");
            }
            const llama_safetensors_tensor * zero_desc = nullptr;
            std::optional<source_bytes> zero;
            if (!group->symmetric) {
                zero_desc = &require_tensor(registry_, binding.auxiliaries.at(2));
                zero.emplace(registry_, *zero_desc);
            }
            return repack_packed_int4(
                primary, packed_weight.data(), scale_desc, scale.data(), zero_desc,
                zero.has_value() ? zero->data() : nullptr, weight_shape, group->group_size, group->symmetric);
        }
        return repack_packed_int8(primary, packed_weight.data(), scale_desc, scale.data(), weight_shape);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::PACKED_INT4_FP8_REPACK) {
        const std::string suffix = ".weight_packed";
        const std::string module = binding.primary.substr(0, binding.primary.size() - suffix.size());
        const auto weight_shape = read_weight_shape(registry_, module);
        const auto & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const auto & channel_scale_desc = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes packed_weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        const source_bytes channel_scale(registry_, channel_scale_desc);
        return repack_packed_int4_fp8(primary, packed_weight.data(), scale_desc, scale.data(),
                                     channel_scale_desc, channel_scale.data(), weight_shape);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::INT4_GROUP_REPACK) {
        const auto & scale_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, scale_desc);
        return repack_int4_group(primary, weight.data(), scale_desc, scale.data());
    }
    if (binding.materialization == llama_safetensors_quant_materialization::RECIPROCAL_F32) {
        if (primary.dtype != llama_safetensors_dtype::F32 || primary.size != sizeof(float)) {
            throw std::runtime_error("invalid global quantization scale '" + primary.name + "'");
        }
        const std::vector<uint8_t> raw = registry_.read(primary);
        float value;
        std::memcpy(&value, raw.data(), sizeof(value));
        if (!(value > 0.0f) || !std::isfinite(value)) {
            throw std::runtime_error("global quantization scale must be finite and positive: '" + primary.name + "'");
        }
        value = 1.0f / value;
        std::vector<uint8_t> result(sizeof(value));
        std::memcpy(result.data(), &value, sizeof(value));
        return result;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::POSITIVE_F32) {
        require_positive_f32_scalar(registry_, primary.name);
        return registry_.read(primary);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::POSITIVE_F32_TO_BF16) {
        if (primary.dtype != llama_safetensors_dtype::F32 ||
                binding.target_shape.size() != 1 || binding.target_shape[0] <= 0 ||
                primary.size != size_t(binding.target_shape[0]) * sizeof(float)) {
            throw std::runtime_error("invalid per-channel FP8 scale '" + primary.name + "'");
        }
        const std::vector<uint8_t> raw = registry_.read(primary);
        std::vector<uint8_t> result(size_t(binding.target_shape[0]) * sizeof(uint16_t));
        for (size_t i = 0; i < size_t(binding.target_shape[0]); ++i) {
            float value;
            std::memcpy(&value, raw.data() + i * sizeof(value), sizeof(value));
            const ggml_bf16_t converted = ggml_fp32_to_bf16(value);
            const float restored = ggml_bf16_to_fp32(converted);
            if (!(value > 0.0f) || !std::isfinite(value) ||
                    !(restored > 0.0f) || !std::isfinite(restored)) {
                throw std::runtime_error(
                    "per-channel FP8 scale must be representable, finite, and positive: '" + primary.name + "'");
            }
            std::memcpy(result.data() + i * sizeof(converted.bits), &converted.bits, sizeof(converted.bits));
        }
        return result;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::BROADCAST_BF16_SCALAR) {
        if ((primary.dtype != llama_safetensors_dtype::BF16 && primary.dtype != llama_safetensors_dtype::F32) ||
            primary.size != llama_safetensors_dtype_size(primary.dtype) ||
            binding.target_shape.size() != 1 || binding.target_shape[0] <= 0) {
            throw std::runtime_error("invalid static tensor quantization scale '" + primary.name + "'");
        }
        const std::vector<uint8_t> raw = registry_.read(primary);
        uint16_t bits;
        float value;
        if (primary.dtype == llama_safetensors_dtype::BF16) {
            std::memcpy(&bits, raw.data(), sizeof(bits));
            const uint32_t f32_bits = uint32_t(bits) << 16;
            std::memcpy(&value, &f32_bits, sizeof(value));
        } else {
            std::memcpy(&value, raw.data(), sizeof(value));
            bits = ggml_fp32_to_bf16(value).bits;
        }
        if (!(value > 0.0f) || !std::isfinite(value)) {
            throw std::runtime_error("static tensor quantization scale must be finite and positive: '" + primary.name + "'");
        }
        std::vector<uint8_t> result(size_t(binding.target_shape[0]) * sizeof(bits));
        for (size_t i = 0; i < size_t(binding.target_shape[0]); ++i) {
            std::memcpy(result.data() + i * sizeof(bits), &bits, sizeof(bits));
        }
        return result;
    }
    return registry_.read(primary);
}

std::vector<uint8_t> llama_safetensors_quant_adapters::finalize(
        const llama_safetensors_quant_binding & binding,
        std::vector<uint8_t> data) const {
    if (binding.materialization == llama_safetensors_quant_materialization::BNB_INT8_SCALE) {
        const llama_safetensors_tensor & source = require_tensor(registry_, binding.primary);
        if (source.dtype == llama_safetensors_dtype::F16) {
            data = llama_safetensors_f16_to_f32(data);
        } else if (source.dtype == llama_safetensors_dtype::BF16) {
            data = llama_safetensors_bf16_to_f32(data);
        } else if (source.dtype != llama_safetensors_dtype::F32) {
            throw std::runtime_error("BitsAndBytes INT8 SCB must be F16, BF16, or F32");
        }
        float * scales = reinterpret_cast<float *>(data.data());
        const size_t count = data.size() / sizeof(float);
        for (size_t i = 0; i < count; ++i) {
            if (!(scales[i] > 0.0f) || !std::isfinite(scales[i])) {
                throw std::runtime_error(
                    "BitsAndBytes INT8 SCB must be finite and positive: '" + binding.primary + "'");
            }
            // BitsAndBytes stores the row absmax. Its integer matrix is
            // reconstructed as CB * SCB / 127; the shared channel-INT8
            // executor expects that complete per-row multiplier.
            scales[i] /= 127.0f;
        }
        return data;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::RAW &&
            binding.target_type == GGML_TYPE_I8 && ends_with(binding.primary, ".weight_scale")) {
        if (std::find(data.begin(), data.end(), uint8_t(0xff)) != data.end()) {
            throw std::runtime_error("E8M0 scale uses the reserved NaN encoding: '" + binding.primary + "'");
        }
        return data;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::RAW &&
            binding.target_type == GGML_TYPE_BF16 && ends_with(binding.primary, ".weight_scale")) {
        for (size_t offset = 0; offset < data.size(); offset += sizeof(uint16_t)) {
            const float value = load_bf16(data.data() + offset);
            if (!(value > 0.0f) || !std::isfinite(value)) {
                throw std::runtime_error(
                    "quantization scale must be finite and positive: '" + binding.primary + "'");
            }
        }
        return data;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::RAW &&
            binding.target_type == GGML_TYPE_F32) {
        const llama_safetensors_tensor & source = require_tensor(registry_, binding.primary);
        if (source.dtype == llama_safetensors_dtype::F16) {
            data = llama_safetensors_f16_to_f32(data);
        } else if (source.dtype == llama_safetensors_dtype::BF16) {
            data = llama_safetensors_bf16_to_f32(data);
        }
        if (ends_with(binding.primary, ".weight_scale") || ends_with(binding.primary, ".input_scale") ||
                ends_with(binding.primary, ".in_scale")) {
            const float * scales = reinterpret_cast<const float *>(data.data());
            const size_t count = data.size() / sizeof(float);
            for (size_t i = 0; i < count; ++i) {
                if (!std::isfinite(scales[i]) || scales[i] <= 0.0f) {
                    throw std::runtime_error(
                        "quantization scale must be finite and positive: '" + binding.primary + "'");
                }
            }
        }
        return data;
    }
    if (binding.materialization == llama_safetensors_quant_materialization::FP8_BLOCK_SCALE_MODELOPT) {
        const llama_safetensors_tensor & source = require_tensor(registry_, binding.primary);
        return transpose_2d(data, source.shape[0], source.shape[2], sizeof(float));
    }
    if (binding.materialization == llama_safetensors_quant_materialization::FP8_BLOCK_SCALE_E8M0) {
        const llama_safetensors_tensor & source = require_tensor(registry_, binding.primary);
        std::vector<uint8_t> converted(data.size() * sizeof(float));
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i] == 0xff) {
                throw std::runtime_error("E8M0 scale uses the reserved NaN encoding: '" + binding.primary + "'");
            }
            const float value = std::ldexp(1.0f, static_cast<int>(data[i]) - 127);
            std::memcpy(converted.data() + i * sizeof(value), &value, sizeof(value));
        }
        return transpose_2d(converted, source.shape[0], source.shape[1], sizeof(float));
    }
    if (binding.materialization != llama_safetensors_quant_materialization::FP8_BLOCK_SCALE) {
        return data;
    }
    const llama_safetensors_tensor & source = require_tensor(registry_, binding.primary);
    return llama_safetensors_bf16_to_f32(
        transpose_2d(data, source.shape[0], source.shape[1], sizeof(uint16_t)));
}

void llama_safetensors_quant_adapters::consume(
        const llama_safetensors_quant_binding & binding) const {
    consumed_.insert(binding.primary);
    for (const std::string & auxiliary : binding.auxiliaries) {
        consumed_.insert(auxiliary);
    }
}

void llama_safetensors_quant_adapters::validate_complete() const {
    for (const auto & [weight, dependencies] : dependencies_) {
        if (consumed_.find(weight) == consumed_.end()) {
            continue;
        }
        for (const std::string & dependency : dependencies) {
            if (consumed_.find(dependency) == consumed_.end()) {
                throw std::runtime_error(
                    "quantized weight '" + weight + "' was bound without required tensor '" + dependency + "'");
            }
        }
    }
}
