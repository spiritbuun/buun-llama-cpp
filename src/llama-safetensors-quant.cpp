#include "llama-safetensors-quant.h"
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

std::vector<uint8_t> bf16_to_f32(const std::vector<uint8_t> & source) {
    if (source.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("invalid BF16 quantization scale byte count");
    }
    std::vector<uint8_t> result(source.size() * 2);
    for (size_t i = 0; i < source.size() / sizeof(uint16_t); ++i) {
        uint16_t bits16;
        std::memcpy(&bits16, source.data() + i * sizeof(bits16), sizeof(bits16));
        const uint32_t bits32 = uint32_t(bits16) << 16;
        std::memcpy(result.data() + i * sizeof(bits32), &bits32, sizeof(bits32));
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

}  // namespace

llama_safetensors_quant_adapters::llama_safetensors_quant_adapters(
        const llama_safetensors_json & config,
        const llama_safetensors_registry & registry) :
    registry_(registry), config_(llama_safetensors_quant_config::from_json(config)) {
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
            return registry_.find(module + ".weight_packed") != nullptr;
        case llama_safetensors_quant_format::AWQ_G128:
        case llama_safetensors_quant_format::GPTQ_G128:
            return registry_.find(module + ".qweight") != nullptr;
        case llama_safetensors_quant_format::PACKED_INT:
            return registry_.find(module + ".weight_packed") != nullptr;
        case llama_safetensors_quant_format::INT8_CHANNEL: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::I8;
        }
        case llama_safetensors_quant_format::FP8_CHANNEL:
        case llama_safetensors_quant_format::FP8_BLOCK: {
            const auto * weight = registry_.find(module + ".weight");
            return weight != nullptr && weight->dtype == llama_safetensors_dtype::F8_E4M3;
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

    if (group->format == llama_safetensors_quant_format::NVFP4_PACK) {
        if (role == llama_safetensors_quant_role::WEIGHT) {
            result.primary        = module + ".weight_packed";
            result.auxiliaries    = { module + ".weight_scale" };
            result.target_type    = GGML_TYPE_NVFP4;
            result.materialization = llama_safetensors_quant_materialization::NVFP4_REPACK;
            const auto & source = require_tensor(registry_, result.primary);
            if (source.shape.size() != 2 || source.shape[1] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 2) {
                throw std::runtime_error("invalid packed NVFP4 dimensions for '" + result.primary + "'");
            }
            result.target_shape = {
                static_cast<int64_t>(source.shape[1] * 2),
                static_cast<int64_t>(source.shape[0]),
            };
        } else {
            result.primary = module + (role == llama_safetensors_quant_role::WEIGHT_SCALE ?
                ".weight_global_scale" : ".input_global_scale");
            result.target_type     = GGML_TYPE_F32;
            result.target_shape    = { 1 };
            result.materialization = llama_safetensors_quant_materialization::RECIPROCAL_F32;
        }
        return result;
    }

    if (group->format == llama_safetensors_quant_format::AWQ_G128) {
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

    if (group->format == llama_safetensors_quant_format::GPTQ_G128) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        result.primary         = module + ".qweight";
        result.auxiliaries     = { module + ".qzeros", module + ".scales", module + ".g_idx" };
        result.target_type     = GGML_TYPE_Q4_1;
        result.materialization = llama_safetensors_quant_materialization::GPTQ_REPACK;
        const auto & source = require_tensor(registry_, result.primary);
        if (source.shape.size() != 2 || source.shape[0] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 8) {
            throw std::runtime_error("invalid packed GPTQ dimensions for '" + result.primary + "'");
        }
        result.target_shape = {
            static_cast<int64_t>(source.shape[0] * 8),
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
            result.auxiliaries.push_back(module + ".weight_zero_point");
            result.target_type     = GGML_TYPE_Q4_A32;
            result.materialization = llama_safetensors_quant_materialization::PACKED_INT4_REPACK;
        } else {
            result.target_type     = GGML_TYPE_Q8_0_G128;
            result.materialization = llama_safetensors_quant_materialization::PACKED_INT8_REPACK;
        }
        return result;
    }

    if (group->format == llama_safetensors_quant_format::INT8_CHANNEL) {
        if (role != llama_safetensors_quant_role::WEIGHT) {
            return std::nullopt;
        }
        result.primary         = module + ".weight";
        result.auxiliaries     = { module + ".weight_scale" };
        result.target_type     = GGML_TYPE_Q8_0;
        result.target_shape    = reversed_shape(require_tensor(registry_, result.primary));
        result.materialization = llama_safetensors_quant_materialization::W8A8_REPACK;
        return result;
    }

    if (role == llama_safetensors_quant_role::INPUT_SCALE) {
        return std::nullopt;
    }
    if (role == llama_safetensors_quant_role::WEIGHT) {
        result.primary      = module + ".weight";
        result.target_type  = GGML_TYPE_F8_E4M3;
        result.target_shape = reversed_shape(require_tensor(registry_, result.primary));
        return result;
    }

    result.primary = weight_scale_name(module);
    const auto & scale = require_tensor(registry_, result.primary);
    if (group->format == llama_safetensors_quant_format::FP8_CHANNEL) {
        result.target_type  = GGML_TYPE_BF16;
        result.target_shape = { static_cast<int64_t>(scale.shape[0]) };
    } else {
        result.target_type     = GGML_TYPE_F32;
        result.target_shape    = {
            static_cast<int64_t>(scale.shape[0]),
            static_cast<int64_t>(scale.shape[1]),
        };
        result.materialization = llama_safetensors_quant_materialization::FP8_BLOCK_SCALE;
    }
    return result;
}

std::string llama_safetensors_quant_adapters::weight_scale_name(const std::string & module) const {
    const llama_safetensors_quant_group * group = match(module);
    if (group != nullptr && group->format == llama_safetensors_quant_format::FP8_BLOCK) {
        return module + ".weight_scale_inv";
    }
    return module + ".weight_scale";
}

const llama_safetensors_quant_summary & llama_safetensors_quant_adapters::summary() const {
    return summary_;
}

uint32_t llama_safetensors_quant_adapters::file_type() const {
    if (summary_.awq + summary_.gptq + summary_.packed_int4 != 0) {
        return LLAMA_FTYPE_MOSTLY_Q4_1;
    }
    if (summary_.w8a8 + summary_.packed_int8 != 0) {
        return LLAMA_FTYPE_MOSTLY_Q8_0;
    }
    if (summary_.nvfp4 != 0) {
        return LLAMA_FTYPE_MOSTLY_NVFP4;
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
            throw std::runtime_error("unsupported zero-point tensor '" + tensor.name + "'");
        }
        if (ends_with(tensor.name, ".zero_point")) {
            throw std::runtime_error("unsupported zero-point tensor '" + tensor.name + "'");
        }

        std::string module;
        llama_safetensors_quant_format expected;
        if (ends_with(tensor.name, ".weight_packed")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".weight_packed").size());
            const auto * group = match(module);
            if (group == nullptr) {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            if (expected == llama_safetensors_quant_format::NVFP4_PACK) {
                ++summary_.nvfp4;
            } else if (expected == llama_safetensors_quant_format::PACKED_INT && group->num_bits == 4) {
                ++summary_.packed_int4;
            } else if (expected == llama_safetensors_quant_format::PACKED_INT && group->num_bits == 8) {
                ++summary_.packed_int8;
            } else {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
        } else if (ends_with(tensor.name, ".qweight")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".qweight").size());
            const llama_safetensors_quant_group * group = match(module);
            if (group == nullptr ||
                (group->format != llama_safetensors_quant_format::AWQ_G128 &&
                 group->format != llama_safetensors_quant_format::GPTQ_G128)) {
                throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
            }
            expected = group->format;
            if (expected == llama_safetensors_quant_format::AWQ_G128) {
                ++summary_.awq;
            } else {
                ++summary_.gptq;
            }
        } else if (ends_with(tensor.name, ".weight")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".weight").size());
            const llama_safetensors_quant_group * group = match(module);
            if (group != nullptr && group->format == llama_safetensors_quant_format::INT8_CHANNEL) {
                if (tensor.dtype != llama_safetensors_dtype::I8) {
                    const std::string scale_name = module + ".weight_scale";
                    if (registry_.find(scale_name) != nullptr) {
                        throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
                    }
                    continue;
                }
                expected = llama_safetensors_quant_format::INT8_CHANNEL;
                ++summary_.w8a8;
            } else {
                const bool fp8_group = group != nullptr &&
                    (group->format == llama_safetensors_quant_format::FP8_CHANNEL ||
                     group->format == llama_safetensors_quant_format::FP8_BLOCK);
                if (!fp8_group) {
                    if (tensor.dtype == llama_safetensors_dtype::F8_E4M3) {
                        throw std::runtime_error("quantization contract does not match source tensor '" + tensor.name + "'");
                    }
                    continue;
                }
                if (tensor.dtype != llama_safetensors_dtype::F8_E4M3) {
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
                if (expected == llama_safetensors_quant_format::FP8_CHANNEL) {
                    ++summary_.fp8_channel;
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

        if (expected == llama_safetensors_quant_format::PACKED_INT) {
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
                const auto & zero = require_tensor(registry_, zero_name);
                dependencies_[tensor.name].push_back(zero.name);
                if (zero.dtype != llama_safetensors_dtype::I32 ||
                    zero.shape != std::vector<uint64_t>({ (rows + 7) / 8, groups })) {
                    throw std::runtime_error(
                        "invalid packed integer zero-point contract for source tensor '" + tensor.name + "'");
                }
            } else if (registry_.find(module + ".weight_zero_point") != nullptr) {
                throw std::runtime_error(
                    "symmetric packed integer tensor has an unexpected zero point: '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::AWQ_G128) {
            const std::string qzeros_name = module + ".qzeros";
            const std::string scales_name = module + ".scales";
            const auto & qzeros = require_tensor(registry_, qzeros_name);
            const auto & scales = require_tensor(registry_, scales_name);
            dependencies_[tensor.name] = { qzeros_name, scales_name };
            if (tensor.dtype != llama_safetensors_dtype::I32 || tensor.shape.size() != 2 ||
                tensor.shape[0] % 128 != 0 || tensor.shape[1] == 0 ||
                qzeros.dtype != llama_safetensors_dtype::I32 || qzeros.shape.size() != 2 ||
                qzeros.shape[0] != tensor.shape[0] / 128 || qzeros.shape[1] != tensor.shape[1] ||
                scales.dtype != llama_safetensors_dtype::F16 || scales.shape.size() != 2 ||
                scales.shape[0] != tensor.shape[0] / 128 || scales.shape[1] != tensor.shape[1] * 8) {
                throw std::runtime_error("invalid AWQ W4A16 group-128 contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::GPTQ_G128) {
            const std::string qzeros_name = module + ".qzeros";
            const std::string scales_name = module + ".scales";
            const std::string g_idx_name  = module + ".g_idx";
            const auto & qzeros = require_tensor(registry_, qzeros_name);
            const auto & scales = require_tensor(registry_, scales_name);
            const auto & g_idx  = require_tensor(registry_, g_idx_name);
            dependencies_[tensor.name] = { qzeros_name, scales_name, g_idx_name };
            if (tensor.shape.size() != 2 || tensor.shape[0] > std::numeric_limits<uint64_t>::max() / 8 ||
                tensor.shape[1] % 8 != 0) {
                throw std::runtime_error("invalid packed GPTQ dimensions for source tensor '" + tensor.name + "'");
            }
            const uint64_t cols = tensor.shape[0] * 8;
            const uint64_t rows = tensor.shape[1];
            if (tensor.dtype != llama_safetensors_dtype::I32 ||
                cols % 128 != 0 || rows == 0 ||
                qzeros.dtype != llama_safetensors_dtype::I32 || qzeros.shape != std::vector<uint64_t>({ cols / 128, rows / 8 }) ||
                scales.dtype != llama_safetensors_dtype::F16 || scales.shape != std::vector<uint64_t>({ cols / 128, rows }) ||
                g_idx.dtype != llama_safetensors_dtype::I32 || g_idx.shape != std::vector<uint64_t>({ cols })) {
                throw std::runtime_error("invalid non-act-order GPTQ W4A16 group-128 contract for source tensor '" + tensor.name + "'");
            }
            const std::vector<uint8_t> indices = registry_.read(g_idx);
            for (size_t col = 0; col < cols; ++col) {
                uint32_t group;
                std::memcpy(&group, indices.data() + col * sizeof(group), sizeof(group));
                if (group != col / 128) {
                    throw std::runtime_error(
                        "GPTQ g_idx is not the identity group map; act-order checkpoints are unsupported");
                }
            }
        } else if (expected == llama_safetensors_quant_format::INT8_CHANNEL) {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (tensor.shape.size() != 2 || tensor.shape[1] % 32 != 0 ||
                scale.dtype != llama_safetensors_dtype::BF16 || scale.shape.empty() || scale.shape.size() > 2 ||
                scale.shape[0] != tensor.shape[0] || (scale.shape.size() == 2 && scale.shape[1] != 1)) {
                throw std::runtime_error("invalid W8A8 channel-scale contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::FP8_CHANNEL) {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (tensor.shape.size() != 2 || scale.dtype != llama_safetensors_dtype::BF16 || scale.shape.empty() ||
                scale.shape.size() > 2 || scale.shape[0] != tensor.shape[0] ||
                (scale.shape.size() == 2 && scale.shape[1] != 1)) {
                throw std::runtime_error("invalid channel-scale contract for source tensor '" + tensor.name + "'");
            }
        } else if (expected == llama_safetensors_quant_format::FP8_BLOCK) {
            const std::string scale_name = module + ".weight_scale_inv";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = { scale_name };
            if (tensor.shape.size() != 2 || tensor.shape[0] % 128 != 0 || tensor.shape[1] % 128 != 0 ||
                scale.dtype != llama_safetensors_dtype::BF16 || scale.shape.size() != 2 ||
                scale.shape[0] != tensor.shape[0] / 128 || scale.shape[1] != tensor.shape[1] / 128) {
                throw std::runtime_error("invalid 128x128 block-scale contract for source tensor '" + tensor.name + "'");
            }
        } else {
            const std::string scale_name = module + ".weight_scale";
            const auto & scale = require_tensor(registry_, scale_name);
            dependencies_[tensor.name] = {
                scale_name,
                module + ".weight_global_scale",
                module + ".input_global_scale",
            };
            if (tensor.shape.size() != 2 || tensor.dtype != llama_safetensors_dtype::U8 ||
                scale.dtype != llama_safetensors_dtype::F8_E4M3 || scale.shape.size() != 2 ||
                scale.shape[0] != tensor.shape[0] || tensor.shape[1] % 8 != 0 ||
                tensor.shape[1] / 8 != scale.shape[1] || scale.shape[1] % 4 != 0) {
                throw std::runtime_error("invalid packed NVFP4 scale contract for source tensor '" + tensor.name + "'");
            }
            require_positive_f32_scalar(registry_, module + ".weight_global_scale");
            require_positive_f32_scalar(registry_, module + ".input_global_scale");
        }
    }
    if (summary_.nvfp4 + summary_.fp8_channel + summary_.fp8_block + summary_.w8a8 + summary_.awq + summary_.gptq +
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
        weight_desc.shape.size() != 2 || scale_desc.shape.size() != 2) {
        throw std::runtime_error("invalid NVFP4 source tensor contract");
    }
    const size_t rows        = weight_desc.shape[0];
    const size_t packed_cols = weight_desc.shape[1];
    const size_t blocks      = scale_desc.shape[1];
    if (scale_desc.shape[0] != rows || packed_cols != blocks * 8 || blocks % 4 != 0 ||
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

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_w8a8(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        size_t weight_size,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        size_t scale_size) const {
    if (weight_desc.dtype != llama_safetensors_dtype::I8 || weight_desc.shape.size() != 2 ||
        scale_desc.dtype != llama_safetensors_dtype::BF16 || scale_desc.shape.empty() || scale_desc.shape.size() > 2) {
        throw std::runtime_error("invalid W8A8 source tensor contract");
    }
    constexpr size_t qk = 32;
    const size_t rows = weight_desc.shape[0];
    const size_t cols = weight_desc.shape[1];
    if (cols % qk != 0 || weight_size != rows * cols || scale_size != rows * sizeof(uint16_t)) {
        throw std::runtime_error("inconsistent W8A8 source tensor shapes");
    }

    std::vector<ggml_fp16_t> row_scales(rows);
    for (size_t row = 0; row < rows; ++row) {
        uint16_t bf16;
        std::memcpy(&bf16, scale + row * sizeof(bf16), sizeof(bf16));
        const uint32_t bits = uint32_t(bf16) << 16;
        float scale_f32;
        std::memcpy(&scale_f32, &bits, sizeof(scale_f32));
        if (!(scale_f32 >= 0.0f) || !std::isfinite(scale_f32)) {
            throw std::runtime_error("W8A8 channel scale must be finite and non-negative");
        }
        row_scales[row] = ggml_fp32_to_fp16(scale_f32);
        if (!std::isfinite(ggml_fp16_to_fp32(row_scales[row]))) {
            throw std::runtime_error("W8A8 channel scale is not representable in Q8_0");
        }
    }

    const size_t block_size = sizeof(ggml_fp16_t) + qk;
    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        const ggml_fp16_t scale_f16 = row_scales[row];
        for (size_t block = 0; block < cols / qk; ++block) {
            uint8_t * out = result.data() + (row * (cols / qk) + block) * block_size;
            std::memcpy(out, &scale_f16, sizeof(scale_f16));
            std::memcpy(out + sizeof(scale_f16), weight + row * cols + block * qk, qk);
        }
    }
    return result;
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_packed_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & zero_desc,
        const uint8_t * zero,
        const std::array<uint64_t, 2> & weight_shape) const {
    constexpr size_t pack_factor = 8;
    constexpr size_t group_size = 32;
    const size_t rows = weight_shape[0];
    const size_t cols = weight_shape[1];
    const size_t groups = cols / group_size;
    const size_t packed_cols = cols / pack_factor;
    const size_t packed_rows = (rows + pack_factor - 1) / pack_factor;
    if (weight_desc.dtype != llama_safetensors_dtype::I32 ||
        weight_desc.shape != std::vector<uint64_t>({ rows, packed_cols }) ||
        scale_desc.dtype != llama_safetensors_dtype::BF16 ||
        scale_desc.shape != std::vector<uint64_t>({ rows, groups }) ||
        zero_desc.dtype != llama_safetensors_dtype::I32 ||
        zero_desc.shape != std::vector<uint64_t>({ packed_rows, groups })) {
        throw std::runtime_error("inconsistent compressed-tensors INT4 group-32 source tensors");
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
            for (size_t local_group = 0; local_group < block_values / group_size; ++local_group) {
                const size_t group = ib * (block_values / group_size) + local_group;
                uint16_t scale_bits;
                std::memcpy(&scale_bits, scale + (row * groups + group) * sizeof(scale_bits), sizeof(scale_bits));
                const float scale_f32 = load_bf16(reinterpret_cast<const uint8_t *>(&scale_bits));
                if (!(scale_f32 > 0.0f) || !std::isfinite(scale_f32)) {
                    throw std::runtime_error("packed INT4 scale must be finite and positive");
                }
                std::memcpy(out + local_group * sizeof(scale_bits), &scale_bits, sizeof(scale_bits));

                uint32_t packed_zero;
                std::memcpy(&packed_zero,
                            zero + ((row / pack_factor) * groups + group) * sizeof(packed_zero),
                            sizeof(packed_zero));
                const uint8_t zero_code = (packed_zero >> (4 * (row % pack_factor))) & 0x0f;
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

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_awq(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + qk / 2;
    constexpr uint32_t shifts[8] = { 0, 16, 4, 20, 8, 24, 12, 28 };

    if (qweight_desc.shape.size() != 2) {
        throw std::runtime_error("invalid packed AWQ dimensions");
    }
    const size_t cols = qweight_desc.shape[0];
    const size_t rows = qweight_desc.shape[1] * 8;
    const size_t groups = cols / 128;
    if (qweight_desc.dtype != llama_safetensors_dtype::I32 ||
        qzeros_desc.dtype != llama_safetensors_dtype::I32 || qzeros_desc.shape != std::vector<uint64_t>({ groups, rows / 8 }) ||
        scales_desc.dtype != llama_safetensors_dtype::F16 || scales_desc.shape != std::vector<uint64_t>({ groups, rows }) ||
        qweight_desc.size != cols * rows / 2 || qzeros_desc.size != groups * rows / 2 ||
        scales_desc.size != groups * rows * sizeof(uint16_t)) {
        throw std::runtime_error("inconsistent AWQ W4A16 group-128 source tensors");
    }

    for (size_t i = 0; i < groups * rows; ++i) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, scales + i * sizeof(scale_bits), sizeof(scale_bits));
        const float value = ggml_fp16_to_fp32(scale_bits);
        if (!(value > 0.0f) || !std::isfinite(value)) {
            throw std::runtime_error("AWQ scale must be finite and positive");
        }
    }

    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        const size_t packed_row = row / 8;
        const uint32_t shift = shifts[row % 8];
        uint8_t * row_out = result.data() + row * (cols / qk) * block_size;
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t group = block / (128 / qk);
            uint16_t scale_bits;
            std::memcpy(&scale_bits, scales + (group * rows + row) * sizeof(scale_bits), sizeof(scale_bits));
            const float scale = ggml_fp16_to_fp32(scale_bits);
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

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_gptq(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales) const {
    constexpr size_t qk = 32;
    constexpr size_t block_size = 2 * sizeof(ggml_fp16_t) + qk / 2;

    if (qweight_desc.shape.size() != 2) {
        throw std::runtime_error("invalid packed GPTQ dimensions");
    }
    const size_t cols = qweight_desc.shape[0] * 8;
    const size_t rows = qweight_desc.shape[1];
    const size_t groups = cols / 128;
    if (qweight_desc.dtype != llama_safetensors_dtype::I32 || rows % 8 != 0 ||
        qzeros_desc.dtype != llama_safetensors_dtype::I32 || qzeros_desc.shape != std::vector<uint64_t>({ groups, rows / 8 }) ||
        scales_desc.dtype != llama_safetensors_dtype::F16 || scales_desc.shape != std::vector<uint64_t>({ groups, rows }) ||
        qweight_desc.size != cols * rows / 2 || qzeros_desc.size != groups * rows / 2 ||
        scales_desc.size != groups * rows * sizeof(uint16_t)) {
        throw std::runtime_error("inconsistent non-act-order GPTQ W4A16 group-128 source tensors");
    }

    for (size_t i = 0; i < groups * rows; ++i) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, scales + i * sizeof(scale_bits), sizeof(scale_bits));
        const float value = ggml_fp16_to_fp32(scale_bits);
        if (!(value > 0.0f) || !std::isfinite(value)) {
            throw std::runtime_error("GPTQ scale must be finite and positive");
        }
    }

    std::vector<uint8_t> result(rows * (cols / qk) * block_size);
    for (size_t row = 0; row < rows; ++row) {
        const size_t packed_row = row / 8;
        const uint32_t row_shift = 4 * (row % 8);
        uint8_t * row_out = result.data() + row * (cols / qk) * block_size;
        for (size_t block = 0; block < cols / qk; ++block) {
            const size_t group = block / (128 / qk);
            uint16_t scale_bits;
            std::memcpy(&scale_bits, scales + (group * rows + row) * sizeof(scale_bits), sizeof(scale_bits));
            const float scale = ggml_fp16_to_fp32(scale_bits);
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

std::vector<uint8_t> llama_safetensors_quant_adapters::read(
        const llama_safetensors_quant_binding & binding) const {
    const llama_safetensors_tensor & primary = require_tensor(registry_, binding.primary);
    if (binding.materialization == llama_safetensors_quant_materialization::NVFP4_REPACK) {
        const llama_safetensors_tensor & auxiliary = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, auxiliary);
        return repack_nvfp4(primary, weight.data(), primary.size, auxiliary, scale.data(), auxiliary.size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::W8A8_REPACK) {
        const llama_safetensors_tensor & auxiliary = require_tensor(registry_, binding.auxiliaries.at(0));
        const source_bytes weight(registry_, primary);
        const source_bytes scale(registry_, auxiliary);
        return repack_w8a8(primary, weight.data(), primary.size, auxiliary, scale.data(), auxiliary.size);
    }
    if (binding.materialization == llama_safetensors_quant_materialization::AWQ_REPACK) {
        const llama_safetensors_tensor & qzeros_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor & scales_desc = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes qweight(registry_, primary);
        const source_bytes qzeros(registry_, qzeros_desc);
        const source_bytes scales(registry_, scales_desc);
        return repack_awq(primary, qweight.data(), qzeros_desc, qzeros.data(), scales_desc, scales.data());
    }
    if (binding.materialization == llama_safetensors_quant_materialization::GPTQ_REPACK) {
        const llama_safetensors_tensor & qzeros_desc = require_tensor(registry_, binding.auxiliaries.at(0));
        const llama_safetensors_tensor & scales_desc = require_tensor(registry_, binding.auxiliaries.at(1));
        const source_bytes qweight(registry_, primary);
        const source_bytes qzeros(registry_, qzeros_desc);
        const source_bytes scales(registry_, scales_desc);
        return repack_gptq(primary, qweight.data(), qzeros_desc, qzeros.data(), scales_desc, scales.data());
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
            const auto & zero_desc = require_tensor(registry_, binding.auxiliaries.at(2));
            const source_bytes zero(registry_, zero_desc);
            return repack_packed_int4(
                primary, packed_weight.data(), scale_desc, scale.data(), zero_desc, zero.data(), weight_shape);
        }
        return repack_packed_int8(primary, packed_weight.data(), scale_desc, scale.data(), weight_shape);
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
    return registry_.read(primary);
}

std::vector<uint8_t> llama_safetensors_quant_adapters::finalize(
        const llama_safetensors_quant_binding & binding,
        std::vector<uint8_t> data) const {
    if (binding.materialization != llama_safetensors_quant_materialization::FP8_BLOCK_SCALE) {
        return data;
    }
    const llama_safetensors_tensor & source = require_tensor(registry_, binding.primary);
    return bf16_to_f32(transpose_2d(data, source.shape[0], source.shape[1], sizeof(uint16_t)));
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
