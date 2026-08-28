#include "llama-safetensors-quant.h"

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

}  // namespace

llama_safetensors_quant_adapters::llama_safetensors_quant_adapters(
        const std::filesystem::path & model_dir,
        const llama_safetensors_registry & registry) :
    registry_(registry), config_(llama_safetensors_quant_config::load(model_dir)) {
    validate();
}

const llama_safetensors_quant_group * llama_safetensors_quant_adapters::match(
        const std::string & module) const {
    return config_.match(module);
}

std::optional<llama_safetensors_quant_binding> llama_safetensors_quant_adapters::bind(
        const std::string & module, llama_safetensors_quant_role role) const {
    const llama_safetensors_quant_group * group = match(module);
    if (group == nullptr) {
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
            result.auxiliary      = module + ".weight_scale";
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

void llama_safetensors_quant_adapters::validate() {
    for (const llama_safetensors_tensor & tensor : registry_.tensors()) {
        for (uint64_t dim : tensor.shape) {
            if (dim == 0 || dim > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("unsupported dimension in source tensor '" + tensor.name + "'");
            }
        }
        if (ends_with(tensor.name, ".weight_zero_point") || ends_with(tensor.name, ".zero_point")) {
            throw std::runtime_error("unsupported zero-point tensor '" + tensor.name + "'");
        }

        std::string module;
        llama_safetensors_quant_format expected;
        if (ends_with(tensor.name, ".weight_packed")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".weight_packed").size());
            expected = llama_safetensors_quant_format::NVFP4_PACK;
            ++summary_.nvfp4;
        } else if (ends_with(tensor.name, ".weight")) {
            module = tensor.name.substr(0, tensor.name.size() - std::string_view(".weight").size());
            const llama_safetensors_quant_group * group = match(module);
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
        } else {
            continue;
        }

        const llama_safetensors_quant_group * group = match(module);
        if (group == nullptr || group->format != expected) {
            throw std::runtime_error("compressed-tensors contract does not match source tensor '" + tensor.name + "'");
        }

        if (expected == llama_safetensors_quant_format::FP8_CHANNEL) {
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
    if (summary_.nvfp4 + summary_.fp8_channel + summary_.fp8_block == 0) {
        throw std::runtime_error("native safetensors model contains no supported quantized weights");
    }
}

std::vector<uint8_t> llama_safetensors_quant_adapters::repack_nvfp4(
        const llama_safetensors_tensor & weight_desc,
        const std::vector<uint8_t> & weight,
        const llama_safetensors_tensor & scale_desc,
        const std::vector<uint8_t> & scale) const {
    if (weight_desc.dtype != llama_safetensors_dtype::U8 || scale_desc.dtype != llama_safetensors_dtype::F8_E4M3 ||
        weight_desc.shape.size() != 2 || scale_desc.shape.size() != 2) {
        throw std::runtime_error("invalid NVFP4 source tensor contract");
    }
    const size_t rows        = weight_desc.shape[0];
    const size_t packed_cols = weight_desc.shape[1];
    const size_t blocks      = scale_desc.shape[1];
    if (scale_desc.shape[0] != rows || packed_cols != blocks * 8 || blocks % 4 != 0 ||
        weight.size() != rows * packed_cols || scale.size() != rows * blocks) {
        throw std::runtime_error("inconsistent NVFP4 source tensor shapes");
    }

    std::vector<uint8_t> result(rows * (blocks / 4) * 36);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t super = 0; super < blocks / 4; ++super) {
            uint8_t * out = result.data() + (row * (blocks / 4) + super) * 36;
            for (size_t block = 0; block < 4; ++block) {
                const size_t  block_index = 4 * super + block;
                const uint8_t scale_byte = scale[row * blocks + block_index];
                if ((scale_byte & 0x80) != 0 || scale_byte == 0x7f) {
                    throw std::runtime_error("NVFP4 block scale is not a non-negative finite E4M3 value");
                }
                out[block] = scale_byte;
                const uint8_t * in = weight.data() + row * packed_cols + block_index * 8;
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

std::vector<uint8_t> llama_safetensors_quant_adapters::read(
        const llama_safetensors_quant_binding & binding) const {
    const llama_safetensors_tensor & primary = require_tensor(registry_, binding.primary);
    if (binding.materialization == llama_safetensors_quant_materialization::NVFP4_REPACK) {
        const llama_safetensors_tensor & auxiliary = require_tensor(registry_, binding.auxiliary);
        return repack_nvfp4(primary, registry_.read(primary), auxiliary, registry_.read(auxiliary));
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
    if (!binding.auxiliary.empty()) {
        consumed_.insert(binding.auxiliary);
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
