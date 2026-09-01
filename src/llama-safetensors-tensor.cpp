#include "llama-safetensors-tensor.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

const llama_safetensors_tensor * find_source(const llama_safetensors_registry &       registry,
                                             const llama_safetensors_tensor_binding & binding) {
    return registry.find(binding.source);
}

ggml_type plain_target_type(const llama_safetensors_tensor & source) {
    switch (source.dtype) {
        case llama_safetensors_dtype::F8_E4M3:
            return GGML_TYPE_F8_E4M3;
        case llama_safetensors_dtype::BF16:
            return source.shape.size() >= 2 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        case llama_safetensors_dtype::F16:
            return source.shape.size() >= 2 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        case llama_safetensors_dtype::F32:
            return GGML_TYPE_F32;
        default:
            throw std::runtime_error("unsupported plain safetensors dtype for '" + source.name + "'");
    }
}

std::vector<int64_t> reverse_shape(const llama_safetensors_tensor & source) {
    std::vector<int64_t> result;
    result.reserve(source.shape.size());
    for (auto it = source.shape.rbegin(); it != source.shape.rend(); ++it) {
        if (*it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("source dimension exceeds runtime limits for '" + source.name + "'");
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

}  // namespace

std::vector<uint8_t> llama_safetensors_bf16_to_f32(const std::vector<uint8_t> & source) {
    if (source.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("invalid BF16 byte count");
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

std::vector<uint8_t> llama_safetensors_f16_to_f32(const std::vector<uint8_t> & source) {
    if (source.size() % sizeof(ggml_fp16_t) != 0) {
        throw std::runtime_error("invalid F16 byte count");
    }
    std::vector<uint8_t> result(source.size() * 2);
    for (size_t i = 0; i < source.size() / sizeof(ggml_fp16_t); ++i) {
        ggml_fp16_t bits;
        std::memcpy(&bits, source.data() + i * sizeof(bits), sizeof(bits));
        const float value = ggml_fp16_to_fp32(bits);
        std::memcpy(result.data() + i * sizeof(value), &value, sizeof(value));
    }
    return result;
}

llama_safetensors_tensor_binding llama_safetensors_bind_tensor(const llama_safetensors_quant_adapters & quant,
                                                               llama_safetensors_source_name            source) {
    if (source.quant_role) {
        if (auto binding = quant.bind(source.module, *source.quant_role)) {
            return { binding->primary, std::move(binding) };
        }
        if (quant.applies(source.module)) {
            return { {}, std::nullopt };
        }
    }
    return { std::move(source.source), std::nullopt };
}

bool llama_safetensors_describe_tensor(const llama_safetensors_registry &       registry,
                                       const llama_safetensors_tensor_binding & binding,
                                       ggml_type &                              type,
                                       std::array<int64_t, GGML_MAX_DIMS> &     ne) {
    const llama_safetensors_tensor * source = find_source(registry, binding);
    if (source == nullptr) {
        return false;
    }
    type                             = binding.quant ? binding.quant->target_type : plain_target_type(*source);
    const std::vector<int64_t> shape = binding.quant ? binding.quant->target_shape : reverse_shape(*source);
    if (shape.size() > GGML_MAX_DIMS) {
        throw std::runtime_error("target tensor rank exceeds GGML_MAX_DIMS for '" + binding.source + "'");
    }
    ne.fill(1);
    std::copy(shape.begin(), shape.end(), ne.begin());
    return true;
}

void llama_safetensors_consume_tensor(const llama_safetensors_quant_adapters & quant,
                                      const llama_safetensors_tensor_binding & binding) {
    if (binding.quant) {
        quant.consume(*binding.quant);
    }
}

bool llama_safetensors_load_tensor_direct(const llama_safetensors_registry &       registry,
                                          const llama_safetensors_tensor_binding & binding,
                                          ggml_tensor *                            destination,
                                          bool                                     check_tensor) {
    if (binding.quant && binding.quant->materialization != llama_safetensors_quant_materialization::RAW) {
        return false;
    }
    const llama_safetensors_tensor * source = find_source(registry, binding);
    if (source == nullptr) {
        return false;
    }
    const ggml_type source_type = binding.quant ? binding.quant->target_type : plain_target_type(*source);
    if (source_type != destination->type || source->size != ggml_nbytes(destination)) {
        return false;
    }
    const uint8_t * data = registry.data(*source);
    if (data == nullptr) {
        return false;
    }
    if (check_tensor && !ggml_validate_row_data(destination->type, data, source->size)) {
        throw std::runtime_error("tensor '" + std::string(destination->name) + "' has invalid data");
    }
    ggml_backend_tensor_set(destination, data, 0, source->size);
    return true;
}

std::vector<uint8_t> llama_safetensors_materialize_tensor(const llama_safetensors_registry &       registry,
                                                          const llama_safetensors_quant_adapters & quant,
                                                          const llama_safetensors_tensor_binding & binding,
                                                          ggml_type                                target_type,
                                                          size_t                                   target_size) {
    const llama_safetensors_tensor * source = find_source(registry, binding);
    if (source == nullptr) {
        throw std::runtime_error("safetensors tensor not found for import: '" + binding.source + "'");
    }
    std::vector<uint8_t> result = binding.quant ? quant.read(*binding.quant) : registry.read(*source);
    if (binding.quant) {
        result = quant.finalize(*binding.quant, std::move(result));
    } else if (source->dtype == llama_safetensors_dtype::BF16 && target_type == GGML_TYPE_F32) {
        result = llama_safetensors_bf16_to_f32(result);
    } else if (source->dtype == llama_safetensors_dtype::F16 && target_type == GGML_TYPE_F32) {
        result = llama_safetensors_f16_to_f32(result);
    }
    if (result.size() != target_size) {
        throw std::runtime_error("produced " + std::to_string(result.size()) + " bytes, expected " +
                                 std::to_string(target_size));
    }
    return result;
}
