#pragma once

#include "ggml.h"
#include "gguf.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// Architecture-facing half of native safetensors loading. Implementations map
// canonical tensors requested by an existing llama_model_* class to source
// tensors and architecture-specific transforms. Backend placement and graph
// construction remain owned by the ordinary model loader.
class llama_safetensors_importer {
  public:
    virtual ~llama_safetensors_importer() = default;

    virtual gguf_context * build_metadata() const = 0;

    virtual bool describe(
        const std::string & canonical_name,
        ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const = 0;

    virtual size_t tensor_capacity_hint() const = 0;
    virtual void bind(const std::string & canonical_name) const = 0;
    virtual bool load(const std::string & canonical_name, ggml_tensor * destination, bool check_tensor) const = 0;
    virtual std::vector<uint8_t> materialize(
        const std::string & canonical_name, ggml_type target_type, size_t target_size) const = 0;
    virtual void validate_complete() const = 0;
};
