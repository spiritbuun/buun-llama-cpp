#pragma once

#include "ggml.h"
#include "llama-safetensors.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_safetensors_quant_summary {
    size_t nvfp4       = 0;
    size_t fp8_channel = 0;
    size_t fp8_block   = 0;
};

enum class llama_safetensors_quant_role {
    WEIGHT,
    WEIGHT_SCALE,
    INPUT_SCALE,
};

enum class llama_safetensors_quant_materialization {
    RAW,
    NVFP4_REPACK,
    RECIPROCAL_F32,
    FP8_BLOCK_SCALE,
};

struct llama_safetensors_quant_binding {
    llama_safetensors_quant_materialization materialization;
    std::string                             primary;
    std::string                             auxiliary;
    ggml_type                               target_type;
    std::vector<int64_t>                    target_shape;
};

// Container-independent compressed-tensors contract and materialization
// helpers. Architecture adapters select canonical names and apply their own
// permutations; numerical-format validation belongs here.
class llama_safetensors_quant_adapters {
  public:
    llama_safetensors_quant_adapters(
        const llama_safetensors_json & config,
        const llama_safetensors_registry & registry);

    std::optional<llama_safetensors_quant_binding> bind(
        const std::string & module, llama_safetensors_quant_role role) const;
    const llama_safetensors_quant_summary & summary() const;

    std::vector<uint8_t> read(const llama_safetensors_quant_binding & binding) const;
    std::vector<uint8_t> finalize(
        const llama_safetensors_quant_binding & binding,
        std::vector<uint8_t> data) const;

    void consume(const llama_safetensors_quant_binding & binding) const;
    void validate_complete() const;

  private:
    const llama_safetensors_registry & registry_;
    llama_safetensors_quant_config      config_;
    llama_safetensors_quant_summary     summary_;
    std::unordered_map<std::string, std::vector<std::string>> dependencies_;
    mutable std::unordered_set<std::string> consumed_;

    const llama_safetensors_quant_group * match(const std::string & module) const;
    std::string weight_scale_name(const std::string & module) const;
    std::vector<uint8_t> repack_nvfp4(
        const llama_safetensors_tensor & weight_desc,
        const std::vector<uint8_t> & weight,
        const llama_safetensors_tensor & scale_desc,
        const std::vector<uint8_t> & scale) const;
    void validate();
};
