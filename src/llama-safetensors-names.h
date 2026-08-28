#pragma once

#include "llama-safetensors-quant.h"

#include <optional>
#include <string>
#include <string_view>

struct llama_safetensors_source_name {
    std::string                                 source;
    std::string                                 module;
    std::optional<llama_safetensors_quant_role> quant_role;
};

// Maps the ordinary tensor names shared by decoder-only HF architectures.
// Architecture adapters retain recurrent, MoE, MTP, and other exceptional
// names, then attach their own layout/value transforms to this result.
std::optional<llama_safetensors_source_name> llama_safetensors_map_decoder_tensor(std::string_view source_layer_prefix,
                                                                                  std::string_view canonical_suffix);
