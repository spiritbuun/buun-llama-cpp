#pragma once

#include "llama-safetensors-names.h"
#include "llama-safetensors-quant.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

// Format-neutral binding between one canonical runtime tensor and its source
// tensor (plus any quantization auxiliaries). Architecture adapters choose the
// name; this layer owns ordinary dtype, shape, and byte materialization.
struct llama_safetensors_tensor_binding {
    std::string                                    source;
    std::optional<llama_safetensors_quant_binding> quant;
};

llama_safetensors_tensor_binding llama_safetensors_bind_tensor(const llama_safetensors_quant_adapters & quant,
                                                               llama_safetensors_source_name            source);

bool llama_safetensors_describe_tensor(const llama_safetensors_registry &       registry,
                                       const llama_safetensors_tensor_binding & binding,
                                       ggml_type &                              type,
                                       std::array<int64_t, GGML_MAX_DIMS> &     ne);

void llama_safetensors_consume_tensor(const llama_safetensors_quant_adapters & quant,
                                      const llama_safetensors_tensor_binding & binding);

std::vector<uint8_t> llama_safetensors_bf16_to_f32(const std::vector<uint8_t> & source);
std::vector<uint8_t> llama_safetensors_f16_to_f32(const std::vector<uint8_t> & source);

// Uploads source bytes directly from a mapped shard when the source and
// canonical runtime layouts are identical. Returns false when materialization
// or an architecture transform is still required.
bool llama_safetensors_load_tensor_direct(const llama_safetensors_registry &       registry,
                                          const llama_safetensors_tensor_binding & binding,
                                          ggml_tensor *                            destination,
                                          bool                                     check_tensor);

std::vector<uint8_t> llama_safetensors_materialize_tensor(const llama_safetensors_registry &       registry,
                                                          const llama_safetensors_quant_adapters & quant,
                                                          const llama_safetensors_tensor_binding & binding,
                                                          ggml_type                                target_type,
                                                          size_t                                   target_size);
