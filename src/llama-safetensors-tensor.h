#pragma once

#include "llama-safetensors-names.h"
#include "llama-safetensors-quant.h"
#include "llama-model-source.h"

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

void llama_safetensors_stream_raw(const llama_safetensors_registry & registry,
                                  const llama_safetensors_tensor & source,
                                  const std::function<void(const void *, size_t)> & write);

llama_safetensors_tensor_binding llama_safetensors_bind_tensor(const llama_safetensors_quant_adapters & quant,
                                                               llama_safetensors_source_name            source);

bool llama_safetensors_describe_tensor(const llama_safetensors_registry &       registry,
                                       const llama_safetensors_tensor_binding & binding,
                                       ggml_type &                              type,
                                       std::array<int64_t, GGML_MAX_DIMS> &     ne);

void llama_safetensors_consume_tensor(const llama_safetensors_quant_adapters & quant,
                                      const llama_safetensors_tensor_binding & binding);

// ggml_backend_tensor_set spread over host threads for large host-resident destinations: the
// source is usually an mmap whose pages fault in on first touch, so one memcpy runs at ~1 GB/s.
// Device destinations and small copies fall through to a plain ggml_backend_tensor_set.
void llama_safetensors_tensor_set_parallel(ggml_tensor * destination, const void * data, size_t offset, size_t size);

std::vector<uint8_t> llama_safetensors_bf16_to_f32(const std::vector<uint8_t> & source);
std::vector<uint8_t> llama_safetensors_f16_to_f32(const std::vector<uint8_t> & source);

// Uploads source bytes directly from a mapped shard when the source and
// canonical runtime layouts are identical. Returns false when materialization
// or an architecture transform is still required.
bool llama_safetensors_load_tensor_direct(const llama_safetensors_registry &       registry,
                                          const llama_safetensors_tensor_binding & binding,
                                          ggml_tensor *                            destination,
                                          bool                                     check_tensor);

std::optional<llama_model_tensor_file_region> llama_safetensors_tensor_file_region(
    const llama_safetensors_registry & registry,
    const llama_safetensors_tensor_binding & binding,
    const ggml_tensor * destination);

std::vector<uint8_t> llama_safetensors_materialize_tensor(const llama_safetensors_registry &       registry,
                                                          const llama_safetensors_quant_adapters & quant,
                                                          const llama_safetensors_tensor_binding & binding,
                                                          ggml_type                                target_type,
                                                          size_t                                   target_size);
