#pragma once

#include "ggml.h"
#include "gguf.h"
#include "llama-safetensors.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

// P0 importer for Qwen3.5 dense compressed-tensors checkpoints. It maps the
// logical GGUF tensor names used by the runtime back to their safetensors
// groups and materializes exactly one final tensor at a time.
class llama_safetensors_qwen35_importer {
  public:
    explicit llama_safetensors_qwen35_importer(const std::filesystem::path & model_dir);

    std::vector<uint8_t> materialize(const std::string & target_name, ggml_type target_type, size_t target_size) const;

    // Builds the complete runtime metadata and tensor-type manifest directly
    // from config/tokenizer JSON and the source tensor registry. The caller
    // owns the returned GGUF context.
    gguf_context * build_metadata() const;

  private:
    std::filesystem::path      model_dir_;
    llama_safetensors_registry registry_;
};
