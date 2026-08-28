#pragma once

#include "ggml.h"
#include "gguf.h"
#include "llama-safetensors-importer.h"
#include "llama-safetensors.h"
#include "llama-safetensors-quant.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Importer for Qwen3.5 dense compressed-tensors checkpoints. It maps the
// logical GGUF tensor names used by the runtime back to their safetensors
// groups and materializes exactly one final tensor at a time.
class llama_safetensors_qwen35_importer final : public llama_safetensors_importer {
  public:
    llama_safetensors_qwen35_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_json config);

    static bool probe(const llama_safetensors_json & config);

    std::vector<uint8_t> materialize(
        const std::string & target_name, ggml_type target_type, size_t target_size) const override;

    bool describe(
        const std::string & target_name,
        ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const override;

    size_t tensor_capacity_hint() const override;
    void bind(const std::string & target_name) const override;
    void validate_complete() const override;

    // Builds model/tokenizer metadata only. Tensor descriptions are answered
    // on demand through describe(); the caller owns the returned context.
    gguf_context * build_metadata() const override;

  private:
    std::filesystem::path      model_dir_;
    llama_safetensors_json     config_;
    llama_safetensors_json     generation_;
    llama_safetensors_json     tokenizer_;
    std::optional<std::string> chat_template_;
    llama_safetensors_registry registry_;
    std::unique_ptr<llama_safetensors_quant_adapters> quant_;
    uint32_t n_layer_         = 0;
    uint32_t n_mtp_           = 0;
    uint32_t n_key_heads_     = 0;
    uint32_t n_value_heads_   = 0;
    uint32_t key_head_dim_    = 0;
    uint32_t value_head_dim_  = 0;
};
