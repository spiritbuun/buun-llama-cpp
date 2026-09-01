#pragma once

#include "llama-safetensors-importer.h"
#include "llama-safetensors-quant.h"
#include "llama-safetensors.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

// Direct source adapter for DeepSeek-V4. The existing DeepSeek-V4 model owns
// execution, placement, splitting, MoE caching, and MTP; this class only maps
// the published safetensors contract into its canonical tensor namespace.
class llama_safetensors_deepseek4_importer final : public llama_safetensors_importer {
  public:
    llama_safetensors_deepseek4_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_json config,
        llama_safetensors_io_mode io_mode = llama_safetensors_io_mode::MMAP);

    static bool probe(const llama_safetensors_json & config);

    gguf_context * build_metadata() const override;
    bool describe(
        const std::string & target_name,
        ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const override;
    size_t tensor_capacity_hint() const override;
    void bind(const std::string & target_name) const override;
    bool load(const std::string & target_name, ggml_tensor * destination, bool check_tensor) const override;
    std::vector<uint8_t> materialize(
        const std::string & target_name, ggml_type target_type, size_t target_size) const override;
    void validate_complete() const override;

  private:
    std::filesystem::path model_dir_;
    llama_safetensors_json config_;
    llama_safetensors_json generation_;
    llama_safetensors_json tokenizer_;
    std::optional<std::string> chat_template_;
    llama_safetensors_registry registry_;
    std::unique_ptr<llama_safetensors_quant_adapters> quant_;
    uint32_t n_layer_ = 0;
    uint32_t n_mtp_ = 0;
    uint32_t n_expert_ = 0;
};
