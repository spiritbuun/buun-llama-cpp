#pragma once

#include "llama-safetensors-importer.h"
#include "llama-safetensors-quant.h"
#include "llama-safetensors.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

class llama_safetensors_metadata_sink;

// Direct source adapter for DeepSeek-V4 and V4.1. The existing DeepSeek-V4 / V4.1 models own
// execution, placement, splitting, MoE caching, and MTP; this class only maps
// the published safetensors contract into their canonical tensor namespace.
class llama_safetensors_deepseek4_importer final : public llama_safetensors_importer {
  public:
    llama_safetensors_deepseek4_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_json config,
        llama_safetensors_io_mode io_mode = llama_safetensors_io_mode::MMAP);

    static bool probe(const llama_safetensors_json & config);
    static bool probe_v41(const llama_safetensors_json & config);

    gguf_context * build_metadata() const override;
    bool describe(
        const std::string & target_name,
        ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const override;
    size_t tensor_capacity_hint() const override;
    void bind(const std::string & target_name) const override;
    std::optional<llama_model_tensor_file_region> file_region(const ggml_tensor * destination) const override;
    bool load(const std::string & target_name, ggml_tensor * destination, bool check_tensor) const override;
    bool can_stream(const std::string & target_name) const override;
    void stream(const std::string & target_name, const std::function<void(const void *, size_t)> & write) const override;
    std::vector<uint8_t> materialize(
        const std::string & target_name, ggml_type target_type, size_t target_size) const override;
    void validate_complete() const override;

  private:
    void emit_v41_metadata(llama_safetensors_metadata_sink & sink, const std::string & arch) const;
    void emit_v41_token_map(gguf_context * ctx) const;

    std::filesystem::path model_dir_;
    llama_safetensors_json config_;
    bool v41_ = false;
    const llama_safetensors_json & hparams_; // config_, or its text_config for V4.1
    llama_safetensors_json generation_;
    llama_safetensors_tokenizer_json tokenizer_;
    std::optional<std::string> chat_template_;
    llama_safetensors_registry registry_;
    std::unique_ptr<llama_safetensors_quant_adapters> quant_;
    uint32_t n_layer_ = 0;
    uint32_t n_mtp_ = 0;
    uint32_t n_expert_ = 0;
};
