#pragma once

#include "gguf.h"
#include "nlohmann/json.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// Preserve declaration order for producer configuration such as
// compressed-tensors target groups.
using llama_safetensors_json = nlohmann::ordered_json;

llama_safetensors_json     llama_safetensors_read_json(const std::filesystem::path & path);
std::string                llama_safetensors_read_text(const std::filesystem::path & path);
std::optional<std::string> llama_safetensors_read_optional_text(const std::filesystem::path & path);

class llama_safetensors_metadata_sink {
  public:
    llama_safetensors_metadata_sink();
    ~llama_safetensors_metadata_sink();

    llama_safetensors_metadata_sink(const llama_safetensors_metadata_sink &)             = delete;
    llama_safetensors_metadata_sink & operator=(const llama_safetensors_metadata_sink &) = delete;

    void set_string(std::string_view key, std::string_view value);
    void set_u32(std::string_view key, uint32_t value);
    void set_i32(std::string_view key, int32_t value);
    void set_f32(std::string_view key, float value);
    void set_bool(std::string_view key, bool value);
    void set_f32_array(std::string_view key, const float * values, size_t count);
    void set_i32_array(std::string_view key, const int32_t * values, size_t count);
    void set_u32_array(std::string_view key, const uint32_t * values, size_t count);
    void set_u64_array(std::string_view key, const uint64_t * values, size_t count);
    void set_string_array(std::string_view key, const std::vector<std::string> & values);

    gguf_context * release();

  private:
    gguf_context * context_ = nullptr;
};

struct llama_safetensors_sampling_defaults {
    int32_t top_k       = 20;
    float   top_p       = 0.95f;
    float   temperature = 1.0f;
};

void llama_safetensors_emit_sampling_defaults(llama_safetensors_metadata_sink &           sink,
                                              const llama_safetensors_json &              generation,
                                              const llama_safetensors_sampling_defaults & defaults = {});

struct llama_safetensors_rope_config {
    float                  theta;
    float                  partial_rotary_factor;
    std::array<int32_t, 4> mrope_sections;
};

llama_safetensors_rope_config llama_safetensors_parse_rope(const llama_safetensors_json & rope,
                                                           std::array<int32_t, 4>         default_sections,
                                                           float default_partial_rotary_factor);

struct llama_safetensors_bpe_policy {
    std::string                pre_tokenizer;
    uint32_t                   vocab_size;
    uint32_t                   bos_token_id;
    uint32_t                   eos_token_id;
    std::optional<std::string> padding_token;
    bool                       angle_pipe_tokens_are_control = false;
    std::vector<std::string>   control_token_prefixes;
    bool                       add_bos_token = false;
};

void llama_safetensors_emit_bpe_tokenizer(llama_safetensors_metadata_sink &    sink,
                                          const llama_safetensors_json &       tokenizer,
                                          const llama_safetensors_bpe_policy & policy,
                                          const std::optional<std::string> &   chat_template);

void llama_safetensors_emit_spm_tokenizer(llama_safetensors_metadata_sink &  sink,
                                          const std::filesystem::path &       tokenizer_model,
                                          uint32_t                            vocab_size,
                                          uint32_t                            bos_token_id,
                                          uint32_t                            eos_token_id,
                                          std::optional<uint32_t>             padding_token_id,
                                          bool                                add_bos_token,
                                          const std::optional<std::string> & chat_template);

uint32_t llama_safetensors_first_token_id(const llama_safetensors_json & value, std::string_view context);
