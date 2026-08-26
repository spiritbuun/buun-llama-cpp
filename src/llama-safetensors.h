#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_model_params;

// Internal source-loader seam. Architecture probing belongs behind this
// boundary so the public model-loading entry point remains format-agnostic.
llama_model * llama_model_load_from_safetensors_dir(
    const std::filesystem::path & model_dir, llama_model_params params);

enum class llama_safetensors_dtype {
    BOOL,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F8_E4M3,
    F8_E5M2,
    F16,
    BF16,
    F32,
    F64,
};

struct llama_safetensors_shard {
    std::filesystem::path path;
    uint64_t              file_size  = 0;
    uint64_t              data_begin = 0;
};

struct llama_safetensors_tensor {
    std::string             name;
    llama_safetensors_dtype dtype;
    std::vector<uint64_t>   shape;
    uint32_t                shard  = 0;
    uint64_t                offset = 0;
    uint64_t                size   = 0;
};

enum class llama_safetensors_quant_format {
    NVFP4_PACK,
    FP8_CHANNEL,
};

struct llama_safetensors_quant_group {
    std::string                    name;
    llama_safetensors_quant_format format;
    uint32_t                       group_size = 0;
    std::vector<uint32_t>          block_structure;
};

// Parsed compressed-tensors contracts from config.json. Matching preserves the
// producer's declaration order and anchors regex matching at the module start.
class llama_safetensors_quant_config {
  public:
    static llama_safetensors_quant_config load(const std::filesystem::path & model_dir);

    const llama_safetensors_quant_group * match(const std::string & module_name) const;
    bool                                  ignored(const std::string & module_name) const;

  private:
    struct rule {
        std::string target;
        bool        is_regex = false;
        std::regex  pattern;
        uint32_t    group = 0;
    };

    static rule make_rule(const std::string & target, uint32_t group);
    static bool rule_matches(const rule & candidate, const std::string & module_name);

    std::vector<llama_safetensors_quant_group> groups_;
    std::vector<rule>                          rules_;
    std::vector<rule>                          ignore_;
};

// Strict, read-only index over a local safetensors model directory. Tensor
// offsets are absolute file offsets and are bounds-checked during parsing.
// This is intentionally independent of GGUF tensor naming and quantization
// contracts; those are layered on top by the model importer.
class llama_safetensors_registry {
  public:
    static llama_safetensors_registry load(const std::filesystem::path & model_dir);

    const llama_safetensors_tensor * find(const std::string & name) const;
    std::vector<uint8_t>             read(const llama_safetensors_tensor & tensor) const;
    std::vector<uint8_t>             read(const std::string & name) const;

    const std::vector<llama_safetensors_shard> &  shards() const;
    const std::vector<llama_safetensors_tensor> & tensors() const;

  private:
    std::vector<llama_safetensors_shard>    shards_;
    std::vector<llama_safetensors_tensor>   tensors_;
    std::unordered_map<std::string, size_t> tensor_index_;
};
