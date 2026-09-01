#pragma once

#include "ggml.h"
#include "llama-safetensors.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_safetensors_quant_summary {
    size_t nvfp4       = 0;
    size_t mxfp4       = 0;
    size_t mxfp8       = 0;
    size_t fp8_group   = 0;
    size_t fp8_channel = 0;
    size_t fp8_tensor  = 0;
    size_t fp8_block   = 0;
    size_t w8a8        = 0;
    size_t w4a8        = 0;
    size_t awq          = 0;
    size_t quark_w4a16  = 0;
    size_t gptq         = 0;
    size_t gptq_int8    = 0;
    size_t eetq         = 0;
    size_t quanto_int4  = 0;
    size_t quanto_int8  = 0;
    size_t quanto_fp8   = 0;
    size_t torchao_int4 = 0;
    size_t torchao_intx = 0;
    size_t hqq_int4     = 0;
    size_t bnb_int8     = 0;
    size_t bnb_nf4      = 0;
    size_t bnb_fp4      = 0;
    size_t packed_int4  = 0;
    size_t packed_int8  = 0;
    size_t w4a8_fp8     = 0;
};

enum class llama_safetensors_quant_role {
    WEIGHT,
    WEIGHT_SCALE,
    INPUT_SCALE,
};

enum class llama_safetensors_quant_materialization {
    RAW,
    NVFP4_REPACK,
    MXFP4_REPACK,
    AWQ_REPACK,
    QUARK_W4A16_REPACK,
    GPTQ_REPACK,
    GPTQ8_REPACK,
    GPTQ_SCALE_BUNDLE,
    EETQ_REPACK,
    QUANTO_INT4_REPACK,
    QUANTO_W4A16_MARKER,
    QUANTO_W8A16_SCALE,
    TORCHAO_INT4_REPACK,
    TORCHAO_INTX_REPACK,
    HQQ_INT4_REPACK,
    BNB_INT8_SCALE,
    BNB_SCALE_BUNDLE,
    PACKED_INT4_REPACK,
    PACKED_INT8_REPACK,
    PACKED_INT4_FP8_REPACK,
    INT4_GROUP_REPACK,
    DYNAMIC_FP8_MARKER,
    DYNAMIC_INT8_MARKER,
    DYNAMIC_MXFP4_MARKER,
    DYNAMIC_MXFP8_MARKER,
    DYNAMIC_FP8_GROUP_MARKER,
    DYNAMIC_W4A8_FP8_MARKER,
    STATIC_INT8_ASYM_PARAMS,
    POSITIVE_F32,
    POSITIVE_F32_TO_BF16,
    BROADCAST_BF16_SCALAR,
    RECIPROCAL_F32,
    FP8_BLOCK_SCALE,
    FP8_BLOCK_SCALE_MODELOPT,
};

struct llama_safetensors_quant_binding {
    llama_safetensors_quant_materialization materialization;
    std::string                             primary;
    std::vector<std::string>                auxiliaries;
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
    bool applies(const std::string & module) const;
    const llama_safetensors_quant_summary & summary() const;
    uint32_t file_type() const;

    std::vector<uint8_t> read(const llama_safetensors_quant_binding & binding) const;
    std::vector<uint8_t> finalize(
        const llama_safetensors_quant_binding & binding,
        std::vector<uint8_t> data) const;

    void consume(const llama_safetensors_quant_binding & binding) const;
    void validate_complete() const;

  private:
    const llama_safetensors_registry & registry_;
    llama_safetensors_quant_config      config_;
    bool                                has_quantization_config_;
    llama_safetensors_quant_summary     summary_;
    std::unordered_map<std::string, std::vector<std::string>> dependencies_;
    mutable std::unordered_set<std::string> consumed_;

    const llama_safetensors_quant_group * match(const std::string & module) const;
    bool format_applies(
        const std::string & module,
        const llama_safetensors_quant_group & group) const;
    std::string weight_scale_name(const std::string & module) const;
    std::vector<uint8_t> repack_nvfp4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        size_t weight_size,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        size_t scale_size) const;
    std::vector<uint8_t> repack_eetq(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale) const;
    std::vector<uint8_t> repack_quanto_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & shift_desc,
        const uint8_t * shift,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const;
    std::vector<uint8_t> repack_torchao_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_zero_desc,
        const uint8_t * scale_zero,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const;
    std::vector<uint8_t> repack_torchao_intx(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & zero_desc,
        const uint8_t * zero,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const;
    std::vector<uint8_t> repack_hqq_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & zero_desc,
        const uint8_t * zero,
        const std::vector<int64_t> & target_shape,
        uint32_t group_size) const;
    std::vector<uint8_t> repack_mxfp4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale) const;
    std::vector<uint8_t> repack_awq(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales,
        uint32_t group_size) const;
    std::vector<uint8_t> repack_quark_w4a16(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor * zero_desc,
        const uint8_t * zero,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        uint32_t group_size,
        bool symmetric) const;
    std::vector<uint8_t> repack_gptq(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales,
        uint32_t group_size) const;
    std::vector<uint8_t> repack_gptq8(
        const llama_safetensors_tensor & qweight_desc,
        const uint8_t * qweight,
        const llama_safetensors_tensor & qzeros_desc,
        const uint8_t * qzeros,
        const llama_safetensors_tensor & scales_desc,
        const uint8_t * scales,
        uint32_t group_size,
        ggml_type target_type) const;
    std::vector<uint8_t> repack_packed_int4(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor * zero_desc,
        const uint8_t * zero,
        const std::array<uint64_t, 2> & weight_shape,
        uint32_t group_size,
        bool symmetric) const;
    std::vector<uint8_t> repack_packed_int8(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const std::array<uint64_t, 2> & weight_shape) const;
    std::vector<uint8_t> repack_packed_int4_fp8(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale,
        const llama_safetensors_tensor & channel_scale_desc,
        const uint8_t * channel_scale,
        const std::array<uint64_t, 2> & weight_shape) const;
    std::vector<uint8_t> repack_int4_group(
        const llama_safetensors_tensor & weight_desc,
        const uint8_t * weight,
        const llama_safetensors_tensor & scale_desc,
        const uint8_t * scale) const;
    void validate();
};
