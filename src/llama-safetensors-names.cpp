#include "llama-safetensors-names.h"

#include <array>

namespace {

struct projection_name {
    std::string_view canonical;
    std::string_view source;
};

std::optional<llama_safetensors_source_name> map_projection(std::string_view prefix,
                                                            std::string_view suffix,
                                                            std::string_view canonical,
                                                            std::string_view source) {
    if (suffix.size() <= canonical.size() || suffix.substr(0, canonical.size()) != canonical) {
        return std::nullopt;
    }

    const std::string_view        tail = suffix.substr(canonical.size());
    llama_safetensors_source_name result;
    result.module = std::string(prefix) + std::string(source);
    if (tail == ".weight") {
        result.source     = result.module + ".weight";
        result.quant_role = llama_safetensors_quant_role::WEIGHT;
    } else if (tail == ".scale") {
        result.source     = result.module + ".weight_scale";
        result.quant_role = llama_safetensors_quant_role::WEIGHT_SCALE;
    } else if (tail == ".input_scale") {
        result.source     = result.module + ".input_global_scale";
        result.quant_role = llama_safetensors_quant_role::INPUT_SCALE;
    } else {
        return std::nullopt;
    }
    return result;
}

}  // namespace

std::optional<llama_safetensors_source_name> llama_safetensors_map_decoder_tensor(std::string_view prefix,
                                                                                  std::string_view suffix) {
    static constexpr std::array<projection_name, 7> projections = {
        {
         { "attn_q", "self_attn.q_proj" },
         { "attn_k", "self_attn.k_proj" },
         { "attn_v", "self_attn.v_proj" },
         { "attn_output", "self_attn.o_proj" },
         { "ffn_gate", "mlp.gate_proj" },
         { "ffn_up", "mlp.up_proj" },
         { "ffn_down", "mlp.down_proj" },
         }
    };
    for (const projection_name & projection : projections) {
        if (auto mapped = map_projection(prefix, suffix, projection.canonical, projection.source)) {
            return mapped;
        }
    }

    static constexpr std::array<projection_name, 4> plain = {
        {
         { "attn_norm.weight", "input_layernorm.weight" },
         { "post_attention_norm.weight", "post_attention_layernorm.weight" },
         { "attn_q_norm.weight", "self_attn.q_norm.weight" },
         { "attn_k_norm.weight", "self_attn.k_norm.weight" },
         }
    };
    for (const projection_name & name : plain) {
        if (suffix == name.canonical) {
            return llama_safetensors_source_name{
                std::string(prefix) + std::string(name.source),
                {},
                std::nullopt,
            };
        }
    }
    return std::nullopt;
}
