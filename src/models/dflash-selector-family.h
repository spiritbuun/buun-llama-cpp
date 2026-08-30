#pragma once

#include <cstdint>
#include <string>

enum class llm_dflash_selector_family {
    none,
    fork_dflash2,
    upstream_compat,
    mixed,
    unidentified,
};

// The fork and upstream compatibility schemas intentionally share GGUF metadata
// keys, so metadata alone cannot select a runtime implementation. Their tensor
// names are distinct and therefore own family selection.
constexpr llm_dflash_selector_family llm_dflash_selector_family_from_identity(
        bool has_selector_metadata,
        bool has_fork_tensor,
        bool has_compat_tensor) {
    if (has_fork_tensor && has_compat_tensor) {
        return llm_dflash_selector_family::mixed;
    }
    if (has_fork_tensor) {
        return llm_dflash_selector_family::fork_dflash2;
    }
    if (has_compat_tensor) {
        return llm_dflash_selector_family::upstream_compat;
    }
    if (has_selector_metadata) {
        return llm_dflash_selector_family::unidentified;
    }
    return llm_dflash_selector_family::none;
}

template<class Loader>
llm_dflash_selector_family llm_dflash_selector_family_from_loader(
        bool has_selector_metadata,
        uint32_t n_layer,
        const Loader & loader) {
    bool has_fork_tensor =
        loader.get_tensor_meta_exact("selector.hidden_proj.weight") != nullptr ||
        loader.get_tensor_meta_exact("selector.pred_codebook") != nullptr ||
        loader.get_tensor_meta_exact("selector.succ_codebook") != nullptr;
    bool has_compat_tensor =
        loader.get_tensor_meta_exact("selector_hidden.weight") != nullptr ||
        loader.get_tensor_meta_exact("selector_predecessor.weight") != nullptr ||
        loader.get_tensor_meta_exact("selector_successor.weight") != nullptr;

    for (uint32_t il = 0; il < n_layer && !(has_fork_tensor && has_compat_tensor); ++il) {
        const std::string prefix = "blk." + std::to_string(il);
        has_fork_tensor |=
            loader.get_tensor_meta_exact((prefix + ".attn_conv.base").c_str()) != nullptr ||
            loader.get_tensor_meta_exact((prefix + ".attn_conv.proj.weight").c_str()) != nullptr ||
            loader.get_tensor_meta_exact((prefix + ".ffn_conv.base").c_str()) != nullptr ||
            loader.get_tensor_meta_exact((prefix + ".ffn_conv.proj.weight").c_str()) != nullptr;
        has_compat_tensor |=
            loader.get_tensor_meta_exact((prefix + ".attn_conv_base").c_str()) != nullptr ||
            loader.get_tensor_meta_exact((prefix + ".attn_conv_proj.weight").c_str()) != nullptr ||
            loader.get_tensor_meta_exact((prefix + ".ffn_conv_base").c_str()) != nullptr ||
            loader.get_tensor_meta_exact((prefix + ".ffn_conv_proj.weight").c_str()) != nullptr;
    }

    return llm_dflash_selector_family_from_identity(
            has_selector_metadata,
            has_fork_tensor,
            has_compat_tensor);
}
