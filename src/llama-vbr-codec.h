#pragma once

#include "llama.h"

#include <cstddef>

// One immutable authority for the representation ladder. The controller owns
// ordering, budgets and transactions; this descriptor owns codec semantics.
struct llama_vbr_codec_ladder {
    const char * name;
    const ggml_type * rungs; // entry-to-bottom, including F16
    size_t n_rungs;
    ggml_type default_floor;
    bool needs_turbo_layout;
};

inline const llama_vbr_codec_ladder & llama_vbr_ladder(llama_vbr_codec codec) {
    static constexpr ggml_type turbo_rungs[] = {
        GGML_TYPE_F16,
        GGML_TYPE_TURBO8_0,
        GGML_TYPE_TURBO4_0,
        GGML_TYPE_TURBO3_TCQ,
        GGML_TYPE_TURBO2_TCQ,
        GGML_TYPE_TURBO1_TCQ,
    };
    static constexpr ggml_type classic_rungs[] = {
        GGML_TYPE_F16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_0,
    };
    static const llama_vbr_codec_ladder turbo = {
        "turbo", turbo_rungs,
        sizeof(turbo_rungs) / sizeof(turbo_rungs[0]), GGML_TYPE_TURBO1_TCQ, true,
    };
    static const llama_vbr_codec_ladder classic = {
        "classic", classic_rungs,
        sizeof(classic_rungs) / sizeof(classic_rungs[0]), GGML_TYPE_Q4_0, false,
    };
    switch (codec) {
        case LLAMA_VBR_CODEC_TURBO:   return turbo;
        case LLAMA_VBR_CODEC_CLASSIC: return classic;
    }
    GGML_ABORT("invalid VBR codec: %d", int(codec));
}

inline bool llama_vbr_codec_contains(llama_vbr_codec codec, ggml_type type) {
    const auto & ladder = llama_vbr_ladder(codec);
    for (size_t i = 0; i < ladder.n_rungs; ++i) {
        if (ladder.rungs[i] == type) {
            return true;
        }
    }
    return false;
}

inline double llama_vbr_type_bits_per_value(ggml_type type) {
    return 8.0 * ggml_type_size(type) / ggml_blck_size(type);
}

inline ggml_type llama_vbr_floor_price_type(llama_vbr_codec codec, double floor_bpv) {
    const auto & ladder = llama_vbr_ladder(codec);
    if (floor_bpv > 0.0) {
        for (size_t i = 0; i < ladder.n_rungs; ++i) {
            const ggml_type type = ladder.rungs[i];
            if (llama_vbr_type_bits_per_value(type) <= floor_bpv + 1e-9) {
                return type;
            }
        }
    }
    return ladder.default_floor;
}

inline bool llama_vbr_codec_full_domain(llama_vbr_codec codec, ggml_type type) {
    if (codec == LLAMA_VBR_CODEC_CLASSIC) {
        return llama_vbr_codec_contains(codec, type);
    }
    return type == GGML_TYPE_F16 || type == GGML_TYPE_TURBO8_0;
}
