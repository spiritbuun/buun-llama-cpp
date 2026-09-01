#pragma once

#include "common.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

struct llama_bench_vbr_plan {
    bool options_selected = false;
    bool matrix_has_alias = false;
    ggml_type entry_type = GGML_TYPE_F16;
    std::string entry = "none";
    double explicit_floor_bits = 0.0;
    double implicit_option_floor_bits = 0.0;
    bool floor_explicit = false;
    uint64_t budget_bytes = 0;
    bool budget_explicit = false;
};

inline llama_bench_vbr_plan llama_bench_vbr_make_plan(
        const std::vector<common_vbr_cache_choice> & types_k,
        const std::vector<common_vbr_cache_choice> & types_v,
        const std::string & entry,
        const std::string & floor,
        const std::string & vram,
        bool entry_explicit,
        bool floor_explicit,
        bool vram_explicit) {
    llama_bench_vbr_plan plan;
    plan.options_selected = entry_explicit || floor_explicit || vram_explicit;
    plan.matrix_has_alias =
        std::any_of(types_k.begin(), types_k.end(), [](const auto & c) { return c.vbr; }) ||
        std::any_of(types_v.begin(), types_v.end(), [](const auto & c) { return c.vbr; });
    if (!plan.options_selected && !plan.matrix_has_alias) {
        return plan;
    }
    plan.entry_type = common_vbr_entry_type(entry);
    plan.entry = ggml_type_name(plan.entry_type);
    plan.floor_explicit = floor_explicit;
    plan.explicit_floor_bits = floor_explicit ? common_vbr_floor_bits(floor) : 0.0;
    plan.implicit_option_floor_bits = common_vbr_floor_bits("t4");
    plan.budget_bytes = common_vbr_vram_bytes(vram);
    plan.budget_explicit = vram_explicit;
    return plan;
}

struct llama_bench_vbr_row {
    bool active = false;
    bool k = false;
    bool v = false;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    std::string entry = "none";
    double floor_bits = 0.0;
    bool floor_explicit = false;
    uint64_t budget_bytes = 0;
    bool budget_explicit = false;
};

inline llama_bench_vbr_row llama_bench_vbr_resolve_row(
        const llama_bench_vbr_plan & plan,
        const common_vbr_cache_choice & tk,
        const common_vbr_cache_choice & tv) {
    llama_bench_vbr_row row;
    row.type_k = tk.type;
    row.type_v = tv.type;
    const bool alias_selected = tk.vbr || tv.vbr;
    const bool option_arms_row = plan.options_selected && !plan.matrix_has_alias;
    if (!alias_selected && !option_arms_row) {
        return row;
    }
    const auto sides = common_vbr_resolve_sides(tk, tv, plan.options_selected, plan.matrix_has_alias);
    row.k = sides.k;
    row.v = sides.v;
    if (!alias_selected && option_arms_row && !row.k && !row.v) {
        throw std::invalid_argument(
            "--vbr-* flags need a VBR cache side: use -ctk vbr / -ctv vbr, or drop the explicit non-vbr cache types");
    }
    row.active = row.k || row.v;
    row.entry = plan.entry;
    if (row.k) row.type_k = plan.entry_type;
    if (row.v) row.type_v = plan.entry_type;
    row.floor_bits = plan.floor_explicit ? plan.explicit_floor_bits
                                         : (alias_selected ? 0.0 : plan.implicit_option_floor_bits);
    row.floor_explicit = plan.floor_explicit;
    if (row.k && row.v) {
        const double entry_bits = 8.0 * ggml_type_size(plan.entry_type) / ggml_blck_size(plan.entry_type);
        if (row.floor_bits > entry_bits + 1e-9) {
            if (row.floor_explicit) {
                throw std::invalid_argument("--vbr-floor cannot exceed --vbr-entry when both cache sides are movable");
            }
            row.floor_bits = entry_bits;
        }
    }
    row.budget_bytes = plan.budget_bytes;
    row.budget_explicit = plan.budget_explicit;
    return row;
}

inline void llama_bench_vbr_apply_row(const llama_bench_vbr_row & row, llama_context_params & cparams) {
    cparams.type_k = row.type_k;
    cparams.type_v = row.type_v;
    if (!row.active) return;
    cparams.vbr_dynamic = true;
    cparams.vbr_pin_k = !row.k;
    cparams.vbr_pin_v = !row.v;
    cparams.vbr_min_bits = row.floor_bits;
    cparams.vbr_min_bits_explicit = row.floor_explicit;
    cparams.vbr_vram_budget_bytes = row.budget_bytes;
    cparams.vbr_budget_explicit = row.budget_explicit;
}
