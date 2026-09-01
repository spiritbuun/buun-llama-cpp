#include "../tools/llama-bench/llama-bench-vbr.h"

#undef NDEBUG
#include <cassert>

int main() {
    const common_vbr_cache_choice unset = { GGML_TYPE_F16, false, false };
    const common_vbr_cache_choice pinned_q8 = { GGML_TYPE_Q8_0, false, true };
    const common_vbr_cache_choice pinned_f16 = { GGML_TYPE_F16, false, true };
    const common_vbr_cache_choice alias = { GGML_TYPE_F16, true, true };

    std::vector<common_vbr_cache_choice> ks = { alias, pinned_f16 };
    std::vector<common_vbr_cache_choice> vs = { unset };
    auto plan = llama_bench_vbr_make_plan(ks, vs, "t8", "t4", "1G", true, true, true);
    auto row = llama_bench_vbr_resolve_row(plan, alias, unset);
    assert(row.active && row.k && row.v);
    assert(row.type_k == GGML_TYPE_TURBO8_0 && row.type_v == GGML_TYPE_TURBO8_0);
    assert(row.floor_bits == 4.125 && row.floor_explicit);
    assert(row.budget_bytes == (1ull << 30) && row.budget_explicit);
    llama_context_params cparams = llama_context_default_params();
    llama_bench_vbr_apply_row(row, cparams);
    assert(cparams.vbr_dynamic && !cparams.vbr_pin_k && !cparams.vbr_pin_v);

    // The concrete comparison row stays non-VBR despite global knobs configuring the alias row.
    row = llama_bench_vbr_resolve_row(plan, pinned_f16, unset);
    assert(!row.active && row.type_k == GGML_TYPE_F16 && row.type_v == GGML_TYPE_F16);
    assert(row.entry == "none");

    ks = { alias };
    vs = { pinned_q8 };
    plan = llama_bench_vbr_make_plan(ks, vs, "t4", "8", "auto", true, true, true);
    row = llama_bench_vbr_resolve_row(plan, alias, pinned_q8);
    assert(row.active && row.k && !row.v);
    assert(row.type_k == GGML_TYPE_TURBO4_0 && row.type_v == GGML_TYPE_Q8_0);
    cparams = llama_context_default_params();
    llama_bench_vbr_apply_row(row, cparams);
    assert(cparams.vbr_pin_v && !cparams.vbr_pin_k);
    assert(cparams.vbr_min_bits == 8.0 && cparams.vbr_min_bits_explicit);
    assert(cparams.vbr_budget_explicit && cparams.vbr_vram_budget_bytes == 0);

    // Option-only plans (no `vbr` alias in the matrix) arm untouched sides while preserving
    // concrete pins. This is the primary `--vbr-entry/--vbr-floor/--vbr-vram` CLI path.
    ks = { unset };
    vs = { unset };
    plan = llama_bench_vbr_make_plan(ks, vs, "t8", "auto", "auto", true, false, false);
    row = llama_bench_vbr_resolve_row(plan, unset, unset);
    assert(row.active && row.k && row.v);
    assert(row.type_k == GGML_TYPE_TURBO8_0 && row.type_v == GGML_TYPE_TURBO8_0);
    assert(row.floor_bits == 4.125 && !row.floor_explicit);
    cparams = llama_context_default_params();
    llama_bench_vbr_apply_row(row, cparams);
    assert(cparams.vbr_dynamic && !cparams.vbr_pin_k && !cparams.vbr_pin_v);
    assert(cparams.type_k == GGML_TYPE_TURBO8_0 && cparams.type_v == GGML_TYPE_TURBO8_0);

    ks = { pinned_q8 };
    vs = { unset };
    plan = llama_bench_vbr_make_plan(ks, vs, "t4", "auto", "auto", true, false, false);
    row = llama_bench_vbr_resolve_row(plan, pinned_q8, unset);
    assert(row.active && !row.k && row.v);
    assert(row.type_k == GGML_TYPE_Q8_0 && row.type_v == GGML_TYPE_TURBO4_0);
    cparams = llama_context_default_params();
    llama_bench_vbr_apply_row(row, cparams);
    assert(cparams.vbr_pin_k && !cparams.vbr_pin_v);

    ks = { pinned_q8 };
    vs = { pinned_f16 };
    plan = llama_bench_vbr_make_plan(ks, vs, "t4", "auto", "auto", true, false, false);
    bool no_free_side_rejected = false;
    try {
        llama_bench_vbr_resolve_row(plan, pinned_q8, pinned_f16);
    } catch (const std::invalid_argument &) {
        no_free_side_rejected = true;
    }
    assert(no_free_side_rejected);

    ks = { unset };
    vs = { unset };
    plan = llama_bench_vbr_make_plan(ks, vs, "t2", "auto", "auto", true, false, false);
    row = llama_bench_vbr_resolve_row(plan, unset, unset);
    const double t2_bits = 8.0 * ggml_type_size(GGML_TYPE_TURBO2_TCQ) /
                           ggml_blck_size(GGML_TYPE_TURBO2_TCQ);
    assert(row.active && row.k && row.v);
    assert(row.type_k == GGML_TYPE_TURBO2_TCQ && row.type_v == GGML_TYPE_TURBO2_TCQ);
    assert(row.floor_bits == t2_bits && !row.floor_explicit);
    cparams = llama_context_default_params();
    llama_bench_vbr_apply_row(row, cparams);
    assert(cparams.vbr_dynamic && !cparams.vbr_pin_k && !cparams.vbr_pin_v);
    assert(cparams.type_k == GGML_TYPE_TURBO2_TCQ && cparams.type_v == GGML_TYPE_TURBO2_TCQ);
    assert(cparams.vbr_min_bits == t2_bits && !cparams.vbr_min_bits_explicit);

    return 0;
}
