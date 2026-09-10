#include "arg.h"
#include "common.h"
#include "llama-batch.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-tree.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

static llama_context_ptr make_ctx(const common_params & params, llama_model * model, uint32_t n_seq_max = 1) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max = n_seq_max;
    cparams.n_rs_seq  = 8;
    cparams.n_batch   = std::max(cparams.n_batch,  n_seq_max * (cparams.n_rs_seq + 1));
    cparams.n_ubatch  = std::max(cparams.n_ubatch, n_seq_max * (cparams.n_rs_seq + 1));
    return llama_context_ptr(llama_init_from_model(model, cparams));
}

static llama_memory_recurrent * get_recurrent(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * recurrent = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return recurrent;
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_recr();
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_recr();
    }
    return nullptr;
}

static llama_pos get_attention_pos_max(llama_context * ctx, llama_seq_id seq_id) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return hybrid->get_mem_attn()->seq_pos_max(seq_id);
    }
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) {
        return hybrid->get_mem_attn()->seq_pos_max(seq_id);
    }
    return -1;
}

static bool check_depth(llama_context * ctx, llama_seq_id seq_id, uint32_t expected, const char * label) {
    const auto * recurrent = get_recurrent(ctx);
    if (recurrent == nullptr || seq_id < 0 || (size_t) seq_id >= recurrent->rollback_valid_depth.size()) {
        fprintf(stderr, "%s : cannot read rollback depth for sequence %d\n", label, seq_id);
        return false;
    }
    const uint32_t actual = recurrent->rollback_valid_depth[seq_id];
    if (actual != expected) {
        fprintf(stderr, "%s : rollback depth mismatch for sequence %d (%u != %u)\n",
                label, seq_id, actual, expected);
        return false;
    }
    return true;
}

static bool decode_range(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         begin,
        uint32_t                         count,
        llama_seq_id                     seq_id = 0) {
    llama_batch batch = llama_batch_init(count, 0, 1);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t pos = begin + i;
        common_batch_add(batch, tokens[pos], pos, { seq_id }, i + 1 == count);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool decode_equal_split(
        llama_context *                  ctx,
        const std::vector<llama_token> & tokens,
        uint32_t                         n_seq_tokens,
        uint32_t                         n_seqs) {
    llama_batch batch = llama_batch_init(n_seq_tokens * n_seqs, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t pos = 0; pos < n_seq_tokens; ++pos) {
            const uint32_t i = s * n_seq_tokens + pos;
            common_batch_add(batch, tokens[i], pos, { (llama_seq_id) s }, pos + 1 == n_seq_tokens);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::vector<uint8_t> save_seq(
        llama_context *       ctx,
        llama_seq_id          seq_id,
        llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_NONE) {
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, seq_id, flags));
    const size_t n = llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq_id, flags);
    if (n != state.size()) {
        state.clear();
    }
    return state;
}

static bool load_seq(llama_context * ctx, const std::vector<uint8_t> & state, llama_seq_id seq_id) {
    return !state.empty() && llama_state_seq_set_data(ctx, state.data(), state.size(), seq_id) == state.size();
}

static bool seq_state_payload_equal(
        const std::vector<uint8_t> & lhs,
        const std::vector<uint8_t> & rhs) {
    // The in-memory sequence envelope starts with magic + source seq_id. The memory
    // payload that follows must be identical when comparing two different sequence ids.
    constexpr size_t envelope_size = sizeof(uint32_t) + sizeof(llama_seq_id);
    return lhs.size() == rhs.size() && lhs.size() >= envelope_size &&
        std::equal(lhs.begin() + envelope_size, lhs.end(), rhs.begin() + envelope_size);
}

static std::vector<float> copy_logits(llama_context * ctx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, 0);
    return logits == nullptr ? std::vector<float>() : std::vector<float>(logits, logits + n_vocab);
}

static bool logits_equal(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const char *               label) {
    constexpr float eps = 1e-5f;
    if (lhs.size() != rhs.size() || lhs.empty()) {
        fprintf(stderr, "%s : missing or differently sized logits\n", label);
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > eps) {
            fprintf(stderr, "%s : logits mismatch at token %zu (%g != %g)\n",
                    label, i, (double) lhs[i], (double) rhs[i]);
            return false;
        }
    }
    return true;
}

static bool abort_decode(void *) {
    return true;
}

static void set_resize_test_fault(bool enabled) {
#ifdef _WIN32
    _putenv_s("LLAMA_RECURRENT_RESIZE_TEST_FAIL", enabled ? "before_publish" : "");
#else
    if (enabled) {
        setenv("LLAMA_RECURRENT_RESIZE_TEST_FAIL", "before_publish", 1);
    } else {
        unsetenv("LLAMA_RECURRENT_RESIZE_TEST_FAIL");
    }
#endif
}

static bool get_recurrent_epoch(llama_memory_recurrent * recurrent, uint64_t & epoch) {
    auto memory_context = recurrent->init_full();
    auto * recurrent_context = dynamic_cast<llama_memory_recurrent_context *>(memory_context.get());
    if (recurrent_context == nullptr) {
        return false;
    }
    epoch = recurrent_context->get_tensor_binding_epoch();
    return true;
}

static bool test_multi_seq_split_replay(const common_params & params, llama_model * model, const int n_vocab) {
    constexpr uint32_t  n_seqs     = 2;
    // Recurrent split keeps the rollback tail together; the ubatch must be
    // larger than n_rs_seq while remaining smaller than n_replay so this still
    // exercises multiple ubatches per sequence.
    constexpr uint32_t  n_ubatch   = 16;
    constexpr uint32_t  n_prompt   = 19;
    constexpr uint32_t  n_rollback = 3;
    constexpr uint32_t  n_replay   = 40; // > n_ubatch so each seq spans multiple ubatches
    constexpr llama_pos p0         = n_prompt - n_rollback;

    const auto make_ctx_multi = [&]() {
        auto cparams = common_context_params_to_llama(params);
        cparams.n_seq_max  = n_seqs;
        cparams.n_rs_seq   = 8;
        cparams.n_ctx      = 256;
        cparams.n_batch    = 256;
        cparams.n_ubatch   = n_ubatch;
        cparams.kv_unified = false;
        return llama_context_ptr(llama_init_from_model(model, cparams));
    };

    auto ctx_roll = make_ctx_multi();
    auto ctx_ref  = make_ctx_multi();
    if (!ctx_roll || !ctx_ref) {
        fprintf(stderr, "%s : failed to init multi-seq contexts\n", __func__);
        return false;
    }

    if (llama_n_rs_seq(ctx_roll.get()) < n_rollback) {
        fprintf(stderr, "%s : n_rs_seq is too small for split replay\n", __func__);
        return false;
    }

    const auto tok = [&](uint32_t seq, llama_pos pos) {
        return (llama_token) ((7*(uint32_t) pos + 31*seq + 1) % (uint32_t) n_vocab);
    };

    bool ok = true;

    // both contexts decode the identical [0, p0) prefill; only ctx_roll decodes
    // the tail, which is then rolled back so its restore is pending at replay
    for (uint32_t s = 0; s < n_seqs && ok; ++s) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        // The rollback plane retains one prior state per transition within a
        // decode batch. Decode the rollback context's boundary token together
        // with its tail so a three-token rollback has three authenticated
        // snapshots; the reference consumes that boundary token separately.
        for (llama_pos pos = 0; pos < p0 - 1; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        ok = ok && llama_decode(ctx_ref.get(),  batch) == 0;

        common_batch_clear(batch);
        common_batch_add(batch, tok(s, p0 - 1), p0 - 1, { (llama_seq_id) s }, false);
        ok = ok && llama_decode(ctx_ref.get(), batch) == 0;

        common_batch_clear(batch);
        for (llama_pos pos = p0 - 1; pos < (llama_pos) n_prompt; ++pos) {
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, false);
        }
        ok = ok && llama_decode(ctx_roll.get(), batch) == 0;
        llama_batch_free(batch);

        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0, -1);

        // a second partial removal while one is pending must be refused
        ok = ok && !llama_memory_seq_rm(llama_get_memory(ctx_roll.get()), (llama_seq_id) s, p0 - 1, -1);
    }
    if (!ok) {
        fprintf(stderr, "%s : multi-seq prefill/rollback failed\n", __func__);
        return false;
    }

    llama_batch batch = llama_batch_init(n_seqs*n_replay, 0, 1);
    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t i = 0; i < n_replay; ++i) {
            const llama_pos pos = p0 + (llama_pos) i;
            common_batch_add(batch, tok(s, pos), pos, { (llama_seq_id) s }, true);
        }
    }
    ok = llama_decode(ctx_roll.get(), batch) == 0;
    ok = ok && llama_decode(ctx_ref.get(), batch) == 0;
    llama_batch_free(batch);
    if (!ok) {
        fprintf(stderr, "%s : multi-seq replay decode failed\n", __func__);
        return false;
    }

    // identical ubatch shapes from bit-exact states: a correct implementation
    // matches bitwise, so eps only allows backend scheduling noise
    constexpr float eps = 1e-7f;

    float    diff_max  = 0.0f;
    uint32_t seq_first = 0;
    int32_t  pos_first = -1;
    for (uint32_t i = 0; i < n_seqs*n_replay; ++i) {
        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), i);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  i);
        if (l_roll == nullptr || l_ref == nullptr) {
            fprintf(stderr, "%s : missing multi-seq logits at index %u\n", __func__, i);
            return false;
        }
        for (int t = 0; t < n_vocab; ++t) {
            const float diff = std::fabs(l_roll[t] - l_ref[t]);
            if (diff > eps && pos_first < 0) {
                seq_first = i/n_replay;
                pos_first = p0 + (int32_t) (i%n_replay);
            }
            diff_max = std::max(diff_max, diff);
        }
    }

    if (diff_max > eps) {
        fprintf(stderr, "%s : multi-seq split replay logits mismatch (max diff %g, first at seq %u pos %d)\n",
                __func__, (double) diff_max, seq_first, pos_first);
        return false;
    }

    fprintf(stderr, "%s : multi-seq split replay matched (max diff %g)\n", __func__, (double) diff_max);

    // seq-1-only decodes must be independent of seq 0's content: diverge seq 0
    // in ctx_ref only, then compare identical seq-1-only continuations bitwise
    constexpr uint32_t n_tail = 4;

    {
        llama_batch batch_tail = llama_batch_init(n_tail, 0, 1);
        for (uint32_t i = 0; i < n_tail; ++i) {
            const llama_pos pos = p0 + (llama_pos) (n_replay + i);
            common_batch_add(batch_tail, tok(0, pos + 7), pos, { 0 }, false);
        }
        ok = llama_decode(ctx_ref.get(), batch_tail) == 0;
        llama_batch_free(batch_tail);
    }

    float diff_tail = 0.0f;
    for (uint32_t i = 0; i < n_tail && ok; ++i) {
        const llama_pos pos = p0 + (llama_pos) (n_replay + i);
        llama_batch batch_one = llama_batch_init(1, 0, 1);
        common_batch_add(batch_one, tok(1, pos), pos, { 1 }, true);
        ok = llama_decode(ctx_roll.get(), batch_one) == 0;
        ok = ok && llama_decode(ctx_ref.get(), batch_one) == 0;
        llama_batch_free(batch_one);
        if (!ok) {
            break;
        }

        const float * l_roll = llama_get_logits_ith(ctx_roll.get(), 0);
        const float * l_ref  = llama_get_logits_ith(ctx_ref.get(),  0);
        ok = l_roll != nullptr && l_ref != nullptr;
        for (int t = 0; ok && t < n_vocab; ++t) {
            diff_tail = std::max(diff_tail, std::fabs(l_roll[t] - l_ref[t]));
        }
    }

    if (!ok || diff_tail > eps) {
        fprintf(stderr, "%s : seq-1-only decode leaked seq 0 state (ok=%d, max diff %g)\n",
                __func__, ok ? 1 : 0, (double) diff_tail);
        return false;
    }

    fprintf(stderr, "%s : seq-1-only decode independent of seq 0 (max diff %g)\n", __func__, (double) diff_tail);
    return true;
}

static bool test_indexed_hybrid_tree_collection(const llama_model & model) {
    const llama_memory_i::layer_filter_cb reject_all = [](int32_t) { return false; };

    // No Qwen4 fixture is needed to pin the topology decision: empty layer
    // filters construct only the composite type, without allocating model
    // payload tensors. The derived-type branch must run before the generic
    // llama_memory_hybrid branch or the QSA owner would be omitted.
    llama_memory_hybrid_idx indexed(
        model,
        GGML_TYPE_F16, GGML_TYPE_F16, false, 8, 1, 0, LLAMA_SWA_TYPE_NONE,
        GGML_TYPE_F32, GGML_TYPE_F32, 1,
        1, 1, false, false,
        reject_all, reject_all, reject_all);

    std::vector<llama_memory_tree_child> tree;
    if (!llama_memory_tree_collect(&indexed, tree) || tree.size() != 2 ||
        tree[0].child_id != 0 || tree[0].attention != indexed.get_mem_attn() ||
        tree[0].recurrent != nullptr || tree[0].qsa_index_owner != &indexed ||
        tree[0].dependency_mode != checkpoint_child_dependency_mode::live_guarded ||
        tree[1].child_id != 1 || tree[1].attention != nullptr ||
        tree[1].recurrent != indexed.get_mem_recr() || tree[1].qsa_index_owner != nullptr ||
        tree[1].dependency_mode != checkpoint_child_dependency_mode::absent) {
        fprintf(stderr, "%s : indexed hybrid topology was incomplete or misclassified\n", __func__);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    set_resize_test_fault(false);

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    std::vector<std::string> parser_args(argv, argv + argc);
    parser_args.push_back("-ct");
    parser_args.push_back("f16");
    std::vector<char *> parser_argv;
    parser_argv.reserve(parser_args.size());
    for (auto & arg : parser_args) {
        parser_argv.push_back(&arg[0]);
    }
    if (!common_params_parse(
            int(parser_argv.size()), parser_argv.data(), params,
            LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // This rollback test is cache-representation agnostic and intentionally
    // pins the static CPU-compatible cache rather than inheriting CLI defaults.
    GGML_ASSERT(params.cache_type_k == GGML_TYPE_F16);
    GGML_ASSERT(params.cache_type_v == GGML_TYPE_F16);
    GGML_ASSERT(!params.vbr_enabled());

    // The production MTP/VBR server uses unified KV. Its single physical stream
    // must remain distinct from the logical multi-sequence graph capacity.
    params.kv_unified = true;

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!test_indexed_hybrid_tree_collection(*model)) {
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> tokens(32);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = (llama_token) ((i + 1) % std::max(n_vocab, 1));
    }

    auto ctx_src      = make_ctx(params, model);
    auto ctx_test     = make_ctx(params, model);
    auto ctx_ref      = make_ctx(params, model);
    auto ctx_parallel = make_ctx(params, model, 3);
    if (!ctx_src || !ctx_test || !ctx_ref || !ctx_parallel) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        return 1;
    }

    // A unified attention cache owns one physical stream while remaining logically
    // capable of the configured sequence count. Composite lowering may narrow that
    // logical cap without changing the physical stream layout.
    llama_memory_t parallel_mem = llama_get_memory(ctx_parallel.get());
    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(parallel_mem)) {
        llama_kv_cache_context unrestricted(hybrid->get_mem_attn());
        llama_kv_cache_context limited(hybrid->get_mem_attn(), 2);
        if (unrestricted.get_max_graph_seqs() != std::numeric_limits<uint32_t>::max() ||
            limited.get_max_graph_seqs() != 2) {
            fprintf(stderr, "%s : unified attention logical capacity was confused with its physical stream\n", __func__);
            return 1;
        }
    } else if (auto * hybrid = dynamic_cast<llama_memory_hybrid_iswa *>(parallel_mem)) {
        llama_kv_cache_iswa_context unrestricted(hybrid->get_mem_attn());
        llama_kv_cache_iswa_context limited(hybrid->get_mem_attn(), 2);
        if (unrestricted.get_max_graph_seqs() != std::numeric_limits<uint32_t>::max() ||
            limited.get_max_graph_seqs() != 2) {
            fprintf(stderr, "%s : unified iSWA attention logical capacity was confused with its physical stream\n", __func__);
            return 1;
        }
    }

    auto * recurrent = get_recurrent(ctx_test.get());
    if (recurrent == nullptr || recurrent->n_rs_seq < 3) {
        fprintf(stderr, "%s : skipping because recurrent rollback depth is less than 3\n", __func__);
        return 0;
    }
    const uint32_t n_rs_seq = recurrent->n_rs_seq;

    // Starting a graph invalidates the planes that graph is about to overwrite, but a
    // rollback selected by the preceding seq_rm() is an input to that graph. Keep its
    // selector alive until s_copy() consumes it while constructing the graph.
    {
        llama_seq_id seq_id = 0;
        llama_seq_id * seq_ids[] = { &seq_id };
        int32_t n_seq_id[] = { 1 };
        llama_ubatch ubatch = {};
        ubatch.b_equal_seqs = 1;
        ubatch.n_tokens = 1;
        ubatch.n_seq_tokens = 1;
        ubatch.n_seqs = 1;
        ubatch.n_seq_id = n_seq_id;
        ubatch.seq_id = seq_ids;

        recurrent->set_rs_idx(0, 2);
        recurrent->rollback_valid_depth[0] = 2;
        recurrent->invalidate_rollback(ubatch);
        if (recurrent->rs_idx[0] != 2 || recurrent->rollback_valid_depth[0] != 0) {
            fprintf(stderr, "%s : graph invalidation discarded a pending rollback selector\n", __func__);
            return 1;
        }
        recurrent->reset_rollback_state(0);
    }

    // Preserve the original regression: a selected rollback plane must serialize as the
    // logical active row and replay identically after restore.
    if (!decode_range(ctx_src.get(), tokens, 0, 4) ||
        !llama_memory_seq_rm(llama_get_memory(ctx_src.get()), 0, 3, -1)) {
        fprintf(stderr, "%s : rolled-back checkpoint setup failed\n", __func__);
        return 1;
    }
    const auto rolled_back_state = save_seq(ctx_src.get(), 0);
    if (!load_seq(ctx_test.get(), rolled_back_state, 0) ||
        !decode_range(ctx_src.get(),  tokens, 3, 1) ||
        !decode_range(ctx_test.get(), tokens, 3, 1) ||
        !logits_equal(copy_logits(ctx_src.get(), n_vocab), copy_logits(ctx_test.get(), n_vocab),
                "rolled-back checkpoint restore")) {
        fprintf(stderr, "%s : rolled-back checkpoint did not replay identically\n", __func__);
        return 1;
    }
    llama_memory_clear(llama_get_memory(ctx_src.get()),  true);
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);

    // A restore installs only the active row. A following one-token decode writes group 0,
    // so rollback one must fail without changing serialized state or the current logits.
    if (!decode_range(ctx_src.get(), tokens, 0, 3)) {
        fprintf(stderr, "%s : source prefix decode failed\n", __func__);
        return 1;
    }
    const auto restored_state = save_seq(ctx_src.get(), 0);
    if (!load_seq(ctx_test.get(), restored_state, 0) ||
        !check_depth(ctx_test.get(), 0, 0, "restore") ||
        !decode_range(ctx_test.get(), tokens, 3, 1) ||
        !check_depth(ctx_test.get(), 0, 0, "restore then one-token decode")) {
        fprintf(stderr, "%s : restore setup failed\n", __func__);
        return 1;
    }

    const auto state_before_rm  = save_seq(ctx_test.get(), 0);
    const auto logits_before_rm = copy_logits(ctx_test.get(), n_vocab);
    const int32_t tail_before_rm = get_recurrent(ctx_test.get())->cells[0].tail;
    const uint32_t rs_before_rm  = get_recurrent(ctx_test.get())->rs_idx[0];
    if (!load_seq(ctx_ref.get(), state_before_rm, 0)) {
        fprintf(stderr, "%s : failed to retain pre-rm reference\n", __func__);
        return 1;
    }
    // Exercise the composite memory operation. For hybrid models, a rejected
    // recurrent rollback must not remove the corresponding attention entries.
    if (llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, 3, -1)) {
        fprintf(stderr, "%s : stale rollback unexpectedly succeeded after restore + decode\n", __func__);
        return 1;
    }
    const auto state_after_rm  = save_seq(ctx_test.get(), 0);
    const auto logits_after_rm = copy_logits(ctx_test.get(), n_vocab);
    const bool state_unchanged  = state_before_rm == state_after_rm;
    const bool logits_unchanged = logits_equal(logits_before_rm, logits_after_rm, "failed rollback");
    const llama_pos pos_after_rm = llama_memory_seq_pos_max(llama_get_memory(ctx_test.get()), 0);
    const auto * recurrent_after_rm = get_recurrent(ctx_test.get());
    const bool metadata_unchanged = recurrent_after_rm->cells[0].tail == tail_before_rm &&
        recurrent_after_rm->rs_idx[0] == rs_before_rm;
    if (!state_unchanged || !logits_unchanged || !metadata_unchanged || pos_after_rm != 3) {
        fprintf(stderr, "%s : failed-rm details: state=%s, logits=%s, metadata=%s, pos=%d\n",
                __func__, state_unchanged ? "same" : "changed",
                logits_unchanged ? "same" : "changed",
                metadata_unchanged ? "same" : "changed", pos_after_rm);
        fprintf(stderr, "%s : failed rollback mutated state, logits, or position\n", __func__);
        return 1;
    }
    if (!decode_range(ctx_test.get(), tokens, 4, 1) ||
        !decode_range(ctx_ref.get(),  tokens, 4, 1) ||
        !logits_equal(copy_logits(ctx_test.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-failed-rm continuation")) {
        fprintf(stderr, "%s : failed rollback changed continuation\n", __func__);
        return 1;
    }

    // A four-token verify writes the active plane plus rollback planes 1..3.
    // Each rollback is checked independently against a retained prefix reference.
    for (uint32_t rollback = 1; rollback <= 3; ++rollback) {
        llama_memory_clear(llama_get_memory(ctx_test.get()), true);
        llama_memory_clear(llama_get_memory(ctx_ref.get()),  true);

        if (!decode_range(ctx_test.get(), tokens, 0, 2) ||
            !decode_range(ctx_test.get(), tokens, 2, 4) ||
            !check_depth(ctx_test.get(), 0, 3, "four-token verify")) {
            fprintf(stderr, "%s : four-token verify setup failed for rollback %u\n", __func__, rollback);
            return 1;
        }

        const uint32_t rollback_pos = 6 - rollback;
        if (!decode_range(ctx_ref.get(), tokens, 0, rollback_pos) ||
            !llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, rollback_pos, -1) ||
            !decode_range(ctx_test.get(), tokens, rollback_pos, 1) ||
            !decode_range(ctx_ref.get(),  tokens, rollback_pos, 1) ||
            !logits_equal(copy_logits(ctx_test.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                    "bounded rollback reference")) {
            fprintf(stderr, "%s : rollback %u did not match retained reference\n", __func__, rollback);
            return 1;
        }
    }

    // A narrow decode replaces, rather than extends, the valid-depth assignment.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, 4) ||
        !check_depth(ctx_test.get(), 0, 3, "verify before narrow decode") ||
        !decode_range(ctx_test.get(), tokens, 4, 1) ||
        !check_depth(ctx_test.get(), 0, 0, "verify then narrow decode")) {
        fprintf(stderr, "%s : verify then narrow decode setup failed\n", __func__);
        return 1;
    }
    const auto narrow_state = save_seq(ctx_test.get(), 0);
    if (get_recurrent(ctx_test.get())->seq_rm(0, 4, -1) ||
        narrow_state != save_seq(ctx_test.get(), 0)) {
        fprintf(stderr, "%s : stale rollback succeeded or mutated after narrow decode\n", __func__);
        return 1;
    }

    // The assignment clamps at n_rs_seq, and full removal is never rejected by the guard.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, n_rs_seq + 1) ||
        !check_depth(ctx_test.get(), 0, n_rs_seq, "clamped verify") ||
        !llama_memory_seq_rm(llama_get_memory(ctx_test.get()), 0, -1, -1) ||
        !check_depth(ctx_test.get(), 0, 0, "full removal")) {
        fprintf(stderr, "%s : clamp or full-removal check failed\n", __func__);
        return 1;
    }

    // A successful detectable copy into an empty destination copies the composite state.
    // Give the empty destination synthetic rollback metadata to prove that the copied active
    // row resets it rather than inheriting the source's valid planes.
    if (!decode_range(ctx_parallel.get(), tokens, 0, 4, 1) ||
        !check_depth(ctx_parallel.get(), 1, 3, "seq_cp source")) {
        fprintf(stderr, "%s : successful seq_cp setup failed\n", __func__);
        return 1;
    }
    const auto source_state = save_seq(ctx_parallel.get(), 1, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    auto * recurrent_parallel = get_recurrent(ctx_parallel.get());
    recurrent_parallel->set_rs_idx(0, 3);
    recurrent_parallel->rollback_valid_depth[0] = 3;
    const bool copy_succeeded = llama_get_memory(ctx_parallel.get())->try_seq_cp(1, 0, -1, -1);
    const auto destination_state = save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if (!copy_succeeded ||
        !seq_state_payload_equal(source_state, destination_state) ||
        !check_depth(ctx_parallel.get(), 0, 0, "seq_cp destination") ||
        !check_depth(ctx_parallel.get(), 1, 3, "seq_cp source")) {
        size_t first_diff = 0;
        while (first_diff < source_state.size() &&
               first_diff < destination_state.size() &&
               source_state[first_diff] == destination_state[first_diff]) {
            ++first_diff;
        }
        fprintf(stderr, "%s : successful seq_cp details: result=%s, source=%zu B, destination=%zu B, first_diff=%zu\n",
                __func__, copy_succeeded ? "true" : "false",
                source_state.size(), destination_state.size(), first_diff);
        fprintf(stderr, "%s : successful seq_cp did not copy state or reset destination depth\n", __func__);
        return 1;
    }

    // Fill every recurrent cell, then exercise the recurrent implementation directly.
    // Reserve-before-clear must report exhaustion while preserving the occupied destination
    // byte-for-byte, including its rollback metadata.
    llama_memory_clear(llama_get_memory(ctx_parallel.get()), true);
    if (!decode_equal_split(ctx_parallel.get(), tokens, 4, 3)) {
        fprintf(stderr, "%s : seq_cp exhaustion setup failed\n", __func__);
        return 1;
    }
    recurrent_parallel = get_recurrent(ctx_parallel.get());
    const auto exhausted_dst_state = save_seq(ctx_parallel.get(), 0);
    const int32_t exhausted_dst_tail = recurrent_parallel->cells[0].tail;
    const llama_pos exhausted_dst_pos = recurrent_parallel->seq_pos_max(0);
    const uint32_t exhausted_dst_rs_idx = recurrent_parallel->rs_idx[0];
    const uint32_t exhausted_dst_depth = recurrent_parallel->rollback_valid_depth[0];
    const uint32_t exhausted_used = recurrent_parallel->used;
    if (recurrent_parallel->try_seq_cp(1, 0, -1, -1) ||
        exhausted_dst_state != save_seq(ctx_parallel.get(), 0) ||
        recurrent_parallel->cells[0].tail != exhausted_dst_tail ||
        recurrent_parallel->seq_pos_max(0) != exhausted_dst_pos ||
        recurrent_parallel->rs_idx[0] != exhausted_dst_rs_idx ||
        recurrent_parallel->rollback_valid_depth[0] != exhausted_dst_depth ||
        recurrent_parallel->used != exhausted_used) {
        fprintf(stderr, "%s : exhausted recurrent seq_cp succeeded or mutated destination\n", __func__);
        return 1;
    }
    if (!recurrent_parallel->seq_rm(0, -1, -1) ||
        !recurrent_parallel->try_seq_cp(1, 0, -1, -1) ||
        recurrent_parallel->cells[0].tail < 0 ||
        recurrent_parallel->used != exhausted_used) {
        fprintf(stderr, "%s : full-pool backup restore did not reuse the freed live cell\n", __func__);
        return 1;
    }

    // The hybrid fallback cannot roll back the attention copy, so detected failure
    // invalidates both children. The destination is empty and therefore never selectable
    // as a half-copied composite checkpoint.
    llama_memory_t mem_parallel = llama_get_memory(ctx_parallel.get());
    if (dynamic_cast<llama_memory_hybrid *>(mem_parallel) != nullptr ||
        dynamic_cast<llama_memory_hybrid_iswa *>(mem_parallel) != nullptr) {
        if (mem_parallel->try_seq_cp(1, 0, -1, -1) ||
            recurrent_parallel->get_cell_count(0) != 0 ||
            recurrent_parallel->seq_pos_max(0) != -1 ||
            get_attention_pos_max(ctx_parallel.get(), 0) != -1 ||
            llama_memory_seq_pos_max(mem_parallel, 0) != -1 ||
            !check_depth(ctx_parallel.get(), 0, 0, "failed hybrid seq_cp destination")) {
            fprintf(stderr, "%s : failed hybrid seq_cp left an incoherent destination\n", __func__);
            return 1;
        }
    }

    // Equal-split batching still publishes each participating sequence independently.
    llama_memory_clear(llama_get_memory(ctx_parallel.get()), true);
    if (!decode_equal_split(ctx_parallel.get(), tokens, 4, 2) ||
        !check_depth(ctx_parallel.get(), 0, 3, "equal split seq 0") ||
        !check_depth(ctx_parallel.get(), 1, 3, "equal split seq 1") ||
        !decode_range(ctx_parallel.get(), tokens, 4, 1, 0) ||
        !check_depth(ctx_parallel.get(), 0, 0, "narrow seq 0") ||
        !check_depth(ctx_parallel.get(), 1, 3, "untouched seq 1")) {
        fprintf(stderr, "%s : per-sequence equal-split assignment failed\n", __func__);
        return 1;
    }

    // A live logical sequence may occupy a high physical cell. Shrinking must
    // refuse atomically instead of truncating that recurrent state while leaving
    // the attention half reusable.
    llama_memory_clear(mem_parallel, true);
    llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
    if (!llama_memory_recurrent_expand(mem_parallel, 3) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 1, 1) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 1, 2) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 1, 0) ||
        !decode_range(ctx_ref.get(), tokens, 0, 1, 0) ||
        !llama_memory_seq_rm(mem_parallel, 1, -1, -1) ||
        !llama_memory_seq_rm(mem_parallel, 2, -1, -1)) {
        fprintf(stderr, "%s : high-cell shrink setup failed\n", __func__);
        return 1;
    }
    recurrent_parallel = get_recurrent(ctx_parallel.get());
    const int32_t high_tail = recurrent_parallel->cells[0].tail;
    const auto high_state_before = save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto high_r_before = recurrent_parallel->r_l;
    const auto high_s_before = recurrent_parallel->s_l;
    const auto high_p_before = recurrent_parallel->p_l;
    uint64_t high_epoch_before = 0;
    uint64_t high_epoch_after = 0;
    if (high_tail < 1 || high_state_before.empty() ||
        !get_recurrent_epoch(recurrent_parallel, high_epoch_before) ||
        llama_memory_recurrent_shrink(mem_parallel, 1) ||
        recurrent_parallel->size != 3 ||
        recurrent_parallel->cells[0].tail != high_tail ||
        recurrent_parallel->r_l != high_r_before ||
        recurrent_parallel->s_l != high_s_before ||
        recurrent_parallel->p_l != high_p_before ||
        save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != high_state_before ||
        !get_recurrent_epoch(recurrent_parallel, high_epoch_after) ||
        high_epoch_after != high_epoch_before ||
        !decode_range(ctx_parallel.get(), tokens, 1, 1, 0) ||
        !decode_range(ctx_ref.get(), tokens, 1, 1, 0) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-refused-high-cell-shrink continuation")) {
        fprintf(stderr, "%s : high-cell shrink was not failure-atomic\n", __func__);
        return 1;
    }

    // A resize failure after all replacement tensors and metadata have been staged must
    // leave the live cache completely untouched and usable. A following successful resize
    // must publish one new binding epoch and preserve the active recurrent row.
    llama_memory_clear(mem_parallel, true);
    llama_memory_clear(llama_get_memory(ctx_ref.get()), true);
    const uint32_t pre_zero_shrink_size = recurrent_parallel->size;
    if (llama_memory_recurrent_shrink(mem_parallel, 0) ||
        recurrent_parallel->size != pre_zero_shrink_size ||
        !llama_memory_recurrent_shrink(mem_parallel, 1) ||
        !decode_range(ctx_parallel.get(), tokens, 0, 4) ||
        !decode_range(ctx_ref.get(), tokens, 0, 4)) {
        fprintf(stderr, "%s : resize failure-atomic setup failed\n", __func__);
        return 1;
    }

    recurrent_parallel = get_recurrent(ctx_parallel.get());
    const auto contracted_mctx = mem_parallel->init_full();
    if (!contracted_mctx || contracted_mctx->get_max_graph_seqs() != 1 ||
        llama_graph_reserve(ctx_parallel.get(), 2, 2, 2) != nullptr ||
        llama_graph_reserve(ctx_parallel.get(), 1, 1, 1) == nullptr) {
        fprintf(stderr, "%s : contracted recurrent graph capacity was not enforced\n", __func__);
        return 1;
    }

    // Force the production sched_reserve() owner through a real topology setter while
    // recurrent storage is contracted. It must reserve the one-sequence graph rather
    // than rebuilding the old configured two-sequence shape.
    llama_set_causal_attn(ctx_parallel.get(), false);
    llama_set_causal_attn(ctx_parallel.get(), true);
    llama_set_causal_attn(ctx_ref.get(), false);
    llama_set_causal_attn(ctx_ref.get(), true);
    if (!decode_range(ctx_parallel.get(), tokens, 4, 1) ||
        !decode_range(ctx_ref.get(), tokens, 4, 1) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "contracted scheduler reserve continuation")) {
        fprintf(stderr, "%s : contracted production scheduler reserve failed\n", __func__);
        return 1;
    }

    const auto resize_state_before = save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto resize_r_before = recurrent_parallel->r_l;
    const auto resize_s_before = recurrent_parallel->s_l;
    const auto resize_p_before = recurrent_parallel->p_l;
    const auto resize_rs_idx_before = recurrent_parallel->rs_idx;
    const auto resize_depth_before = recurrent_parallel->rollback_valid_depth;
    const uint32_t resize_size_before = recurrent_parallel->size;
    const uint32_t resize_head_before = recurrent_parallel->head;
    const uint32_t resize_used_before = recurrent_parallel->used;
    const uint32_t resize_n_before = recurrent_parallel->n;
    const int32_t resize_rs_z_before = recurrent_parallel->rs_z;
    uint64_t resize_epoch_before = 0;
    if (resize_state_before.empty() || !get_recurrent_epoch(recurrent_parallel, resize_epoch_before)) {
        fprintf(stderr, "%s : failed to snapshot recurrent resize state\n", __func__);
        return 1;
    }

    set_resize_test_fault(true);
    const bool injected_resize_succeeded = llama_memory_recurrent_expand(mem_parallel, 2);
    set_resize_test_fault(false);

    uint64_t resize_epoch_after_failure = 0;
    if (injected_resize_succeeded ||
        recurrent_parallel->size != resize_size_before ||
        recurrent_parallel->head != resize_head_before ||
        recurrent_parallel->used != resize_used_before ||
        recurrent_parallel->n != resize_n_before ||
        recurrent_parallel->rs_z != resize_rs_z_before ||
        recurrent_parallel->r_l != resize_r_before ||
        recurrent_parallel->s_l != resize_s_before ||
        recurrent_parallel->p_l != resize_p_before ||
        recurrent_parallel->rs_idx != resize_rs_idx_before ||
        recurrent_parallel->rollback_valid_depth != resize_depth_before ||
        save_seq(ctx_parallel.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != resize_state_before ||
        !get_recurrent_epoch(recurrent_parallel, resize_epoch_after_failure) ||
        resize_epoch_after_failure != resize_epoch_before) {
        fprintf(stderr, "%s : failed recurrent resize mutated live state or bindings\n", __func__);
        return 1;
    }

    if (!decode_range(ctx_parallel.get(), tokens, 5, 1) ||
        !decode_range(ctx_ref.get(), tokens, 5, 1) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-failed-resize continuation")) {
        fprintf(stderr, "%s : failed recurrent resize broke continuation\n", __func__);
        return 1;
    }

    if (!llama_memory_recurrent_expand(mem_parallel, 2)) {
        fprintf(stderr, "%s : recurrent resize did not recover after injected failure\n", __func__);
        return 1;
    }
    const auto expanded_mctx = mem_parallel->init_full();
    if (!expanded_mctx || expanded_mctx->get_max_graph_seqs() != 2 ||
        llama_graph_reserve(ctx_parallel.get(), 2, 2, 2) == nullptr) {
        fprintf(stderr, "%s : unified hybrid graph capacity did not follow recurrent expansion\n", __func__);
        return 1;
    }
    uint64_t resize_epoch_after_success = 0;
    const uint64_t expected_resize_epoch = resize_epoch_before == UINT64_MAX ? 1 : resize_epoch_before + 1;
    bool tensors_rebound = false;
    for (size_t i = 0; i < resize_r_before.size(); ++i) {
        tensors_rebound = tensors_rebound ||
            (resize_r_before[i] != nullptr && resize_r_before[i] != recurrent_parallel->r_l[i]) ||
            (resize_s_before[i] != nullptr && resize_s_before[i] != recurrent_parallel->s_l[i]) ||
            (resize_p_before[i] != nullptr && resize_p_before[i] != recurrent_parallel->p_l[i]);
    }
    if (recurrent_parallel->size != 2 || !tensors_rebound ||
        !get_recurrent_epoch(recurrent_parallel, resize_epoch_after_success) ||
        resize_epoch_after_success != expected_resize_epoch ||
        !decode_range(ctx_parallel.get(), tokens, 6, 1) ||
        !decode_range(ctx_ref.get(), tokens, 6, 1) ||
        !logits_equal(copy_logits(ctx_parallel.get(), n_vocab), copy_logits(ctx_ref.get(), n_vocab),
                "post-successful-resize continuation")) {
        fprintf(stderr, "%s : successful recurrent resize did not publish one usable binding epoch\n", __func__);
        return 1;
    }

    // Invalidation happens before graph submission, so an aborted decode cannot leave a
    // positive depth that is not backed by a successfully committed write.
    llama_memory_clear(llama_get_memory(ctx_test.get()), true);
    if (!decode_range(ctx_test.get(), tokens, 0, 4) ||
        !check_depth(ctx_test.get(), 0, 3, "pre-abort verify")) {
        fprintf(stderr, "%s : abort setup failed\n", __func__);
        return 1;
    }
    llama_set_abort_callback(ctx_test.get(), abort_decode, nullptr);
    llama_batch abort_batch = llama_batch_init(1, 0, 1);
    common_batch_add(abort_batch, tokens[4], 4, { 0 }, true);
    const int32_t abort_result = llama_decode(ctx_test.get(), abort_batch);
    llama_batch_free(abort_batch);
    llama_set_abort_callback(ctx_test.get(), nullptr, nullptr);
    if (abort_result != 2 || !check_depth(ctx_test.get(), 0, 0, "aborted decode")) {
        fprintf(stderr, "%s : aborted decode returned %d or retained positive depth\n",
                __func__, abort_result);
        return 1;
    }

    if (!test_multi_seq_split_replay(params, model, n_vocab)) {
        return 1;
    }

    fprintf(stderr, "%s : recurrent rollback-plane validity checks passed\n", __func__);
    return 0;
}
