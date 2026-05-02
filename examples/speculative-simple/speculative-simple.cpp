#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "pflash.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <random>
#include <string>
#include <vector>
#include <utility>

struct spec_checkpoint {
    int64_t n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }

    bool empty() const {
        return data.empty();
    }
};

// Rejection sampling verification for speculative decoding at temp > 0.
// Accepts draft token x with probability min(1, p_target(x) / q_draft(x)).
// On rejection, uses the target's sampled token instead.
// Returns accepted tokens (at least 1, at most draft.size()+1).
static std::vector<llama_token> speculative_reject_sample(
        struct common_sampler * smpl,
        struct llama_context  * ctx_tgt,
        const llama_tokens    & draft,
        const std::vector<float> & draft_log_probs,
        float                   temp,
        std::mt19937          & rng) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_tgt)));
    const float inv_temp = (temp > 0.0f) ? (1.0f / temp) : 1.0f;

    std::vector<llama_token> result;
    result.reserve(draft.size() + 1);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    for (size_t i = 0; i < draft.size(); i++) {
        // sample from target at position i (through the full sampler chain)
        const llama_token target_token = common_sampler_sample(smpl, ctx_tgt, (int)i);

        if (target_token == draft[i]) {
            // exact match — always accept
            common_sampler_accept(smpl, target_token, true);
            result.push_back(target_token);
            continue;
        }

        // extension tokens (beyond draft_log_probs): exact match only
        if (i >= draft_log_probs.size()) {
            common_sampler_accept(smpl, target_token, true);
            result.push_back(target_token);
            break;
        }

        // compute target's log-probability for the draft token
        float * target_logits = llama_get_logits_ith(ctx_tgt, (int)i);
        if (!target_logits) {
            // no target logits — fall back to exact match (reject)
            common_sampler_accept(smpl, target_token, true);
            result.push_back(target_token);
            break;
        }

        // compute log p_target(draft[i]) via single-pass online softmax
        float lse_max = -INFINITY;
        float lse_sum = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            float scaled = target_logits[v] * inv_temp;
            if (scaled > lse_max) {
                lse_sum = lse_sum * expf(lse_max - scaled) + 1.0f;
                lse_max = scaled;
            } else {
                lse_sum += expf(scaled - lse_max);
            }
        }
        float p_log = target_logits[draft[i]] * inv_temp - lse_max - logf(lse_sum);

        // acceptance probability = min(1, p_target / q_draft) = min(1, exp(p_log - q_log))
        float q_log = draft_log_probs[i];
        float accept_prob = expf(p_log - q_log);
        if (accept_prob > 1.0f) accept_prob = 1.0f;

        if (uniform(rng) < accept_prob) {
            // accept draft token
            common_sampler_accept(smpl, draft[i], true);
            result.push_back(draft[i]);
        } else {
            // reject — use target's sample
            common_sampler_accept(smpl, target_token, true);
            result.push_back(target_token);
            break;
        }
    }

    // if all draft tokens accepted, sample one more from target
    if (result.size() == draft.size()) {
        const llama_token bonus = common_sampler_sample(smpl, ctx_tgt, (int)draft.size());
        common_sampler_accept(smpl, bonus, true);
        result.push_back(bonus);
    }

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.mparams_dft.path.empty() &&
            params.speculative.draft.mparams.path.empty() &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_MOD &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_NGRAM_CACHE &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_SUFFIX &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_COPYSPEC &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_RECYCLE) {
        LOG_ERR("%s: --model-draft is required (unless using a model-free --spec-type)\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // speculative decoding with recurrent/hybrid models needs seq_backup=1 for state rollback
    {
        if (params.n_parallel < 2) {
            params.n_parallel = 2;
        }
    }

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    // check if the context supports partial sequence removal
    const auto ctx_seq_rm = common_context_can_seq_rm(ctx_tgt);
    const bool use_ckpt = (ctx_seq_rm == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (skip for model-free spec types)
    llama_model_ptr model_dft;

    if (!params.speculative.mparams_dft.path.empty() || !params.speculative.draft.mparams.path.empty()) {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;

        params_dft.n_parallel   = 1;
        // DFlash drafter uses sliding window (ctx_window=512) at runtime.
        // n_ctx only affects graph reservation size; 256 keeps compute buffer small.
        params_dft.n_ctx        = params_spec.n_ctx > 0 ? params_spec.n_ctx : 256;
        // drafter only processes block_size tokens per call — keep batch small to save VRAM
        params_dft.n_batch      = std::min((int32_t)64, params_dft.n_ctx);
        params_dft.n_ubatch     = params_dft.n_batch;
        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.draft.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        params.speculative.model_dft = model_dft.get();
        params.speculative.cparams_dft = common_context_params_to_llama(params_dft);
        params.speculative.draft.model = model_dft.get();
        params.speculative.draft.cparams = common_context_params_to_llama(params_dft);

        // Auto-detect DFlash from model architecture
        if (llama_model_dflash_block_size(model_dft.get()) > 0 &&
            params.speculative.type != COMMON_SPECULATIVE_TYPE_DFLASH) {
            params.speculative.type = COMMON_SPECULATIVE_TYPE_DFLASH;
            LOG_INF("auto-detected DFlash drafter (block_size=%d)\n", llama_model_dflash_block_size(model_dft.get()));
        }

        // DFlash: share tok_embd/output from target BEFORE creating drafter context
        if (params.speculative.type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            llama_model_share_tensors(model_dft.get(), model_tgt);
        }
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (!params.speculative.pflash_scorer_path.empty() &&
        (int)inp.size() >= params.speculative.pflash_min_tokens) {
        auto pcfg = pflash_config::from_params(params.speculative);
        const int orig_len = (int)inp.size();
        inp = pflash_compress(inp, pcfg);
        LOG_INF("pflash: %d -> %d tokens (%.1f%% kept)\n",
            orig_len, (int)inp.size(), 100.0f * inp.size() / orig_len);
    }

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // per-position rejection histogram: reject_pos[i] = how many times position i caused rejection
    // position 0 = first draft token, position N-1 = last. "all_accepted" counted separately.
    std::vector<int> reject_pos(16, 0);
    int n_all_accepted = 0;

    // used to determine end of generation
    bool has_eos = false;

    // hybrid models with recurrent layers need state management after partial rejection
    const bool needs_reeval = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    // backup sequence ID for saving recurrent state before speculative batches
    const llama_seq_id seq_backup = 1;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // init the speculator BEFORE prefill so DFlash can configure hidden state capture
    // enable Gumbel sampling for DFlash drafter when target uses temp > 0
    if (params.sampling.temp > 0.0f && params.speculative.sample_temp == 0.0f) {
        params.speculative.sample_temp = params.sampling.temp;
    }
    const auto & params_spec = params.speculative;

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    // eval the prompt (with hidden state capture if DFlash is active)
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    common_speculative_begin(spec, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    size_t n_draft = 0;

    llama_tokens draft;
    spec_checkpoint spec_ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    // timing accumulators (microseconds)
    int64_t t_draft_total   = 0;  // suffix tree draft generation
    int64_t t_backup_total  = 0;  // copy_cell backup
    int64_t t_decode1_total = 0;  // main verification decode
    int64_t t_sample_total  = 0;  // sampling
    int64_t t_restore_total = 0;  // seq_rm + seq_cp restore
    int64_t t_decode2_total = 0;  // re-evaluation decode
    int64_t t_other_total   = 0;  // everything else (token output, bookkeeping)
    int     n_iters         = 0;
    int     n_reeval_tokens = 0;  // total tokens re-evaluated
    int     n_reeval_calls  = 0;  // number of re-eval decode calls
    int     n_reeval_skipped = 0; // iterations where all drafts accepted (no re-eval needed)

    const int tree_budget = params_spec.tree_budget;

    // rejection sampling for temp > 0: enable when drafter provides log-probs
    const float sample_temp = params.sampling.temp;
    const bool use_rejection_sampling = (sample_temp > 0.0f && params_spec.sample_temp > 0.0f);
    std::mt19937 reject_rng(params.sampling.seed != 0 ? params.sampling.seed : 42);

    std::vector<float> draft_log_probs;
    if (use_rejection_sampling) {
        draft_log_probs.reserve(params_spec.n_max);
        LOG_INF("rejection sampling enabled (temp=%.2f)\n", sample_temp);
    }


    while (true) {
        n_iters++;

        llama_tokens ids;
        common_sampler_ptr smpl_save;
        int n_draft_this_iter = 0;
        int main_path_len = 0;
        bool has_backup = false;
        bool accepted_on_main_path = true;
        common_speculative_tree tree;
        int commit_n = 0;
        std::vector<int32_t> linear_parents;

        if (tree_budget > 0) {
            // === DDTree path: single-pass tree-structured speculative decoding ===
            {
                common_time_meas tm(t_draft_total);
                tree = common_speculative_draft_tree(spec, params_spec, prompt_tgt, id_last, tree_budget);
            }

            common_batch_clear(batch_tgt);

            if (tree.n_nodes == 0) {
                // no tree nodes — single token decode (same as empty draft)
                common_batch_add(batch_tgt, id_last, n_past++, {0}, true);
                {
                    common_time_meas tm(t_decode1_total);
                    llama_decode(ctx_tgt, batch_tgt);
                }
                {
                    llama_tokens batch_tokens = { id_last };
                    common_speculative_update_logits(spec, ctx_tgt, batch_tokens, 1);
                }
                {
                    common_time_meas tm(t_sample_total);
                    llama_token t = common_sampler_sample(smpl.get(), ctx_tgt, 0);
                    common_sampler_accept(smpl.get(), t, true);
                    ids.push_back(t);
                }
            } else {
                // Single-pass tree verify: batch ALL tree nodes with tree mask + parent_ids
                // All tokens on seq_id=0 so ubatch allocator keeps them together for GDN kernel.
                // KV cleanup uses backup-restore path (branches share positions with main chain).
                common_batch_add(batch_tgt, id_last, n_past++, {0}, true);
                for (int i = 0; i < tree.n_nodes; ++i) {
                    common_batch_add(batch_tgt, tree.tokens[i], n_past - 1 + tree.depths[i], {0}, true);
                }

                n_draft_this_iter = tree.n_nodes;
                main_path_len = tree.main_path_len;

                if (needs_reeval) {
                    llama_tape_replay_sync(ctx_tgt); // ensure previous async replay is done
                    common_time_meas tm(t_backup_total);
                    auto * mem = llama_get_memory(ctx_tgt);
                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    llama_memory_seq_cp(mem, 0, seq_backup, -1, -1);
                    has_backup = true;
                }

                // Set tree mask (attention layers) + parent_ids (tree-aware SSM kernel)
                llama_set_tree_mask(ctx_tgt, tree.visibility.data(), tree.n_nodes + 1);
                llama_set_tree_parent_ids(ctx_tgt, tree.parents.data(), tree.n_nodes + 1);

                // Enable tape recording for conv state rollback
                if (needs_reeval) {
                    llama_set_tape_recording(ctx_tgt, true);
                }
                {
                    common_time_meas tm(t_decode1_total);
                    llama_decode(ctx_tgt, batch_tgt);
                }
                if (needs_reeval) {
                    llama_set_tape_recording(ctx_tgt, false);
                }

                llama_clear_tree_mask(ctx_tgt);
                // Note: tree_parent_ids stays active until tree_rollback clears it

                LOG_DBG("[iter %d] tree decode: n_nodes=%d, n_past=%d\n", n_iters, tree.n_nodes, n_past);

                // Tree walk: all logits available from single pass
                {
                    int current = 0;

                    while (true) {
                        llama_token target_token;
                        {
                            common_time_meas tm_s(t_sample_total);
                            target_token = common_sampler_sample(smpl.get(), ctx_tgt, current);
                            common_sampler_accept(smpl.get(), target_token, true);
                            ids.push_back(target_token);
                        }

                        auto it = tree.child_maps[current].find(target_token);
                        if (it == tree.child_maps[current].end()) {
                            break; // mismatch = bonus token
                        }

                        int next = it->second;
                        current = next;
                        commit_n = next;

                        if (next > tree.main_path_len) {
                            accepted_on_main_path = false;
                        }
                    }
                }

                // Update drafter hidden states with accepted path tokens
                {
                    llama_tokens accepted_tokens;
                    accepted_tokens.push_back(id_last);
                    // Walk accepted path to collect tokens
                    int node = commit_n;
                    std::vector<int> path;
                    while (node > 0) {
                        path.push_back(node);
                        node = tree.parents[node];
                    }
                    std::reverse(path.begin(), path.end());
                    for (int idx : path) {
                        accepted_tokens.push_back(tree.tokens[idx - 1]);
                    }
                    common_speculative_update_logits(spec, ctx_tgt, accepted_tokens, (int)ids.size());
                }
            }
        } else {
            // === Flat path: linear speculative decoding ===
            if (draft.empty()) {
                draft_log_probs.clear();
                {
                    common_time_meas tm(t_draft_total);
                    draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last,
                        use_rejection_sampling ? &draft_log_probs : nullptr);
                }

                // save the original draft size
                n_draft = draft.size();

                // save a checkpoint of the target context before evaluating the draft
                if (!draft.empty() && use_ckpt) {
                    const size_t ckpt_size = llama_state_seq_get_size_ext(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    spec_ckpt.data.resize(ckpt_size);

                    const size_t n = llama_state_seq_get_data_ext(ctx_tgt, spec_ckpt.data.data(), ckpt_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    GGML_ASSERT(n == ckpt_size);

                    spec_ckpt.n_tokens = (int64_t) prompt_tgt.size();
                    LOG_DBG("created speculative checkpoint (n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                            spec_ckpt.n_tokens, (float) spec_ckpt.data.size() / 1024 / 1024);
                }
            } else {
                // we have a previous (partial) draft to reuse from checkpoint restoration
                if (use_ckpt) {
                    GGML_ASSERT(!spec_ckpt.empty());
                }
            }

            common_batch_clear(batch_tgt);
            common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

            {
                if (draft.size() < (size_t) params_spec.n_min) {
                    draft.clear();
                }

                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
                }

                // Set linear parent_ids to trigger tree kernel (stores intermediates for fast rollback)
                if (needs_reeval && !draft.empty()) {
                    llama_tape_replay_sync(ctx_tgt); // ensure previous async replay is done
                    const int n_batch_tokens = 1 + (int)draft.size();
                    linear_parents.resize(n_batch_tokens);
                    linear_parents[0] = -1; // root loads initial state
                    for (int i = 1; i < n_batch_tokens; i++) {
                        linear_parents[i] = i - 1;
                    }
                    llama_set_tree_parent_ids(ctx_tgt, linear_parents.data(), n_batch_tokens);

                    // Take backup for rollback
                    auto * mem = llama_get_memory(ctx_tgt);
                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    llama_memory_seq_cp(mem, 0, seq_backup, -1, -1);
                    has_backup = true;
                }

                // Enable tape recording so GPU tape captures k/v/g/b for tape_replay
                if (needs_reeval && !draft.empty()) {
                    llama_set_tape_recording(ctx_tgt, true);
                }
                {
                    common_time_meas tm(t_decode1_total);
                    llama_decode(ctx_tgt, batch_tgt);
                }
                if (needs_reeval && !draft.empty()) {
                    llama_set_tape_recording(ctx_tgt, false);
                }
            }

            // save sampler state before sampling if we use checkpoints
            if (use_ckpt) {
                smpl_save.reset(common_sampler_clone(smpl.get()));
            }

            {
                common_time_meas tm(t_sample_total);
                if (use_rejection_sampling && !draft_log_probs.empty()) {
                    ids = speculative_reject_sample(smpl.get(), ctx_tgt, draft, draft_log_probs,
                        sample_temp, reject_rng);
                } else {
                    ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
                }
            }

            n_draft_this_iter = (int)draft.size();

            // update draft strategies with logits (e.g. token recycling adjacency matrix)
            {
                llama_tokens batch_tokens;
                batch_tokens.push_back(id_last);
                batch_tokens.insert(batch_tokens.end(), draft.begin(), draft.end());
                common_speculative_update_logits(spec, ctx_tgt, batch_tokens, (int)ids.size());
            }

        } // end flat path

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            draft = std::move(ids);

            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, spec_ckpt.data.data(), spec_ckpt.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            GGML_ASSERT(n == spec_ckpt.size());

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, spec_ckpt.n_tokens, -1);

            prompt_tgt.resize(spec_ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft_this_iter;
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // track rejection position
        if (n_draft_this_iter > 0) {
            if ((int)ids.size() == n_draft_this_iter + 1) {
                n_all_accepted++;
            } else {
                int rej_pos = (int)ids.size() - 1;
                if (rej_pos < (int)reject_pos.size()) {
                    reject_pos[rej_pos]++;
                }
            }

        }

        {
            common_time_meas tm(t_other_total);
            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);

                id_last = ids[i];

                if (llama_vocab_is_eog(vocab, id_last)) {
                    has_eos = true;
                    break;
                }

                const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

                if (params.use_color && i + 1 < ids.size()) {
                    LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
                } else {
                    LOG("%s", token_str.c_str());
                }
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, n_draft_this_iter, id_last);

        // clear the draft since it has been consumed
        draft.clear();

        if (has_backup && !has_eos) {
            if (tree_budget > 0) {
                // DDTree rollback: tree_rollback for SSM state, backup-restore for KV cache
                const int n_past_before = n_past - (int)ids.size();
                auto * mem = llama_get_memory(ctx_tgt);

                // tree_rollback: restore SSM state from f16 intermediates + conv from tape
                // Target pos = n_past_before + depth of committed node
                const int commit_depth = (commit_n > 0) ? tree.depths[commit_n - 1] : 0;
                {
                    common_time_meas tm(t_restore_total);
                    llama_tree_rollback(ctx_tgt, commit_n, tree.parents.data(), n_past_before + commit_depth);
                }

                // KV cache cleanup
                {
                    common_time_meas tm(t_decode2_total);
                    const int n_branches = tree.n_nodes - tree.main_path_len;
                    const bool all_accepted = (commit_depth == tree.main_path_len);

                    if (n_branches == 0 && accepted_on_main_path) {
                        // Fast path: no branches, just trim rejected tail from KV
                        if (!all_accepted) {
                            llama_memory_seq_rm(mem, 0, n_past_before + commit_depth + 1, -1);
                        }
                        llama_memory_seq_rm(mem, seq_backup, -1, -1);
                        n_reeval_skipped++;
                    } else {
                        // Slow path: branches create position collisions on seq 0.
                        // Backup-restore + re-decode accepted tokens.
                        llama_memory_seq_rm(mem, 0, n_past_before, -1);
                        llama_memory_seq_cp(mem, seq_backup, 0, -1, -1);
                        llama_memory_seq_rm(mem, seq_backup, -1, -1);

                        // Re-decode accepted tokens for KV
                        common_batch_clear(batch_tgt);
                        for (int i = n_past_before; i < (int)prompt_tgt.size(); ++i) {
                            common_batch_add(batch_tgt, prompt_tgt[i], i, { 0 }, false);
                        }
                        if (batch_tgt.n_tokens > 0) {
                            llama_decode(ctx_tgt, batch_tgt);
                            n_reeval_tokens += batch_tgt.n_tokens;
                            n_reeval_calls++;
                        }
                    }
                }
                n_past = (int)prompt_tgt.size();
            } else {
                // Flat path rollback: tree_rollback (GPU intermediates) for SSM + conv state
                const bool all_accepted = ((int)ids.size() == n_draft_this_iter + 1);

                if (all_accepted) {
                    llama_clear_tree_parent_ids(ctx_tgt);
                    auto * mem = llama_get_memory(ctx_tgt);
                    llama_memory_seq_rm(mem, seq_backup, -1, -1);
                    llama_memory_seq_rm(mem, 0, n_past, -1);
                    n_reeval_skipped++;
                } else {
                    const int n_past_before = n_past - (int)ids.size();

                    llama_clear_tree_parent_ids(ctx_tgt);

                    {
                        common_time_meas tm(t_decode2_total);
                        llama_dflash_rollback(ctx_tgt, 0, seq_backup, n_past_before, (int)ids.size());
                    }

                    n_reeval_tokens += (int)ids.size();
                    n_reeval_calls++;
                    n_past = (int)prompt_tgt.size();
                }
            }
        } else {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
            // Clear stale parent_ids if no rollback happened
            llama_clear_tree_parent_ids(ctx_tgt);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int64_t t_dec_total = t_dec_end - t_dec_start;
    const int64_t t_accounted = t_draft_total + t_backup_total + t_decode1_total + t_sample_total + t_restore_total + t_decode2_total + t_other_total;

    LOG_INF("\n");
    LOG_INF("=== SPECULATIVE LOOP TIMING BREAKDOWN ===\n");
    LOG_INF("iterations:     %d\n", n_iters);
    LOG_INF("total decode:   %8.2f ms (100%%)\n", t_dec_total / 1e3);
    LOG_INF("  draft gen:    %8.2f ms (%5.1f%%)\n", t_draft_total / 1e3,   100.0 * t_draft_total / t_dec_total);
    LOG_INF("  backup:       %8.2f ms (%5.1f%%)\n", t_backup_total / 1e3,  100.0 * t_backup_total / t_dec_total);
    LOG_INF("  decode1 (verify): %8.2f ms (%5.1f%%)  [%d tok/call avg]\n", t_decode1_total / 1e3, 100.0 * t_decode1_total / t_dec_total, n_iters > 0 ? (int)(n_predict + n_drafted) / n_iters : 0);
    LOG_INF("  sampling:     %8.2f ms (%5.1f%%)\n", t_sample_total / 1e3,  100.0 * t_sample_total / t_dec_total);
    LOG_INF("  restore:      %8.2f ms (%5.1f%%)\n", t_restore_total / 1e3, 100.0 * t_restore_total / t_dec_total);
    LOG_INF("  decode2 (reeval): %8.2f ms (%5.1f%%)  [%d calls, %d tok total, %.1f tok/call avg, %d skipped (all-accept)]\n", t_decode2_total / 1e3, 100.0 * t_decode2_total / t_dec_total, n_reeval_calls, n_reeval_tokens, n_reeval_calls > 0 ? (float)n_reeval_tokens / n_reeval_calls : 0.0f, n_reeval_skipped);
    LOG_INF("  other:        %8.2f ms (%5.1f%%)\n", t_other_total / 1e3,   100.0 * t_other_total / t_dec_total);
    LOG_INF("  unaccounted:  %8.2f ms (%5.1f%%)\n", (t_dec_total - t_accounted) / 1e3, 100.0 * (t_dec_total - t_accounted) / t_dec_total);
    LOG_INF("=========================================\n");

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    // per-position rejection histogram
    {
        int n_rounds_with_draft = n_all_accepted;
        for (int i = 0; i < (int)reject_pos.size(); ++i) {
            n_rounds_with_draft += reject_pos[i];
        }
        if (n_rounds_with_draft > 0) {
            LOG_INF("\nrejection histogram (position → count [%%]):\n");
            for (int i = 0; i < (int)reject_pos.size() && i < params_spec.n_max; ++i) {
                if (reject_pos[i] > 0) {
                    LOG_INF("  pos %2d: %4d (%5.1f%%)\n", i, reject_pos[i], 100.0f * reject_pos[i] / n_rounds_with_draft);
                }
            }
            LOG_INF("  all ok: %4d (%5.1f%%)\n", n_all_accepted, 100.0f * n_all_accepted / n_rounds_with_draft);
        }
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    common_speculative_print_stats(spec);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
