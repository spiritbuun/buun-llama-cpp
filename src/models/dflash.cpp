#include "models.h"
#include "dflash-selector-family.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <atomic>
#include <cstdlib>

// GPU top-K/argmax draft sampling tail (opt-in via llama_set_dflash_argmax): computes
// K ids + log-probs per draft position in-graph so the draft loop skips the full-vocab
// logits transfer + CPU scan. Mirrors the fork drafter's tail (dflash_draft.cpp).
static void build_dflash_draft_argmax(llm_graph_context & g) {
    if (!g.cparams.dflash_argmax || !g.res->t_logits) {
        return;
    }

    const float sample_temp = g.cparams.dflash_sample_temp;
    static std::atomic<uint64_t> gumbel_counter{1};
    const uint64_t seed = (sample_temp > 0.0f) ? gumbel_counter.fetch_add(1) : 0;

    const int topk = g.cparams.dflash_topk;
    ggml_tensor * t = topk > 1
        ? ggml_topk_ext  (g.ctx0, g.res->t_logits, topk, sample_temp, seed)
        : ggml_argmax_ext(g.ctx0, g.res->t_logits,       sample_temp, seed);

    g.res->t_logits_argmax = t;
    ggml_build_forward_expand(g.gf, t);
}

class llm_graph_input_dflash_uniforms final : public llm_graph_input_i {
public:
    llm_graph_input_dflash_uniforms(
            const llama_cross * cross,
            int64_t n_steps,
            int64_t n_blocks,
            int64_t block_size) :
        cross(cross), n_steps(n_steps), n_blocks(n_blocks), block_size(block_size),
        values((size_t) n_steps * n_blocks, 0.5f) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch && ubatch->seq_id &&
                ubatch->n_tokens >= (uint32_t) (n_blocks * block_size));
        for (int64_t block = 0; block < n_blocks; ++block) {
            const int64_t token = block * block_size;
            GGML_ASSERT(ubatch->n_seq_id[token] > 0);
            const llama_seq_id seq_id = ubatch->seq_id[token][0];
            const auto it = cross->dflash_proposal_uniforms.find(seq_id);
            if (it == cross->dflash_proposal_uniforms.end() ||
                    it->second.size() < (size_t) n_steps) {
                // Context warmup has no request-owned speculative state yet.
                std::fill_n(values.begin() + block * n_steps, n_steps, 0.5f);
                continue;
            }
            std::copy_n(it->second.begin(), n_steps,
                    values.begin() + block * n_steps);
        }
        ggml_backend_tensor_set(uniforms, values.data(), 0,
                values.size() * sizeof(float));
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.cross == cross &&
            params.ubatch.n_tokens == (uint32_t) (n_blocks * block_size);
    }

    ggml_tensor * uniforms = nullptr;

private:
    const llama_cross * cross;
    int64_t n_steps;
    int64_t n_blocks;
    int64_t block_size;
    std::vector<float> values;
};

// DFlash2 predicts every block position in parallel. The selector does the same
// for its expensive work: it materializes the complete [successor, predecessor]
// score lattice for every draft position before walking the seven selected edges.
// Only that tiny index walk is sequential, matching the reference implementation.
// Row zero of each returned block is a dummy (the consumer skips the anchor row).
static bool llm_build_dflash2_selector(
        llm_graph_context & g,
        const llama_model & model,
        ggml_tensor * inp_tokens,
        ggml_tensor * hidden,
        ggml_tensor * logits) {
    if (!g.cparams.dflash_argmax || !model.dflash2_selector_hidden) {
        return false;
    }
    const int64_t n_tokens = logits->ne[1];
    const int64_t block_size = g.hparams.dflash_block_size;
    const int64_t n_blocks = n_tokens / block_size;
    const int64_t n_steps  = block_size - 1;
    const int64_t rank     = g.hparams.dflash2_selector_rank;
    const int64_t top_k    = g.hparams.dflash2_selector_top_k;
    GGML_ASSERT(block_size > 2);
    // Context initialization also probes one-token and other generic graph
    // shapes. Those are not DFlash2 cycles, so keep the ordinary sampler tail
    // for them; real DFlash2 decode batches are whole anchor+mask blocks.
    if (n_tokens % block_size != 0) {
        return false;
    }
    GGML_ASSERT(top_k >= 1 && top_k <= 64);

    // Row zero is the committed anchor. The trained selector consumes decoder
    // rows 1..block_size-1 and predicts the seven draft tokens in parallel.
    ggml_tensor * hidden_steps = ggml_view_3d(g.ctx0, hidden,
            hidden->ne[0], n_steps, n_blocks,
            hidden->nb[1], (size_t) block_size * hidden->nb[1], hidden->nb[1]);
    if (!ggml_is_contiguous(hidden_steps)) {
        hidden_steps = ggml_cont(g.ctx0, hidden_steps);
    }
    ggml_tensor * projected = g.build_lora_mm(model.dflash2_selector_hidden, hidden_steps);
    GGML_ASSERT(projected->ne[0] == rank);

    ggml_tensor * logits_steps = ggml_view_3d(g.ctx0, logits,
            logits->ne[0], n_steps, n_blocks,
            logits->nb[1], (size_t) block_size * logits->nb[1], logits->nb[1]);
    if (!ggml_is_contiguous(logits_steps)) {
        logits_steps = ggml_cont(g.ctx0, logits_steps);
    }
    // Stochastic DFlash2 assigns proposal probabilities to candidate positions, so it
    // requires a stable candidate set and order.  Greedy selection is a point mass:
    // candidate ordering is immaterial and target verification remains authoritative,
    // so avoid the stable CUDA repair's extra vocabulary scan on that hot path.
    ggml_tensor * candidates = ggml_top_k_ext(g.ctx0, logits_steps, top_k,
            g.cparams.dflash_sample_temp > 0.0f); // [K, steps, blocks]

    // Step zero has one anchor predecessor per block. Score it separately so we
    // do not materialize a repeated I32 anchor tensor (CUDA's generic REPEAT
    // does not support I32). The remaining steps use every candidate from the
    // preceding step as their predecessor lattice.
    ggml_tensor * anchors = ggml_view_2d(g.ctx0, inp_tokens, 1, n_blocks,
            (size_t) block_size * inp_tokens->nb[0], 0);
    anchors = ggml_cont(g.ctx0, anchors);
    ggml_tensor * anchor_ids_flat = ggml_reshape_1d(g.ctx0, anchors, n_blocks);
    ggml_tensor * prior_ids = ggml_view_3d(g.ctx0, candidates, top_k, n_steps - 1, n_blocks,
            candidates->nb[1], candidates->nb[2], 0);
    prior_ids = ggml_cont(g.ctx0, prior_ids);

    ggml_tensor * candidate_ids_flat = ggml_reshape_1d(g.ctx0, candidates,
            top_k * n_steps * n_blocks);
    ggml_tensor * prior_ids_flat = ggml_reshape_1d(g.ctx0, prior_ids,
            top_k * (n_steps - 1) * n_blocks);

    ggml_tensor * successors = ggml_get_rows(g.ctx0,
            model.dflash2_selector_succ, candidate_ids_flat);
    successors = ggml_reshape_4d(g.ctx0, successors, rank, top_k, n_steps, n_blocks);
    ggml_tensor * hidden_rank = ggml_reshape_4d(g.ctx0, projected, rank, 1, n_steps, n_blocks);

    ggml_tensor * anchor_pred = ggml_get_rows(g.ctx0,
            model.dflash2_selector_pred, anchor_ids_flat);
    anchor_pred = ggml_reshape_4d(g.ctx0, anchor_pred, rank, 1, 1, n_blocks);
    ggml_tensor * successors_first = ggml_view_4d(g.ctx0, successors,
            rank, top_k, 1, n_blocks,
            successors->nb[1], successors->nb[2], successors->nb[3], 0);
    ggml_tensor * hidden_first = ggml_view_4d(g.ctx0, hidden_rank,
            rank, 1, 1, n_blocks,
            hidden_rank->nb[1], hidden_rank->nb[2], hidden_rank->nb[3], 0);
    ggml_tensor * scores_first = ggml_mul_mat(g.ctx0, successors_first,
            ggml_mul(g.ctx0, anchor_pred, hidden_first)); // [successor K, 1, 1, block]

    ggml_tensor * predecessors_rest = ggml_get_rows(g.ctx0,
            model.dflash2_selector_pred, prior_ids_flat);
    predecessors_rest = ggml_reshape_4d(g.ctx0, predecessors_rest,
            rank, top_k, n_steps - 1, n_blocks);
    ggml_tensor * successors_rest = ggml_view_4d(g.ctx0, successors,
            rank, top_k, n_steps - 1, n_blocks,
            successors->nb[1], successors->nb[2], successors->nb[3], successors->nb[2]);
    ggml_tensor * hidden_rest = ggml_view_4d(g.ctx0, hidden_rank,
            rank, 1, n_steps - 1, n_blocks,
            hidden_rank->nb[1], hidden_rank->nb[2], hidden_rank->nb[3], hidden_rank->nb[2]);
    ggml_tensor * scores_rest = ggml_mul_mat(g.ctx0, successors_rest,
            ggml_mul(g.ctx0, predecessors_rest, hidden_rest)); // [successor K, predecessor K, step-1, block]

    // Gather candidate unary logits in one operation by viewing each vocabulary
    // column as a scalar row. Broadcasting adds them across predecessor choices.
    ggml_tensor * scalar_logits = ggml_view_4d(g.ctx0, logits,
            1, logits->ne[0], n_steps, n_blocks,
            logits->nb[0], logits->nb[1], (size_t) block_size * logits->nb[1], logits->nb[1]);
    ggml_tensor * unary = ggml_get_rows(g.ctx0, scalar_logits, candidates);
    unary = ggml_reshape_4d(g.ctx0, unary, top_k, 1, n_steps, n_blocks);
    ggml_tensor * unary_first = ggml_view_4d(g.ctx0, unary,
            top_k, 1, 1, n_blocks,
            unary->nb[1], unary->nb[2], unary->nb[3], 0);
    ggml_tensor * unary_rest = ggml_view_4d(g.ctx0, unary,
            top_k, 1, n_steps - 1, n_blocks,
            unary->nb[1], unary->nb[2], unary->nb[3], unary->nb[2]);
    scores_first = ggml_add(g.ctx0, scores_first, unary_first);
    scores_rest  = ggml_add(g.ctx0, scores_rest,  unary_rest);

    ggml_tensor * proposal_uniforms = nullptr;
    if (g.cparams.dflash_sample_temp > 0.0f) {
        auto inp = std::make_unique<llm_graph_input_dflash_uniforms>(
                g.cross, n_steps, n_blocks, block_size);
        inp->uniforms = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32,
                n_steps, n_blocks);
        ggml_set_input(inp->uniforms);
        proposal_uniforms = inp->uniforms;
        g.res->add_input(std::move(inp));
    }
    ggml_tensor * all_paths = nullptr;
    ggml_tensor * all_q_rows = nullptr;
    for (int64_t block = 0; block < n_blocks; ++block) {
        const size_t block_logits_off = (size_t) block * block_size * logits->nb[1];
        ggml_tensor * row0_logits = ggml_view_1d(g.ctx0, logits, logits->ne[0], block_logits_off);
        ggml_tensor * path = ggml_argmax(g.ctx0, row0_logits);
        ggml_tensor * block_q_rows = nullptr;
        ggml_tensor * previous = nullptr; // step zero uses predecessor slot zero (all K are the anchor)

        for (int64_t step = 0; step < n_steps; ++step) {
            ggml_tensor * selected_scores;
            if (step == 0) {
                selected_scores = ggml_view_1d(g.ctx0, scores_first, top_k,
                        (size_t) block * scores_first->nb[3]);
            } else {
                const size_t score_off = (size_t) (step - 1) * scores_rest->nb[2] +
                        (size_t) block * scores_rest->nb[3];
                ggml_tensor * score_matrix = ggml_view_2d(g.ctx0, scores_rest,
                        top_k, top_k, scores_rest->nb[1], score_off);
                selected_scores = ggml_get_rows(g.ctx0, score_matrix, previous);
            }
            selected_scores = ggml_reshape_2d(g.ctx0, selected_scores, top_k, 1);

            ggml_tensor * q_row = nullptr;
            if (g.cparams.dflash_sample_temp > 0.0f) {
                q_row = ggml_soft_max_ext(g.ctx0, selected_scores,
                        nullptr, 1.0f / g.cparams.dflash_sample_temp, 0.0f);
                block_q_rows = block_q_rows
                    ? ggml_concat(g.ctx0, block_q_rows, q_row, 1)
                    : q_row;
            }

            ggml_tensor * best;
            if (g.cparams.dflash_sample_temp > 0.0f) {
                GGML_ASSERT(proposal_uniforms);
                const size_t uniform_off =
                    ((size_t) block * n_steps + step) * proposal_uniforms->nb[0];
                ggml_tensor * uniform = ggml_view_1d(g.ctx0,
                        proposal_uniforms, 1, uniform_off);
                ggml_tensor * cdf = ggml_cumsum(g.ctx0, q_row);
                ggml_tensor * above = ggml_step(g.ctx0,
                        ggml_sub(g.ctx0, cdf, uniform));
                ggml_tensor * n_above = ggml_sum(g.ctx0, above);
                ggml_tensor * index = ggml_scale_bias(g.ctx0,
                        n_above, -1.0f, (float) top_k);
                index = ggml_clamp(g.ctx0, index, 0.0f, (float) top_k - 1.0f);
                best = ggml_cast(g.ctx0, index, GGML_TYPE_I32);
            } else {
                best = ggml_argmax(g.ctx0, selected_scores);
            }
            previous = best;

            ggml_tensor * ids = ggml_view_1d(g.ctx0, candidates, top_k,
                    (size_t) step * candidates->nb[1] + (size_t) block * candidates->nb[2]);
            ggml_tensor * id_rows = ggml_reshape_2d(g.ctx0, ids, 1, top_k);
            ggml_tensor * selected = ggml_get_rows(g.ctx0, id_rows, best);
            path = ggml_concat(g.ctx0, path, ggml_reshape_1d(g.ctx0, selected, 1), 0);
        }
        all_paths = all_paths ? ggml_concat(g.ctx0, all_paths, path, 0) : path;
        if (block_q_rows) {
            block_q_rows = ggml_reshape_3d(g.ctx0, block_q_rows, top_k, n_steps, 1);
            all_q_rows = all_q_rows
                ? ggml_concat(g.ctx0, all_q_rows, block_q_rows, 2)
                : block_q_rows;
        }
    }

    g.res->t_logits_argmax = all_paths;
    if (all_q_rows) {
        g.res->t_dflash_candidate_ids = candidates;
        g.res->t_dflash_q_rows = all_q_rows;
        ggml_build_forward_expand(g.gf, candidates);
        ggml_build_forward_expand(g.gf, all_q_rows);
    }
    ggml_build_forward_expand(g.gf, all_paths);
    return true;
}

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LOGIT_SCALE,                 hparams.f_logit_scale, false);
    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     hparams.f_final_logit_softcapping, false);

    // drafts for M-RoPE targets carry degenerate sections [n_rot/2, 0, 0, 0]
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, false);

    uint32_t selector_conv_kernel_size = 0;
    uint32_t selector_conv_group_size  = 0;
    uint32_t selector_rank             = 0;
    uint32_t selector_top_k            = 0;

    bool has_selector_metadata = false;
    has_selector_metadata |= ml.get_key(LLM_KV_DFLASH2_CONV_KERNEL_SIZE, selector_conv_kernel_size, false);
    has_selector_metadata |= ml.get_key(LLM_KV_DFLASH2_CONV_GROUP_SIZE,  selector_conv_group_size,  false);
    has_selector_metadata |= ml.get_key(LLM_KV_DFLASH2_SELECTOR_RANK,     selector_rank,             false);
    has_selector_metadata |= ml.get_key(LLM_KV_DFLASH2_SELECTOR_TOP_K,    selector_top_k,            false);

    switch (llm_dflash_selector_family_from_loader(has_selector_metadata, hparams.n_layer(), ml)) {
        case llm_dflash_selector_family::none:
            break;
        case llm_dflash_selector_family::fork_dflash2:
        case llm_dflash_selector_family::upstream_compat:
            if (selector_rank == 0) {
                throw std::runtime_error("DFlash2 selector tensors require selector_rank metadata");
            }
            hparams.dflash2_conv_kernel_size = selector_conv_kernel_size;
            hparams.dflash2_conv_group_size  = selector_conv_group_size;
            hparams.dflash2_selector_rank    = selector_rank;
            hparams.dflash2_selector_top_k   = selector_top_k;
            break;
        case llm_dflash_selector_family::mixed:
            throw std::runtime_error(
                    "DFlash model mixes mutually exclusive fork DFlash2 and upstream selector tensor schemas");
        case llm_dflash_selector_family::unidentified:
            throw std::runtime_error("DFlash selector metadata has no recognized selector tensor schema");
    }

    // DFlash block size: default 16, overridable via GGUF. Must be set here (not as a
    // struct default) so only genuine DFlash drafters report block_size > 0.
    // Upstream conversions write the bare %s.block_size key (e.g. "dflash.block_size" —
    // DSpark sidecars carry 5 there); the fork-prefixed %s.dflash.block_size flavor is
    // kept as a fallback. Missing both keys keeps the historic default of 16.
    hparams.dflash_block_size = 16;
    if (!ml.get_key(LLM_KV_BLOCK_SIZE, hparams.dflash_block_size, false)) {
        ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);
    }
    if (!ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID, hparams.dflash_mask_token_id, false)) {
        ml.get_key(LLM_KV_TOKENIZER_MASK_ID, hparams.dflash_mask_token_id, false);
    }

    // DFlash2 extends the DFlash backbone with a learned two-sided grouped
    // convolution around attention/MLP and a rank-factorized path selector.
    // A zero selector rank distinguishes ordinary DFlash/DSpark sidecars.
    if (hparams.dflash2_selector_rank > 0) {
        if (hparams.dflash2_conv_kernel_size != 2 ||
            hparams.dflash2_conv_group_size == 0 ||
            hparams.n_embd % hparams.dflash2_conv_group_size != 0 ||
            hparams.dflash2_selector_top_k == 0) {
            throw std::runtime_error("unsupported DFlash2 convolution/selector geometry");
        }
        if (const char * value = std::getenv("GGML_DFLASH2_BLOCK_SIZE_OVERRIDE")) {
            const int override = std::atoi(value);
            if (override < 3 || override > 64) {
                throw std::runtime_error("GGML_DFLASH2_BLOCK_SIZE_OVERRIDE must be between 3 and 64");
            }
            LLAMA_LOG_WARN("%s: overriding DFlash2 block size %u -> %d\n",
                    __func__, hparams.dflash_block_size, override);
            hparams.dflash_block_size = override;
        } else if (hparams.dflash_block_size == 8) {
            // The released Qwen3.8 sidecar advertises eight positions, but
            // anchor + 12 nearly doubled code throughput and also won on prose
            // in the RTX 3090 sweep. Preserve metadata for other trained widths;
            // GGML_DFLASH2_BLOCK_SIZE_OVERRIDE=8 restores this one.
            LLAMA_LOG_INFO("%s: using tuned DFlash2 block size 13 (metadata: 8)\n", __func__);
            hparams.dflash_block_size = 13;
        }
    }

    if (!ml.get_arr(LLM_KV_TARGET_LAYERS, target_layer_ids, false)) {
        throw std::runtime_error("DFlash model requires 'target_layers' in GGUF metadata");
    }

    hparams.n_embd_inp_enc_impl = (uint32_t) target_layer_ids.size() * hparams.n_embd;

    if (target_layer_ids.size() > std::size(hparams.dflash_target_layer_ids)) {
        throw std::runtime_error("DFlash supports at most 8 target layers");
    }
    hparams.dflash_n_target_layers   = (uint32_t) target_layer_ids.size();
    hparams.dflash_n_target_features = hparams.n_embd_inp_enc_impl;
    for (size_t i = 0; i < target_layer_ids.size(); ++i) {
        hparams.dflash_target_layer_ids[i] = (uint32_t) target_layer_ids[i];
    }

    std::string layers;
    const char * sep = "";
    for (const auto id : target_layer_ids) {
        layers += sep;
        layers += std::to_string(id);
        sep = ", ";
    }
    LLAMA_LOG_INFO("%s: DFlash extract_layers = [%s]\n", __func__, layers.c_str());

    // DeepSeek-V4 DSpark backbone: stages are full DSV4 blocks, uniform sliding window (the draft KV ring)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT, hparams.dsv4_hc_mult, false);
    if (hparams.dsv4_hc_mult > 0) {
        ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,                hparams.n_lora_q);
        ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,             hparams.n_swa);
        ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,           hparams.n_ff_exp);
        ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,                  hparams.n_expert_shared);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,                 hparams.expert_weights_scale);
        ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,                  hparams.expert_weights_norm);
        ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                   hparams.expert_gating_func);
        ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,              hparams.swiglu_clamp_exp, hparams.n_layer_all);
        if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,       hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
            hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
        }
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
        ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
        ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
        ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
        ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS,            hparams.dsv4_compress_ratios, false);

        GGML_ASSERT(hparams.dsv4_o_group_count > 0); // avoid div by zero

        if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
            throw std::runtime_error("DSpark DSV4 draft expects sqrtsoftplus MoE scoring");
        }
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (hparams.dsv4_compress_ratios[il] != 0) {
                throw std::runtime_error("DSpark DSV4 draft expects uncompressed attention on all stages");
            }
        }

        GGML_ASSERT(hparams.n_swa > 0);
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        hparams.set_swa_pattern(0);
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            hparams.is_swa_impl[il] = true;
        }
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

        type = LLM_TYPE_UNKNOWN;
        return;
    }

    // optional interleaved sliding-window attention with per-layer pattern array.
    // DFlash has a single rope, so the SWA rope == main rope.
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false) && hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
        hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
        hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_inp = hparams.n_embd_inp_enc();

    const auto selector_family = llm_dflash_selector_family_from_loader(
            hparams.dflash2_selector_rank > 0, hparams.n_layer(), *ml);
    const auto selector_schema = llm_dflash_selector_tensor_schema_for_family(selector_family);
    if (hparams.dflash2_selector_rank > 0 && !selector_schema.valid) {
        throw std::runtime_error("DFlash2 tensor schema changed between metadata and tensor loading");
    }

    tok_embd        = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,       "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // reduced draft vocab (optional): d2t maps draft rows to target token ids
    int64_t n_vocab_draft = n_vocab;
    const struct ggml_tensor * d2t_meta = ml->get_tensor_meta("d2t");
    if (d2t_meta) {
        n_vocab_draft = d2t_meta->ne[0];
        d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_vocab_draft }, 0);
        LLAMA_LOG_INFO("%s: DFlash using d2t mapping (draft_vocab_size = %lld)\n", __func__, (long long) n_vocab_draft);
    }

    // DSpark = DFlash + a semi-autoregressive Markov head and Confidence head
    //
    // TODO: only Qwen3-style backbones are supported for now; other backbones (e.g. Gemma4)
    //       need their own conversion path and graph tweaks
    const struct ggml_tensor * markov_meta = ml->get_tensor_meta("markov_w1.weight");
    if (markov_meta) {
        const int64_t dspark_markov_rank = markov_meta->ne[0];

        dspark_markov_w1   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), { dspark_markov_rank, n_vocab }, 0);
        dspark_markov_w2   = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), { dspark_markov_rank, n_vocab_draft }, 0);
        dspark_markov_w2_s = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "scale"),  { 1 }, TENSOR_NOT_REQUIRED);

        dspark_conf_proj   = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "weight"), { n_embd + dspark_markov_rank, 1 }, TENSOR_NOT_REQUIRED);
        dspark_conf_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_CONF_PROJ, "bias"),   { 1 },             TENSOR_NOT_REQUIRED);

        LLAMA_LOG_INFO("%s: DFlash with DSpark markov head (rank = %lld)\n", __func__, (long long) dspark_markov_rank);
    }

    if (hparams.dflash2_selector_rank > 0) {
        const int64_t rank = hparams.dflash2_selector_rank;
        const char * codebook_suffix = selector_schema.selector_codebooks_have_weight_suffix ? "weight" : nullptr;
        dflash2_selector_hidden = create_tensor(tn(selector_schema.selector_hidden, "weight"), { n_embd, rank }, 0);
        dflash2_selector_pred   = create_tensor(tn(selector_schema.selector_pred, codebook_suffix), { rank, n_vocab }, 0);
        dflash2_selector_succ   = create_tensor(tn(selector_schema.selector_succ, codebook_suffix), { rank, n_vocab }, 0);
        const char * wire_schema = selector_family == llm_dflash_selector_family::upstream_compat
                ? "upstream-compatible" : "fork";
        LLAMA_LOG_INFO("%s: DFlash2 selector (rank = %lld, top-k = %u, wire schema = %s, graph = optimized fork)\n",
                __func__, (long long) rank, hparams.dflash2_selector_top_k, wire_schema);
    }

    fc              = create_tensor(tn(LLM_TENSOR_FC,              "weight"), { n_embd_inp, n_embd }, 0);
    fc_s            = create_tensor(tn(LLM_TENSOR_FC,              "scale"),  { 1 }, TENSOR_NOT_REQUIRED);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), { n_embd }, 0); // encoder hidden_norm (after fc)
    output_norm     = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,    "weight"), { n_embd }, 0); // decoder final norm

    // optional: reduced-vocab drafts ship their own lm head, full-vocab drafts can share the target's via ctx_other
    // a draft with its own embeddings + head references no target tensors and can run on devices the target does not use (e.g. -devd with a tensor-split target)
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), { n_embd, n_vocab_draft }, TENSOR_NOT_REQUIRED);

    if (hparams.dsv4_hc_mult > 0) {
        const int64_t q_lora_rank     = hparams.n_lora_q;
        const int64_t n_ff_exp        = hparams.n_ff_exp;
        const int64_t n_expert_shared = hparams.n_expert_shared;
        const int64_t n_embd_head     = hparams.n_embd_head_k();
        const int64_t o_groups        = hparams.dsv4_o_group_count;
        const int64_t o_lora_rank     = hparams.dsv4_o_lora_rank;
        const int64_t hc_mult         = hparams.dsv4_hc_mult;
        const int64_t hc_dim          = hc_mult * n_embd;
        const int64_t hc_mix_dim      = (2 + hc_mult) * hc_mult;

        hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
        hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
        hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
            layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
            layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
            layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
            layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        }
        return;
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_embd, n_embd_head_k * n_head }, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), { n_embd, n_embd_k_gqa }, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), { n_embd, n_embd_v_gqa }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);

        // optional per-head attention sinks (e.g. Nemotron DSpark)
        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), { n_head }, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_embd, n_ff }, 0);

        if (hparams.dflash2_selector_rank > 0) {
            const int64_t n_groups = n_embd / hparams.dflash2_conv_group_size;
            const int64_t n_coeff  = 2 * hparams.dflash2_conv_kernel_size * n_groups;
            layer.dflash2_attn_conv_base = create_tensor(tn(selector_schema.attn_conv_base, nullptr, i),
                    { n_embd, (int64_t) hparams.dflash2_conv_kernel_size, 2 }, 0);
            layer.dflash2_attn_conv_proj = create_tensor(tn(selector_schema.attn_conv_proj, "weight", i),
                    { n_embd, n_coeff }, 0);
            layer.dflash2_ffn_conv_base = create_tensor(tn(selector_schema.ffn_conv_base, nullptr, i),
                    { n_embd, (int64_t) hparams.dflash2_conv_kernel_size, 2 }, 0);
            layer.dflash2_ffn_conv_proj = create_tensor(tn(selector_schema.ffn_conv_proj, "weight", i),
                    { n_embd, n_coeff }, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            if (hparams.dsv4_hc_mult > 0) {
                return std::make_unique<graph_dsv4>(*this, params);
            }
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type: %d", (int) params.gtype);
    };
}

template <>
ggml_tensor * llama_model_dflash::graph<true>::build_inp_embd_enc() const {
    auto inp_target = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());

    inp_target->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp_enc(), n_tokens);
    ggml_set_input(inp_target->embd);

    ggml_tensor * cur = inp_target->embd;
    cb(cur, "inp_embd", -1);

    res->add_input(std::move(inp_target));

    return cur;
}

// DFlash Encoder: processes target model features through feature fusion layer
template <>
llama_model_dflash::graph<true>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inp_embd_enc();

    cur = build_lora_mm(model.fc, cur, model.fc_s);
    cb(cur, "fc_out", -1);

    cur = build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_h_nextn = cur;

    ggml_build_forward_expand(gf, cur);
}

// Injection-graph feature input, three flavors:
//   * staged (fused + inject_stage bound): gather rows on-device from the target's
//     capture stage via a per-decode row-index input — no host feature upload at all
//   * fused: raw concatenated target features from the batch (H2D), fc + enc-norm here
//   * unfused: pre-encoded g rows from the batch (H2D)
// staged-rows gather + encoder (fc + enc-norm): shared by the standalone staged inject
// graph and the fused-cycle inject rows so the two paths' math cannot drift
static ggml_tensor * build_dflash_staged_enc(llm_graph_context & g, const llama_model & model,
        ggml_tensor * stage, int64_t n_rows) {
    auto inp = std::make_unique<llm_graph_input_dflash_stage_rows>(g.cparams);
    inp->rows = ggml_new_tensor_1d(g.ctx0, GGML_TYPE_I32, n_rows);
    ggml_set_input(inp->rows);
    ggml_tensor * cur = ggml_get_rows(g.ctx0, stage, inp->rows);
    g.res->add_input(std::move(inp));
    g.cb(cur, "inp_g_embeddings", -1);

    cur = g.build_lora_mm(model.fc, cur);
    g.cb(cur, "fc_out", -1);

    cur = g.build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
    g.cb(cur, "enc_norm_out", -1);

    return cur;
}

static ggml_tensor * build_dflash_inject_input(llm_graph_context & g, const llama_model & model, int64_t n_embd) {
    const auto & cparams = g.cparams;
    const bool fused = cparams.dflash_fused_inject;

    if (fused && cparams.dflash_inject_stage) {
        return build_dflash_staged_enc(g, model, cparams.dflash_inject_stage, g.n_tokens);
    }

    const int64_t n_embd_in = fused ? g.hparams.n_embd_inp_enc() : n_embd;
    auto inp = std::make_unique<llm_graph_input_embd>(n_embd_in);
    inp->embd = ggml_new_tensor_2d(g.ctx0, GGML_TYPE_F32, n_embd_in, g.n_tokens);
    ggml_set_input(inp->embd);
    ggml_tensor * cur = inp->embd;
    g.res->add_input(std::move(inp));
    g.cb(cur, "inp_g_embeddings", -1);

    if (fused) {
        cur = g.build_lora_mm(model.fc, cur);
        g.cb(cur, "fc_out", -1);

        cur = g.build_norm(cur, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
        g.cb(cur, "enc_norm_out", -1);
    }

    return cur;
}

// Apply one side of DFlash2's learned two-tap grouped convolution. The token
// graph is one independent anchor+mask block; injection uses a separate embd
// graph and therefore never enters here. Keeping that invariant explicit also
// prevents the previous-token tap from crossing request/block boundaries.
static ggml_tensor * build_dflash2_grouped_conv(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * projected,
        ggml_tensor * base,
        int side) {
    const auto & hp = g.hparams;
    const int64_t n_embd     = hp.n_embd;
    const int64_t group_size = hp.dflash2_conv_group_size;
    const int64_t n_groups   = n_embd / group_size;
    const int64_t taps       = hp.dflash2_conv_kernel_size;
    const int64_t n_tokens   = hidden->ne[1];

    GGML_ASSERT(side == 0 || side == 1);
    GGML_ASSERT(taps == 2);
    GGML_ASSERT(hidden->ne[0] == n_embd);
    GGML_ASSERT(projected->ne[0] == 2 * taps * n_groups);

    const char * fused_env = std::getenv("GGML_DFLASH2_FUSED_CONV");
    if (fused_env && std::atoi(fused_env) != 0) {
        if (!ggml_is_contiguous(hidden)) {
            hidden = ggml_cont(g.ctx0, hidden);
        }
        if (!ggml_is_contiguous(projected)) {
            projected = ggml_cont(g.ctx0, projected);
        }
        return ggml_dflash2_conv(g.ctx0, hidden, projected, base,
                side, group_size, g.hparams.dflash_block_size);
    }

    ggml_tensor * blocks = ggml_reshape_3d(g.ctx0, hidden, group_size, n_groups, n_tokens);

    const size_t delta_side_off = (size_t) side * taps * n_groups * projected->nb[0];
    ggml_tensor * delta = ggml_view_4d(g.ctx0, projected, 1, n_groups, taps, n_tokens,
            projected->nb[0], n_groups * projected->nb[0], projected->nb[1], delta_side_off);
    delta = ggml_repeat_4d(g.ctx0, delta, group_size, n_groups, taps, n_tokens);

    ggml_tensor * base_side = ggml_view_4d(g.ctx0, base,
            group_size, n_groups, taps, 1,
            group_size * base->nb[0], base->nb[1], base->nb[2], (size_t) side * base->nb[2]);
    base_side = ggml_cast(g.ctx0, base_side, projected->type);

    ggml_tensor * coeff = ggml_add(g.ctx0, delta, base_side);
    ggml_tensor * c0 = ggml_view_3d(g.ctx0, coeff, group_size, n_groups, n_tokens,
            coeff->nb[1], coeff->nb[3], 0);
    ggml_tensor * c1 = ggml_view_3d(g.ctx0, coeff, group_size, n_groups, n_tokens,
            coeff->nb[1], coeff->nb[3], coeff->nb[2]);

    // Shift once, then mask every block anchor. This keeps graph size constant
    // when reservation covers a large ubatch (the prototype emitted one concat
    // chain per reserved block and could exhaust the graph-node budget).
    const int64_t block_size = hp.dflash_block_size;
    GGML_ASSERT(block_size > 1);
    ggml_tensor * first = ggml_view_3d(g.ctx0, blocks, group_size, n_groups, 1,
            blocks->nb[1], blocks->nb[2], 0);
    ggml_tensor * shifted = ggml_scale(g.ctx0, first, 0.0f);
    ggml_tensor * prior = ggml_view_3d(g.ctx0, blocks, group_size, n_groups, n_tokens - 1,
            blocks->nb[1], blocks->nb[2], 0);
    shifted = ggml_concat(g.ctx0, shifted, prior, 2);

    if (n_tokens != block_size) {
        ggml_tensor * position_mask = ggml_arange(g.ctx0, 0.0f, (float) block_size, 1.0f);
        position_mask = ggml_clamp(g.ctx0, position_mask, 0.0f, 1.0f);
        position_mask = ggml_reshape_3d(g.ctx0, position_mask, 1, 1, block_size);
        // Reservation probes are not required to be complete DFlash2 blocks. Tile
        // the anchor mask through the next whole block and slice it back so those
        // generic shapes still reserve a valid graph without changing runtime math.
        const int64_t mask_tokens = ((n_tokens + block_size - 1) / block_size) * block_size;
        position_mask = ggml_repeat_4d(g.ctx0, position_mask, 1, 1, mask_tokens, 1);
        position_mask = ggml_view_3d(g.ctx0, position_mask, 1, 1, n_tokens,
                position_mask->nb[1], position_mask->nb[2], 0);
        shifted = ggml_mul(g.ctx0, shifted, position_mask);
    }

    ggml_tensor * out = ggml_add(g.ctx0,
            ggml_mul(g.ctx0, c0, blocks),
            ggml_mul(g.ctx0, c1, shifted));
    return ggml_reshape_2d(g.ctx0, out, n_embd, n_tokens);
}

static ggml_tensor * llm_build_dflash2_conv_prepare(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * base,
        ggml_tensor * projection,
        ggml_tensor ** coefficients);

static ggml_tensor * llm_build_dflash2_conv_finish(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * coefficients,
        ggml_tensor * base);

static ggml_tensor * build_dflash2_conv_prepare_tail(
        llm_graph_context & g,
        ggml_tensor * hidden,
        int64_t n_prefix,
        ggml_tensor * base,
        ggml_tensor * projection,
    ggml_tensor ** coefficients) {
    if (n_prefix == 0) {
        return llm_build_dflash2_conv_prepare(g, hidden, base, projection, coefficients);
    }
    ggml_tensor * tail = ggml_view_2d(g.ctx0, hidden, hidden->ne[0], hidden->ne[1] - n_prefix,
            hidden->nb[1], (size_t) n_prefix * hidden->nb[1]);
    ggml_tensor * convolved = llm_build_dflash2_conv_prepare(g, tail, base, projection, coefficients);
    ggml_tensor * prefix = ggml_view_2d(g.ctx0, hidden, hidden->ne[0], n_prefix, hidden->nb[1], 0);
    return ggml_concat(g.ctx0, prefix, convolved, 1);
}

static ggml_tensor * build_dflash2_conv_finish_tail(
        llm_graph_context & g,
        ggml_tensor * hidden,
        int64_t n_prefix,
        ggml_tensor * coefficients,
    ggml_tensor * base) {
    if (n_prefix == 0) {
        return llm_build_dflash2_conv_finish(g, hidden, coefficients, base);
    }
    ggml_tensor * tail = ggml_view_2d(g.ctx0, hidden, hidden->ne[0], hidden->ne[1] - n_prefix,
            hidden->nb[1], (size_t) n_prefix * hidden->nb[1]);
    ggml_tensor * convolved = build_dflash2_grouped_conv(g, tail, coefficients, base, 1);
    ggml_tensor * prefix = ggml_view_2d(g.ctx0, hidden, hidden->ne[0], n_prefix, hidden->nb[1], 0);
    return ggml_concat(g.ctx0, prefix, convolved, 1);
}

static ggml_tensor * llm_build_dflash2_conv_prepare(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * base,
        ggml_tensor * projection,
        ggml_tensor ** coefficients) {
    *coefficients = g.build_lora_mm(projection, hidden);
    return build_dflash2_grouped_conv(g, hidden, *coefficients, base, 0);
}

static ggml_tensor * llm_build_dflash2_conv_finish(
        llm_graph_context & g,
        ggml_tensor * hidden,
        ggml_tensor * coefficients,
        ggml_tensor * base) {
    return build_dflash2_grouped_conv(g, hidden, coefficients, base, 1);
}

// DSpark (DFlash + Markov & Confidence head): Markov bias on the draft logits, chained per block position
static void build_dspark_markov_head(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    ggml_tensor * w1 = model.dspark_markov_w1;
    ggml_tensor * w2 = model.dspark_markov_w2;
    GGML_ASSERT(w1 && w2 && "DSpark markov weights not loaded");

    // confidence head is optional
    const bool has_conf = model.dspark_conf_proj != nullptr;

    ggml_tensor * base = res->t_logits; // [n_vocab, n_tokens]
    const int64_t n_vocab = base->ne[0];
    const int64_t n_tok   = base->ne[1];

    const auto it = model.gguf_kv.find("dflash.block_size");
    GGML_ASSERT(it != model.gguf_kv.end() && "DSpark draft requires 'dflash.block_size' in GGUF metadata");
    const int64_t block_size = std::stoi(it->second);
    GGML_ASSERT(block_size > 0);

    int64_t n_blocks = g.ubatch.n_seqs_unq;
    if (g.cparams.dflash_oneg_n_inject > 0) {
        n_blocks -= 1; // fused cycles carry a scratch padding seq that has no noise rows
    }

    // bonus anchor (SpecForge exports): slot 0 is a bonus token, not a prediction slot
    const auto it_anchor          = model.gguf_kv.find("dflash.sample_from_anchor");
    const bool sample_from_anchor = it_anchor == model.gguf_kv.end() || it_anchor->second == "true";
    const int64_t i_draft_beg     = sample_from_anchor ? 0 : 1;
    GGML_ASSERT(n_blocks > 0 && n_tok % n_blocks == 0 && "DSpark markov head requires equal-size blocks");
    // runtime tokens per block in this ubatch (anchor + drafted positions), bounded by training block_size
    const int64_t block_drafts = n_tok / n_blocks;
    if (block_drafts > block_size) {
        return;
    }

    // anchor (committed last) token of every block: token 0 of each block, i.e. a strided view
    const size_t token_stride = (size_t) block_drafts * tokens->nb[0];
    const size_t base_stride = (size_t) block_drafts * base->nb[1];

    ggml_tensor * prev = ggml_view_2d(ctx0, tokens, 1, n_blocks, token_stride, 0);
    prev = ggml_cont_1d(ctx0, prev, n_blocks);

    ggml_tensor * cat      = nullptr;
    ggml_tensor * cat_conf = nullptr;

    if (!sample_from_anchor) {
        // bonus anchor slot: pass the logits through unbiased, pad the (unread) confidence column
        cat = ggml_cont(ctx0, ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, 0));
        if (has_conf) {
            cat_conf = ggml_sigmoid(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, base, 1, n_blocks, base_stride, 0)));
        }
    }

    // TODO: the in-graph chain is greedy (argmax); sampling params affect only the final
    //       token pick, not the Markov conditioning path
    for (int64_t i = i_draft_beg; i < block_drafts; ++i) {
        ggml_tensor * w1_prev = ggml_get_rows(ctx0, w1, prev);                          // [R, n_blocks]
        ggml_tensor * bias    = g.build_lora_mm(w2, w1_prev, model.dspark_markov_w2_s); // [n_vocab_draft, n_blocks]
        if (model.d2t) {
            // reduced draft vocab: scatter the bias to the target rows (base is -inf on the others)
            const int64_t n_draft_vocab = bias->ne[0];
            ggml_tensor * full = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_blocks), 0.0f);
            bias = ggml_set_rows(ctx0, full,
                    ggml_reshape_3d(ctx0, bias,      1,             n_draft_vocab, n_blocks),
                    ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
            bias = ggml_reshape_2d(ctx0, bias, n_vocab, n_blocks);
        }

        // position i of every block: strided view [n_vocab, n_blocks]
        ggml_tensor * base_i = ggml_view_2d(ctx0, base, n_vocab, n_blocks, base_stride, i*base->nb[1]);
        ggml_tensor * col    = ggml_add(ctx0, base_i, bias);

        cat = cat ? ggml_concat(ctx0, cat, col, 1) : col;

        if (has_conf) {
            // confidence head input: predicts per-position acceptance
            ggml_tensor * conf_inp   = res->t_embd; // [n_embd, n_tok]
            // conf(i) = sigmoid(conf_proj . [conf_inp(i); markov_w1[prev(i)]] + b)  -- [1, n_blocks]
            ggml_tensor * conf_inp_i = ggml_view_2d(ctx0, conf_inp, conf_inp->ne[0], n_blocks,
                                                    (size_t) block_drafts * conf_inp->nb[1], i*conf_inp->nb[1]);
            ggml_tensor * feat = ggml_concat(ctx0, ggml_cont(ctx0, conf_inp_i), w1_prev, 0);
            ggml_tensor * conf = ggml_mul_mat(ctx0, model.dspark_conf_proj, feat);
            if (model.dspark_conf_proj_b) {
                conf = ggml_add(ctx0, conf, model.dspark_conf_proj_b);
            }
            conf = ggml_sigmoid(ctx0, conf);

            cat_conf = cat_conf ? ggml_concat(ctx0, cat_conf, conf, 1) : conf;
        }

        if (i + 1 < block_drafts) {
            prev = ggml_argmax(ctx0, col);
        }
    }

    // cat is position-major; restore ubatch block-major order
    ggml_tensor * out = ggml_reshape_3d(ctx0, cat, n_vocab, n_blocks, block_drafts);
    out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3)); // [n_vocab, block_drafts, n_blocks]
    out = ggml_reshape_2d(ctx0, out, n_vocab, n_tok);

    if (has_conf) {
        ggml_tensor * conf = ggml_reshape_3d(ctx0, cat_conf, 1, n_blocks, block_drafts);
        conf = ggml_cont(ctx0, ggml_permute(ctx0, conf, 0, 2, 1, 3));
        conf = ggml_reshape_2d(ctx0, conf, 1, n_tok);

        // note: broadcast the [1, n_tok] confidences to n_embd-wide rows to be able to reuse `llama_get_embeddings_nextn`
        conf = ggml_repeat(ctx0, conf, res->t_embd);
        res->t_h_nextn = conf;
        ggml_build_forward_expand(g.gf, conf);
    }

    res->t_logits = out;
    ggml_build_forward_expand(g.gf, out);
}

// DFlash decoder, dual-mode by batch type:
// DFlash decoder, dual-mode by batch type:
//   * embd batch  -> fused target features: project + inject K/V into the cache.
//   * token batch -> noise-block diffusion: attend over [committed, MASK...] to generate draft tokens
template <>
llama_model_dflash::graph<false>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos  = build_inp_pos();

    // optional iSWA: pick the matching attention input
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;

    llm_graph_input_attn_kv      * inp_attn      = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    // drafts for M-RoPE targets use degenerate sections (temporal dim only)
    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto build_rope = [&](ggml_tensor * cur, ggml_tensor * pos) {
        return rope_type == GGML_ROPE_TYPE_MROPE
            ? ggml_rope_multi(ctx0, cur, pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow)
            : ggml_rope_ext(ctx0, cur, pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
    };

    // KV cache injection
    if (ubatch.embd) {
        ggml_tensor * inp_g = build_dflash_inject_input(*this, model, n_embd);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            ggml_tensor * Kcur = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * Vcur = build_lora_mm(layer.wv, inp_g);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur = build_rope(Kcur, inp_pos);
            cb(Kcur, "Kcur_injected", il);
            cb(Vcur, "Vcur_injected", il);

            if (use_iswa) {
                // route each layer's K/V to its sub-cache: SWA layers -> sliding cache, full -> dense
                const bool    is_swa = hparams.is_swa(il);
                const auto  * kv     = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                // rotate K/V into the cache's rotated space
                ggml_tensor * k_rot  = is_swa ? inp_attn_iswa->self_k_rot_swa : inp_attn_iswa->self_k_rot;
                ggml_tensor * v_rot  = is_swa ? inp_attn_iswa->self_v_rot_swa : inp_attn_iswa->self_v_rot;
                if (k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
                }
                if (v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
                }
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, Kcur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, Vcur, v_idxs, il));
            } else {
                // rotate K/V into the cache's rotated space
                if (inp_attn->self_k_rot) {
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
                }
                if (inp_attn->self_v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
                }
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, inp_attn->mctx->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // tok_embd from the target model — llama_model_share_tensors fills model.tok_embd
    // with a drafter-schedulable pointer/copy; the raw ctx_other fallback references the
    // target tensor directly and aborts at sched split when it lives on a device this
    // context cannot schedule (e.g. -sm layer target + pinned drafter)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DFlash decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    // single-graph fused cycle: rows [0, n_inj) are staged KV injections gathered from
    // the carry tensor (their token ids are placeholders, their attention output is
    // discarded), rows [n_inj, n_tokens) are the noise block. The constant n_inj keeps
    // one token-graph topology across generation cycles.
    const int64_t n_inj = cparams.dflash_oneg_stage ? cparams.dflash_oneg_n_inject : 0;
    GGML_ASSERT(n_inj < n_tokens);

    ggml_tensor * inp_g = nullptr;
    if (n_inj > 0) {
        inp_g = build_dflash_staged_enc(*this, model, cparams.dflash_oneg_stage, n_inj);
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    res->t_inp_tokens = inp->tokens;

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * noise_norm = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(noise_norm, "noise_norm", il);

        ggml_tensor * attn_coeff = nullptr;
        ggml_tensor * attn_inp = noise_norm;
        if (layer.dflash2_attn_conv_base) {
            attn_inp = build_dflash2_conv_prepare_tail(*this, noise_norm, n_inj,
                    layer.dflash2_attn_conv_base, layer.dflash2_attn_conv_proj, &attn_coeff);
            cb(attn_inp, "attn_conv_in", il);
        }

        ggml_tensor * Qcur = build_lora_mm(layer.wq, attn_inp);
        ggml_tensor * Kcur;
        ggml_tensor * Vcur;
        if (inp_g) {
            // K/V rows [0, n_inj) come from the encoder output (injection), the rest
            // from the noise tokens — per-row math matches both standalone graphs
            ggml_tensor * tail = ggml_view_2d(ctx0, attn_inp, n_embd, n_tokens - n_inj,
                    attn_inp->nb[1], (size_t) n_inj * attn_inp->nb[1]);
            Kcur = ggml_concat(ctx0, build_lora_mm(layer.wk, inp_g), build_lora_mm(layer.wk, tail), 1);
            Vcur = ggml_concat(ctx0, build_lora_mm(layer.wv, inp_g), build_lora_mm(layer.wv, tail), 1);
        } else {
            Kcur = build_lora_mm(layer.wk, attn_inp);
            Vcur = build_lora_mm(layer.wv, attn_inp);
        }

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        Qcur = build_norm(Qcur, layer.attn_q_norm, NULL, LLM_NORM_RMS, il);
        Kcur = build_norm(Kcur, layer.attn_k_norm, NULL, LLM_NORM_RMS, il);

        Qcur = build_rope(Qcur, inp_pos);
        Kcur = build_rope(Kcur, inp_pos);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // cache-aware, non-causal attention
        ggml_tensor * cur = use_iswa
            ? build_attn(inp_attn_iswa, layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il)
            : build_attn(inp_attn,      layer.wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, layer.attn_sinks, nullptr, kq_scale, il);

        if (attn_coeff) {
            cur = build_dflash2_conv_finish_tail(*this, cur, n_inj,
                    attn_coeff, layer.dflash2_attn_conv_base);
            cb(cur, "attn_conv_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn_coeff = nullptr;
        if (layer.dflash2_ffn_conv_base) {
            cur = build_dflash2_conv_prepare_tail(*this, cur, n_inj,
                    layer.dflash2_ffn_conv_base, layer.dflash2_ffn_conv_proj, &ffn_coeff);
            cb(cur, "ffn_conv_in", il);
        }

        cur = build_ffn(cur,
                layer.ffn_up,   NULL, layer.ffn_up_s,
                layer.ffn_gate, NULL, layer.ffn_gate_s,
                layer.ffn_down, NULL, layer.ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        if (ffn_coeff) {
            cur = build_dflash2_conv_finish_tail(*this, cur, n_inj,
                    ffn_coeff, layer.dflash2_ffn_conv_base);
            cb(cur, "ffn_conv_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = inpL;
    if (n_inj > 0) {
        // only the noise rows produce outputs — drop the injection rows here so the
        // logits/nextn tails line up with the batch's output rows
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens - n_inj, cur->nb[1], (size_t) n_inj * cur->nb[1]);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    res->t_embd = cur;

    // lm_head from the target model — see the tok_embd note above (share_tensors
    // provides a drafter-schedulable model.output; the fallback is not device-safe)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DFlash decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = build_lora_mm(output, cur, output_s);

    // reduced-draft-vocab exports: scatter the draft logits to the target vocabulary via d2t
    if (model.d2t) {
        const int64_t n_draft_vocab = cur->ne[0];
        const int64_t n_outputs     = cur->ne[1];
        const int64_t n_vocab       = (int64_t) model.vocab.n_tokens();

        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64);
        GGML_ASSERT(model.d2t->ne[0] == n_draft_vocab);

        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,             n_draft_vocab, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_draft_vocab, 1,             1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // DSpark: bias the draft logits with the Markov head
    if (model.dspark_markov_w1) {
        ggml_tensor * tok = inp_tokens;
        if (n_inj > 0) {
            tok = ggml_view_1d(ctx0, inp_tokens, n_tokens - n_inj, (size_t) n_inj * inp_tokens->nb[0]);
        }
        build_dspark_markov_head(*this, model, tok);
    }

    if (model.dflash2_selector_hidden) {
        ggml_tensor * selector_tokens = inp_tokens;
        if (n_inj > 0) {
            selector_tokens = ggml_view_1d(ctx0, inp_tokens, n_tokens - n_inj,
                    (size_t) n_inj * inp_tokens->nb[0]);
        }
        if (!llm_build_dflash2_selector(*this, model, selector_tokens, res->t_embd, res->t_logits)) {
            build_dflash_draft_argmax(*this);
        }
    } else {
        build_dflash_draft_argmax(*this);
    }

}

// DSV4 DSpark decoder, dual-mode by batch type (see the DFlash decoder above):
//   * embd batch  -> project main_x through each stage's wkv and inject K into the ring cache
//   * token batch -> noise block through 3 full DSV4 stages (hc + MLA + MoE), markov + confidence heads
llama_model_dflash::graph_dsv4::graph_dsv4(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;

    ggml_tensor * inp_pos = build_inp_pos();

    llm_graph_input_attn_k_iswa * inp_attn = build_attn_inp_k_iswa();

    // KV cache injection: fused target features from the encoder
    if (ubatch.embd) {
        ggml_tensor * inp_g = build_dflash_inject_input(*this, model, n_embd);

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];

            // main-track KV: kv_norm(wkv(main_x)) with rope on the trailing dims, same
            // rope parameters as the uncompressed layers in build_attention_impl
            ggml_tensor * kv = build_lora_mm(layer.wkv, inp_g);
            kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_tokens);

            kv = ggml_rope_ext(ctx0, kv, inp_pos, nullptr, n_embd_head_rope, rope_type, 0,
                    freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            kv = ggml_rope_set_offset(kv, n_embd_head_nope);
            cb(kv, "kv_injected", il);

            if (inp_attn->self_k_rot_swa) {
                kv = llama_mul_mat_hadamard(ctx0, kv, inp_attn->self_k_rot_swa);
            }
            ggml_build_forward_expand(gf, inp_attn->mctx->get_swa()->cpy_k(ctx0, kv, inp_attn->get_k_idxs_swa(), il));
        }

        res->t_embd = inp_g;

        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    // single-graph fused cycle (see the plain-backbone token branch): rows [0, n_inj)
    // are staged injections gathered from the carry tensor. They are spliced into the
    // post-attn-norm stream each layer, so their MLA latent takes the SAME
    // wkv/kv_norm/rope path as the standalone inject graph and the single cpy_k write
    // inside build_attn (the cached K is read back as V — one write, no K/V divergence).
    // Their q/attention outputs are row-local garbage, dropped by the output slice.
    const int64_t n_inj = cparams.dflash_oneg_stage ? cparams.dflash_oneg_n_inject : 0;
    GGML_ASSERT(n_inj < n_tokens);

    ggml_tensor * inp_g = nullptr;
    if (n_inj > 0) {
        inp_g = build_dflash_staged_enc(*this, model, cparams.dflash_oneg_stage, n_inj);
    }

    // tok_embd from the target model — llama_model_share_tensors fills model.tok_embd
    // with a drafter-schedulable pointer/copy (the ctx_other fallback is not device-safe,
    // see the DFlash decoder above)
    auto * tok_embd = model.tok_embd;
    if (tok_embd == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);

        GGML_ASSERT(model_other->tok_embd != nullptr && "DSpark decoder requires the target model's token embeddings");
        tok_embd = model_other->tok_embd;
    }

    auto inp = std::make_unique<llm_graph_input_embd>(n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    ggml_tensor * inp_tokens = inp->tokens;

    ggml_tensor * inpL = ggml_get_rows(ctx0, tok_embd, inp->tokens);
    cb(inpL, "inp_noise_embd", -1);

    res->add_input(std::move(inp));

    const int64_t hc = hparams.dsv4_hc_mult;
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * cur = build_hc_pre(inpL,
                layer.hc_attn_fn,
                layer.hc_attn_scale,
                layer.hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        if (inp_g) {
            // rows [0, n_inj) take the injection math: wkv applies to the encoder
            // output directly (no attn_norm), as in the standalone inject graph
            ggml_tensor * tail = ggml_view_2d(ctx0, cur, n_embd, n_tokens - n_inj,
                    cur->nb[1], (size_t) n_inj * cur->nb[1]);
            cur = ggml_concat(ctx0, inp_g, tail, 1);
        }

        cur = build_attention(model, inp_attn, cur, inp_pos, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                layer.hc_ffn_fn,
                layer.hc_ffn_scale,
                layer.hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    if (n_inj > 0) {
        // only the noise rows produce outputs — drop the injection rows here so the
        // logits/nextn tails line up with the batch's output rows
        cur = ggml_view_2d(ctx0, cur, n_embd, n_tokens - n_inj, cur->nb[1], (size_t) n_inj * cur->nb[1]);
    }

    // confidence head input: the reference scores the pre-norm collapsed hidden state
    res->t_embd = cur;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head from the target model — see the tok_embd note above (share_tensors
    // provides a drafter-schedulable model.output; the fallback is not device-safe)
    auto * output   = model.output;
    auto * output_s = model.output_s;
    if (output == nullptr) {
        GGML_ASSERT(cparams.ctx_other != nullptr);
        const auto * model_other = llama_get_model(cparams.ctx_other);
        GGML_ASSERT(model_other->output != nullptr && "DSpark decoder requires the target model's output projection");
        output   = model_other->output;
        output_s = model_other->output_s;
    }

    cur = build_lora_mm(output, cur, output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    if (model.dspark_markov_w1) {
        ggml_tensor * tok = inp_tokens;
        if (n_inj > 0) {
            tok = ggml_view_1d(ctx0, inp_tokens, n_tokens - n_inj, (size_t) n_inj * inp_tokens->nb[0]);
        }
        build_dspark_markov_head(*this, model, tok);
    }

    build_dflash_draft_argmax(*this);
}
