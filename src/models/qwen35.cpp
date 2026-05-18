#include "models.h"

#include "llama-memory-recurrent.h"
#include "llama-context.h"

llm_build_qwen35::llm_build_qwen35(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int n_main_layers = n_layer - (int)hparams.nextn_predict_layers;

    for (int il = 0; il < n_main_layers; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_main_layers - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // Dense FFN layer - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);

    // Gated on mtp_enabled to avoid baseline penalty when not active
    if (hparams.nextn_predict_layers > 0 && n_outputs == n_tokens && cparams.mtp_enabled) {
        const int mtp_il = n_main_layers;
        const int64_t n_embd_head = hparams.n_embd_head_v();
        const int64_t n_head_q    = hparams.n_head();
        const int64_t n_head_kv   = hparams.n_head_kv();
        const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
        const int n_chain = 2; // chain depth 2 = 3 total MTP predictions

        ggml_tensor * mtp_norm_w = model.layers[mtp_il].nextn.shared_head_norm
                                 ? model.layers[mtp_il].nextn.shared_head_norm
                                 : model.output_norm;

        const int64_t mtp_vocab = std::min(model.output->ne[1], (int64_t)32768);
        ggml_tensor * lm_head_reduced = ggml_view_2d(ctx0, model.output,
            model.output->ne[0], mtp_vocab, model.output->nb[1], 0);

        // Chain position input: MRoPE positions for chain steps on the LAST token
        // Chain always runs on last token only (avoids cross-contamination with multi-token batches)
        ggml_tensor * chain_pos_all = nullptr;
        const bool build_chain = (n_chain > 0);
        if (build_chain) {
            auto inp_chain = std::make_unique<llm_graph_input_pos_mtp_chain>(n_chain, 1);
            inp_chain->chain_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_chain * 4);
            ggml_set_input(inp_chain->chain_pos);
            chain_pos_all = inp_chain->chain_pos;
            res->add_input(std::move(inp_chain));
        }

        struct mtp_step_result { ggml_tensor * hidden; ggml_tensor * K; ggml_tensor * V; };

        auto mtp_chain_step = [&](ggml_tensor * projected, ggml_tensor * pos_tensor, int64_t n_tok,
                            ggml_tensor * K_accum, ggml_tensor * V_accum) -> mtp_step_result {
            ggml_tensor * cur = build_norm(projected, model.layers[mtp_il].attn_norm, nullptr, LLM_NORM_RMS, mtp_il);

            ggml_tensor * Qfull = build_lora_mm(model.layers[mtp_il].wq, cur);
            ggml_tensor * Q = ggml_view_3d(ctx0, Qfull, n_embd_head, n_head_q, n_tok,
                ggml_element_size(Qfull) * n_embd_head * 2,
                ggml_element_size(Qfull) * n_embd_head * 2 * n_head_q, 0);
            Q = build_norm(Q, model.layers[mtp_il].attn_q_norm, nullptr, LLM_NORM_RMS, mtp_il);

            ggml_tensor * K = build_lora_mm(model.layers[mtp_il].wk, cur);
            K = ggml_reshape_3d(ctx0, K, n_embd_head, n_head_kv, n_tok);
            K = build_norm(K, model.layers[mtp_il].attn_k_norm, nullptr, LLM_NORM_RMS, mtp_il);

            ggml_tensor * gate = ggml_view_3d(ctx0, Qfull, n_embd_head, n_head_q, n_tok,
                ggml_element_size(Qfull) * n_embd_head * 2,
                ggml_element_size(Qfull) * n_embd_head * 2 * n_head_q,
                ggml_element_size(Qfull) * n_embd_head);
            gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head_q, n_tok);

            ggml_tensor * V = build_lora_mm(model.layers[mtp_il].wv, cur);
            V = ggml_reshape_3d(ctx0, V, n_embd_head, n_head_kv, n_tok);

            Q = ggml_rope_multi(ctx0, Q, pos_tensor, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            K = ggml_rope_multi(ctx0, K, pos_tensor, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            ggml_tensor * K_full = K_accum ? ggml_concat(ctx0, K_accum, K, 2) : K;
            ggml_tensor * V_full = V_accum ? ggml_concat(ctx0, V_accum, V, 2) : V;

            cur = build_attn_mha(Q, K_full, V_full, nullptr, nullptr, nullptr, nullptr, kq_scale, mtp_il);

            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
            cur = build_lora_mm(model.layers[mtp_il].wo, cur);
            cur = ggml_add(ctx0, cur, projected);

            ggml_tensor * ffn_res = cur;
            cur = build_layer_ffn(build_norm(cur, model.layers[mtp_il].attn_post_norm, nullptr, LLM_NORM_RMS, mtp_il), mtp_il);
            cur = ggml_add(ctx0, cur, ffn_res);

            return { cur, ggml_cont(ctx0, K), ggml_cont(ctx0, V) };
        };

        // === Base MTP step (depth 0): predict position N+1 using KV-cached attention ===
        ggml_tensor * greedy = ggml_argmax(ctx0, res->t_logits);
        ggml_tensor * emb = ggml_get_rows(ctx0, model.tok_embd, greedy);
        ggml_tensor * enorm = build_norm(emb, model.layers[mtp_il].nextn.enorm, nullptr, LLM_NORM_RMS, mtp_il);
        ggml_tensor * hnorm = build_norm(inpL, model.layers[mtp_il].nextn.hnorm, nullptr, LLM_NORM_RMS, mtp_il);
        ggml_tensor * projected_base = build_lora_mm(model.layers[mtp_il].nextn.eh_proj,
            ggml_concat(ctx0, enorm, hnorm, 0));

        ggml_tensor * mtp_cur;
        ggml_tensor * base_K;
        ggml_tensor * base_V;
        {
            ggml_tensor * cur = build_norm(projected_base, model.layers[mtp_il].attn_norm, nullptr, LLM_NORM_RMS, mtp_il);

            ggml_tensor * Qfull = build_lora_mm(model.layers[mtp_il].wq, cur);
            ggml_tensor * Q = ggml_view_3d(ctx0, Qfull, n_embd_head, n_head_q, n_tokens,
                ggml_element_size(Qfull) * n_embd_head * 2,
                ggml_element_size(Qfull) * n_embd_head * 2 * n_head_q, 0);
            Q = build_norm(Q, model.layers[mtp_il].attn_q_norm, nullptr, LLM_NORM_RMS, mtp_il);

            ggml_tensor * K = build_lora_mm(model.layers[mtp_il].wk, cur);
            K = ggml_reshape_3d(ctx0, K, n_embd_head, n_head_kv, n_tokens);
            K = build_norm(K, model.layers[mtp_il].attn_k_norm, nullptr, LLM_NORM_RMS, mtp_il);

            ggml_tensor * gate = ggml_view_3d(ctx0, Qfull, n_embd_head, n_head_q, n_tokens,
                ggml_element_size(Qfull) * n_embd_head * 2,
                ggml_element_size(Qfull) * n_embd_head * 2 * n_head_q,
                ggml_element_size(Qfull) * n_embd_head);
            gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head_q, n_tokens);

            ggml_tensor * V = build_lora_mm(model.layers[mtp_il].wv, cur);
            V = ggml_reshape_3d(ctx0, V, n_embd_head, n_head_kv, n_tokens);

            Q = ggml_rope_multi(ctx0, Q, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            K = ggml_rope_multi(ctx0, K, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            base_K = ggml_cont(ctx0, K);
            base_V = ggml_cont(ctx0, V);

            cur = build_attn(inp->get_attn(), nullptr, nullptr, nullptr,
                    Q, K, V, nullptr, nullptr, nullptr, kq_scale, mtp_il);

            cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
            cur = build_lora_mm(model.layers[mtp_il].wo, cur);
            cur = ggml_add(ctx0, cur, projected_base);

            ggml_tensor * ffn_res = cur;
            cur = build_layer_ffn(build_norm(cur, model.layers[mtp_il].attn_post_norm, nullptr, LLM_NORM_RMS, mtp_il), mtp_il);
            cur = ggml_add(ctx0, cur, ffn_res);

            mtp_cur = cur;
        }

        ggml_tensor * mtp_logits = build_lora_mm(lm_head_reduced,
            build_norm(mtp_cur, mtp_norm_w, nullptr, LLM_NORM_RMS, mtp_il));
        cb(mtp_logits, "result_output_mtp", -1);

        ggml_tensor * mtp_logits_safe = ggml_cont(ctx0, mtp_logits);
        ggml_set_name(mtp_logits_safe, "result_output_mtp_safe");
        res->t_logits_mtp = mtp_logits_safe;
        ggml_build_forward_expand(gf, mtp_logits_safe);

        // === MTP chain: predict N+2, N+3, ... with in-graph K/V accumulation on LAST token ===
        if (build_chain) {
            const int64_t last_off = (n_tokens - 1);
            ggml_tensor * last_hidden = ggml_view_2d(ctx0, mtp_cur, n_embd, 1,
                mtp_cur->nb[1], last_off * ggml_element_size(mtp_cur) * n_embd);
            ggml_tensor * last_K = ggml_cont(ctx0, ggml_view_3d(ctx0, base_K, n_embd_head, n_head_kv, 1,
                base_K->nb[1], base_K->nb[2], last_off * base_K->nb[2]));
            ggml_tensor * last_V = ggml_cont(ctx0, ggml_view_3d(ctx0, base_V, n_embd_head, n_head_kv, 1,
                base_V->nb[1], base_V->nb[2], last_off * base_V->nb[2]));
            ggml_tensor * last_logits = ggml_view_2d(ctx0, mtp_logits, mtp_vocab, 1,
                mtp_logits->nb[1], last_off * ggml_element_size(mtp_logits) * mtp_vocab);

            ggml_tensor * K_accum = last_K;
            ggml_tensor * V_accum = last_V;
            ggml_tensor * chain_logits = last_logits;
            ggml_tensor * chain_hidden = last_hidden;

            for (int ck = 0; ck < n_chain; ++ck) {
                ggml_tensor * ck_greedy = ggml_argmax(ctx0, chain_logits);
                ggml_tensor * ck_emb = ggml_get_rows(ctx0, model.tok_embd, ck_greedy);
                ggml_tensor * ck_enorm = build_norm(ck_emb, model.layers[mtp_il].nextn.enorm, nullptr, LLM_NORM_RMS, mtp_il);
                ggml_tensor * ck_hnorm = build_norm(chain_hidden, model.layers[mtp_il].nextn.hnorm, nullptr, LLM_NORM_RMS, mtp_il);
                ggml_tensor * ck_proj = build_lora_mm(model.layers[mtp_il].nextn.eh_proj,
                    ggml_concat(ctx0, ck_enorm, ck_hnorm, 0));

                ggml_tensor * ck_pos = ggml_view_1d(ctx0, chain_pos_all,
                    4, ck * 4 * ggml_element_size(chain_pos_all));

                auto [ck_hidden, ck_K, ck_V] = mtp_chain_step(ck_proj, ck_pos, 1, K_accum, V_accum);

                K_accum = ggml_concat(ctx0, K_accum, ck_K, 2);
                V_accum = ggml_concat(ctx0, V_accum, ck_V, 2);
                chain_hidden = ck_hidden;

                chain_logits = build_lora_mm(lm_head_reduced,
                    build_norm(ck_hidden, mtp_norm_w, nullptr, LLM_NORM_RMS, mtp_il));
                res->t_logits_mtp_chain[ck] = chain_logits;
                ggml_build_forward_expand(gf, chain_logits);
            }
        }

    }
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen35::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llm_build_qwen35::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen35::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Per-layer n_head_kv (may differ from scalar in RYS models)
    const int64_t n_head_kv_il = hparams.n_head_kv(il);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv_il, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv_il, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llm_build_qwen35::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    ggml_tensor * beta_presigmoid = beta; // save for GPU tape capture (pre-sigmoid)
    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    cb(qkv_mixed, "qkv_mixed_pretranspose", il);  // tape captures contiguous [conv_channels, n_tokens]
    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Tree-mode conv kernel is sized for a single sequence — skip on multi-seq batches
    // (see build_delta_net() for the same constraint on the attention tree kernel).
    const bool tree_mode = (tree_parent_ids != nullptr && n_seq_tokens > 1 && n_seqs == 1 &&
                             n_seq_tokens <= ggml_nelements(tree_parent_ids));

    // Update convolution state cache
    // Note: in tree mode with linear parents, last d_conv-1 columns are correct.
    // For actual tree branches, tree_rollback will overwrite with correct state.
    {
        ggml_tensor * last_conv_states =
            ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                         conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
        cb(last_conv_states, "last_conv_states", il);

        ggml_tensor * state_update_target =
            ggml_view_2d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels, n_seqs, conv_states_all->nb[1],
                         kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
        cb(state_update_target, "state_update_target", il);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));
    }

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper;
    if (tree_mode) {
        conv_output_proper = ggml_ssm_conv_tree(ctx0, conv_input, conv_kernel, tree_parent_ids);
        cb(conv_output_proper, "conv_output_tree", il);
    } else {
        conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);
    }

    ggml_tensor * conv_output_silu;
    if (tree_mode) {
        conv_output_silu = conv_output_proper; // silu already fused in tree conv kernel
    } else {
        conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    }
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // GPU tape: copy k/v/gate/beta directly to persistent per-slot GPU buffers.
    // Per-seq loop extracts each seq's 3D slice from the 4D source tensors.
    // For n_seqs==1 this degenerates to a single copy at offset 0.
    if (cparams.tape_gpu_n_seqs > 0) {
        for (int s = 0; s < (int)n_seqs && s < cparams.tape_gpu_n_seqs; ++s) {
            auto * tgpu = cparams.tape_gpu_seqs[s];
            if (!tgpu) continue;

            int li = -1;
            for (int i = 0; i < (int)tgpu->layer_ids.size(); ++i) {
                if (tgpu->layer_ids[i] == il) { li = i; break; }
            }
            if (li < 0 || n_seq_tokens > tgpu->max_tokens) continue;

            auto & tl = tgpu->layers[li];

            // extract per-seq 3D slice from 4D [dim, heads, n_seq_tokens, n_seqs]
            ggml_tensor * k_slice = ggml_view_3d(ctx0, k_conv,
                k_conv->ne[0], k_conv->ne[1], n_seq_tokens,
                k_conv->nb[1], k_conv->nb[2], s * k_conv->nb[3]);
            ggml_tensor * v_slice = ggml_view_3d(ctx0, v_conv,
                v_conv->ne[0], v_conv->ne[1], n_seq_tokens,
                v_conv->nb[1], v_conv->nb[2], s * v_conv->nb[3]);
            ggml_tensor * g_slice = ggml_view_3d(ctx0, gate,
                gate->ne[0], gate->ne[1], n_seq_tokens,
                gate->nb[1], gate->nb[2], s * gate->nb[3]);
            ggml_tensor * b_slice = ggml_view_3d(ctx0, beta_presigmoid,
                beta_presigmoid->ne[0], beta_presigmoid->ne[1], n_seq_tokens,
                beta_presigmoid->nb[1], beta_presigmoid->nb[2], s * beta_presigmoid->nb[3]);

            // make contiguous (v_conv is a non-contiguous view)
            ggml_tensor * k_cont = ggml_cont(ctx0, k_slice);
            ggml_tensor * v_cont = ggml_cont(ctx0, v_slice);
            ggml_tensor * g_cont = ggml_cont(ctx0, g_slice);
            ggml_tensor * b_cont = ggml_cont(ctx0, b_slice);

            // destination views (sliced to actual n_tokens from max_tokens allocation)
            ggml_tensor * k_dst = ggml_view_3d(ctx0, tl.k,
                tl.k->ne[0], tl.k->ne[1], (int64_t)n_seq_tokens,
                tl.k->nb[1], tl.k->nb[2], 0);
            ggml_tensor * v_dst = ggml_view_3d(ctx0, tl.v,
                tl.v->ne[0], tl.v->ne[1], (int64_t)n_seq_tokens,
                tl.v->nb[1], tl.v->nb[2], 0);
            ggml_tensor * g_dst = ggml_view_3d(ctx0, tl.gate,
                tl.gate->ne[0], tl.gate->ne[1], (int64_t)n_seq_tokens,
                tl.gate->nb[1], tl.gate->nb[2], 0);
            ggml_tensor * b_dst = ggml_view_3d(ctx0, tl.beta,
                tl.beta->ne[0], tl.beta->ne[1], (int64_t)n_seq_tokens,
                tl.beta->nb[1], tl.beta->nb[2], 0);

            ggml_build_forward_expand(gf, ggml_cpy(ctx0, k_cont, k_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, v_cont, v_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, g_cont, g_dst));
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, b_cont, b_dst));
        }
    }

    auto attn_out = build_delta_net(q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    // Update the recurrent states (tree_rollback will overwrite if needed)
    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_2d(ctx0, ssm_states_all, hparams.n_embd_s(), n_seqs, ssm_states_all->nb[1],
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen35::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Qwen3.5 does not use MoE FFN
    GGML_ASSERT(model.layers[il].ffn_gate_inp == nullptr);

    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}
