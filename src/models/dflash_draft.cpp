#include "models.h"

#include <algorithm>
#include <atomic>
#include <vector>

// ---------------------------------------------------------------------------
// llama_model_dflash_draft — model class methods
// ---------------------------------------------------------------------------

void llama_model_dflash_draft::load_arch_hparams(llama_model_loader & ml) {
    auto & hparams = this->hparams;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,           hparams.dflash_block_size,        false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID,        hparams.dflash_mask_token_id,     false);
    ml.get_key(LLM_KV_DFLASH_N_TARGET_FEATURES,    hparams.dflash_n_target_features, false);

    {
        const std::string key = ml.llm_kv(LLM_KV_DFLASH_TARGET_LAYER_IDS);
        const int64_t kid = gguf_find_key(ml.metadata, key.c_str());
        if (kid >= 0) {
            const size_t n = gguf_get_arr_n(ml.metadata, kid);
            hparams.dflash_n_target_layers = std::min((uint32_t) n, (uint32_t) 8);
            const void * data = gguf_get_arr_data(ml.metadata, kid);
            for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
                hparams.dflash_target_layer_ids[i] = ((const uint32_t *) data)[i];
            }
        }
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer(), false);
    }
}

void llama_model_dflash_draft::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    dflash_fc          = create_tensor(tn(LLM_TENSOR_DFLASH_FC,          "weight"), {(int64_t) hparams.dflash_n_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        // Recurrent layers (DeltaNet) — optional, probed by existence
        layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {0}, TENSOR_NOT_REQUIRED);
        if (layer.ssm_conv1d) {
            int64_t d_conv  = layer.ssm_conv1d->ne[0];
            int64_t d_inner = layer.ssm_conv1d->ne[1];

            layer.ssm_in   = create_tensor(tn(LLM_TENSOR_SSM_IN,   "weight", i), {n_embd, d_inner},     TENSOR_NOT_REQUIRED);
            layer.ssm_out  = create_tensor(tn(LLM_TENSOR_SSM_OUT,  "weight", i), {d_inner / 2, n_embd}, TENSOR_NOT_REQUIRED);
            layer.ssm_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {d_inner / 2},         TENSOR_NOT_REQUIRED);

            GGML_UNUSED(d_conv);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash_draft::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<llm_build_dflash_draft>(*this, params);
}

// Max cross-attention context for DFlash drafter (caps VRAM growth).
// Override with GGML_DFLASH_MAX_CTX env var. 0 = unlimited.
static int64_t dflash_max_cross_ctx() {
    static const int64_t val = [] {
        const char * e = getenv("GGML_DFLASH_MAX_CTX");
        return e ? (int64_t) atoi(e) : (int64_t) 4096;
    }();
    return val;
}

// DFlash drafter custom graph input
// Holds the target hidden states, context positions, and asymmetric non-causal attention mask
class llm_graph_input_dflash : public llm_graph_input_i {
public:
    llm_graph_input_dflash(const llama_cross * cross, int64_t ctx_len, int64_t n_block, uint32_t n_swa)
        : cross(cross), ctx_len(ctx_len), n_block(n_block), n_swa(n_swa) {}

    void set_input(const llama_ubatch * ubatch) override;

    // set_input() rewrites every buffer (hidden window, positions, masks) from the live
    // cross state on each call, so the graph is reusable whenever the baked shapes match.
    // Without this override the base class refuses reuse and the drafter rebuilds its
    // graph (and re-captures CUDA graphs) on every draft call.
    bool can_reuse(const llm_graph_params & params) override {
        if (cross != params.cross || n_block != (int64_t) params.ubatch.n_tokens) {
            return false;
        }
        // ctx_len is baked into every tensor here but derived from live cross state at
        // build time — recompute the builder's formula and refuse reuse across a window
        // change instead of trusting the set-cross path to have invalidated the graph.
        const int n_slots = std::clamp(params.cparams.dflash_n_slots, 1, (int) LLAMA_DFLASH_MAX_SLOTS);
        int64_t want_ctx;
        if (n_slots == 1) {
            want_ctx = (cross && cross->n_enc > 0) ? cross->n_enc : (int64_t) LLAMA_DFLASH_PER_SLOT_CTX;
            const int64_t max_ctx = dflash_max_cross_ctx();
            if (max_ctx > 0 && want_ctx > max_ctx) {
                want_ctx = max_ctx;
            }
        } else {
            want_ctx = (int64_t) n_slots * LLAMA_DFLASH_PER_SLOT_CTX;
        }
        return want_ctx == ctx_len;
    }

    ggml_tensor * target_hidden     = nullptr; // [n_target_features, ctx_len]
    ggml_tensor * pos_ctx           = nullptr; // [ctx_len]
    ggml_tensor * kq_mask           = nullptr; // [ctx_len + n_block, n_block, 1, 1]
    ggml_tensor * kq_mask_cnv       = nullptr;
    // Only allocated when hparams.is_swa_any(); same shape as kq_mask
    ggml_tensor * kq_mask_swa       = nullptr;
    ggml_tensor * kq_mask_swa_cnv   = nullptr;

    const llama_cross * cross;
    int64_t ctx_len;
    int64_t n_block;
    uint32_t n_swa;
};

void llm_graph_input_dflash::set_input(const llama_ubatch * ubatch) {
    const int n_seqs = (ubatch && ubatch->n_seqs_unq > 1) ? (int) ubatch->n_seqs_unq : 1;

    if (n_seqs == 1) {
        // === Single-slot path ===
        // resolve cross data for the active seq (GPU path preferred, CPU fallback)
        const float * src_data  = nullptr;
        const void *  src_gpu   = nullptr;
        int64_t       src_n_enc  = 0;
        int64_t       src_n_real = 0;
        if (cross) {
            llama_seq_id active_seq = -1;
            if (ubatch && ubatch->n_seqs_unq > 0 && ubatch->seq_id_unq) {
                active_seq = ubatch->seq_id_unq[0];
            }
            if (active_seq >= 0) {
                auto it = cross->v_embd_per_seq.find(active_seq);
                if (it != cross->v_embd_per_seq.end()) {
                    if (it->second.v_embd_gpu) {
                        src_gpu    = it->second.v_embd_gpu;
                        src_n_enc  = it->second.n_enc;
                        src_n_real = it->second.v_embd_gpu_n_enc_real;
                    } else if (!it->second.v_embd.empty()) {
                        src_data   = it->second.v_embd.data();
                        src_n_enc  = it->second.n_enc;
                        src_n_real = it->second.n_enc_real;
                    }
                }
            }
            if (!src_data && !src_gpu) {
                if (cross->v_embd_gpu) {
                    src_gpu    = cross->v_embd_gpu;
                    src_n_enc  = cross->n_enc;
                    src_n_real = cross->v_embd_gpu_n_enc_real;
                } else if (!cross->v_embd.empty()) {
                    src_data   = cross->v_embd.data();
                    src_n_enc  = cross->n_enc;
                    src_n_real = cross->n_enc_real;
                }
            }
        }

        // Sliding window: if src has more tokens than ctx_len, take the most recent
        const int64_t src_real = src_n_real > 0 ? src_n_real : 0;
        const int64_t n_copy  = std::min(src_real, ctx_len);
        const int64_t win_off = (src_real > ctx_len) ? (src_real - ctx_len) : 0;

        if (target_hidden && (src_data || src_gpu) && n_copy > 0) {
            const int64_t n_feat = cross->n_embd;
            const size_t copy_bytes  = (size_t) n_feat * (size_t) n_copy * sizeof(float);
            const size_t tensor_bytes = ggml_nbytes(target_hidden);
            const size_t actual_bytes = std::min(copy_bytes, tensor_bytes);

            if (src_gpu && cross->fn_set_tensor_d2d) {
                // GPU D2D path
                const void * gpu_src = (const char *)src_gpu + (size_t)win_off * n_feat * sizeof(float);
                cross->fn_set_tensor_d2d(target_hidden->data, gpu_src, 0, actual_bytes);
            } else {
                // CPU H2D path (fallback)
                const float * src = src_data + win_off * n_feat;
                ggml_backend_tensor_set(target_hidden, src, 0, actual_bytes);
            }
            if (copy_bytes < tensor_bytes) {
                ggml_backend_tensor_memset(target_hidden, 0, copy_bytes, tensor_bytes - copy_bytes);
            }
        } else if (target_hidden) {
            ggml_backend_tensor_memset(target_hidden, 0, 0, ggml_nbytes(target_hidden));
        }

        const int64_t n_real = n_copy;

        if (pos_ctx && pos_ctx->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
            int32_t * data = (int32_t *) pos_ctx->data;
            for (int64_t i = 0; i < ctx_len; ++i) {
                data[i] = (i < n_real) ? (int32_t) (win_off + i) : 0;
            }
        }

        if (kq_mask && kq_mask->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
            float * data = (float *) kq_mask->data;
            const int64_t n_kv = ctx_len + n_block;
            for (int64_t q = 0; q < n_block; ++q) {
                for (int64_t k = 0; k < n_kv; ++k) {
                    if (k >= n_real && k < ctx_len) {
                        data[q * n_kv + k] = -INFINITY;
                    } else {
                        data[q * n_kv + k] = 0.0f;
                    }
                }
            }
        }

        if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask_swa->buffer));
            float * data = (float *) kq_mask_swa->data;
            const int64_t n_kv   = ctx_len + n_block;
            const int32_t window = (int32_t) n_swa;
            const bool    have_pos = (ubatch != nullptr) && (ubatch->pos != nullptr)
                                   && ((int64_t) ubatch->n_tokens >= n_block);
            for (int64_t q = 0; q < n_block; ++q) {
                const int32_t q_pos = have_pos ? ubatch->pos[q] : (int32_t) (n_real + q);
                for (int64_t k = 0; k < n_kv; ++k) {
                    float v = 0.0f;
                    if (k < n_real) {
                        if (q_pos - (int32_t) k > window) v = -INFINITY;
                    } else if (k < ctx_len) {
                        v = -INFINITY;
                    } else {
                        const int64_t b_k = k - ctx_len;
                        if (b_k > q) v = -INFINITY;
                    }
                    data[q * n_kv + k] = v;
                }
            }
        }
    } else {
        // === Multi-slot batched draft path ===
        // Pack each slot's cross data at per-slot offsets in target_hidden.
        // Build per-slot isolating masks so each slot's block queries only
        // attend to that slot's cross keys and block keys.
        GGML_ASSERT(ctx_len % n_seqs == 0);
        GGML_ASSERT(n_block % n_seqs == 0);
        const int per_slot_ctx    = (int)(ctx_len / n_seqs);
        const int n_seq_tokens    = (int)(n_block / n_seqs);
        const size_t n_feat       = cross ? (size_t) cross->n_embd : 0;

        // collect per-slot cross data
        struct { const float * data; int64_t n_real; } slot_info[LLAMA_DFLASH_MAX_SLOTS] = {};
        for (int s = 0; s < n_seqs && s < LLAMA_DFLASH_MAX_SLOTS; s++) {
            llama_seq_id seq = ubatch->seq_id_unq[s];
            if (!cross) { continue; }
            auto it = cross->v_embd_per_seq.find(seq);
            if (it != cross->v_embd_per_seq.end() && !it->second.v_embd.empty()) {
                slot_info[s] = { it->second.v_embd.data(), it->second.n_enc_real };
            }
        }

        // pack target_hidden (f16): slot s at offset [s * per_slot_ctx, (s+1) * per_slot_ctx)
        // Sliding window: if a slot has more tokens than per_slot_ctx, take the most recent.
        int64_t slot_win_off[LLAMA_DFLASH_MAX_SLOTS] = {};
        int64_t slot_n_copy[LLAMA_DFLASH_MAX_SLOTS]  = {};
        for (int s = 0; s < n_seqs && s < LLAMA_DFLASH_MAX_SLOTS; s++) {
            const int64_t nr = slot_info[s].n_real > 0 ? slot_info[s].n_real : 0;
            slot_n_copy[s] = std::min(nr, (int64_t) per_slot_ctx);
            slot_win_off[s] = (nr > per_slot_ctx) ? (nr - per_slot_ctx) : 0;
        }

        if (target_hidden && n_feat > 0) {
            ggml_backend_tensor_memset(target_hidden, 0, 0, ggml_nbytes(target_hidden));
            for (int s = 0; s < n_seqs; s++) {
                if (!slot_info[s].data || slot_n_copy[s] <= 0) { continue; }
                const float * src = slot_info[s].data + slot_win_off[s] * n_feat;
                const size_t copy_bytes = n_feat * (size_t) slot_n_copy[s] * sizeof(float);
                const size_t dst_offset = (size_t) s * (size_t) per_slot_ctx * n_feat * sizeof(float);
                ggml_backend_tensor_set(target_hidden, src, dst_offset, copy_bytes);
            }
        }

        // pos_ctx: per-slot position patterns with window offset
        if (pos_ctx && pos_ctx->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
            int32_t * data = (int32_t *) pos_ctx->data;
            for (int s = 0; s < n_seqs; s++) {
                const int64_t nc  = slot_n_copy[s];
                const int64_t wo  = slot_win_off[s];
                const int64_t off = (int64_t) s * per_slot_ctx;
                for (int64_t i = 0; i < per_slot_ctx; i++) {
                    data[off + i] = (i < nc) ? (int32_t) (wo + i) : 0;
                }
            }
        }

        // kq_mask: per-slot isolation — query in slot S sees only slot S's
        // cross keys (real only) and slot S's block keys (causal).
        if (kq_mask && kq_mask->buffer) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
            float * data = (float *) kq_mask->data;
            const int64_t n_kv = ctx_len + n_block;
            for (int64_t q = 0; q < n_block; q++) {
                const int qs = (int)(q / n_seq_tokens);
                const int ql = (int)(q % n_seq_tokens);
                const int64_t nc = slot_n_copy[qs];
                for (int64_t k = 0; k < n_kv; k++) {
                    float v = -INFINITY;
                    if (k < ctx_len) {
                        const int ks = (int)(k / per_slot_ctx);
                        const int kl = (int)(k % per_slot_ctx);
                        if (ks == qs && kl < nc) { v = 0.0f; }
                    } else {
                        const int bi = (int)(k - ctx_len);
                        const int ks = bi / n_seq_tokens;
                        const int kl = bi % n_seq_tokens;
                        if (ks == qs && kl <= ql) { v = 0.0f; }
                    }
                    data[q * n_kv + k] = v;
                }
            }
        }

        // kq_mask_swa: per-slot isolation + sliding window on cross keys
        if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
            GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask_swa->buffer));
            float * data = (float *) kq_mask_swa->data;
            const int64_t n_kv   = ctx_len + n_block;
            const int32_t window = (int32_t) n_swa;
            const bool have_pos  = (ubatch->pos != nullptr)
                                 && ((int64_t) ubatch->n_tokens >= n_block);
            for (int64_t q = 0; q < n_block; q++) {
                const int qs = (int)(q / n_seq_tokens);
                const int ql = (int)(q % n_seq_tokens);
                const int64_t nc = slot_n_copy[qs];
                const int64_t full_nr = slot_info[qs].n_real > 0 ? slot_info[qs].n_real : 0;
                const int32_t q_pos = have_pos ? ubatch->pos[q] : (int32_t)(full_nr + ql);
                for (int64_t k = 0; k < n_kv; k++) {
                    float v = -INFINITY;
                    if (k < ctx_len) {
                        const int ks = (int)(k / per_slot_ctx);
                        const int kl = (int)(k % per_slot_ctx);
                        if (ks == qs && kl < nc && q_pos - (int32_t)(slot_win_off[qs] + kl) <= window) {
                            v = 0.0f;
                        }
                    } else {
                        const int bi = (int)(k - ctx_len);
                        const int ks = bi / n_seq_tokens;
                        const int kl = bi % n_seq_tokens;
                        if (ks == qs && kl <= ql) { v = 0.0f; }
                    }
                    data[q * n_kv + k] = v;
                }
            }
        }
    }
}

llm_build_dflash_draft::llm_build_dflash_draft(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = hparams.dflash_n_target_features;

    // Drafter graph shape:
    //   n_slots == 1: ctx_len = cross->n_enc (power-of-2 bucket of actual data length).
    //                 set_cross_data triggers sched_need_reserve when the bucket changes,
    //                 so the graph re-reserves at each bucket boundary. This matches the
    //                 original single-slot path and keeps throughput unchanged.
    //   n_slots >= 2: ctx_len = n_slots × PER_SLOT_CTX (fixed). The shared drafter ctx
    //                 services multiple slots whose bucket-of-n_enc would otherwise
    //                 thrash sched_need_reserve as different slots write data of
    //                 different lengths. Fixed width avoids that thrash; multi-slot
    //                 users pay flat n_slots × PER_SLOT_CTX attention cost.
    const int n_slots = std::clamp(cparams.dflash_n_slots, 1, (int) LLAMA_DFLASH_MAX_SLOTS);
    int64_t ctx_len;
    if (n_slots == 1) {
        ctx_len = (cross && cross->n_enc > 0) ? cross->n_enc : (int64_t) LLAMA_DFLASH_PER_SLOT_CTX;
        const int64_t max_ctx = dflash_max_cross_ctx();
        if (max_ctx > 0 && ctx_len > max_ctx) {
            ctx_len = max_ctx;
        }
    } else {
        ctx_len = (int64_t) n_slots * LLAMA_DFLASH_PER_SLOT_CTX;
    }

    const int64_t n_kv_total = ctx_len + n_tokens;

    // --- DFlash-specific inputs ---
    const bool have_swa = hparams.is_swa_any();
    auto inp_dflash = std::make_unique<llm_graph_input_dflash>(cross, ctx_len, n_tokens, hparams.n_swa);

    // concatenated target hidden states [n_target_features, ctx_len]
    inp_dflash->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(inp_dflash->target_hidden);
    cb(inp_dflash->target_hidden, "dflash_target_hidden", -1);

    // context positions for K RoPE [ctx_len]
    inp_dflash->pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(inp_dflash->pos_ctx);
    cb(inp_dflash->pos_ctx, "dflash_pos_ctx", -1);

    // asymmetric non-causal mask [n_kv_total, n_tokens, 1, 1] — full-attention layers
    inp_dflash->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
    ggml_set_input(inp_dflash->kq_mask);
    inp_dflash->kq_mask_cnv = cparams.flash_attn
        ? ggml_cast(ctx0, inp_dflash->kq_mask, GGML_TYPE_F16)
        : inp_dflash->kq_mask;

    if (have_swa) {
        inp_dflash->kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
        ggml_set_input(inp_dflash->kq_mask_swa);
        cb(inp_dflash->kq_mask_swa, "dflash_kq_mask_swa", -1);
        inp_dflash->kq_mask_swa_cnv = cparams.flash_attn
            ? ggml_cast(ctx0, inp_dflash->kq_mask_swa, GGML_TYPE_F16)
            : inp_dflash->kq_mask_swa;
    }

    ggml_tensor * kq_mask_full  = inp_dflash->kq_mask_cnv;
    ggml_tensor * kq_mask_swa   = inp_dflash->kq_mask_swa_cnv; // may be null if no SWA
    ggml_tensor * pos_ctx       = inp_dflash->pos_ctx;
    ggml_tensor * target_hidden = inp_dflash->target_hidden;

    res->add_input(std::move(inp_dflash));

    // --- Embedding ---
    // tok_embd/output may be nullptr during graph reservation (shared from target at runtime)
    // Use Q4_0 placeholder to avoid 4.8 GB F32 allocation during reservation
    ggml_tensor * tok_embd_use = model.tok_embd;
    if (!tok_embd_use) {
        tok_embd_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    ggml_tensor * inpL = build_inp_embd(tok_embd_use);

    // block positions from ubatch.pos
    ggml_tensor * inp_pos = build_inp_pos();

    // --- Fusion layer: project concatenated target hidden states ---
    ggml_tensor * fused_target = build_lora_mm(model.dflash_fc, target_hidden);
    fused_target = build_norm(fused_target, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, -1);
    cb(fused_target, "fused_target", -1);

    // --- Transformer layers ---
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * kq_mask = (hparams.is_swa(il) && kq_mask_swa) ? kq_mask_swa : kq_mask_full;

        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // --- KV-injection attention ---
        {
            // Q from drafter hidden only
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            // K from drafter (noise tokens)
            ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
            Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head, n_head_kv, n_tokens);
            Kcur_noise = build_norm(Kcur_noise, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                                       n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                       ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_noise, "Kcur_noise", il);

            // K from target (context features)
            ggml_tensor * Kcur_ctx = build_lora_mm(model.layers[il].wk, fused_target);
            Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head, n_head_kv, ctx_len);
            Kcur_ctx = build_norm(Kcur_ctx, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, pos_ctx, nullptr,
                                     n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_ctx, "Kcur_ctx", il);

            // V from drafter (noise tokens)
            ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
            Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_noise, "Vcur_noise", il);

            // V from target (context features)
            ggml_tensor * Vcur_ctx = build_lora_mm(model.layers[il].wv, fused_target);
            Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head, n_head_kv, ctx_len);
            cb(Vcur_ctx, "Vcur_ctx", il);

            // concatenate K: [ctx, noise] along sequence dim (dim 2)
            ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 2);
            cb(Kcur, "Kcur", il);

            // concatenate V: [ctx, noise] along sequence dim (dim 2)
            ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 2);
            cb(Vcur, "Vcur", il);

            // prevent reordering
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            // asymmetric attention: Q [head_dim, n_head, n_tokens]
            //                       K [head_dim, n_head_kv, n_kv_total]
            //                    mask [n_kv_total, n_tokens, 1, 1]
            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, kq_mask, nullptr, nullptr,
                                 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            // output projection
            cur = build_lora_mm(model.layers[il].wo, cur);
        }

        // residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        // post-attention RMSNorm
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // SwiGLU FFN
        cur = build_ffn(cur,
            model.layers[il].ffn_up,   nullptr, nullptr,
            model.layers[il].ffn_gate, nullptr, nullptr,
            model.layers[il].ffn_down, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        // FFN residual
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // final RMSNorm
    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head — may be nullptr during reservation (shared from target at runtime)
    // Use Q4_0 placeholder to avoid 4.8 GB F32 allocation during reservation
    ggml_tensor * output_use = model.output;
    if (!output_use) {
        output_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    cur = build_lora_mm(output_use, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // GPU top-K or argmax — avoids 15.9MB logits transfer + CPU scan for DFlash draft
    const float sample_temp = cparams.dflash_sample_temp;
    static std::atomic<uint64_t> gumbel_counter{1};
    const uint64_t seed = (sample_temp > 0.0f) ? gumbel_counter.fetch_add(1) : 0;
    const int topk = cparams.dflash_topk;
    if (topk > 1) {
        res->t_logits_argmax = ggml_topk_ext(ctx0, cur, topk, sample_temp, seed);
    } else {
        res->t_logits_argmax = ggml_argmax_ext(ctx0, cur, sample_temp, seed);
    }

    ggml_build_forward_expand(gf, res->t_logits_argmax);
}
