#include "models.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <vector>

// ---------------------------------------------------------------------------
// gemma4-dflash-draft — a DFlash speculative-decoding drafter for Gemma-4 TARGETS.
// The drafter's OWN architecture is Qwen3, NOT Gemma-4: z-lab's DFlashDraftModel subclasses
// Qwen3 (Qwen3RMSNorm, Qwen3MLP/SiLU, 1/sqrt(head_dim) attn scale, raw embeddings). "gemma-4"
// refers only to the target it drafts FOR — it reads the gemma target's hidden states (fc
// fusion), shares the target's tied token_embd/output at runtime, and applies the target's
// final logit softcapping (tanh, cap = 30). Otherwise it is the generic `dflash-draft`
// KV-injection graph. It carries only ONE gemma-specific graph graft:
//   1. final logit softcapping (tanh, cap = 30) — from the drafter's config
// (Qwen3 conventions kept: SiLU FFN, 1/sqrt(head_dim) attn scale, raw embed, QK-norm,
// pre-FFN `ffn_norm`, RoPE freq_base=1e6.)
// Tensor names use the dot form (`dflash.fc`, `dflash.hidden_norm`) and the
// fusion width is derived from the #concatenated target layers × n_embd, since
// the Lucebox GGUF omits `.dflash.n_target_features`.
// ---------------------------------------------------------------------------

void llama_model_gemma4_dflash_draft::load_arch_hparams(llama_model_loader & ml) {
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

    // The Lucebox gemma4-dflash GGUF has no `.dflash.n_target_features` key, and the
    // hparams default (25600, a Qwen-draft value) is wrong here. Derive the fusion
    // input width straight from the fc tensor's own ne[0] (= #concatenated target
    // layers × n_embd; here 6 × 5376 = 32256).
    if (const ggml_tensor * fc = ml.get_tensor_meta("dflash.fc.weight")) {
        hparams.dflash_n_target_features = (uint32_t) fc->ne[0];
    }

    // Gemma-4 target: final logit softcapping (the drafter's one gemma-specific graft).
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // SWA disabled: the drafter's KV-injection SWA mask is buggy — enabling it drops draft
    // acceptance even at short context, where a 2048 window should be a no-op. z-lab's drafter
    // does declare a sliding window, so this is a validated workaround (+~7pp), not the final
    // answer; the per-slot cross+noise SWA mask needs a proper fix before re-enabling.
    hparams.n_swa = 0;
    hparams.swa_type = LLAMA_SWA_TYPE_NONE;
}

void llama_model_gemma4_dflash_draft::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    // tied embeddings — shared from the target model at runtime (absent in this GGUF)
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    dflash_fc          = create_tensor(tn(LLM_TENSOR_GEMMA4_DFLASH_FC,          "weight"), {(int64_t) hparams.dflash_n_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_GEMMA4_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        // pre-FFN norm (this draft has ffn_norm, NOT attn_post_norm)
        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_dflash_draft::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<llm_build_gemma4_dflash_draft>(*this, params);
}

// Max cross-attention context for the DFlash drafter (caps VRAM growth).
// Override with GGML_DFLASH_MAX_CTX env var. 0 = unlimited.
static int64_t gemma4_dflash_max_cross_ctx() {
    static const int64_t val = [] {
        const char * e = getenv("GGML_DFLASH_MAX_CTX");
        return e ? (int64_t) atoi(e) : (int64_t) 4096;
    }();
    return val;
}

// DFlash drafter custom graph input (identical semantics to the generic
// dflash-draft input; renamed to avoid an ODR clash with dflash_draft.cpp).
// Holds the target hidden states, context positions, and asymmetric non-causal mask.
// NOTE: set_input below is byte-identical to dflash_draft.cpp's. Kept duplicated
// deliberately so the validated generic-dflash path is untouched by this new arch.
// TODO(dedup): once both drafters are settled, factor the shared class + set_input
// body into a common helper (e.g. in llama-graph) so the ~230 lines live once.
class llm_graph_input_gemma4_dflash : public llm_graph_input_i {
public:
    llm_graph_input_gemma4_dflash(const llama_cross * cross, int64_t ctx_len, int64_t n_block, uint32_t n_swa)
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
            const int64_t max_ctx = gemma4_dflash_max_cross_ctx();
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

void llm_graph_input_gemma4_dflash::set_input(const llama_ubatch * ubatch) {
    const int n_seqs = (ubatch && ubatch->n_seqs_unq > 1) ? (int) ubatch->n_seqs_unq : 1;

    if (n_seqs == 1) {
        // === Single-slot path ===
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

        const int64_t src_real = src_n_real > 0 ? src_n_real : 0;
        const int64_t n_copy  = std::min(src_real, ctx_len);
        const int64_t win_off = (src_real > ctx_len) ? (src_real - ctx_len) : 0;

        if (target_hidden && (src_data || src_gpu) && n_copy > 0) {
            const int64_t n_feat = cross->n_embd;
            const size_t copy_bytes  = (size_t) n_feat * (size_t) n_copy * sizeof(float);
            const size_t tensor_bytes = ggml_nbytes(target_hidden);
            const size_t actual_bytes = std::min(copy_bytes, tensor_bytes);

            if (src_gpu && cross->fn_set_tensor_d2d) {
                const void * gpu_src = (const char *)src_gpu + (size_t)win_off * n_feat * sizeof(float);
                cross->fn_set_tensor_d2d(target_hidden->data, gpu_src, 0, actual_bytes);
            } else {
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
        GGML_ASSERT(ctx_len % n_seqs == 0);
        GGML_ASSERT(n_block % n_seqs == 0);
        const int per_slot_ctx    = (int)(ctx_len / n_seqs);
        const int n_seq_tokens    = (int)(n_block / n_seqs);
        const size_t n_feat       = cross ? (size_t) cross->n_embd : 0;

        struct { const float * data; int64_t n_real; } slot_info[LLAMA_DFLASH_MAX_SLOTS] = {};
        for (int s = 0; s < n_seqs && s < LLAMA_DFLASH_MAX_SLOTS; s++) {
            llama_seq_id seq = ubatch->seq_id_unq[s];
            if (!cross) { continue; }
            auto it = cross->v_embd_per_seq.find(seq);
            if (it != cross->v_embd_per_seq.end() && !it->second.v_embd.empty()) {
                slot_info[s] = { it->second.v_embd.data(), it->second.n_enc_real };
            }
        }

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

llm_build_gemma4_dflash_draft::llm_build_gemma4_dflash_draft(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = hparams.dflash_n_target_features;

    const int n_slots = std::clamp(cparams.dflash_n_slots, 1, (int) LLAMA_DFLASH_MAX_SLOTS);
    int64_t ctx_len;
    if (n_slots == 1) {
        ctx_len = (cross && cross->n_enc > 0) ? cross->n_enc : (int64_t) LLAMA_DFLASH_PER_SLOT_CTX;
        const int64_t max_ctx = gemma4_dflash_max_cross_ctx();
        if (max_ctx > 0 && ctx_len > max_ctx) {
            ctx_len = max_ctx;
        }
    } else {
        ctx_len = (int64_t) n_slots * LLAMA_DFLASH_PER_SLOT_CTX;
    }

    const int64_t n_kv_total = ctx_len + n_tokens;

    // --- DFlash-specific inputs ---
    const bool have_swa = hparams.is_swa_any();
    auto inp_dflash = std::make_unique<llm_graph_input_gemma4_dflash>(cross, ctx_len, n_tokens, hparams.n_swa);

    inp_dflash->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(inp_dflash->target_hidden);
    cb(inp_dflash->target_hidden, "dflash_target_hidden", -1);

    inp_dflash->pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(inp_dflash->pos_ctx);
    cb(inp_dflash->pos_ctx, "dflash_pos_ctx", -1);

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

    // --- Embedding: raw, like Qwen3 / the generic dflash-draft graph (no gemma sqrt(n_embd)
    //     scale, no bf16 round-trip — the drafter is Qwen3-arch). ---
    ggml_tensor * tok_embd_use = model.tok_embd;
    if (!tok_embd_use) {
        tok_embd_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    ggml_tensor * inpL = build_inp_embd(tok_embd_use);
    cb(inpL, "inp_embd", -1);

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
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
            Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head, n_head_kv, n_tokens);
            Kcur_noise = build_norm(Kcur_noise, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                                       n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                       ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_noise, "Kcur_noise", il);

            ggml_tensor * Kcur_ctx = build_lora_mm(model.layers[il].wk, fused_target);
            Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head, n_head_kv, ctx_len);
            Kcur_ctx = build_norm(Kcur_ctx, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, pos_ctx, nullptr,
                                     n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_ctx, "Kcur_ctx", il);

            ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
            Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_noise, "Vcur_noise", il);

            ggml_tensor * Vcur_ctx = build_lora_mm(model.layers[il].wv, fused_target);
            Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head, n_head_kv, ctx_len);
            cb(Vcur_ctx, "Vcur_ctx", il);

            ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 2);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 2);
            cb(Vcur, "Vcur", il);

            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            // z-lab drafter is Qwen3-arch: standard softmax scale = 1/sqrt(head_dim), not gemma's 1.0
            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, kq_mask, nullptr, nullptr,
                                 1.0f/sqrtf((float) hparams.n_embd_head_k(il)), il);
            cb(cur, "kqv_out", il);

            cur = build_lora_mm(model.layers[il].wo, cur);
        }

        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        // pre-FFN RMSNorm (ffn_norm)
        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        // z-lab drafter uses Qwen3MLP -> SiLU (SwiGLU), not gemma's GELU
        cur = build_ffn(cur,
            model.layers[il].ffn_up,   nullptr, nullptr,
            model.layers[il].ffn_gate, nullptr, nullptr,
            model.layers[il].ffn_down, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    ggml_tensor * output_use = model.output;
    if (!output_use) {
        output_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    cur = build_lora_mm(output_use, cur);

    // Gemma 4 final logit softcapping (tanh, cap = 30)
    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    // GPU top-K or argmax for the DFlash draft.
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
