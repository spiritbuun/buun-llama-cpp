#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)
#include "../src/llama-io.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cinttypes>
#include <limits>
#include <queue>
#include <random>

#define SPC_DBG(fmt, ...) LOG_DBG("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_TRC(fmt, ...) LOG_TRC("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_ERR(fmt, ...) LOG_ERR("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"draft-dflash",  COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
    {"draft-dspark",  COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX},
    {"copyspec",      COMMON_SPECULATIVE_TYPE_COPYSPEC},
    {"recycle",       COMMON_SPECULATIVE_TYPE_RECYCLE},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"mtp",           COMMON_SPECULATIVE_TYPE_DRAFT_MTP}
};

bool common_speculative_mtp_carry_lifecycle::draft_ready() const noexcept {
    return ready;
}

const float * common_speculative_mtp_carry_lifecycle::draft_carry(
        const float * pending_h) const noexcept {
    return ready ? pending_h : nullptr;
}

common_speculative_mtp_carry_lifecycle::process_mode
common_speculative_mtp_carry_lifecycle::target_process_mode(
        llama_pos first_position) const noexcept {
    if (ready) {
        return process_mode::retained_carry;
    }
    return first_position == 0 ? process_mode::cold_zero : process_mode::target_only;
}

void common_speculative_mtp_carry_lifecycle::target_process_refreshed() noexcept {
    ready = true;
}

void common_speculative_mtp_carry_lifecycle::target_process_skipped() noexcept {
    ready = false;
}

void common_speculative_mtp_carry_lifecycle::sequence_transition(
        common_speculative_sequence_event event) noexcept {
    switch (event) {
        case common_speculative_sequence_event::prompt_rewind:
        case common_speculative_sequence_event::target_restored_without_draft:
        case common_speculative_sequence_event::draft_image_restored:
        case common_speculative_sequence_event::composite_image_restored:
        case common_speculative_sequence_event::live_range_shift:
        case common_speculative_sequence_event::target_replaced:
        case common_speculative_sequence_event::full_clear:
            ready = false;
            break;
    }
}

common_speculative_mtp_process_preflight
common_speculative_mtp_process_preflight_resolve(
        const std::vector<common_speculative_mtp_carry_lifecycle> & lifecycles,
        const std::vector<int32_t> & active_batch_beg,
        const llama_pos * positions) noexcept {
    if (lifecycles.size() != active_batch_beg.size() || positions == nullptr) {
        return common_speculative_mtp_process_preflight::target_only;
    }
    for (size_t seq = 0; seq < lifecycles.size(); ++seq) {
        if (active_batch_beg[seq] < 0) {
            continue;
        }
        if (lifecycles[seq].target_process_mode(positions[active_batch_beg[seq]]) ==
                common_speculative_mtp_carry_lifecycle::process_mode::target_only) {
            return common_speculative_mtp_process_preflight::target_only;
        }
    }
    return common_speculative_mtp_process_preflight::cold_or_retained;
}

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    SPC_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    SPC_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        SPC_WRN("draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        SPC_WRN("draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        SPC_WRN("draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            SPC_DBG("draft model vocab must closely match target model to use speculation but "
                    "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                SPC_DBG("draft model vocab must match target model to use speculation but "
                        "token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;
    int32_t n_max; // maximum draft length after implementation-specific limits

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // Set only when the most recent draft() call completed at least one
    // draft-model decode. Producing an empty token list is not equivalent to a
    // failed decode (for example p_min may reject the first candidate).
    bool last_draft_model_decode_succeeded = false;

    std::vector<size_t> n_acc_tokens_per_pos; // number of tokens accepted per draft position.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq, int32_t n_max) : type(type), n_seq(n_seq), n_max(n_max) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // (optional) serialize/restore per-seq internal state (e.g. eagle3's deferred boundary).
    virtual bool get_state(llama_seq_id /*seq_id*/, std::vector<uint8_t> & /*data*/) const { return false; }
    virtual void set_state(llama_seq_id /*seq_id*/, const std::vector<uint8_t> & /*data*/) {}
    // Called after an external sequence lifecycle mutation. Most implementations
    // have no branch-local state beyond their serialized state and need no action.
    virtual void sequence_transition(
            llama_seq_id /*seq_id*/,
            common_speculative_sequence_event /*event*/) {}

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const { return false; }

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_nextn() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        if (!ctx_dft) {
            throw std::runtime_error("draft-simple requires a draft context");
        }

        SPC_TRC("%s", "adding speculative implementation 'draft-simple'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f\n", this->params.n_max, this->params.n_min, this->params.p_min);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        SPC_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            SPC_ERR("%s", "the target and draft vocabs are not compatible\n");

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            SPC_ERR("n_seq mismatch: %d != %d\n", n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        // draft() has already evaluated the speculative suffix. Verification
        // starts at the sampled token and may replace that entire suffix, so
        // restore each participating sequence to the first incoming position
        // before decoding the verification batch. Prompt chunks are naturally
        // idempotent here because there is no suffix at or beyond their start.
        if (batch.pos != nullptr) {
            std::vector<llama_pos> first_pos(n_seq, std::numeric_limits<llama_pos>::max());
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
                    const llama_seq_id seq_id = batch.seq_id[i][j];
                    if (seq_id >= 0 && (uint32_t) seq_id < n_seq) {
                        first_pos[seq_id] = std::min(first_pos[seq_id], batch.pos[i]);
                    }
                }
            }
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (first_pos[seq_id] != std::numeric_limits<llama_pos>::max() &&
                        !llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, first_pos[seq_id], -1)) {
                    SPC_ERR("failed to trim draft sequence %d at position %d\n", seq_id, first_pos[seq_id]);
                    return false;
                }
            }
        }

        llama_batch batch_dft = batch;
        batch_dft.logits = nullptr;

        const int ret = llama_decode(ctx_dft, batch_dft);

        if (ret != 0) {
            SPC_ERR("failed to decode draft batch, ret = %d\n", ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, dp.n_past, -1);

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }
        last_draft_model_decode_succeeded = true;

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};


// EAGLE3 speculative decoding state
//
// Input of draft decoder: (This is different compared to MTP)
//   At "pos P", the decoder takes input pair (t_{P+1}, g_P), with RoPE at P.
//     - t_{P+1} = token at sequence pos P+1 (the *next* token after P)
//     - g_P     = encoder output = projection of target's extracted hidden states at P
//
// Deferred boundary (MTP doesn't have this issue):
//   Within a single process() call with n_tokens, we can only write decoder KV for
//   training pos 0..n_tokens-2. The last training pos (n_tokens-1) needs t_{n_tokens}
//   which lies *outside* this batch — it is the token target will sample next or the first token from next ubatch.
//   So the last training pos of each process() call is *deferred* to whichever next call has
//   the missing token in hand:
//     - multi-ubatch prefill: the next process()'s first token completes the pair
//                              (handled by the per-seq "cross-ubatch bridge")
//     - single-ubatch prefill / after verify: draft()'s seed step uses "dp.id_last"
//                              (target's freshest sample) to complete the pair
//
// Per-seq carry-over state:
//   pending_g_last    [n_embd_dec]  ┐  the deferred boundary's (g, pos). Set by
//   pending_pos_last  llama_pos     ┘  process() at end of ubatch (= last row);
//                                       rebased by accept() to first-non-accepted pos.
//   verify_g          [N × n_embd_dec] snapshot of process()'s encoder output;
//   verify_pos_first  llama_pos         consumed by accept() to recover the right
//   verify_g_rows     int32_t           pending_g_last row for any n_accepted value.
//
// Performance is overall good but there is waste in verify cycle:
//   process() runs encoder + decoder on the *full* verify batch including rows for
//   rejected drafts. The KV at those positions is then dropped.
//
// TODO: Not sure if we need optimization for this waste?
// If so we may need hybrid stash:
//      in verify mode, have process() only stash features and let draft() seed run
//      encoder+decoder on n_accepted+1 rows).
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;
    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd_dec = 0;       // draft hidden size
    int32_t n_embd_enc = 0;       // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;       // target model hidden size
    int32_t n_layer_tgt = 0;      // target model layer count

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;

    // [per-seq] deferred boundary state
    std::vector<std::vector<float>> pending_g_last;
    std::vector<llama_pos>          pending_pos_last;

    // [per-seq] snapshot of the most recent process()'s encoder output
    std::vector<std::vector<float>> verify_g;         // [n_seq][n_rows * n_embd_dec]
    std::vector<llama_pos>          verify_pos_first; // [n_seq] — pos of verify_g[seq][0]
    std::vector<int32_t>            verify_g_rows;    // [n_seq] — number of rows

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;
    std::vector<float> g_embd_buf;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        SPC_TRC("%s", "adding speculative implementation 'draft-eagle3'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%f, backend_sampling=%d\n", params.draft.n_max, params.draft.n_min, params.draft.p_min, (int) params.draft.backend_sampling);

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "EAGLE3 requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n != 3) {
            throw std::runtime_error("draft model is not eagle3 (expected 3 extract layers, got " +
                                     std::to_string(target_layer_ids_n) + ")");
        }

        n_embd_tgt = llama_model_n_embd(model_tgt);
        n_embd_dec = llama_model_n_embd(model_dft);
        n_embd_enc = (int32_t) target_layer_ids_n * n_embd_tgt;
        n_layer_tgt = llama_model_n_layer(model_tgt);

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd_dec, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; eagle3 decoder needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        // turn on extraction of the target layers' hidden states
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            if (target_layer_ids[k] < n_layer_tgt) {
                llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
            } else if (target_layer_ids[k] == n_layer_tgt) {
                llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
            } else {
                GGML_ABORT("EAGLE3: target layer id %d exceeds target n_layer %d", target_layer_ids[k], n_layer_tgt);
            }
        }

        // turn on extraction of the draft model's pre-norm hidden state
        // (used both for the encoder output g_embd and the decoder pre-norm output).
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_g_last.assign(n_seq, std::vector<float>(n_embd_dec, 0.0f));
        pending_pos_last.assign(n_seq, -1);

        verify_g.assign(n_seq, std::vector<float>());
        verify_pos_first.assign(n_seq, -1);
        verify_g_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_eagle3() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        // expected state after prefill: ctx_dft has pos 0..N-2 (last position is deferred to
        // draft()'s seed step). Warn only if more than one position is missing.
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 2) {
            SPC_WRN("ctx_dft pos_max=%d < N-2=%d — process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 2);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // i_batch_beg[seq] / i_batch_end[seq]: inclusive batch indices of this seq's
        // first/last token in batch_in. Assumes per-seq tokens are contiguous within
        // the ubatch (server's default ordering).
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // Interleave each extract_layer's hidden state into a contiguous buffer of
        // shape [n_tokens, target_layer_ids_n * n_embd_tgt]. Then run EAGLE3 encoder
        // to get one g_embd row per token.
        features_buf.resize((size_t) n_tokens * n_embd_enc, 0.0f);

        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            const float * layer = target_layer_ids[k] < n_layer_tgt
                ? llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k])
                : llama_get_embeddings_nextn(ctx_tgt);
            if (!layer) {
                GGML_ABORT("EAGLE3: target layer %d input not extracted.", target_layer_ids[k]);
            }
            for (int32_t i = 0; i < n_tokens; ++i) {
                float * dst = features_buf.data() + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                const float * src = layer + (size_t) i * n_embd_tgt;
                std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
            }
        }

        g_embd_buf.resize((size_t) n_tokens * n_embd_dec);

        // llama_encode() requires the full encoder batch to fit in n_ubatch.
        // Allow batch > ubatch: eagle3's per-token encoder can be chunked safely.
        const int32_t n_ubatch_dft = (int32_t) llama_n_ubatch(ctx_dft);
        for (int32_t i = 0; i < n_tokens; i += n_ubatch_dft) {
            const int32_t n_chunk = std::min(n_ubatch_dft, n_tokens - i);

            llama_batch enc_batch = {
                /*.n_tokens =*/ n_chunk,
                /*.token    =*/ nullptr,
                /*.embd     =*/ features_buf.data() + (size_t) i * n_embd_enc,
                /*.pos      =*/ nullptr,
                /*.n_seq_id =*/ nullptr,
                /*.seq_id   =*/ nullptr,
                /*.logits   =*/ nullptr,
            };
            const int32_t rc = llama_encode(ctx_dft, enc_batch);
            if (rc != 0) {
                SPC_ERR("llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                        rc, (int) n_chunk, (int) i);
                return false;
            }

            // g_embd has shape [n_chunk, n_embd_dec] in ctx_dft's pre-norm embeddings buffer.
            const float * g_embd_chunk = llama_get_embeddings_nextn(ctx_dft);
            GGML_ASSERT(g_embd_chunk && "EAGLE3 encoder produced no output.");
            std::memcpy(g_embd_buf.data() + (size_t) i * n_embd_dec,
                        g_embd_chunk,
                        (size_t) n_chunk * n_embd_dec * sizeof(float));
        }

        const float * g_embd = g_embd_buf.data();

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // EAGLE3 decoder input convention: at memory pos P the input pair is
        // (token[P+1], g_embd[P]). This shifts the token index "left by one" relative to g_embd.
        //
        // Per seq, in order:
        //   (a) cross-ubatch bridge — when applicable, write the previously-deferred
        //       pos using this ubatch's first token + pending_g_last.
        //   (b) main write loop — for k in [beg, end-1], write (token[k+1], g_embd[k])
        //       at pos[k]. The last training pos (k=end) is left unwritten = new
        //       deferred boundary, completed by the next process() or draft() call.
        //   (c) refresh deferred state — stash this ubatch's full g_embd into verify_g,
        //       update pending_g_last / pending_pos_last to the last row.
        common_batch_clear(batch);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            const int32_t beg = i_batch_beg[seq_id];
            const int32_t end = i_batch_end[seq_id];
            if (beg < 0 || end < 0) {
                continue;
            }

            // cross-ubatch bridge — complete the prior ubatch's deferred boundary.
            // Fires iff all three preconditions hold:
            //   1) pending_pos_last >= 0
            //   2) pending_pos_last + 1 == pos[beg]
            //   3) pending_pos_last > dft_pos_max // TODO: is this check needed?
            const llama_pos pending_pos = pending_pos_last[seq_id];
            if (pending_pos >= 0 && pending_pos + 1 == batch_in.pos[beg]) {
                const llama_pos dft_pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
                if (pending_pos > dft_pos_max) {
                    common_batch_add(batch, batch_in.token[beg], pending_pos, { seq_id }, /*logits=*/ false);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                                pending_g_last[seq_id].data(), row_bytes);
                }
            }

            for (int32_t k = beg; k < end; ++k) {
                common_batch_add(batch, batch_in.token[k + 1], batch_in.pos[k], { seq_id }, /*logits=*/ false);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                            g_embd + (size_t) k * n_embd_dec, row_bytes);
            }

            // refresh deferred state
            const int32_t n_rows = end - beg + 1;
            verify_pos_first[seq_id] = batch_in.pos[beg];
            pending_pos_last[seq_id] = batch_in.pos[end];
            verify_g_rows[seq_id]    = n_rows;
            verify_g[seq_id].resize((size_t) n_rows * n_embd_dec, 0.0f);
            std::memcpy(verify_g[seq_id].data(),       g_embd + (size_t) beg * n_embd_dec, row_bytes * n_rows);
            std::memcpy(pending_g_last[seq_id].data(), g_embd + (size_t) end * n_embd_dec, row_bytes);
        }

        if (batch.n_tokens > 0) {
            const int32_t rc = llama_decode(ctx_dft, batch);
            if (rc != 0) {
                SPC_ERR("llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, ubatch_pos[0]=%d)\n",
                        rc, (int) batch.n_tokens, (int) batch_in.pos[0]);
                return false;
            }
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd_dec * sizeof(float);

        // Complete the deferred boundary pair (dp.id_last, pending_g_last) at memory
        // pos pending_pos_last. dp.id_last is target's freshest sample (= corrected
        // token after verify, or first generated token after prefill), matching the
        // EAGLE3 input convention (token[P+1], g_embd[P]) at pos P.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }
            if (pending_pos_last[seq_id] < 0) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pending_pos_last[seq_id], -1);

            common_batch_add(batch, dp.id_last, pending_pos_last[seq_id], { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec,
                        pending_g_last[seq_id].data(),
                        row_bytes);
        }

        if (batch.n_tokens == 0) {
            return;
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            SPC_ERR("llama_decode returned %d\n", ret);
            return;
        }
        last_draft_model_decode_succeeded = true;

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                // pre-norm hidden state of this position becomes g_embd for the next step
                const float * prenorm = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                // (configurable via --spec-draft-p-min, set to 0.0 to disable early-stop)
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, pending_pos_last[seq_id] + (i + 1), { seq_id }, true);
                std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd_dec, prenorm, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_g_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_g = std::min<int32_t>(n_accepted, n_rows - 1);
        pending_pos_last[seq_id] = verify_pos_first[seq_id] + i_g;
        std::memcpy(pending_g_last[seq_id].data(),
                    verify_g[seq_id].data() + (size_t) i_g * n_embd_dec,
                    (size_t) n_embd_dec * sizeof(float));
    }

    // we only need to stash the deferred boundary's g_embd row for recurrent/hybrid targets:
    // their single-position checkpoints drop it on restore
    bool need_boundary_stash() const {
        const llama_model * model_tgt = llama_get_model(params.ctx_tgt);
        return llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        if (!need_boundary_stash()) {
            return false;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq || pending_pos_last[seq_id] < 0) {
            return false;
        }

        const llama_pos          pos = pending_pos_last[seq_id];
        const std::vector<float> & g = pending_g_last[seq_id];

        data.resize(sizeof(llama_pos) + g.size() * sizeof(float));
        std::memcpy(data.data(),                     &pos,     sizeof(llama_pos));
        std::memcpy(data.data() + sizeof(llama_pos), g.data(), g.size() * sizeof(float));
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        if (!need_boundary_stash()) {
            return;
        }
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        if (data.size() != sizeof(llama_pos) + (size_t) n_embd_dec * sizeof(float)) {
            return;
        }

        llama_pos pos = -1;
        std::memcpy(&pos, data.data(), sizeof(llama_pos));

        pending_pos_last[seq_id] = pos;
        pending_g_last[seq_id].resize(n_embd_dec);
        std::memcpy(pending_g_last[seq_id].data(), data.data() + sizeof(llama_pos), (size_t) n_embd_dec * sizeof(float));
    }
};

// DFlash: block-diffusion drafting with a draft-side KV cache injection
// default-on env kill switches: set NAME=0 to disable
static bool env_on(const char * name) {
    const char * v = getenv(name);
    return !(v && atoi(v) == 0);
}

struct common_speculative_impl_draft_dflash : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;        // noise tokens
    llama_batch batch_inject; // target features for KV cache injection

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd_dec = 0;  // draft hidden size
    int32_t n_embd_enc = 0;  // target_layer_ids_n * target_hidden_size
    int32_t n_embd_tgt = 0;  // target model hidden size

    int32_t     block_size    = 0;
    llama_token mask_token_id = 0;

    bool is_mrope = false;

    // draft-dspark: the draft carries a Markov head and uses an anchor-first block layout
    const bool is_dspark;
    const bool is_dflash2;
    const float sample_temp;

    // GPU draft sampling: top-K ids + log-probs computed in-graph (t_logits_argmax tail),
    // skipping the full-vocab logits D2H + CPU top-k scan per draft position
    bool    gpu_sample = false;
    int32_t gpu_topk   = 10; // matches the CPU sampler chain's top_k

    // fused encoder+injection: inject decode takes raw target features and applies the
    // encoder (fc + enc-norm) in-graph — no llama_encode round-trip per prefill chunk
    bool fused_inject = false;

    // device-staged capture->inject chain: target graph writes captured features into a
    // persistent device tensor (interleaved); the inject decode gathers rows on-device.
    // No host D2H capture, no host interleave, no H2D feature upload.
    bool                 staged = false;
    void *               stage_handle = nullptr;
    std::vector<int32_t> stage_rows;

    // phase C single-graph fused cycle: generation-phase injections are deferred (rows
    // D2D-copied into the carry tensor so they survive the next target decode) and
    // folded into the draft decode — one constant-topology drafter graph per cycle, so
    // gf_res_prev reuse and CUDA graph capture hold. Only committed rows get injected;
    // padding rows repeat the last committed row into a scratch seq the KV mask hides
    // from real attention. Kill switch: GGML_DFLASH_ONEGRAPH=0 (requires staged).
    void *       carry_handle       = nullptr;
    int32_t      oneg_n_inject      = 0;  // fixed inject rows per fused decode (carry_rows_per_seq + 1)
    int32_t      carry_rows_per_seq = 0;  // carry capacity per seq (n_max + 1)
    llama_seq_id oneg_scratch_seq   = -1;

    bool oneg() const { return carry_handle != nullptr; }
    struct oneg_stash {
        bool      pending = false;
        llama_pos pos0    = 0;
        int32_t   n_rows  = 0;
    };
    std::vector<oneg_stash> stash;      // [n_seq] deferred injection per seq
    std::vector<bool>       seq_in_gen; // [n_seq] generation phase (first draft seen)

    // The target may use multimodal M-RoPE positions, while the text-only drafter owns
    // an ordinary one-dimensional cache. Image rows therefore cannot be inserted using
    // the target's positions. Drop them and restart the drafter at the first following
    // text row; that row's captured target features already summarize the image. The
    // offset also covers a drafter context recreated by --mmproj-gpu-swap.
    struct position_seq_state {
        llama_pos target_base = 0;
        bool      rebase_pending = false;
        bool      rebased = false;
    };
    std::vector<position_seq_state> positions;

    // Adaptive depth is per sequence. Ordinary DFlash retains its lightweight
    // acceptance controller. DFlash2 measures wall time per emitted target token for
    // full, target-only, and depth-2 verification because its full selector lattice
    // has constant draft cost and target-side batch kernels are not monotonic in
    // emitted depth.
    bool adaptive = false;
    struct adaptive_seq_state {
        int32_t n_draft_last = 0;
        int32_t n_low_acc = 0;
        int32_t cap = -1; // -1 = caller/model ceiling; 0 = target-only

        bool sweep = false;
        bool sweep_pending = false;
        int32_t hold = 0;
        int32_t candidate = 0;
        int32_t sample_cycles = 0;
        int64_t sample_us = 0;
        int32_t sample_tokens = 0;
        double best_us_per_token = std::numeric_limits<double>::infinity();

        int64_t cycle_start_us = 0;
        int32_t cycle_tokens = 0;
        bool cycle_is_sample = false;

        int64_t full_us = 0;
        int32_t full_tokens = 0;

        int32_t recovery_cycles = 0;
        int32_t recovery_drafts = 0;
        int32_t recovery_accepts = 0;
    };
    std::vector<adaptive_seq_state> adpt;

    // Request-seeded stochastic DFlash2 proposal state, kept independently for
    // every server sequence so a batched decode cannot couple slot RNGs.
    std::vector<std::mt19937> proposal_rngs;
    std::vector<std::vector<float>> proposal_uniforms;
    std::vector<common_speculative_proposal> proposals;

    // pre-gate: after consecutive p_min-gated EMPTY drafts, skip whole draft calls with
    // exponential backoff (1,2,4,8 cap) — on expensive-verify targets (cpu-moe MoE) the
    // wasted calls are the entire speculative overhead. Reset on any emit; re-probes at
    // least every 9 cycles. Kill switch: GGML_DFLASH_DRAFT_PREGATE=0.
    bool                 pregate = false;
    std::vector<int32_t> pregate_empty; // [n_seq] consecutive attempted-but-empty drafts
    std::vector<int32_t> pregate_skip;  // [n_seq] draft calls left to skip

    // dspark speculators
    bool sample_from_anchor = true;

    const int32_t * target_layer_ids   = nullptr; // model_dft's extract layer indices
    uint32_t        target_layer_ids_n = 0;
    int32_t         target_layer_ids_buf[8] = {}; // backing store for fork-arch drafters

    // scratch buffer for concatenated target features [n_tokens, n_embd_enc]
    std::vector<float> features_buf;

    common_speculative_impl_draft_dflash(const common_params_speculative & params, uint32_t n_seq,
            common_speculative_type type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)
        : common_speculative_impl(type, n_seq, params.draft.n_max)
        , params(params.draft)
        , is_dspark(type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)
        , is_dflash2(llama_model_dflash2_has_selector(llama_get_model(params.draft.ctx_dft)))
        , sample_temp(params.sample_temp)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "DFlash requires ctx_tgt and ctx_dft to be set");

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        target_layer_ids   = llama_model_target_layer_ids  (model_dft);
        target_layer_ids_n = llama_model_target_layer_ids_n(model_dft);
        if (target_layer_ids_n == 0) {
            // fork drafter archs (dflash-draft, gemma4-dflash-draft) publish their layer ids
            // via hparams (%s.dflash.target_layer_ids), not the upstream model vector
            // (%s.target_layers) that eagle3/upstream-dflash fill — read the fork plumbing too
            int32_t n = llama_model_dflash_target_layer_ids(model_dft, target_layer_ids_buf,
                    (int32_t) (sizeof(target_layer_ids_buf) / sizeof(target_layer_ids_buf[0])));
            if (n > 0) {
                target_layer_ids   = target_layer_ids_buf;
                target_layer_ids_n = (uint32_t) n;
            }
        }
        GGML_ASSERT(target_layer_ids_n > 0 && "DFlash model has no target_layer_ids");

        n_embd_tgt    = llama_model_n_embd(model_tgt);
        n_embd_dec    = llama_model_n_embd(model_dft);
        n_embd_enc    = (int32_t) target_layer_ids_n * n_embd_tgt;

        // read the trained block size — fork drafter archs load it into hparams from
        // the arch-prefixed %s.dflash.block_size key (the bare-key metadata lookup
        // below misses those and silently keeps the default)
        block_size = 16;
        if (int32_t bs = llama_model_dflash_block_size(model_dft); bs > 0) {
            block_size = bs;
        } else {
            char buf[32] = {};
            if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
                block_size = std::atoi(buf);
            }
        }
        char sample_from_anchor_buf[16] = {};
        if (llama_model_meta_val_str(model_dft, "dflash.sample_from_anchor",
                    sample_from_anchor_buf, sizeof(sample_from_anchor_buf)) >= 0) {
            sample_from_anchor = std::strcmp(sample_from_anchor_buf, "true") == 0;
        }
        // fork drafters carry the mask token in %s.dflash.mask_token_id (hparams),
        // not as a tokenizer-level mask token — a -1 here poisons every draft batch
        // (llama_decode rejects the invalid token, so drafting silently never works)
        mask_token_id = llama_vocab_mask(llama_model_get_vocab(model_dft));
        if (mask_token_id < 0) {
            mask_token_id = (llama_token) llama_model_dflash_mask_token_id(model_dft);
        }

        // n_max < 0 = auto (--spec-dflash-default): the full block depth strictly wins
        // (EXP-37i). The server resolves this at model load; this covers the other
        // binaries (llama-cli, speculative-simple).
        if (this->params.n_max < 0) {
            this->params.n_max = block_size > 1 ? block_size - 1 : 12;
        }

        if (is_dspark && this->params.p_min > 0.0f) {
            char buf[16] = {};
            const bool has_conf =
                llama_model_meta_val_str(model_dft, "dflash.has_confidence_head", buf, sizeof(buf)) < 0 ||
                std::strcmp(buf, "true") == 0;
            if (!has_conf) {
                throw std::runtime_error("DSpark draft has no confidence head: please set --spec-draft-p-min 0");
            }
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(type).c_str());
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - block_size=%d, mask_token_id=%d, n_extract=%u, sample_from_anchor=%s\n", __func__,
                block_size, mask_token_id, target_layer_ids_n, sample_from_anchor ? "true" : "false");

        // DFlash input is [id_last, <mask> * (block_size-1)]: in-place denoising yields at most
        // block_size-1 draft tokens, anchor-first DSpark yields a full block_size draft tokens
        const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
        if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
            LOG_WRN("%s: requested draft size (n_max=%d, n_min=%d) exceeds the trained block size %d -- clamping to %d\n",
                    __func__, this->params.n_max, this->params.n_min, block_size, n_draft_max);
            this->params.n_max = std::min(this->params.n_max, n_draft_max);
            this->params.n_min = std::min(this->params.n_min, n_draft_max);
        }
        if (is_dflash2 && this->params.p_min > 0.0f) {
            LOG_WRN("%s: DFlash2 path scores are not token probabilities; ignoring p_min %.2f\n",
                    __func__, this->params.p_min);
            this->params.p_min = 0.0f;
        }
        if (is_dflash2 && sample_temp > 0.0f) {
            llama_set_dflash_sample_temp(ctx_dft, sample_temp);
        }
        const char * dflash2_mmq_env = getenv("GGML_DFLASH2_TARGET_MMQ");
        if (is_dflash2 && dflash2_mmq_env && atoi(dflash2_mmq_env) != 0) {
            LOG_INF("%s: - server target verification forces CUDA MMQ at batch width %d (experimental)\n",
                    __func__, block_size);
        }
        this->n_max = this->params.n_max;

        // fused encoder+injection: the inject decode carries raw concatenated target
        // features and applies fc + enc-norm in-graph, replacing the per-chunk
        // llama_encode + readback round-trip. Kill switch: GGML_DFLASH_FUSE=0.
        {
            fused_inject = env_on("GGML_DFLASH_FUSE");
            if (fused_inject) {
                llama_set_dflash_fused_inject(ctx_dft, true);
                LOG_INF("%s: - fused encoder+injection enabled\n", __func__);
            }
        }

        // device-staged capture->inject: keep the whole capture->interleave->inject
        // chain on the GPU whenever a target batch fits one ubatch (all generation-phase
        // verify batches). Multi-ubatch prefill chunks fall back to the host path per
        // decode. Kill switch: GGML_DFLASH_STAGED=0 (requires fused injection).
        if (fused_inject) {
            if (env_on("GGML_DFLASH_STAGED")) {
                // phase C wants a scratch drafter seq (the server reserves one when the
                // env is on) and a carry buffer for the deferred inject rows. The DSV4
                // backbone's fused cycle has its own kill switch on top of the master
                // one: GGML_DFLASH_ONEGRAPH_DSV4=0 restores its two-decode path.
                const bool want_oneg = env_on("GGML_DFLASH_ONEGRAPH") &&
                        llama_n_seq_max(ctx_dft) > n_seq &&
                        (!llama_model_dflash_dsv4_backbone(model_dft) ||
                          env_on("GGML_DFLASH_ONEGRAPH_DSV4"));
                if (want_oneg) {
                    carry_rows_per_seq = this->params.n_max + 1;
                }
                stage_handle = llama_dflash_draft_stage_init(ctx_tgt, ctx_dft,
                        target_layer_ids, (int32_t) target_layer_ids_n, n_embd_enc,
                        (int32_t) n_seq * carry_rows_per_seq);
                if (stage_handle) {
                    llama_set_dflash_inject_stage(ctx_dft, stage_handle);
                    staged = true;
                    LOG_INF("%s: - device-staged capture->inject enabled\n", __func__);
                    if (want_oneg) {
                        carry_handle = llama_dflash_draft_stage_carry_tensor(ctx_tgt);
                        if (carry_handle) {
                            // always >= 1 padding row, so the scratch seq is present in
                            // every fused batch (build_dspark_markov_head counts on it)
                            oneg_n_inject    = carry_rows_per_seq + 1;
                            oneg_scratch_seq = (llama_seq_id) n_seq;
                            llama_set_dflash_oneg_inject(ctx_dft, carry_handle, 0);
                            LOG_INF("%s: - single-graph fused cycle enabled (inject rows=%d, scratch seq=%d)\n",
                                    __func__, oneg_n_inject, (int) oneg_scratch_seq);
                        }
                    }
                }
            }
        }
        stash     .resize(n_seq);
        seq_in_gen.assign(n_seq, false);
        positions .resize(n_seq);

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, n_seq);
        // process() chunks by n_ubatch, so the (wider) fused injection batch only needs
        // n_ubatch rows; the unfused batch keeps the historic n_batch sizing
        batch_inject = fused_inject
            ? llama_batch_init(llama_n_ubatch(ctx_dft), n_embd_enc, n_seq)
            : llama_batch_init(llama_n_batch (ctx_dft), n_embd_dec, n_seq);

        // embd batches on an M-RoPE draft need 4 position rows per token
        is_mrope = llama_model_rope_type(model_dft) == LLAMA_ROPE_TYPE_MROPE;
        if (is_mrope) {
            free(batch_inject.pos);
            batch_inject.pos = (llama_pos *) malloc(sizeof(llama_pos) * 4 * llama_n_batch(ctx_dft));
        }

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(model_dft, sparams));
        }

        // GPU draft sampling: build the in-graph top-K tail on the drafter decode graph.
        // draft() then reads K ids + log-probs per position (tiny transfer) instead of
        // pulling full-vocab logits to the host and scanning them per position.
        // Ordinary DFlash kill switches: --no-spec-draft-backend-sampling or
        // GGML_DFLASH_DRAFT_GPUSAMPLE=0. DFlash2's learned selector is part of
        // the model contract rather than an optional sampling optimization.
        {
            gpu_sample = is_dflash2 || (this->params.backend_sampling && env_on("GGML_DFLASH_DRAFT_GPUSAMPLE"));
            if (gpu_sample) {
                if (!is_dflash2) {
                    llama_set_dflash_topk(ctx_dft, gpu_topk);
                }
                llama_set_dflash_argmax(ctx_dft, true);
                if (is_dflash2) {
                    LOG_INF("%s: - in-graph DFlash2 path selector enabled\n", __func__);
                } else {
                    LOG_INF("%s: - GPU draft sampling enabled (in-graph top-%d)\n", __func__, gpu_topk);
                }
            }
        }

        // adaptive draft length controller. Kill switch: GGML_DFLASH_DRAFT_ADAPTIVE=0.
        {
            adaptive = env_on("GGML_DFLASH_DRAFT_ADAPTIVE");
        }
        adpt.resize(n_seq);
        proposal_rngs.resize(n_seq);
        proposal_uniforms.resize(n_seq);
        proposals.resize(n_seq);

        pregate = env_on("GGML_DFLASH_DRAFT_PREGATE");
        pregate_empty.assign(n_seq, 0);
        pregate_skip .assign(n_seq, 0);
        // turn on extraction of the target layers' input embeddings
        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        llama_set_causal_attn(ctx_dft, false); // DFlash needs non-causal attention

        // warm up both drafter graph modes (embd inject + token draft) so the first
        // request doesn't pay backend module load / pool growth (fork 2a491077d).
        // Kill switch: GGML_DFLASH_DRAFT_WARMUP=0.
        {
            if (env_on("GGML_DFLASH_DRAFT_WARMUP")) {
                llama_set_warmup(ctx_dft, true);

                // injection graph: one zero-feature row at pos 0
                batch_inject.n_tokens = 1;
                std::memset(batch_inject.embd, 0,
                        (size_t) (fused_inject ? n_embd_enc : n_embd_dec) * sizeof(float));
                if (staged) {
                    // staged inject graph reads stage row 0 (content irrelevant here)
                    const int32_t r0 = 0;
                    llama_set_dflash_inject_rows(ctx_dft, &r0, 1);
                }
                batch_inject.pos[0]       = 0;
                batch_inject.n_seq_id[0]  = 1;
                batch_inject.seq_id[0][0] = 0;
                batch_inject.logits[0]    = false;
                if (llama_decode(ctx_dft, batch_inject) != 0) {
                    LOG_WRN("%s: drafter inject warmup decode failed (non-fatal)\n", __func__);
                }

                // draft graph: one noise block (mask tokens are valid vocab ids)
                if (mask_token_id >= 0) {
                    common_batch_clear(batch);
                    const int32_t n_wtok = is_dflash2
                        ? block_size
                        : this->params.n_max + (is_dspark && sample_from_anchor ? 0 : 1);
                    for (int32_t i = 0; i < n_wtok; ++i) {
                        common_batch_add(batch, mask_token_id, i + 1, { 0 }, true);
                    }
                    if (llama_decode(ctx_dft, batch) != 0) {
                        LOG_WRN("%s: drafter draft warmup decode failed (non-fatal)\n", __func__);
                    } else if (gpu_sample && !is_dflash2 && !llama_get_logits_argmax_gpu(ctx_dft)) {
                        // the sched placed the sampling tail on the CPU backend (e.g. -ngld 0
                        // or --spec-draft-device none): only the GPU argmax kernels implement
                        // the extended top-K ids/log-probs layout, so the in-graph results
                        // would be uninitialized garbage — sample on the host instead
                        gpu_sample = false;
                        llama_set_dflash_argmax(ctx_dft, false);
                        llama_set_dflash_topk(ctx_dft, 1);
                        LOG_INF("%s: - draft sampling on host (drafter logits on CPU)\n", __func__);
                    }

                    // fused single-graph cycle: warm the steady-state shape (carry
                    // contents are junk here; every output is discarded)
                    if (oneg()) {
                        llama_memory_t mem = llama_get_memory(ctx_dft);
                        if (mem) {
                            llama_memory_clear(mem, true);
                        }
                        common_batch_clear(batch);
                        common_batch_add(batch, mask_token_id, 0, { 0 }, false);
                        for (int32_t i = 1; i < oneg_n_inject; ++i) {
                            common_batch_add(batch, mask_token_id, 0, { oneg_scratch_seq }, false);
                        }
                        for (int32_t i = 0; i < n_wtok; ++i) {
                            common_batch_add(batch, mask_token_id, i + 1, { 0 }, true);
                        }
                        stage_rows.assign(oneg_n_inject, 0);
                        llama_set_dflash_inject_rows(ctx_dft, stage_rows.data(), oneg_n_inject);
                        llama_set_dflash_oneg_inject(ctx_dft, carry_handle, oneg_n_inject);
                        if (llama_decode(ctx_dft, batch) != 0) {
                            LOG_WRN("%s: drafter fused warmup decode failed (non-fatal)\n", __func__);
                        }
                        llama_set_dflash_oneg_inject(ctx_dft, carry_handle, 0);
                    }
                }

                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                if (mem_dft) {
                    llama_memory_clear(mem_dft, true);
                }
                llama_synchronize(ctx_dft);
                llama_perf_context_reset(ctx_dft);
                llama_set_warmup(ctx_dft, false);

                LOG_INF("%s: - drafter warmup complete (inject + draft graphs)\n", __func__);
            }
        }
    }

    ~common_speculative_impl_draft_dflash() override {
        llama_batch_free(batch);
        llama_batch_free(batch_inject);
    }

    void discard_branch_state(llama_seq_id seq_id) {
        stash[seq_id] = {};
        seq_in_gen[seq_id] = false;
        proposals[seq_id] = {};
        adpt[seq_id] = {};
        pregate_empty[seq_id] = 0;
        pregate_skip [seq_id] = 0;
    }

    void sequence_transition(
            llama_seq_id seq_id,
            common_speculative_sequence_event event) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // Deferred carry, proposal probabilities, and adaptive/pregate history
        // describe the branch that was live before the restore. None of them is
        // part of a context checkpoint, so they must not cross the frontier swap.
        discard_branch_state(seq_id);

        switch (event) {
            case common_speculative_sequence_event::target_restored_without_draft:
                llama_memory_seq_rm(
                    llama_get_memory(params.ctx_dft), seq_id, -1, -1);
                positions[seq_id].rebase_pending = true;
                positions[seq_id].rebased = true;
                break;
            case common_speculative_sequence_event::target_replaced:
            case common_speculative_sequence_event::full_clear:
                positions[seq_id] = {};
                break;
            case common_speculative_sequence_event::prompt_rewind:
            case common_speculative_sequence_event::live_range_shift:
                break;
            case common_speculative_sequence_event::draft_image_restored:
            case common_speculative_sequence_event::composite_image_restored: {
                // The compact text drafter and an M-RoPE target can use different
                // physical positions. A full sequence image restores only the KV,
                // not this lightweight offset; derive it before the first suffix
                // injection rather than waiting for draft()-side recovery.
                const llama_pos target_max = llama_memory_seq_pos_max(
                    llama_get_memory(params.ctx_tgt), seq_id);
                const llama_pos draft_max = llama_memory_seq_pos_max(
                    llama_get_memory(params.ctx_dft), seq_id);
                if (target_max >= 0 && draft_max >= 0 && target_max >= draft_max) {
                    positions[seq_id].target_base = target_max - draft_max;
                    positions[seq_id].rebase_pending = false;
                    positions[seq_id].rebased = positions[seq_id].target_base != 0;
                } else {
                    positions[seq_id].rebase_pending = true;
                    positions[seq_id].rebased = true;
                }
                break;
            }
        }
    }

    // standalone injection of a deferred stash out of the carry tensor — the old
    // immediate injection, one cycle late. Used when the fused draft decode did not
    // consume the stash (skipped-draft cycles, request tails, fallback shapes).
    bool flush_stash(llama_seq_id seq_id) {
        auto & st = stash[seq_id];
        st.pending = false;

        auto * ctx_dft = params.ctx_dft;
        llama_memory_t mem = llama_get_memory(ctx_dft);

        // A gap below the deferred block proves that a restore/rewind discarded
        // the carry's branch. Clear the unauthenticated transient drafter
        // sequence and let ordinary prefill reconstruct it from target hidden
        // state. A frontier at or beyond the block is a normal DFlash repair
        // shape: provisional rows are removed below and the authenticated
        // carry is reapplied.
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, seq_id);
        if (pos_max < st.pos0 - 1) {
            LOG_INF("%s: drafter seq %d rewind crossed deferred inject "
                    "(pos_max=%d, pos0=%d) - clearing transient sequence\n",
                    __func__, (int) seq_id, (int) pos_max, (int) st.pos0);
            llama_memory_seq_rm(mem, seq_id, -1, -1);
            seq_in_gen[seq_id] = false;
            proposals[seq_id] = {};
            adpt[seq_id] = {};
            pregate_empty[seq_id] = 0;
            pregate_skip [seq_id] = 0;
            return true;
        }

        llama_memory_seq_rm(mem, seq_id, st.pos0, -1);

        stage_rows.resize(st.n_rows);
        for (int32_t i = 0; i < st.n_rows; ++i) {
            stage_rows[i] = seq_id * carry_rows_per_seq + i;
        }
        llama_set_dflash_inject_stage(ctx_dft, carry_handle);
        llama_set_dflash_inject_rows(ctx_dft, stage_rows.data(), st.n_rows);

        batch_inject.n_tokens = st.n_rows;
        for (int32_t i = 0; i < st.n_rows; ++i) {
            batch_inject.pos[i]       = st.pos0 + i;
            batch_inject.n_seq_id[i]  = 1;
            batch_inject.seq_id[i][0] = seq_id;
            batch_inject.logits[i]    = false;
        }
        const int32_t rc = llama_decode(ctx_dft, batch_inject);
        llama_set_dflash_inject_stage(ctx_dft, stage_handle);
        if (rc != 0) {
            LOG_ERR("%s: deferred inject flush failed rc=%d (seq=%d, n=%d)\n",
                    __func__, rc, (int) seq_id, (int) st.n_rows);
            return false;
        }
        return true;
    }

    void set_rng_seed(llama_seq_id seq_id, uint32_t seed) {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        proposal_rngs[seq_id].seed(seed ^ 0x85ebca6bU);
    }

    void prepare_proposal_uniforms(llama_seq_id seq_id) {
        if (!is_dflash2 || sample_temp <= 0.0f ||
                seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }
        auto & values = proposal_uniforms[seq_id];
        values.resize((size_t) block_size - 1);
        for (float & value : values) {
            value = std::generate_canonical<float, 24>(proposal_rngs[seq_id]);
        }
        llama_set_dflash_proposal_uniforms(params.ctx_dft, seq_id,
                values.data(), (int32_t) values.size());
    }

    bool capture_proposal(
            llama_seq_id seq_id,
            const llama_tokens & selected,
            const llama_dflash_proposal_view & view,
            int32_t block) {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return false;
        }
        auto & proposal = proposals[seq_id];
        proposal = {};
        if (selected.empty() || !view.candidate_ids || !view.q_rows ||
                view.top_k <= 0 || view.n_steps < (int32_t) selected.size() ||
                block < 0 || block >= view.n_blocks) {
            return false;
        }

        proposal.seq_id = seq_id;
        proposal.selected = selected;
        proposal.top_k = view.top_k;
        proposal.q_covered_tokens = selected.size();
        proposal.candidate_ids.resize(selected.size() * (size_t) view.top_k);
        proposal.q_rows.resize(proposal.candidate_ids.size());

        const size_t block_stride = (size_t) view.n_steps * view.top_k;
        const size_t src = (size_t) block * block_stride;
        std::copy_n(view.candidate_ids + src, proposal.candidate_ids.size(),
                proposal.candidate_ids.begin());
        std::copy_n(view.q_rows + src, proposal.q_rows.size(),
                proposal.q_rows.begin());

        constexpr float sum_tol = 2e-4f;
        for (size_t row = 0; row < selected.size(); ++row) {
            float sum = 0.0f;
            bool found = false;
            for (int32_t k = 0; k < proposal.top_k; ++k) {
                const size_t i = row * (size_t) proposal.top_k + k;
                const float q = proposal.q_rows[i];
                if (!std::isfinite(q) || q < 0.0f) {
                    proposal = {};
                    return false;
                }
                sum += q;
                found |= proposal.candidate_ids[i] == selected[row] && q > 0.0f;
            }
            if (!found || std::abs(sum - 1.0f) > sum_tol) {
                proposal = {};
                return false;
            }
        }
        proposal.exact_q = true;
        return true;
    }

    void dflash2_finish_adaptive_cycle(llama_seq_id seq_id, int64_t now) {
        auto & st = adpt[seq_id];
        if (st.cycle_start_us <= 0 || st.cycle_tokens <= 0) {
            st.cycle_start_us = 0;
            st.cycle_tokens = 0;
            st.cycle_is_sample = false;
            return;
        }

        const int64_t elapsed = std::max<int64_t>(1, now - st.cycle_start_us);
        if (st.cycle_is_sample) {
            st.sample_us += elapsed;
            st.sample_tokens += st.cycle_tokens;
            ++st.sample_cycles;
        } else {
            if (st.cap < 0) {
                st.full_us += elapsed;
                st.full_tokens += st.cycle_tokens;
            }
            if (!st.sweep && st.hold > 0 && --st.hold == 0) {
                st.cap = -1;
                st.n_low_acc = 0;
                st.recovery_cycles = 0;
                st.recovery_drafts = 0;
                st.recovery_accepts = 0;
                st.full_us = 0;
                st.full_tokens = 0;
                LOG_DBG("%s: seq %d periodic full-depth probe\n", __func__, (int) seq_id);
            }
        }
        st.cycle_start_us = 0;
        st.cycle_tokens = 0;
        st.cycle_is_sample = false;

        if (!st.sweep) {
            return;
        }

        if (st.candidate == 0) {
            if (st.sample_cycles < 4) {
                return;
            }
            const double target_us_per_token =
                (double) st.sample_us / std::max(1, st.sample_tokens);
            if (st.best_us_per_token < 0.80 * target_us_per_token) {
                st.cap = -1;
                st.sweep = false;
                st.hold = 512;
                st.sample_cycles = 0;
                st.sample_us = 0;
                st.sample_tokens = 0;
                st.full_us = 0;
                st.full_tokens = 0;
                LOG_DBG("%s: seq %d full %.2f vs target-only %.2f ms/token - retaining full depth\n",
                        __func__, (int) seq_id, st.best_us_per_token / 1e3,
                        target_us_per_token / 1e3);
                return;
            }

            st.candidate = 1;
            st.cap = 2;
            st.sample_cycles = 0;
            st.sample_us = 0;
            st.sample_tokens = 0;
            LOG_DBG("%s: seq %d full %.2f vs target-only %.2f ms/token - measuring depth 2\n",
                    __func__, (int) seq_id, st.best_us_per_token / 1e3,
                    target_us_per_token / 1e3);
            return;
        }

        if (st.sample_cycles < 16) {
            return;
        }

        const double depth2_us_per_token =
            (double) st.sample_us / std::max(1, st.sample_tokens);
        // The exhaustive panel establishes depth 2 as the robust low-match prior.
        // Favor it inside a 5% noise band; reject it when the measured arm is clearly
        // slower (as on medium-match prose), where full batching is decisive.
        const bool depth2_wins = depth2_us_per_token <= 1.05 * st.best_us_per_token;
        st.cap = depth2_wins ? 2 : -1;
        st.sweep = false;
        st.hold = depth2_wins ? 256 : 512;
        st.n_low_acc = 0;
        st.sample_cycles = 0;
        st.sample_us = 0;
        st.sample_tokens = 0;
        st.recovery_cycles = 0;
        st.recovery_drafts = 0;
        st.recovery_accepts = 0;
        st.full_us = 0;
        st.full_tokens = 0;
        LOG_DBG("%s: seq %d full %.2f vs depth-2 %.2f ms/token - selected %s\n",
                __func__, (int) seq_id, st.best_us_per_token / 1e3,
                depth2_us_per_token / 1e3, depth2_wins ? "depth 2" : "full depth");
    }

    void dflash2_start_adaptive_sweep(llama_seq_id seq_id) {
        auto & st = adpt[seq_id];
        if (st.full_tokens <= 0) {
            st.sweep_pending = false;
            return;
        }

        st.best_us_per_token = (double) st.full_us / st.full_tokens;
        st.sweep = true;
        st.sweep_pending = false;
        st.hold = 0;
        st.candidate = 0;
        st.sample_cycles = 0;
        st.sample_us = 0;
        st.sample_tokens = 0;
        st.cap = 0;
        st.n_low_acc = 0;
        st.recovery_cycles = 0;
        st.recovery_drafts = 0;
        st.recovery_accepts = 0;
        st.full_us = 0;
        st.full_tokens = 0;
        LOG_DBG("%s: seq %d full-depth baseline %.2f ms/token - measuring target-only\n",
                __func__, (int) seq_id, st.best_us_per_token / 1e3);
    }

    int32_t dflash2_adaptive_depth(llama_seq_id seq_id, int32_t n_max) {
        auto & st = adpt[seq_id];
        dflash2_finish_adaptive_cycle(seq_id, ggml_time_us());
        if (st.sweep_pending) {
            dflash2_start_adaptive_sweep(seq_id);
        }

        const int32_t depth = st.cap >= 0 ? std::min(st.cap, n_max) : n_max;
        st.cycle_start_us = ggml_time_us();
        st.cycle_is_sample = st.sweep;
        if (depth == 0) {
            // No speculative implementation owns this cycle, so accept() will not be
            // called. The next draft invocation closes a one-target-token sample.
            st.cycle_tokens = 1;
        }
        return depth;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // fresh request: reset the adaptive draft-length state for this seq
        adpt[seq_id] = {};
        proposals[seq_id] = {};

        pregate_empty[seq_id] = 0;
        pregate_skip [seq_id] = 0;

        // back to prefill: injections go through the standalone path again
        seq_in_gen[seq_id] = false;

        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(params.ctx_dft), seq_id);
        if (stash[seq_id].pending) {
            // a deferred injection still covers these rows (flushed on the next process)
            pos_max = std::max(pos_max, stash[seq_id].pos0 + stash[seq_id].n_rows - 1);
        }
        if (!positions[seq_id].rebased && pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - process() did not run on every prefill ubatch. "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // Target processing supplies either token IDs or multimodal embeddings.
        // Mixed/empty representations do not carry an injectable DFlash batch.
        const bool has_tokens     = batch_in.token != nullptr;
        const bool has_embeddings = batch_in.embd  != nullptr;
        if (has_tokens == has_embeddings) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // per-seq inclusive batch range (assumes each seq's tokens are contiguous in the batch)
        std::vector<int32_t> i_batch_beg(n_seq, -1);
        std::vector<int32_t> i_batch_end(n_seq, -1);
        for (int32_t k = 0; k < n_tokens; ++k) {
            GGML_ASSERT(batch_in.n_seq_id[k] == 1);
            const llama_seq_id seq_id = batch_in.seq_id[k][0];
            if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                continue;
            }
            i_batch_end[seq_id] = k;
            if (i_batch_beg[seq_id] < 0) {
                i_batch_beg[seq_id] = k;
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // A vision/audio batch carries multi-axis target positions and is outside the
        // text drafter's training contract. Clear only the affected sequences and wait
        // for the next text capture to establish a compact local position stream.
        if (has_embeddings) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }
                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, -1, -1)) {
                    LOG_ERR("%s: failed to clear drafter sequence %d at media boundary\n",
                            __func__, (int) seq_id);
                    return false;
                }
                stash[seq_id] = {};
                seq_in_gen[seq_id] = false;
                adpt[seq_id] = {};
                pregate_empty[seq_id] = 0;
                pregate_skip [seq_id] = 0;
                positions[seq_id].rebase_pending = true;
                positions[seq_id].rebased = true;
            }
            return true;
        }

        const int32_t n_ubatch = (int32_t) llama_n_ubatch(ctx_dft);

        const int64_t t_proc0 = ggml_time_us();

        // device-staged path: the target graph already wrote this batch's features into
        // the stage tensor (interleaved) — the inject decodes gather rows on-device and
        // the whole host gather is skipped. Multi-ubatch target batches (stage_valid_n
        // == 0) fall back to the host path: unbind the stage so the inject graph builds
        // its H2D feature input, rebind at the end.
        const bool use_stage = staged &&
                llama_dflash_draft_stage_valid_n(ctx_tgt) == batch_in.n_tokens;

        // phase C: flush stashes the fused draft did not consume (skipped-draft cycles,
        // request tails, multi-view iterations) — the carry keeps their rows valid, so
        // this is the old immediate injection one cycle late
        if (oneg()) {
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (stash[s].pending && !flush_stash(s)) {
                    return false;
                }
            }
        }

        if (staged && !use_stage) {
            llama_set_dflash_inject_stage(ctx_dft, nullptr);
        }
        if (use_stage) {
            // the drafter's sched reads the stage on its own streams: fence the target's
            // in-flight compute (the host path synced implicitly via the D2H readback)
            llama_synchronize(ctx_tgt);
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }
            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            const llama_pos target_pos0 = batch_in.pos[i_batch_beg[seq_id]];
            auto & pstate = positions[seq_id];
            const llama_pos dft_pos_max = llama_memory_seq_pos_max(
                    llama_get_memory(ctx_dft), seq_id);
            if (target_pos0 == 0) {
                pstate = {};
            } else if (pstate.rebase_pending || dft_pos_max < 0) {
                pstate.target_base = target_pos0;
                pstate.rebase_pending = false;
                pstate.rebased = true;
                LOG_DBG("%s: seq %d rebased target position %d to drafter position 0\n",
                        __func__, (int) seq_id, (int) target_pos0);
            }
            llama_pos dft_pos0 = target_pos0 - pstate.target_base;
            const bool discontinuous = dft_pos0 < 0 ||
                (dft_pos_max >= 0 &&
                 int64_t(dft_pos0) > int64_t(dft_pos_max) + 1);
            if (discontinuous) {
                // A lifecycle owner should have announced every external
                // target/draft replacement. Keep this as a final fail-safe:
                // an unannounced prompt-cache swap may leave the lightweight
                // target base describing a different conversation even though
                // both context images are individually valid. Cold-rebase the
                // drafter rather than turning the request into an HTTP 500.
                LOG_WRN(
                    "%s: seq %d drafter stream discontinuous "
                    "(dft_pos_max=%d, dft_pos0=%d, target_pos0=%d, base=%d) - "
                    "resetting drafter sequence\n",
                    __func__, (int) seq_id, (int) dft_pos_max, (int) dft_pos0,
                    (int) target_pos0, (int) pstate.target_base);
                sequence_transition(
                    seq_id,
                    common_speculative_sequence_event::target_restored_without_draft);
                pstate.target_base = target_pos0;
                pstate.rebase_pending = false;
                pstate.rebased = true;
                dft_pos0 = 0;
            }

            // phase C: defer generation-cycle injections into the next draft decode.
            // Only rows committed by then get injected, dropping the historical
            // inject-rejected-then-trim round trip on this path. The carry copy keeps
            // the rows alive past the next target decode.
            if (oneg() && use_stage && seq_in_gen[seq_id] &&
                n_rows <= carry_rows_per_seq &&
                llama_dflash_draft_stage_carry(ctx_tgt, i_batch_beg[seq_id], n_rows,
                        (int32_t) seq_id * carry_rows_per_seq)) {
                stash[seq_id] = { true, dft_pos0, n_rows };
                continue;
            }

            // trim stale drafter cells at/after this injection's start (leftover noise
            // or rejected-draft injections from cycles where the server truncated or
            // skipped drafting) — otherwise the batch position-continuity check rejects
            // the injection. Mirrors eagle3's process()-side seq_rm.
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id,
                    dft_pos0, -1);

            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                if (use_stage) {
                    stage_rows.resize(n_chunk);
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        stage_rows[i] = i_batch_beg[seq_id] + offset + i;
                    }
                    llama_set_dflash_inject_rows(ctx_dft, stage_rows.data(), n_chunk);
                } else {
                    // gather this chunk's target features, interleaved by extract layer;
                    // fused mode writes straight into the injection batch (the decode
                    // graph applies the encoder itself), unfused goes through
                    // features_buf + encode
                    float * gather_dst = fused_inject ? batch_inject.embd : nullptr;
                    if (!fused_inject) {
                        features_buf.resize((size_t) n_chunk * n_embd_enc);
                        gather_dst = features_buf.data();
                    }
                    for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
                        const float * layer = llama_get_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k]);
                        if (!layer) {
                            GGML_ABORT("DFlash: target layer %d input not extracted.", target_layer_ids[k]);
                        }
                        for (int32_t i = 0; i < n_chunk; ++i) {
                            float       * dst = gather_dst + (size_t) i * n_embd_enc + k * (size_t) n_embd_tgt;
                            const float * src = layer + (size_t) (i_batch_beg[seq_id] + offset + i) * n_embd_tgt;
                            std::memcpy(dst, src, (size_t) n_embd_tgt * sizeof(float));
                        }
                    }
                }

                if (fused_inject) {
                    batch_inject.n_tokens = n_chunk;
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        batch_inject.pos[i]       = dft_pos0 + offset + i;
                        batch_inject.n_seq_id[i]  = 1;
                        batch_inject.seq_id[i][0] = seq_id;
                        batch_inject.logits[i]    = false;
                    }
                    const int32_t rc = llama_decode(ctx_dft, batch_inject);
                    if (rc != 0) {
                        LOG_ERR("%s: llama_decode(ctx_dft) fused inject failed rc=%d (n_tokens=%d, offset=%d)\n",
                                __func__, rc, (int) n_chunk, (int) offset);
                        return false;
                    }
                    continue;
                }

                // fuse extracted features through DFlash encoder
                // M-RoPE drafts read 4 position rows per token from embd batches, so pass them explicitly
                std::vector<llama_pos> enc_pos;
                if (is_mrope) {
                    enc_pos.resize((size_t) 4 * n_chunk);
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const llama_pos p = batch_in.pos[i_batch_beg[seq_id] + offset + i];
                        enc_pos[0 * n_chunk + i] = p;
                        enc_pos[1 * n_chunk + i] = p;
                        enc_pos[2 * n_chunk + i] = p;
                        enc_pos[3 * n_chunk + i] = 0;
                    }
                }

                llama_batch enc_batch = {
                    /*.n_tokens =*/ n_chunk,
                    /*.token    =*/ nullptr,
                    /*.embd     =*/ features_buf.data(),
                    /*.pos      =*/ is_mrope ? enc_pos.data() : nullptr,
                    /*.n_seq_id =*/ nullptr,
                    /*.seq_id   =*/ nullptr,
                    /*.logits   =*/ nullptr,
                };

                int32_t rc = llama_encode(ctx_dft, enc_batch);
                if (rc != 0) {
                    LOG_ERR("%s: llama_encode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }

                const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
                GGML_ASSERT(inp_g && "DFlash encoder produced no output.");

                // inject the DFlash decoder K/V cache at the tokens' target positions
                batch_inject.n_tokens = n_chunk;
                std::memcpy(batch_inject.embd, inp_g, (size_t) n_chunk * n_embd_dec * sizeof(float));

                for (int32_t i = 0; i < n_chunk; ++i) {
                    const llama_pos p = dft_pos0 + offset + i;
                    batch_inject.pos[i] = p;
                    if (is_mrope) {
                        batch_inject.pos[1 * n_chunk + i] = p;
                        batch_inject.pos[2 * n_chunk + i] = p;
                        batch_inject.pos[3 * n_chunk + i] = 0;
                    }
                    batch_inject.n_seq_id[i]  = 1;
                    batch_inject.seq_id[i][0] = seq_id;
                    batch_inject.logits[i]    = false;
                }
                rc = llama_decode(ctx_dft, batch_inject);
                if (rc != 0) {
                    LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                            __func__, rc, (int) n_chunk, (int) offset);
                    return false;
                }
            }
        }

        if (staged && !use_stage) {
            llama_set_dflash_inject_stage(ctx_dft, stage_handle);
        }

        LOG_DBG("%s: process (capture+%s+inject) %.2f ms (%d tokens)\n",
                __func__, use_stage ? "staged" : fused_inject ? "fused" : "encode",
                (ggml_time_us() - t_proc0) / 1e3, (int) n_tokens);

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (dparams[seq_id].drafting) {
                proposals[seq_id] = {};
            }
        }

        // pre-gate: decline the whole call for seqs in backoff (like a null draft —
        // the server re-arms drafting every cycle; a pending stash flushes through the
        // standard paths). Note: clearing the flag also skips later-chained impls.
        if (pregate) {
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (dparams[s].drafting && pregate_skip[s] > 0) {
                    pregate_skip[s]--;
                    dparams[s].drafting = false;
                }
            }
        }

        // phase C: fuse the deferred injection into this decode when the batch has the
        // single-drafting-seq shape (the server drives this path per slot). Other
        // shapes flush their stashes standalone and keep the two-decode path.
        llama_seq_id seq_fused = -1;
        if (oneg()) {
            int n_armed = 0;
            for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                if (dparams[s].drafting) {
                    n_armed++;
                    seq_fused     = s;
                    seq_in_gen[s] = true;
                }
            }
            if (n_armed != 1 || !stash[seq_fused].pending) {
                seq_fused = -1;
            }
            if (seq_fused < 0) {
                for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
                    // the noise decode below reads the drafter cache — flush first
                    if (stash[s].pending && dparams[s].drafting && !flush_stash(s)) {
                        return;
                    }
                }
            }
        }

        // build one batch holding every drafting sequence's noise block into a single decode)
        // record where each block starts and its size
        std::vector<int32_t> i_block_beg(n_seq, -1);
        std::vector<int32_t> n_block    (n_seq,  0);
        std::vector<int32_t> n_emit     (n_seq,  0);

        int32_t out_off = 0; // output rows start after the fused inject rows

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (!gpu_sample) {
                common_sampler_reset(smpls[seq_id].get());
            }

            auto & pstate = positions[seq_id];
            llama_pos n_local = dp.n_past - pstate.target_base;
            const llama_pos pos_max = llama_memory_seq_pos_max(
                    llama_get_memory(ctx_dft), seq_id);
            // A multimodal prompt-cache entry can restore compact drafter cells into a
            // different slot without restoring this lightweight offset. Recover it from
            // the only valid frontier relation before constructing the next noise block.
            if (!stash[seq_id].pending && pos_max >= 0 && n_local > pos_max + 1) {
                pstate.target_base = dp.n_past - (pos_max + 1);
                pstate.rebased = true;
                n_local = pos_max + 1;
                LOG_DBG("%s: seq %d recovered drafter base %d from cached frontier\n",
                        __func__, (int) seq_id, (int) pstate.target_base);
            }
            if (n_local < 0 || n_local > INT32_MAX) {
                LOG_WRN("%s: seq %d cannot map target frontier %d from base %d\n",
                        __func__, (int) seq_id, (int) dp.n_past,
                        (int) pstate.target_base);
                continue;
            }
            const int32_t n = (int32_t) n_local;

            int32_t n_max_eff = params.n_max;
            // honor the per-call cap (e.g. remaining token budget) instead of drafting
            // past it and having the wrapper truncate — avoids wasted mask positions
            // and stale drafter cells beyond the verify range
            if (dp.n_max >= 0) {
                n_max_eff = std::min(n_max_eff, dp.n_max);
            }
            int32_t n_draft = n_max_eff;
            if (adaptive) {
                if (is_dflash2) {
                    n_draft = dflash2_adaptive_depth(seq_id, n_max_eff);
                } else if (adpt[seq_id].cap > 0) {
                    n_draft = std::min(adpt[seq_id].cap, n_max_eff);
                }
            }
            if (n_draft <= 0) {
                continue;
            }

            llama_pos trim_pos = n;

            if (seq_id == seq_fused) {
                auto & st = stash[seq_id];
                const int32_t n_acc = n - st.pos0;
                if (n_acc < 1 || n_acc > st.n_rows ||
                    llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) < st.pos0 - 1) {
                    // rollback or frontier drift — not the steady-state shape
                    if (n_acc <= 0) {
                        st.pending = false; // every deferred row was rejected: drop
                    } else if (!flush_stash(seq_id)) {
                        return;
                    }
                    seq_fused = -1;
                } else {
                    // committed rows at [pos0, n_past), padded to the fixed row count by
                    // repeating the last row into the scratch seq (the KV mask hides it
                    // from real attention; its cells are dropped right after the decode)
                    stage_rows.resize(oneg_n_inject);
                    for (int32_t i = 0; i < oneg_n_inject; ++i) {
                        const int32_t k = std::min(i, n_acc - 1);
                        stage_rows[i] = (int32_t) seq_id * carry_rows_per_seq + k;
                        common_batch_add(batch, mask_token_id, st.pos0 + k,
                                { i < n_acc ? seq_id : oneg_scratch_seq }, false);
                    }
                    st.pending = false;
                    out_off    = oneg_n_inject;
                    trim_pos   = st.pos0;
                }
            }

            // trim stale drafter cells at/after the anchor position (rejected verify
            // tokens' injections and any prior noise) — without this the batch position
            // continuity check rejects the draft decode after every partial-accept
            // cycle (~1/3 of cycles on Muse). Same pre-decode seq_rm draft_simple does.
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, trim_pos, -1);

            const int32_t n_block_tokens = is_dflash2
                ? block_size
                : n_draft + (is_dspark && sample_from_anchor ? 0 : 1);
            i_block_beg[seq_id] = batch.n_tokens;
            n_block    [seq_id] = n_block_tokens;
            n_emit     [seq_id] = n_draft;
            prepare_proposal_uniforms(seq_id);
            for (int32_t i = 0; i < n_block_tokens; ++i) {
                // The fork DFlash2 path consumes the in-graph argmax rows below
                // (including temperature-zero drafting), so every proposal row
                // must remain an output. The upstream compatibility path's
                // proposal-only flag cannot be applied to this implementation.
                common_batch_add(batch, i == 0 ? dp.id_last : mask_token_id, n + i, { seq_id }, true);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        if (out_off > 0) {
            llama_set_dflash_inject_rows(ctx_dft, stage_rows.data(), oneg_n_inject);
            llama_set_dflash_oneg_inject(ctx_dft, carry_handle, oneg_n_inject);
        }

        // decode all sequence's noise block in a single batch
        const int64_t t_dec0 = ggml_time_us();
        int ret = llama_decode(ctx_dft, batch);
        if (out_off > 0) {
            llama_set_dflash_oneg_inject(ctx_dft, carry_handle, 0);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), oneg_scratch_seq, -1, -1);
        }
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }
        llama_synchronize(ctx_dft);
        LOG_DBG("%s: draft decode %.2f ms (%d tokens%s)\n",
                __func__, (ggml_time_us() - t_dec0) / 1e3, (int) batch.n_tokens,
                out_off > 0 ? ", fused inject" : "");
        last_draft_model_decode_succeeded = true;

        // GPU sampling results: K ids + log-probs per OUTPUT row, row order == batch
        // order (every noise token requests logits; fused inject rows produce none, so
        // output row = batch index - out_off)
        const int32_t * g_ids = nullptr;
        const float   * g_lps = nullptr;
        int32_t         g_K   = 0;
        if (gpu_sample && !is_dflash2 && !llama_get_logits_argmax_gpu(ctx_dft)) {
            // the sched placed the sampling tail on the CPU backend, whose plain argmax
            // kernel leaves the extended top-K layout uninitialized (the warmup decode
            // normally catches this at init; cover the warmup-disabled path too). Raw
            // logits were skipped for THIS decode, so produce no drafts this round and
            // sample on the host from the next decode on
            gpu_sample = false;
            llama_set_dflash_argmax(ctx_dft, false);
            llama_set_dflash_topk(ctx_dft, 1);
            LOG_INF("%s: draft sampling on host (drafter logits on CPU)\n", __func__);
            return;
        }
        if (gpu_sample) {
            g_ids = llama_get_logits_argmax(ctx_dft);
            g_lps = llama_get_logits_argmax_probs(ctx_dft);
            g_K   = llama_get_logits_argmax_k(ctx_dft);
            if (!g_ids || (!is_dflash2 && !g_lps) || g_K <= 0 ||
                    llama_get_logits_argmax_n(ctx_dft) != batch.n_tokens - out_off) {
                // raw logits were skipped in favor of the in-graph tail, so there is no
                // CPU fallback for THIS decode — produce no drafts (safe) and warn
                LOG_WRN("%s: GPU draft sampling results unavailable (rows=%d, batch=%d) - skipping draft\n",
                        __func__, (int) llama_get_logits_argmax_n(ctx_dft), (int) (batch.n_tokens - out_off));
                return;
            }
        }

        llama_dflash_proposal_view proposal_view;
        const bool have_proposal = is_dflash2 && sample_temp > 0.0f &&
            llama_get_dflash_proposal(ctx_dft, &proposal_view);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
                continue;
            }
            auto & dp = dparams[seq_id];

            const int32_t beg            = i_block_beg[seq_id];
            const int32_t n_block_tokens = n_block[seq_id];
            const int32_t n_emit_tokens  = n_emit[seq_id];

            auto * smpl = smpls[seq_id].get();

            auto & result = *dp.result;

            // top-1 prob renormalized over the row's top-K == the CPU chain's softmax over
            // its top-k candidates (the full-vocab log-normalizer cancels in the ratio)
            auto gpu_top_p = [&](int32_t idx) {
                const float * lp = g_lps + (size_t) idx * g_K;
                float sum = 0.0f;
                for (int32_t j = 0; j < g_K; ++j) {
                    sum += expf(lp[j] - lp[0]);
                }
                return 1.0f / sum;
            };

            if (is_dspark) {
                // DSpark predicts the next token from position 0 and optionally truncates
                // at the first position below the confidence threshold.
                const float * conf = params.p_min > 0.0f ? llama_get_embeddings_nextn(ctx_dft) : nullptr;
                const int32_t i_draft_beg = sample_from_anchor ? 0 : 1;
                for (int32_t i = i_draft_beg; i < n_block_tokens; ++i) {
                    const int32_t idx = beg + i;
                    const int32_t row = idx - out_off; // conf/argmax rows are output rows

                    if (conf && conf[(size_t) row * n_embd_dec] < params.p_min) {
                        break;
                    }

                    if (g_ids) {
                        const llama_token id = (llama_token) g_ids[(size_t) row * g_K];
                        LOG_DBG(" - seq_id %d, draft pos %3d: %6d (lp %8.3f) '%s'\n",
                                seq_id, i, id, g_lps[(size_t) row * g_K],
                                common_token_to_piece(ctx_dft, id).c_str());
                        result.push_back(id);
                        continue;
                    }

                    common_sampler_sample(smpl, ctx_dft, idx, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            } else {
                // greedily read the predicted block at this sequence's noise positions 1..n_block_tokens-1
                for (int32_t i = 1; i < n_block_tokens && i <= n_emit_tokens; ++i) {
                    if (g_ids) {
                        const int32_t row = beg + i - out_off; // argmax rows are output rows
                        const llama_token id = (llama_token) g_ids[(size_t) row * g_K];
                        if (params.p_min > 0.0f && g_lps && gpu_top_p(row) < params.p_min) {
                            break;
                        }
                        if (g_lps) {
                            LOG_DBG(" - seq_id %d, draft pos %3d: %6d (lp %8.3f) '%s'\n",
                                    seq_id, i - 1, id, g_lps[(size_t) row * g_K],
                                    common_token_to_piece(ctx_dft, id).c_str());
                        } else {
                            LOG_DBG(" - seq_id %d, draft pos %3d: %6d '%s'\n",
                                    seq_id, i - 1, id,
                                    common_token_to_piece(ctx_dft, id).c_str());
                        }
                        result.push_back(id);
                        continue;
                    }

                    common_sampler_sample(smpl, ctx_dft, beg + i, true);

                    const auto * cur_p = common_sampler_get_candidates(smpl, true);

                    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                        LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                                seq_id, k, i - 1, cur_p->data[k].id, cur_p->data[k].p,
                                common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }

                    const llama_token id = cur_p->data[0].id;

                    if (cur_p->data[0].p < params.p_min) {
                        break;
                    }

                    common_sampler_accept(smpl, id, true);

                    result.push_back(id);
                }
            }

            if (result.size() < (size_t) params.n_min) {
                result.clear();
            }

            if (have_proposal && !result.empty()) {
                const int32_t block = (beg - out_off) / block_size;
                capture_proposal(seq_id, result, proposal_view, block);
            }

            adpt[seq_id].n_draft_last = (int32_t) result.size();
            if (adaptive && is_dflash2 && result.empty()) {
                // No implementation will receive accept() for an empty result.
                adpt[seq_id].cycle_tokens = 1;
            }

            // pre-gate accounting (attempted drafts only — skipped/declined seqs never
            // reach this loop): first empty is free, then backoff 1,2,4,8
            if (pregate) {
                if (result.empty()) {
                    pregate_empty[seq_id] = std::min(pregate_empty[seq_id] + 1, 5);
                    if (pregate_empty[seq_id] >= 2) {
                        pregate_skip[seq_id] = 1 << (pregate_empty[seq_id] - 2);
                    }
                } else {
                    pregate_empty[seq_id] = 0;
                    pregate_skip [seq_id] = 0;
                }
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (!adaptive || seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        auto & st = adpt[seq_id];
        const int32_t n_last = st.n_draft_last;
        st.n_draft_last = 0;
        if (n_last <= 0) {
            return;
        }
        const int32_t n_accepted_own = std::min<int32_t>(n_accepted, n_last);

        if (is_dflash2) {
            // cycle_tokens measures total target progress, including a CopySpec
            // extension. Adaptive acceptance, however, must only score rows the
            // DFlash2 model itself proposed.
            st.cycle_tokens = (int32_t) n_accepted + 1;
            if (st.sweep) {
                return;
            }

            const int32_t n_max = std::min(params.n_max, block_size - 1);
            if (st.cap < 0 || st.cap >= n_max) {
                if (st.hold > 0) {
                    return;
                }
                ++st.recovery_cycles;
                st.recovery_drafts += n_last;
                st.recovery_accepts += n_accepted_own;
                if (n_accepted_own <= 1) {
                    ++st.n_low_acc;
                } else {
                    st.n_low_acc = 0;
                }

                const bool low_window = st.recovery_cycles >= 4 &&
                    10 * st.recovery_accepts <= 3 * st.recovery_drafts;
                if (low_window) {
                    st.sweep_pending = true;
                    st.n_low_acc = 0;
                    st.recovery_cycles = 0;
                    st.recovery_drafts = 0;
                    st.recovery_accepts = 0;
                } else if (st.recovery_cycles >= 4) {
                    st.recovery_cycles = 0;
                    st.recovery_drafts = 0;
                    st.recovery_accepts = 0;
                }
                return;
            }

            // A shallow winner must notice when prose turns into an easy code-like
            // region. Use a wide, high-threshold window so ordinary prose bursts do
            // not bounce repeatedly between depth 2 and full verification.
            ++st.recovery_cycles;
            st.recovery_drafts += n_last;
            st.recovery_accepts += n_accepted_own;
            if (st.recovery_cycles >= 32) {
                if (st.recovery_drafts > 0 &&
                        10 * st.recovery_accepts >= 9 * st.recovery_drafts) {
                    st.cap = -1;
                    st.hold = 0;
                    st.n_low_acc = 0;
                    LOG_DBG("%s: seq %d high-match region - restoring full depth\n",
                            __func__, (int) seq_id);
                }
                st.recovery_cycles = 0;
                st.recovery_drafts = 0;
                st.recovery_accepts = 0;
            }
            return;
        }

        const float f_acc = (float) n_accepted_own / (float) n_last;
        if (f_acc < 0.3f) {
            if (++st.n_low_acc >= 3) {
                const int32_t base = st.cap > 0 ? st.cap : params.n_max;
                st.cap = std::max(1, base / 2);
                st.n_low_acc = 0;
                LOG_DBG("%s: seq %d low-acceptance streak - draft cap -> %d\n",
                        __func__, (int) seq_id, st.cap);
            }
        } else {
            st.n_low_acc = 0;
            if (f_acc > 0.6f && st.cap > 0) {
                st.cap++;
                if (st.cap >= params.n_max) {
                    st.cap = -1; // fully recovered
                }
            }
        }
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // One MTP draft driver, three modes (set once in the ctor):
    //   is_mem_shared (gemma4): shares the target KV, runs all heads in one graph.
    //   chain_heads (step35): n_mtp_layers trained heads, one per draft step.
    //   neither (qwen35 / qwen35moe): a single trained MTP head.
    int32_t n_mtp_layers  = 1;
    bool    is_mem_shared = false;   // gemma4
    bool    chain_heads   = false;   // derived in the ctor: n_mtp_layers > 1 && !is_mem_shared

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]
    std::vector<common_speculative_mtp_carry_lifecycle> pending_h_lifecycle;

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // A single recurrent MTP head is commonly reused recursively.  The third
    // recursive prediction is model-dependent: it is valuable on Qwen3.6 but
    // not on Qwen3.8.  Probe it once per slot, then retain depth three only when
    // its marginal acceptance pays for the larger verify graph.
    bool adaptive_recursive_depth = false;
    std::vector<int32_t> adaptive_cap;
    std::vector<int32_t> adaptive_depth3_attempts;
    std::vector<int32_t> adaptive_depth3_accepts;
    std::vector<int32_t> adaptive_last_draft_size;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq, params.draft.n_max)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        if (this->params.ctx_mtp != nullptr) {
            // Keep every subsequent MTP operation (samplers, memory maintenance,
            // decode and teardown) on the same selected context.
            this->params.ctx_dft = this->params.ctx_mtp;
        }
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd_out(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");
        n_mtp_layers = std::max(1, (int) llama_model_n_layer_nextn(llama_get_model(ctx_dft)));

        SPC_TRC("%s", "adding speculative implementation 'draft-mtp'\n");
        SPC_TRC("- n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        SPC_TRC("- gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n",
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    SPC_WRN("backend offload failed for seq_id=%d; using CPU sampler\n", (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        // Every MTP context points at its target through ctx_other, but Qwen-family
        // MTP contexts own a separate filtered cache. Ask the memory about the cells
        // that the drafting algorithm actually depends on.
        is_mem_shared = llama_memory_has_shared_cells(llama_get_memory(ctx_dft));
        chain_heads   = n_mtp_layers > 1 && !is_mem_shared;

        const char * adaptive_env = getenv("GGML_MTP_DRAFT_ADAPTIVE");
        adaptive_recursive_depth = n_mtp_layers == 1 && !is_mem_shared && this->params.n_max == 3 &&
                                   !(adaptive_env && atoi(adaptive_env) == 0);
        adaptive_cap.assign(n_seq, this->params.n_max);
        adaptive_depth3_attempts.assign(n_seq, 0);
        adaptive_depth3_accepts.assign(n_seq, 0);
        adaptive_last_draft_size.assign(n_seq, 0);

        if (chain_heads) {
            this->params.n_max = std::min(this->params.n_max, n_mtp_layers);

            chain_h.assign(n_seq, {});
            for (auto & c : chain_h) {
                c.reserve((size_t) (this->params.n_max + 1) * n_embd);
            }
        }
        this->n_max = this->params.n_max;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));
        pending_h_lifecycle.resize(n_seq);

        i_last.assign(n_seq, -1);
        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);

        if (pos_max < N - 1 && !is_mem_shared) {
            SPC_WRN("ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        // pending_h is an activation, not part of either serialized sequence
        // image. A restored/rewound nonzero frontier therefore cannot safely
        // catch the draft model up: the predecessor target-hidden row is
        // unavailable. Stay target-only without touching the restored draft
        // sequence. A true cold frontier has a defined zero predecessor and can
        // start/re-arm normal MTP processing.
        if (!is_mem_shared) {
            const auto preflight = common_speculative_mtp_process_preflight_resolve(
                pending_h_lifecycle, i_batch_beg, batch_in.pos);
            if (preflight == common_speculative_mtp_process_preflight::target_only) {
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (i_batch_beg[seq_id] >= 0) {
                        pending_h_lifecycle[seq_id].target_process_skipped();
                        verify_h_rows[seq_id] = 0;
                    }
                }
                return true;
            }
        }

        // draft() may have advanced the MTP KV beyond this target batch. Rewind
        // to the verified frontier before replaying target hidden rows; M-RoPE
        // requires the stored position to be no greater than the incoming one.
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] >= 0) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id,
                    batch_in.pos[i_batch_beg[seq_id]], -1);
            }
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // if kv is shared with target (e.g Gemma4), then we can skip this catch-up decode
        if (!is_mem_shared) {
            common_batch_clear(batch);

            for (int k = 0; k < n_tokens; ++k) {
                common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
            }

            // shift the tgt embeddings to the right by one position
            // assumes that the tokens in the batch are sequential for each sequence
            // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
            //                                                       ^--- this is a problem
            // TODO:this is generally true, but would be nice to assert it
            {
                const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
                std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));
            }

            // fill the pending embeddings from a previous run
            auto set_h = [&](int idx, const float * h_row) {
                std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
            };

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (i_batch_beg[seq_id] < 0) {
                    continue;
                }

                const auto mode = pending_h_lifecycle[seq_id].target_process_mode(
                    batch_in.pos[i_batch_beg[seq_id]]);
                if (mode == common_speculative_mtp_carry_lifecycle::process_mode::cold_zero) {
                    std::fill(pending_h[seq_id].begin(), pending_h[seq_id].end(), 0.0f);
                }
                GGML_ASSERT(mode != common_speculative_mtp_carry_lifecycle::process_mode::target_only);
                set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
            }

            auto * mem_dft = llama_get_memory(ctx_dft);

            bool ok = true;
            for (int head = 0; head < n_mtp_layers; ++head) {
                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340/changes#r3413498544
                    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                        if (i_batch_beg[seq_id] < 0) {
                            continue;
                        }
                        llama_memory_seq_rm(mem_dft, seq_id, batch_in.pos[i_batch_beg[seq_id]], -1);
                    }
                    llama_set_nextn_layer_offset(ctx_dft, head);
                }

                const int32_t rc = llama_decode(ctx_dft, batch);
                if (rc != 0) {
                    SPC_ERR("llama_decode(ctx_dft) head=%d failed rc=%d (pos=%d)\n",
                            head, (int) rc, (int) batch_in.pos[0]);
                    ok = false;
                    break;
                }
            }

            if (chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
            }
            if (!ok) {
                return false;
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
            pending_h_lifecycle[seq_id].target_process_refreshed();
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            const float * carry = pending_h_lifecycle[seq_id].draft_carry(
                    pending_h[seq_id].data());
            if (carry == nullptr) {
                continue;
            }

            // Truncate stale draft positions: process() only cleans sequences
            // present in the verify batch, so a previous draft() may have
            // advanced this sequence past dp.n_past.
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, dp.n_past, -1);

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
            std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, carry, row_bytes);

            i_last[seq_id] = batch.n_tokens - 1;

            if (chain_heads) {
                chain_h[seq_id].assign(carry, carry + n_embd);
            }
        }

        int i = 0;

        while (n_drafting > 0) {
            // each step decodes under a different head, i.e. a different decoder layer, and
            // KV is per layer. process() filled this layer's KV only for positions < n_past
            // (prompt + accepted prefix) — nothing in the draft region yet. so reset the
            // draft region (the seq_rm lower bound is n_past, leaving the prompt KV intact)
            // and select head i so it rebuilds its own layer's KV there; decoding just the
            // latest token would leave its attention reading cells only another head wrote.
            if (chain_heads) {
                auto * mem_dft = llama_get_memory(ctx_dft);
                for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                    if (drafting[seq_id]) {
                        llama_memory_seq_rm(mem_dft, seq_id, dparams[seq_id].n_past, -1);
                    }
                }
                llama_set_nextn_layer_offset(ctx_dft, i);
            }

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                SPC_ERR("llama_decode[%d] returned %d\n", i, ret);
                break;
            }
            last_draft_model_decode_succeeded = true;

            // rebuild the batch for the next step: the growing-KV paths re-add only the
            // new token (the KV already holds the prefix), while chained heads re-add the
            // whole prefix at the next head. dropped sequences are simply not re-added.
            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_last[seq_id], true);
                const float * h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last[seq_id]);

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    SPC_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                const int32_t n_max_eff = adaptive_recursive_depth
                    ? std::min(params.n_max, adaptive_cap[seq_id])
                    : params.n_max;
                if (n_max_eff <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                if (chain_heads) {
                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546
                    chain_h[seq_id].insert(chain_h[seq_id].end(), h_row, h_row + n_embd);

                    const int n_rows = (int) result.size() + 1; // id_last + tokens drafted so far
                    for (int t = 0; t < n_rows; ++t) {
                        const llama_token tok = (t == 0) ? dp.id_last : result[t - 1];
                        common_batch_add(batch, tok, dp.n_past + t, { seq_id }, t == n_rows - 1);
                        std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd,
                                    chain_h[seq_id].data() + (size_t) t * n_embd, row_bytes);
                    }
                } else if (is_mem_shared) {
                    // note: with shared memory (e.g. Gemma4 assistants) we use the same position for all draft tokens
                    // ref: https://github.com/huggingface/transformers/blob/effde20942e3f82a1b97449f60b3a48c5ff96145/docs/source/en/model_doc/gemma4_assistant.md?plain=1#L36-L37
                    common_batch_add(batch, id, dp.n_past, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                } else {
                    common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                    std::memcpy(batch.embd + (size_t) (batch.n_tokens - 1) * n_embd, h_row, row_bytes);
                }

                i_last[seq_id] = batch.n_tokens - 1;
            }

            if (batch.n_tokens == 0) {
                break;
            }

            ++i;
        }

        if (chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0); // restore default for non-draft decodes
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
            if (adaptive_recursive_depth) {
                adaptive_last_draft_size[seq_id] = (int32_t) dp.result->size();
            }
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        if (adaptive_recursive_depth && adaptive_last_draft_size[seq_id] == 3) {
            adaptive_depth3_attempts[seq_id]++;
            adaptive_depth3_accepts[seq_id] += n_accepted >= 3;
            adaptive_last_draft_size[seq_id] = 0;

            if (adaptive_depth3_attempts[seq_id] >= 16) {
                const float p3 = (float) adaptive_depth3_accepts[seq_id] /
                                 (float) adaptive_depth3_attempts[seq_id];
                if (p3 < 0.50f) {
                    adaptive_cap[seq_id] = 2;
                    SPC_DBG("MTP seq %d marginal depth-3 acceptance %.3f; draft cap -> 2\n",
                            (int) seq_id, p3);
                }
                adaptive_depth3_attempts[seq_id] = 0;
                adaptive_depth3_accepts[seq_id] = 0;
            }
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    void sequence_transition(
            llama_seq_id seq_id,
            common_speculative_sequence_event event) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        pending_h_lifecycle[seq_id].sequence_transition(event);
        verify_h_rows[seq_id] = 0;
        i_last[seq_id] = -1;
        adaptive_last_draft_size[seq_id] = 0;
        if (chain_heads) {
            chain_h[seq_id].clear();
        }
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq, params.ngram_simple.size_m)
        , params(params.ngram_simple)
        , config(config)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-simple'\n");
        SPC_TRC("- size_n=%d, size_m=%d, min_hits=%d\n",
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(config.key_only ? COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K
            : COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, n_seq, config.size_value)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        SPC_TRC("adding speculative implementation '%s'\n", common_speculative_type_to_str(this->type).c_str());
        SPC_TRC("- size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n",
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq, params.ngram_mod.n_max)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        SPC_TRC("%s", "adding speculative implementation 'ngram-mod'\n");
        SPC_TRC("- n_match=%d, n_max=%d, n_min=%d\n",
                this->params.n_match, this->params.n_max, this->params.n_min);
        SPC_TRC("- mod size=%zu (%.3f MB)\n",
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            SPC_WRN("ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        SPC_TRC("ngram_mod occupancy = %zu/%zu (%.2f)\n", mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            SPC_WRN("ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        SPC_TRC("low acceptance streak (%d) - resetting ngram_mod\n", sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq, n_draft)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        SPC_TRC("%s", "adding speculative implementation 'ngram-cache'\n");
        SPC_TRC("- n_draft=%d, cache_static=%s, cache_dynamic=%s\n",
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                SPC_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                SPC_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }
};

// ============================================================================
// Fork-specific speculative implementations
// ============================================================================

// Fork-specific speculative decoding classes, ported to common_speculative_impl base.

// Checkpoint struct used by server for DFlash ring persistence
struct common_speculative_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;

    int64_t   n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }
};

// ---- Suffix tree speculative decoding ----

struct common_speculative_impl_suffix : public common_speculative_impl {
    SuffixTree tree;
    static constexpr int SEQ_ID = 1;

    int32_t max_depth;
    int32_t n_draft_max;
    float   spec_factor;
    float   spec_offset;
    float   min_prob;

    size_t tree_size = 0;  // number of tokens fed to the tree (prompt_tgt.size() + 1)

    common_speculative_impl_suffix(
            common_speculative_type type,
            uint32_t n_seq,
            int32_t max_depth,
            int32_t n_draft_max,
            float   spec_factor,
            float   spec_offset,
            float   min_prob)
        : common_speculative_impl(type, n_seq, n_draft_max)
        , tree(max_depth)
        , max_depth(max_depth)
        , n_draft_max(n_draft_max)
        , spec_factor(spec_factor)
        , spec_offset(spec_offset)
        , min_prob(min_prob)
    {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        tree = SuffixTree(max_depth);
        tree_size = 0;
        if (!prompt.empty()) {
            tree.extend(SEQ_ID, prompt.data(), prompt.size());
            tree_size = prompt.size();
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const llama_tokens & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            llama_tokens & result = *dp.result;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : n_draft_max;

            // feed new tokens to suffix tree (same pattern as ngram_cache)
            if (tree_size < prompt_tgt.size() + 1) {
                for (size_t j = tree_size; j < prompt_tgt.size(); ++j) {
                    tree.append(SEQ_ID, prompt_tgt[j]);
                }
                tree.append(SEQ_ID, id_last);
                tree_size = prompt_tgt.size() + 1;
            }

            // build full context for pattern matching
            std::vector<int32_t> context;
            context.reserve(prompt_tgt.size() + 1);
            for (size_t i = 0; i < prompt_tgt.size(); i++) {
                context.push_back(prompt_tgt[i]);
            }
            context.push_back(id_last);

            if (context.size() < 2) { continue; }

            SuffixDraft draft = tree.speculate(
                context.data(), context.size(),
                n_max_eff, spec_factor, spec_offset, min_prob, false);

            for (size_t i = 0; i < draft.token_ids.size(); i++) {
                result.push_back(draft.token_ids[i]);
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }

    bool need_embd() const override {
        return false;
    }
};

// ---- CopySpec: draft by copying matching subsequences from the prompt context ----
// Builds a rolling-hash index of all gamma-length windows in the prompt.
// On each draft call, hashes the last gamma tokens of output and looks up matches.

struct common_speculative_impl_copyspec : public common_speculative_impl {
    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    int32_t gamma; // window size for matching

    struct sequence_state {
        // hash of gamma-length window -> position after the window in the prompt
        std::unordered_multimap<uint64_t, int32_t> index;
        llama_tokens prompt_tokens;
        int32_t original_prompt_size = 0;
    };

    std::vector<sequence_state> states;
    bool has_model_drafter = false; // true when paired with DFlash/draft (apply primary threshold)

    common_speculative_impl_copyspec(
            common_speculative_type type, uint32_t n_seq, int32_t gamma, int32_t n_max)
        : common_speculative_impl(type, n_seq, n_max)
        , gamma(gamma)
        , states(n_seq)
    {}

    static uint64_t hash_window(const llama_token * tokens, int32_t len) {
        uint64_t h = FNV_OFFSET;
        for (int32_t i = 0; i < len; i++) {
            h ^= (uint64_t)(uint32_t)tokens[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) states.size());
        auto & state = states[seq_id];
        state.index.clear();
        state.prompt_tokens = prompt;
        state.original_prompt_size = (int32_t)prompt.size();
        if ((int32_t)prompt.size() <= gamma) {
            return;
        }
        for (int32_t i = 0; i <= (int32_t)prompt.size() - gamma; i++) {
            uint64_t h = hash_window(prompt.data() + i, gamma);
            state.index.emplace(h, i + gamma);
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const llama_tokens & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            llama_tokens & result = *dp.result;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : gamma;
            auto & state = states[seq_id];

            // build the full context (prompt_tgt + id_last)
            const int32_t ctx_len = (int32_t)prompt_tgt.size() + 1;
            if (ctx_len < gamma) {
                continue;
            }

            // hash the last gamma tokens of context
            std::vector<llama_token> window(gamma);
            const int32_t start = ctx_len - gamma;
            for (int32_t i = 0; i < gamma; i++) {
                const int32_t pos = start + i;
                window[i] = (pos < (int32_t)prompt_tgt.size()) ? prompt_tgt[pos] : id_last;
            }
            uint64_t h = hash_window(window.data(), gamma);

            // find longest match in prompt
            int32_t best_pos = -1;
            int32_t best_avail = 0; // uncapped available tokens after match
            auto range = state.index.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                int32_t pos = it->second;
                // verify hash match (collision check)
                if (pos < gamma || pos > (int32_t)state.prompt_tokens.size()) {
                    continue;
                }
                bool match = true;
                for (int32_t j = 0; j < gamma; j++) {
                    if (state.prompt_tokens[pos - gamma + j] != window[j]) {
                        match = false;
                        break;
                    }
                }
                if (!match) {
                    continue;
                }
                int32_t avail = (int32_t)state.prompt_tokens.size() - pos;
                if (avail > best_avail) {
                    best_avail = avail;
                    best_pos = pos;
                }
            }

            if (best_pos < 0) {
                continue;
            }

            // when paired with a model-based drafter, only fire as primary if the match
            // has enough original prompt tokens to justify skipping the model drafter
            if (has_model_drafter) {
                const int32_t avail_in_orig = std::max(0, state.original_prompt_size - best_pos);
                if (avail_in_orig < 2 * n_max_eff) {
                    continue;
                }
            }

            const int32_t draft_len = std::min(n_max_eff, best_avail);
            for (int32_t i = 0; i < draft_len; i++) {
                result.push_back(state.prompt_tokens[best_pos + i]);
            }
        }
    }

    // Extend an existing draft by looking for suffix matches at the end of (prompt + draft)
    void extend(
            llama_seq_id seq_id,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            int32_t n_max_ext) {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) states.size());
        auto & state = states[seq_id];
        if (result.empty() || state.index.empty()) {
            return;
        }

        // build full context: prompt_tgt + id_last + result
        // hash the last gamma tokens of this extended context
        const int32_t ctx_len = (int32_t)prompt_tgt.size() + 1 + (int32_t)result.size();
        if (ctx_len < gamma) {
            return;
        }

        std::vector<llama_token> window(gamma);
        const int32_t start = ctx_len - gamma;
        for (int32_t i = 0; i < gamma; i++) {
            const int32_t pos = start + i;
            if (pos < (int32_t)prompt_tgt.size()) {
                window[i] = prompt_tgt[pos];
            } else if (pos == (int32_t)prompt_tgt.size()) {
                window[i] = id_last;
            } else {
                window[i] = result[pos - (int32_t)prompt_tgt.size() - 1];
            }
        }
        uint64_t h = hash_window(window.data(), gamma);

        // find longest match
        int32_t best_pos = -1;
        int32_t best_len = 0;
        const int32_t max_ext = n_max_ext - (int32_t)result.size();
        if (max_ext <= 0) {
            return;
        }

        auto range = state.index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            int32_t pos = it->second;
            if (pos < gamma || pos > (int32_t)state.prompt_tokens.size()) {
                continue;
            }
            bool match = true;
            for (int32_t j = 0; j < gamma; j++) {
                if (state.prompt_tokens[pos - gamma + j] != window[j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            int32_t avail = std::min(max_ext, (int32_t)state.prompt_tokens.size() - pos);
            if (avail > best_len) {
                best_len = avail;
                best_pos = pos;
            }
        }

        if (best_pos < 0) {
            return;
        }

        for (int32_t i = 0; i < best_len; i++) {
            result.push_back(state.prompt_tokens[best_pos + i]);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }

    // incrementally extend index with accepted tokens
    void update_logits(llama_seq_id seq_id, const llama_tokens & batch_tokens, int n_accepted) {
        GGML_ASSERT(seq_id >= 0 && seq_id < (llama_seq_id) states.size());
        auto & state = states[seq_id];
        // batch_tokens = [id_last, draft0, draft1, ...], n_accepted of which were accepted
        for (int i = 0; i < n_accepted && i < (int)batch_tokens.size(); i++) {
            state.prompt_tokens.push_back(batch_tokens[i]);
            // add new gamma-length window ending at this position
            if ((int32_t)state.prompt_tokens.size() >= gamma) {
                int32_t start = (int32_t)state.prompt_tokens.size() - gamma;
                uint64_t h = hash_window(state.prompt_tokens.data() + start, gamma);
                state.index.emplace(h, (int32_t)state.prompt_tokens.size());
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

// ---- Token Recycling: adjacency matrix tracking top-k successors per token ----
// Seeded from observed bigrams, then updated from model logits after each
// verification decode. Logit-based entries have much higher scores and
// dominate the adjacency matrix after the first few iterations.

struct common_speculative_impl_recycle : public common_speculative_impl {
    int32_t k; // top-k successors per token

    // adjacency: token -> vector of (score, successor) pairs, sorted by score descending
    // scores: bigram observations use small integer counts (1, 2, ...),
    //         logit-derived entries use logit values (typically 10-30+ for top tokens)
    std::unordered_map<llama_token, std::vector<std::pair<float, llama_token>>> adj;

    size_t n_fed = 0;
    int32_t n_vocab = 0;

    common_speculative_impl_recycle(
            common_speculative_type type, uint32_t n_seq, int32_t k, int32_t n_max)
        : common_speculative_impl(type, n_seq, n_max)
        , k(k)
    {}

    void set_successors(llama_token tok, const float * logits, int32_t vocab_size) {
        // partial sort to find top-k logits
        std::vector<std::pair<float, llama_token>> top(k, std::make_pair(-INFINITY, (llama_token)-1));
        for (int32_t i = 0; i < vocab_size; i++) {
            if (logits[i] > top[k-1].first) {
                top[k-1] = std::make_pair(logits[i], (llama_token)i);
                // bubble up
                for (int32_t j = k-2; j >= 0; j--) {
                    if (top[j+1].first > top[j].first) {
                        std::swap(top[j], top[j+1]);
                    } else {
                        break;
                    }
                }
            }
        }
        // remove unfilled slots
        while (!top.empty() && top.back().second < 0) {
            top.pop_back();
        }
        adj[tok] = std::move(top);
    }

    void add_bigram(llama_token a, llama_token b) {
        auto & succs = adj[a];
        for (size_t i = 0; i < succs.size(); i++) {
            if (succs[i].second == b) {
                succs[i].first += 1.0f;
                while (i > 0 && succs[i].first > succs[i-1].first) {
                    std::swap(succs[i], succs[i-1]);
                    i--;
                }
                return;
            }
        }
        if ((int32_t)succs.size() < k) {
            succs.push_back(std::make_pair(1.0f, b));
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        adj.clear();
        n_fed = 0;
        for (size_t i = 0; i + 1 < prompt.size(); i++) {
            add_bigram(prompt[i], prompt[i + 1]);
        }
        n_fed = prompt.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const llama_tokens & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            llama_tokens & result = *dp.result;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : k;

            // feed new bigrams from generated tokens
            if (n_fed < prompt_tgt.size() + 1) {
                size_t start = (n_fed > 0) ? n_fed - 1 : 0;
                for (size_t i = start; i < prompt_tgt.size(); i++) {
                    llama_token next = (i + 1 < prompt_tgt.size()) ? prompt_tgt[i + 1] : id_last;
                    add_bigram(prompt_tgt[i], next);
                }
                n_fed = prompt_tgt.size() + 1;
            }

            // greedy walk through adjacency matrix
            llama_token cur = id_last;
            for (int32_t i = 0; i < n_max_eff; i++) {
                auto it = adj.find(cur);
                if (it == adj.end() || it->second.empty()) {
                    break;
                }
                cur = it->second[0].second;
                result.push_back(cur);
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }

    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
        if (n_vocab == 0) {
            const llama_model * model = llama_get_model(ctx);
            n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        }
        // update adjacency from logits for each position that had logits computed
        // batch_tokens[i] is the token at position i; logits at i predict its successor
        const int n_positions = std::min(n_accepted, (int)batch_tokens.size());
        for (int i = 0; i < n_positions; i++) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (logits) {
                set_successors(batch_tokens[i], logits, n_vocab);
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

// ---- DFlash block-diffusion speculative decoding ----
// Uses an external drafter model conditioned on target hidden states via KV injection

struct common_speculative_tree {
    std::vector<llama_token> tokens;
    std::vector<int32_t>     parents;
    std::vector<int32_t>     depths;
    std::vector<std::unordered_map<llama_token, int>> child_maps;
    std::vector<uint8_t>     visibility;
    int n_nodes = 0;
    int main_path_len = 0;
};

struct common_speculative_impl_dflash : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    llama_model   * model_dft;
    bool            owns_ctx_dft; // when false, ctx_dft is externally owned (shared across slots)
    llama_seq_id    seq_id = 0;   // which server slot this state owns

    float p_min; // minimum probability threshold from config

    int block_size;
    llama_token mask_token_id;
    int n_target_layers;
    int n_embd;
    int n_target_features;

    // Ring buffer for target hidden states — fixed memory regardless of context length
    // Stores last RING_SIZE tokens per layer in circular fashion
    static constexpr int RING_SIZE = 4096;

    // ring_buf[layer][slot * n_embd ... (slot+1) * n_embd - 1], slot = pos % RING_SIZE
    std::vector<std::vector<float>> ring_buf; // [n_target_layers][RING_SIZE * n_embd]
    int ring_write_pos = 0;    // next write slot (0..RING_SIZE-1)
    int ring_filled = 0;       // how many valid slots (0..RING_SIZE)
    int committed_len = 0;     // total tokens committed (unbounded counter)
    uint64_t ring_mutation_epoch = 1;
    bool prefill_flushed = false; // true if flush_prefill() was called during this request

    // Interleaved cross-attention buffer — rebuilt from ring on each draft call
    // Only holds ctx_window tokens worth of data
    std::vector<float> cross_buf;

    // Sliding-window limit for the drafter context (0 = unlimited).
    static constexpr int ctx_window = LLAMA_DFLASH_PER_SLOT_CTX;

    // GPU cross-attention ring (nullptr = CPU fallback)
    void * gpu_ring_handle = nullptr;

    // Projected cross-KV cache (nullptr = legacy full-recompute path).
    // crosskv_projected_len counts committed tokens whose projections are cached;
    // ring resets / checkpoint loads zero it, forcing a cold window refill.
    void * crosskv_handle = nullptr;
    int crosskv_projected_len = 0;

    // true when D2D staged writes have bypassed the host ring since the last
    // sync_cpu_ring_from_gpu() (checkpoint save rebuilds the host mirror lazily)
    bool cpu_ring_stale = false;
    int staged_since_sync = 0;      // D2D-written tokens not yet mirrored to the host ring
    std::vector<float> sync_tmp;    // scratch for sync_cpu_ring_from_gpu (avoid per-call alloc)

    // Adaptive draft length tracking
    int n_low_accept = 0;
    int n_draft_last = 0;
    int adaptive_n_draft = -1; // -1 = use default

    // build interleaved cross-attention data from ring buffer (GPU or CPU path)
    int build_cross_data(llama_context * ctx) {
        if (gpu_ring_handle) {
            int gpu_write_pos = ring_write_pos % ctx_window;
            int gpu_filled = std::min(ring_filled, ctx_window);
            if (crosskv_handle) {
                // project only the tokens ringed since the last draft call; a cold or
                // desynced counter (reset, prefill burst > window) refills the window
                int n_new = committed_len - crosskv_projected_len;
                if (n_new < 0 || n_new > gpu_filled) {
                    n_new = gpu_filled;
                }
                bool ok = true;
                if (n_new > 0) {
                    ok = llama_dflash_crosskv_project(ctx, crosskv_handle, gpu_ring_handle,
                                                      gpu_write_pos, n_new);
                }
                if (ok) {
                    crosskv_projected_len = committed_len;
                    llama_dflash_crosskv_set_cross(ctx, crosskv_handle, seq_id,
                                                   gpu_write_pos, gpu_filled);
                    return gpu_filled;
                }
                // projection failed — permanently fall back to the legacy path
                LOG_WRN("dflash: cross-KV projection failed, disabling the projected cache\n");
                llama_dflash_crosskv_free(crosskv_handle);
                crosskv_handle = nullptr;
            }
            llama_dflash_cross_ring_gpu_set_cross(ctx, gpu_ring_handle, seq_id,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, ctx_window);
            return gpu_filled;
        }
        int cross_len = std::min(ring_filled, ctx_window > 0 ? ctx_window : ring_filled);
        cross_buf.resize((size_t)n_target_features * cross_len);
        int read_start = (ring_write_pos - cross_len + RING_SIZE) % RING_SIZE;
        for (int t = 0; t < cross_len; ++t) {
            int slot = (read_start + t) % RING_SIZE;
            for (int layer = 0; layer < n_target_layers; ++layer) {
                memcpy(&cross_buf[(size_t)(layer * n_embd) + (size_t)t * n_target_features],
                       ring_buf[layer].data() + (size_t)slot * n_embd,
                       n_embd * sizeof(float));
            }
        }
        llama_set_cross_data_seq(ctx, seq_id, cross_buf.data(), n_target_features, cross_len);
        return cross_len;
    }

    llama_batch batch_dft;

    common_speculative_impl_dflash(
            common_speculative_type type,
            uint32_t n_seq,
            llama_context * ctx_tgt_,
            llama_context * ctx_dft_,
            llama_model   * model_dft_,
            int32_t         n_max_,
            bool            owns_ctx_dft_ = true,
            float           p_min_ = 0.0f)
        : common_speculative_impl(type, n_seq, n_max_)
        , ctx_tgt(ctx_tgt_)
        , ctx_dft(ctx_dft_)
        , model_dft(model_dft_)
        , owns_ctx_dft(owns_ctx_dft_)
        , p_min(p_min_)
    {
        block_size        = llama_model_dflash_block_size(model_dft_);
        mask_token_id     = (llama_token) llama_model_dflash_mask_token_id(model_dft_);
        n_target_layers   = llama_model_dflash_n_target_layers(model_dft_);
        n_embd            = llama_model_n_embd(model_dft_);
        n_target_features = llama_model_dflash_n_target_features(model_dft_);
        this->n_max       = std::min(std::max(0, this->n_max), std::max(0, block_size - 1));

        ring_buf.resize(n_target_layers);
        for (int i = 0; i < n_target_layers; ++i) {
            ring_buf[i].resize((size_t)RING_SIZE * n_embd, 0.0f);
        }

        // tok_embd/output sharing must happen BEFORE context creation
        // (done in speculative-simple.cpp before common_speculative_init)

        // configure target context to capture hidden states
        std::vector<int32_t> capture_layers(n_target_layers);
        llama_model_dflash_target_layer_ids(model_dft_, capture_layers.data(), n_target_layers);
        llama_set_dflash_capture(ctx_tgt, capture_layers.data(), n_target_layers);

        batch_dft = llama_batch_init(block_size, 0, 1);

        // try to allocate GPU ring buffer on drafter's GPU
        gpu_ring_handle = llama_dflash_cross_ring_gpu_init(ctx_dft, n_target_layers, n_embd, ctx_window);
        if (gpu_ring_handle) {
            LOG_INF("dflash: GPU cross ring enabled (%d layers x %d slots x %d embd)\n",
                    n_target_layers, ctx_window, n_embd);
        }

        // GPU capture staging (graph-embedded l_out copies on the target) is only safe
        // when the ring's D2D write path exists — without it the staged data would have
        // no route into the ring. The no-op probe (layer -1) just checks the proc.
        if (gpu_ring_handle && llama_dflash_cross_ring_gpu_write_d2d(gpu_ring_handle, -1, 0, nullptr, 0, 0)) {
            llama_dflash_set_capture_stage_enabled(ctx_tgt, true);
            LOG_INF("dflash: GPU capture staging enabled (device-side l_out -> ring)\n");
        }

        // projected cross-KV cache: single-slot GPU-ring path only; GGML_DFLASH_CROSSKV=0 kills it
        if (gpu_ring_handle && n_seq == 1) {
            if (env_on("GGML_DFLASH_CROSSKV")) {
                crosskv_handle = llama_dflash_crosskv_init(ctx_dft, gpu_ring_handle, ctx_window);
                if (crosskv_handle) {
                    LOG_INF("dflash: projected cross-KV cache enabled (%d slots)\n", ctx_window);
                }
            }
        }

        {
            std::string ids_str;
            for (int i = 0; i < n_target_layers; ++i) {
                if (i) ids_str += ", ";
                ids_str += std::to_string(capture_layers[i]);
            }
            LOG_INF("dflash: block_size=%d, mask_token=%d, n_target_layers=%d, n_embd=%d, target_ids=[%s]\n",
                    block_size, mask_token_id, n_target_layers, n_embd, ids_str.c_str());
        }
    }

    ~common_speculative_impl_dflash() override {
        llama_dflash_crosskv_free(crosskv_handle);
        llama_dflash_cross_ring_gpu_free(gpu_ring_handle);
        llama_batch_free(batch_dft);
        if (owns_ctx_dft) {
            llama_free(ctx_dft);
        }
    }

    void set_seq_id(llama_seq_id seq_id_) {
        seq_id = seq_id_;
    }

    // prepare cross-attention data for batched draft decode.
    // Returns cross_len (position offset for tokens), or -1 if no committed tokens.
    int prepare_batch_draft(llama_context * ctx_dft_ext) {
        if (committed_len == 0) {
            return -1;
        }

        return build_cross_data(ctx_dft_ext);
    }

    // called after initial prefill — extract hidden states from target
    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        if (prefill_flushed) {
            // ring was already populated incrementally by flush_prefill() calls
            // during checkpoint-split prefill — nothing to do
            prefill_flushed = false;
            return;
        }
        capture_target_hiddens();
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void flush_prefill() {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;

        int64_t n_tokens = captured_n_tokens();
        if (n_tokens <= 0) return;

        if (!prefill_flushed) {
            // first flush for this request — reset ring
            ring_write_pos = 0;
            ring_filled = 0;
            committed_len = 0;
            crosskv_projected_len = 0; // ring reset invalidates all cached projections
        }

        ring_write((int)n_tokens);
        committed_len += (int)n_tokens;
        prefill_flushed = true;
    }

    // Ring state serialization for checkpoint persistence.
    // Format: [ring_write_pos:i32][ring_filled:i32][committed_len:i32]
    //         [n_target_layers:i32][n_embd:i32][n_entries:i32]
    //         [layer_0 data: n_entries * n_embd * f32] ...

    size_t ring_state_size() const {
        int n_entries = std::min(ring_filled, RING_SIZE);
        return 6 * sizeof(int32_t) +
               (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
    }

    template <typename Sink>
    bool ring_state_emit(Sink && sink) const {
        auto * self = const_cast<common_speculative_impl_dflash *>(this);
        const int n_entries = std::min(ring_filled, RING_SIZE);
        const int32_t header[6] = {
            ring_write_pos, ring_filled, committed_len,
            n_target_layers, n_embd, n_entries,
        };
        if (!sink(reinterpret_cast<const uint8_t *>(header), sizeof(header))) {
            return false;
        }
        constexpr size_t MAX_QUANTUM = 1u << 20;
        const size_t layer_bytes = size_t(n_entries)*n_embd*sizeof(float);
        const int stale_entries = cpu_ring_stale && gpu_ring_handle
            ? std::min({ ring_filled, ctx_window, staged_since_sync })
            : 0;
        if (stale_entries > 0) {
            self->sync_tmp.resize(size_t(stale_entries)*n_embd);
        }
        const int gpu_pos = stale_entries > 0
            ? ((ring_write_pos-stale_entries)%ctx_window+ctx_window)%ctx_window
            : 0;
        for (int l = 0; l < n_target_layers; ++l) {
            // Materialize and emit one layer at a time. The chain writer
            // checks cancellation at each <=1 MiB write, so request arrival
            // never waits for a whole multi-layer ring image to be rebuilt
            // before the provider can yield.
            if (stale_entries > 0) {
                if (!llama_dflash_cross_ring_gpu_read(
                        gpu_ring_handle, l, gpu_pos,
                        self->sync_tmp.data(), stale_entries, n_embd)) {
                    return false;
                }
                for (int t = 0; t < stale_entries; ++t) {
                    const int cpu_slot =
                        (ring_write_pos-stale_entries+t+RING_SIZE)%RING_SIZE;
                    memcpy(
                        self->ring_buf[l].data()+size_t(cpu_slot)*n_embd,
                        self->sync_tmp.data()+size_t(t)*n_embd,
                        size_t(n_embd)*sizeof(float));
                }
            }
            const auto * src = reinterpret_cast<const uint8_t *>(
                ring_buf[l].data());
            for (size_t offset = 0; offset < layer_bytes;) {
                const size_t quantum = std::min(
                    MAX_QUANTUM, layer_bytes-offset);
                if (!sink(src+offset, quantum)) {
                    return false;
                }
                offset += quantum;
            }
        }
        if (stale_entries > 0 || (cpu_ring_stale && gpu_ring_handle)) {
            self->cpu_ring_stale = false;
            self->staged_since_sync = 0;
        }
        return true;
    }

    void ring_state_save(uint8_t * buf, size_t size) const {
        const size_t expected = ring_state_size();
        if (!buf || size < expected) {
            return;
        }
        size_t offset = 0;
        const bool emitted = ring_state_emit(
            [&](const uint8_t * data, size_t n) {
                if (n > size-offset) {
                    return false;
                }
                memcpy(buf+offset, data, n);
                offset += n;
                return true;
            });
        GGML_ASSERT(!emitted || offset == expected);
    }

    bool ring_state_write(llama_io_write_i & output) const {
        const size_t before = output.n_bytes();
        if (!ring_state_emit([&](const uint8_t * data, size_t n) {
                output.write(data, n);
                return true;
            })) {
            return false;
        }
        return output.n_bytes() >= before &&
            output.n_bytes()-before == ring_state_size();
    }

    bool ring_state_load(const uint8_t * buf, size_t size) {
        int saved_write_pos = 0;
        int saved_filled = 0;
        int saved_committed = 0;
        int saved_entries = 0;
        if (!ring_state_header(
                buf, size, saved_write_pos, saved_filled,
                saved_committed, saved_entries)) {
            return false;
        }

        size_t layer_bytes = (size_t)saved_entries * n_embd * sizeof(float);

        ring_write_pos = saved_write_pos;
        ring_filled    = saved_filled;
        committed_len  = saved_committed;

        const uint8_t * src = buf + 6 * sizeof(int32_t);
        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(ring_buf[l].data(), src, layer_bytes);
            src += layer_bytes;
        }

        // sync GPU ring with restored CPU ring — batch per layer to avoid N*L individual H2D calls
        if (gpu_ring_handle) {
            int gpu_entries = std::min(ring_filled, ctx_window);
            std::vector<float> tmp((size_t)gpu_entries * n_embd);
            for (int l = 0; l < n_target_layers; ++l) {
                for (int t = 0; t < gpu_entries; ++t) {
                    int cpu_slot = (ring_write_pos - gpu_entries + t + RING_SIZE) % RING_SIZE;
                    memcpy(tmp.data() + (size_t)t * n_embd,
                           ring_buf[l].data() + (size_t)cpu_slot * n_embd,
                           n_embd * sizeof(float));
                }
                int gpu_pos = ((ring_write_pos - gpu_entries) % ctx_window + ctx_window) % ctx_window;
                llama_dflash_cross_ring_gpu_write(gpu_ring_handle, l, gpu_pos,
                    tmp.data(), gpu_entries, n_embd);
            }
        }

        // mark as flushed so subsequent flush_prefill() calls from suffix
        // decoding APPEND to the restored ring instead of resetting it
        prefill_flushed = true;

        // restored ring contents are new to the projected cache — force cold refill
        crosskv_projected_len = 0;
        cpu_ring_stale = false;
        staged_since_sync = 0;

        bump_ring_mutation_epoch();

        return true;
    }

    bool ring_state_read(llama_io_read_i & input, size_t size) {
        constexpr size_t HEADER_BYTES = 6*sizeof(int32_t);
        constexpr size_t MAX_QUANTUM = size_t(1) << 20;
        if (size < HEADER_BYTES) {
            return false;
        }

        const size_t before = input.n_bytes();
        int32_t header[6] = {};
        input.read(header, sizeof(header));

        int saved_write_pos = 0;
        int saved_filled = 0;
        int saved_committed = 0;
        int saved_entries = 0;
        if (!ring_state_header_values(
                header, size, saved_write_pos, saved_filled,
                saved_committed, saved_entries)) {
            return false;
        }

        const size_t row_bytes = size_t(n_embd)*sizeof(float);
        if (row_bytes == 0 || row_bytes > MAX_QUANTUM) {
            return false;
        }
        const size_t layer_bytes = size_t(saved_entries)*row_bytes;
        for (int l = 0; l < n_target_layers; ++l) {
            auto * destination = reinterpret_cast<uint8_t *>(
                ring_buf[l].data());
            for (size_t offset = 0; offset < layer_bytes;) {
                const size_t quantum = std::min(
                    MAX_QUANTUM, layer_bytes-offset);
                input.read(destination+offset, quantum);
                offset += quantum;
            }
        }
        if (input.n_bytes() < before || input.n_bytes()-before != size) {
            return false;
        }

        ring_write_pos = saved_write_pos;
        ring_filled = saved_filled;
        committed_len = saved_committed;

        // Keep both the host read and each backend upload bounded to 1 MiB.
        // A row is indivisible for the backend API, so reject an unsupported
        // geometry rather than silently issuing an oversized operation.
        if (gpu_ring_handle) {
            const int gpu_entries = std::min(ring_filled, ctx_window);
            const int max_chunk_entries = std::max<int>(
                1, int(MAX_QUANTUM/row_bytes));
            std::vector<float> tmp(
                size_t(std::min(gpu_entries, max_chunk_entries))*n_embd);
            const int gpu_begin =
                ((ring_write_pos-gpu_entries)%ctx_window+ctx_window)%ctx_window;
            for (int l = 0; l < n_target_layers; ++l) {
                for (int first = 0; first < gpu_entries;
                        first += max_chunk_entries) {
                    const int count = std::min(
                        max_chunk_entries, gpu_entries-first);
                    for (int t = 0; t < count; ++t) {
                        const int cpu_slot =
                            (ring_write_pos-gpu_entries+first+t+RING_SIZE)%
                            RING_SIZE;
                        memcpy(
                            tmp.data()+size_t(t)*n_embd,
                            ring_buf[l].data()+size_t(cpu_slot)*n_embd,
                            row_bytes);
                    }
                    llama_dflash_cross_ring_gpu_write(
                        gpu_ring_handle, l,
                        (gpu_begin+first)%ctx_window,
                        tmp.data(), count, n_embd);
                }
            }
        }

        prefill_flushed = true;
        cpu_ring_stale = false;
        staged_since_sync = 0;
        crosskv_projected_len = 0;
        bump_ring_mutation_epoch();
        return true;
    }

    bool ring_state_header(
            const uint8_t * buf,
            size_t size,
            int & saved_write_pos,
            int & saved_filled,
            int & saved_committed,
            int & saved_entries) const {
        if (!buf || size < 6 * sizeof(int32_t)) {
            return false;
        }
        const int32_t * hdr = reinterpret_cast<const int32_t *>(buf);
        return ring_state_header_values(
            hdr, size, saved_write_pos, saved_filled,
            saved_committed, saved_entries);
    }

    bool ring_state_header_values(
            const int32_t * hdr,
            size_t size,
            int & saved_write_pos,
            int & saved_filled,
            int & saved_committed,
            int & saved_entries) const {
        if (!hdr || size < 6*sizeof(int32_t)) {
            return false;
        }
        saved_write_pos = hdr[0];
        saved_filled = hdr[1];
        saved_committed = hdr[2];
        const int saved_layers = hdr[3];
        const int saved_embd = hdr[4];
        saved_entries = hdr[5];
        if (saved_layers != n_target_layers || saved_embd != n_embd ||
            saved_write_pos < 0 || saved_write_pos >= RING_SIZE ||
            saved_filled < 0 || saved_filled > RING_SIZE ||
            saved_entries < 0 || saved_entries > RING_SIZE ||
            saved_entries != std::min(saved_filled, RING_SIZE) ||
            saved_committed <= 0 || saved_filled > saved_committed) {
            return false;
        }
        const size_t rows = size_t(saved_entries);
        const size_t columns = size_t(n_embd);
        const size_t layers = size_t(n_target_layers);
        if (columns != 0 && rows > SIZE_MAX/columns) {
            return false;
        }
        const size_t values = rows*columns;
        if (values > SIZE_MAX/sizeof(float) ||
            (layers != 0 && values*sizeof(float) >
                (SIZE_MAX-6*sizeof(int32_t))/layers)) {
            return false;
        }
        return size == 6*sizeof(int32_t)+
            values*sizeof(float)*layers;
    }

    bool ring_state_matches_frontier(
            const uint8_t * buf,
            size_t size,
            llama_pos expected_terminal) const {
        int saved_write_pos = 0;
        int saved_filled = 0;
        int saved_committed = 0;
        int saved_entries = 0;
        return expected_terminal >= 0 &&
            ring_state_header(
                buf, size, saved_write_pos, saved_filled,
                saved_committed, saved_entries) &&
            int64_t(saved_committed) == int64_t(expected_terminal)+1;
    }

    bool ring_state_serialized_size(
            const uint8_t * buf,
            size_t available,
            size_t & serialized_size) const {
        serialized_size = 0;
        if (!buf || available < 6*sizeof(int32_t)) {
            return false;
        }
        const auto * hdr = reinterpret_cast<const int32_t *>(buf);
        const int saved_entries = hdr[5];
        if (saved_entries < 0 || saved_entries > RING_SIZE) {
            return false;
        }
        const size_t rows = size_t(saved_entries);
        const size_t columns = size_t(n_embd);
        const size_t layers = size_t(n_target_layers);
        if (columns != 0 && rows > SIZE_MAX/columns) {
            return false;
        }
        const size_t values = rows*columns;
        if (values > SIZE_MAX/sizeof(float) ||
            (layers != 0 && values*sizeof(float) >
                (SIZE_MAX-6*sizeof(int32_t))/layers)) {
            return false;
        }
        const size_t exact = 6*sizeof(int32_t)+
            values*sizeof(float)*layers;
        if (exact > available) {
            return false;
        }
        int saved_write_pos = 0;
        int saved_filled = 0;
        int saved_committed = 0;
        int checked_entries = 0;
        if (!ring_state_header(
                buf, exact, saved_write_pos, saved_filled,
                saved_committed, checked_entries)) {
            return false;
        }
        serialized_size = exact;
        return true;
    }

    bool ring_state_terminal(llama_pos & terminal) const {
        terminal = -1;
        if (committed_len <= 0) {
            return false;
        }
        terminal = llama_pos(committed_len-1);
        return true;
    }

    bool ring_state_empty() const {
        return ring_filled == 0 && committed_len == 0;
    }

    void ring_state_reset() {
        const bool changed = ring_write_pos != 0 || ring_filled != 0 ||
            committed_len != 0 || prefill_flushed || cpu_ring_stale ||
            staged_since_sync != 0;
        ring_write_pos = 0;
        ring_filled = 0;
        committed_len = 0;
        crosskv_projected_len = 0;
        prefill_flushed = false;
        cpu_ring_stale = false;
        staged_since_sync = 0;
        if (changed) {
            bump_ring_mutation_epoch();
        }
    }

    void sequence_transition(
            llama_seq_id /*seq_id*/,
            common_speculative_sequence_event event) override {
        switch (event) {
            case common_speculative_sequence_event::prompt_rewind:
            case common_speculative_sequence_event::target_restored_without_draft:
            case common_speculative_sequence_event::draft_image_restored:
            case common_speculative_sequence_event::live_range_shift:
            case common_speculative_sequence_event::target_replaced:
            case common_speculative_sequence_event::full_clear:
                // Checkpoint restoration publishes this transition before it
                // installs an authenticated ring image. Rewinds/replacements
                // invalidate the old committed frontier outright. A live-range
                // shift can overlap the retained ring and carries no row-level
                // proof, so it also resets; the next ordinary target decode
                // appends one verified hidden row and resumes drafting safely.
                ring_state_reset();
                break;
            case common_speculative_sequence_event::composite_image_restored:
                // The composite VBR transaction has already installed and
                // authenticated this ring together with the draft image.
                // Preserve it while other speculative implementations repair
                // their lightweight restored-frontier metadata.
                break;
        }
    }

    void ring_state_currency(
            common_speculative_ring_state_currency & output) const {
        output.serialized_bytes = ring_state_size();
        output.terminal = committed_len > 0 ? llama_pos(committed_len-1) : -1;
        output.mutation_epoch = ring_mutation_epoch;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id sid = 0; sid < (llama_seq_id) n_seq; ++sid) {
            auto & dp = dparams[sid];
            if (!dp.drafting) {
                continue;
            }

            llama_tokens & result = *dp.result;
            const llama_token id_last = dp.id_last;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : (block_size - 1);

            const int n_draft_base = adaptive_n_draft > 0 ? adaptive_n_draft : (block_size - 1);
            const int n_draft = std::min(n_draft_base, n_max_eff);
            if (committed_len == 0) {
                continue;
            }

            const int64_t t0 = ggml_time_us();

            int cross_len = build_cross_data(ctx_dft);

            const int64_t t1 = ggml_time_us();

            // build drafter batch: [id_last, mask, mask, ..., mask]
            // positions are relative to the context window fed to the drafter
            // batch size adapts to n_draft+1 (saves compute when n_max < block_size-1)
            const int batch_len = n_draft + 1;
            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, id_last, cross_len, { seq_id }, true);
            for (int i = 1; i < batch_len; ++i) {
                common_batch_add(batch_dft, mask_token_id, cross_len + i, { seq_id }, true);
            }

            const int64_t t2 = ggml_time_us();

            // run drafter forward pass
            int ret = llama_decode(ctx_dft, batch_dft);
            if (ret != 0) {
                LOG_ERR("dflash: drafter decode failed with %d\n", ret);
                continue;
            }
            last_draft_model_decode_succeeded = true;

            const int64_t t3 = ggml_time_us();

            // read argmax tokens for positions 1..batch_len-1 (skip position 0 = staged_first)
            {
                int32_t * argmax = llama_get_logits_argmax(ctx_dft);
                if (argmax && !llama_get_logits_argmax_gpu(ctx_dft)) {
                    // the sched ran the sampling tail on the CPU backend, whose plain
                    // argmax kernel leaves the extended ids/log-probs layout
                    // uninitialized (the init warmup normally catches this; cover the
                    // failed-warmup path too). Raw logits were skipped for THIS decode,
                    // so draft nothing this round; disabling the tail drops the stale
                    // results and routes every later round through the host fallback
                    llama_set_dflash_argmax(ctx_dft, false);
                    LOG_INF("dflash: draft sampling on host (drafter sampling tail on CPU)\n");
                    continue;
                }
                float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
                const int K_flat = llama_get_logits_argmax_k(ctx_dft);
                if (argmax) {
                    // GPU argmax path — only 64-128 bytes transferred instead of 15.9MB
                    for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                        if (argmax_probs && p_min > 0.0f && i > 1) {
                            float log_prob = argmax_probs[i * K_flat];
                            float log_p_min = logf(p_min);
                            if (log_prob < log_p_min) {
                                LOG_DBG("dflash: early stop at position %d/%d (prob %.3f < p_min %.3f)\n",
                                        i, batch_len, expf(log_prob), p_min);
                                break;
                            }
                        }
                        result.push_back((llama_token) argmax[i * K_flat]);
                    }
                } else {
                    // fallback: CPU argmax over full vocab
                    const int n_vocab_dft = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
                    for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                        float * logits = llama_get_logits_ith(ctx_dft, i);
                        if (!logits) {
                            break;
                        }
                        llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab_dft) - logits);
                        result.push_back(best);
                    }
                }
            }

            const int64_t t4 = ggml_time_us();

            n_draft_last = (int) result.size();

            LOG_DBG("dflash draft breakdown (ctx=%d): concat=%.1fms cross=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
                    committed_len,
                    (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3, (t4 - t0) / 1e3);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t n_accepted, bool /*is_other*/) override {
        if (n_draft_last > 0) {
            float f_acc = (float) n_accepted / (float) n_draft_last;
            if (f_acc < 0.3f) {
                n_low_accept++;
                if (n_low_accept >= 3) {
                    int base = adaptive_n_draft > 0 ? adaptive_n_draft : (block_size - 1);
                    adaptive_n_draft = std::max(1, base / 2);
                    LOG_DBG("dflash: low acceptance streak (%d) — reducing draft to %d\n",
                            n_low_accept, adaptive_n_draft);
                    n_low_accept = 0;
                }
            } else {
                n_low_accept = 0;
                if (f_acc > 0.6f && adaptive_n_draft > 0) {
                    adaptive_n_draft = std::min(block_size - 1, adaptive_n_draft + 1);
                }
            }
        }
    }

    void draft_tree(
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            int n_max_eff,
            int tree_budget,
            common_speculative_tree & tree) {
        const int n_draft = std::min(n_max_eff, block_size - 1);
        if (n_draft <= 0 || committed_len == 0) {
            return;
        }

        // run drafter forward pass (same as flat draft)
        // --- begin shared draft setup ---
        const int64_t t0 = ggml_time_us();

        int cross_len = build_cross_data(ctx_dft);

        common_batch_clear(batch_dft);
        common_batch_add(batch_dft, id_last, cross_len, { seq_id }, true);
        for (int i = 1; i < block_size; ++i) {
            common_batch_add(batch_dft, mask_token_id, cross_len + i, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch_dft);
        if (ret != 0) {
            LOG_ERR("dflash: drafter decode failed with %d\n", ret);
            return;
        }
        // --- end shared draft setup ---

        const int draft_horizon = std::min(n_draft, block_size - 1);
        const int depth_limit = draft_horizon;

        // Use GPU argmax/topk for tree building
        int32_t * argmax = llama_get_logits_argmax(ctx_dft);
        if (argmax && !llama_get_logits_argmax_gpu(ctx_dft)) {
            // CPU-scheduled tail — the extended layout is uninitialized (see draft())
            llama_set_dflash_argmax(ctx_dft, false);
            LOG_INF("dflash: draft sampling on host (drafter sampling tail on CPU)\n");
            argmax = nullptr;
        }
        if (!argmax) {
            LOG_ERR("draft_tree: no GPU argmax available\n");
            return;
        }
        const int K = llama_get_logits_argmax_k(ctx_dft);
        float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);

        // Build tree using best-first heap expansion with chain-seed backbone
        tree.tokens.clear();
        tree.parents.clear();
        tree.depths.clear();
        tree.child_maps.clear();
        tree.visibility.clear();

        tree.parents.push_back(-1); // root parent
        tree.child_maps.push_back({}); // root child_map
        tree.n_nodes = 0;
        tree.main_path_len = 0;

        // Chain-seed: pre-insert greedy backbone (top-1 at each depth)
        {
            int parent = 0;
            for (int d = 1; d <= depth_limit && tree.n_nodes < tree_budget; ++d) {
                llama_token token_id = (llama_token) argmax[d * K];

                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back(token_id);
                tree.parents.push_back(parent);
                tree.depths.push_back(d);
                tree.child_maps.push_back({});
                tree.child_maps[parent][token_id] = current_idx;
                tree.n_nodes++;

                parent = current_idx;
            }
            tree.main_path_len = tree.n_nodes;
        }

        // Best-first expansion using log-prob heap (DDTree Algorithm 1)
        if (K > 1 && tree.n_nodes < tree_budget && argmax_probs) {
            // Heap entry: (cumulative_log_prob, parent_tree_idx, depth, rank)
            struct heap_entry {
                float  log_w;
                int    parent_idx;
                int    depth;  // 1-based position in draft sequence
                int    rank;   // rank within top-K at this depth
                bool operator<(const heap_entry & o) const { return log_w < o.log_w; }
            };

            std::priority_queue<heap_entry> heap;

            // Seed heap: siblings of the main chain at each depth
            // cumulative log-prob along main path up to parent
            float cum_log_prob = 0.0f;
            int main_parent = 0;
            for (int d = 1; d <= depth_limit; ++d) {
                float sibling_lp = cum_log_prob + argmax_probs[d * K + 1];
                heap.push({sibling_lp, main_parent, d, 1});
                cum_log_prob += argmax_probs[d * K + 0];
                main_parent = d; // main path nodes are 1-indexed
            }

            while (!heap.empty() && tree.n_nodes < tree_budget) {
                auto top = heap.top();
                heap.pop();

                llama_token token_id = (llama_token) argmax[top.depth * K + top.rank];
                if (token_id < 0) continue;
                if (tree.child_maps[top.parent_idx].count(token_id)) continue;

                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back(token_id);
                tree.parents.push_back(top.parent_idx);
                tree.depths.push_back(top.depth);
                tree.child_maps.push_back({});
                tree.child_maps[top.parent_idx][token_id] = current_idx;
                tree.n_nodes++;

                // Push sibling: same depth, next rank
                if (top.rank + 1 < K) {
                    float sib_lp = top.log_w - argmax_probs[top.depth * K + top.rank]
                                             + argmax_probs[top.depth * K + top.rank + 1];
                    heap.push({sib_lp, top.parent_idx, top.depth, top.rank + 1});
                }

                // Push child: extend this branch one depth deeper (rank 0)
                if (top.depth < depth_limit) {
                    int child_depth = top.depth + 1;
                    float child_lp = top.log_w + argmax_probs[child_depth * K + 0];
                    heap.push({child_lp, current_idx, child_depth, 0});
                }
            }
        } else if (K > 1 && tree.n_nodes < tree_budget) {
            // Fallback without log-probs: uniform sibling addition
            int main_parent = 0;
            for (int d = 1; d <= depth_limit && tree.n_nodes < tree_budget; ++d) {
                for (int ki = 1; ki < K && tree.n_nodes < tree_budget; ++ki) {
                    llama_token alt_token = (llama_token) argmax[d * K + ki];
                    if (alt_token < 0) continue;
                    if (tree.child_maps[main_parent].count(alt_token)) continue;

                    int current_idx = tree.n_nodes + 1;
                    tree.tokens.push_back(alt_token);
                    tree.parents.push_back(main_parent);
                    tree.depths.push_back(d);
                    tree.child_maps.push_back({});
                    tree.child_maps[main_parent][alt_token] = current_idx;
                    tree.n_nodes++;
                }
                main_parent = d;
            }
        }

        // build visibility matrix [(n_nodes+1) × (n_nodes+1)]
        int n = tree.n_nodes + 1;
        tree.visibility.assign(n * n, false);
        tree.visibility[0] = true; // root sees itself
        for (int i = 1; i < n; ++i) {
            int parent = tree.parents[i];
            // inherit parent's visibility row
            for (int j = 0; j < i; ++j) {
                tree.visibility[i * n + j] = tree.visibility[parent * n + j];
            }
            tree.visibility[i * n + i] = true; // see itself
        }

        const int64_t t1 = ggml_time_us();
        LOG_INF("ddtree: built tree with %d nodes (%d main + %d branch, budget %d) in %.1fms\n",
                tree.n_nodes, tree.main_path_len, tree.n_nodes - tree.main_path_len,
                tree_budget, (t1 - t0) / 1e3);

        GGML_UNUSED(prompt_tgt);
    }

    // called after target verification decode — capture and append new hidden states
    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
        GGML_UNUSED(ctx);
        GGML_UNUSED(batch_tokens);
        // n_accepted includes the bonus token: [id_last, draft0, ..., draftN-1] → accepted count
        // the verification batch had (1 + n_draft) tokens
        // only the first n_accepted tokens' hidden states should be kept
        append_target_hiddens(n_accepted);
    }

    bool need_embd() const override {
        return true;
    }

private:
    // write n_tokens into ring buffer from captured hidden states
    // write n_tokens from the capture buffer into the ring, starting at
    // src_offset in the capture buffer. wraps circularly in the ring.
    // tokens captured by the target's last decode: host buffers, or GPU staging when
    // the graph-embedded capture covered the decode (host buffers then stay empty)
    int64_t captured_n_tokens() {
        int64_t n = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n <= 0) {
            const void * p = nullptr;
            n = llama_dflash_capture_stage_get(ctx_tgt, 0, &p);
        }
        return n;
    }

    void ring_write(int n_tokens, int src_offset = 0) {
        const void * dev_ptr = nullptr;
        const int staged = llama_dflash_capture_stage_get(ctx_tgt, 0, &dev_ptr);
        if (staged > 0) {
            // GPU-staged capture: the data never touched the host — D2D it into the ring.
            // Staging is only enabled when the GPU ring + D2D proc exist (see ctor), and a
            // staged decode is always single-ubatch, so all layers share one token count.
            const int to_write = std::min(n_tokens, staged - src_offset);
            if (to_write > 0) {
                llama_synchronize(ctx_tgt); // staging is written by the decode graph
                const int gpu_pos = ring_write_pos % ctx_window;
                const size_t src_off_bytes = (size_t) src_offset * n_embd * sizeof(float);
                for (int layer = 0; layer < n_target_layers; ++layer) {
                    llama_dflash_capture_stage_get(ctx_tgt, layer, &dev_ptr);
                    llama_dflash_cross_ring_gpu_write_d2d(gpu_ring_handle, layer, gpu_pos,
                        (const uint8_t *)dev_ptr + src_off_bytes, to_write, n_embd);
                }
                cpu_ring_stale = true; // host mirror rebuilt lazily at checkpoint save
                staged_since_sync += to_write;
            }
        } else {
            const int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
            for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
                float * data = llama_get_layer_hidden(ctx_tgt, layer);
                int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
                int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
                if (!data || ntok <= 0) continue;

                int to_write = std::min(n_tokens, (int)ntok - src_offset);
                for (int t = 0; t < to_write; ++t) {
                    int slot = (ring_write_pos + t) % RING_SIZE;
                    memcpy(ring_buf[layer].data() + (size_t)slot * embd,
                           data + (size_t)(src_offset + t) * embd,
                           embd * sizeof(float));
                }

                // GPU ring upload (capture buffer is contiguous, write fn handles wrap)
                if (gpu_ring_handle && to_write > 0) {
                    int gpu_pos = ring_write_pos % ctx_window;
                    llama_dflash_cross_ring_gpu_write(gpu_ring_handle, layer, gpu_pos,
                        data + (size_t)src_offset * embd, to_write, embd);
                }
            }
        }
        ring_write_pos = (ring_write_pos + n_tokens) % RING_SIZE;
        ring_filled = std::min(ring_filled + n_tokens, RING_SIZE);
        bump_ring_mutation_epoch();
    }

    void bump_ring_mutation_epoch() {
        ++ring_mutation_epoch;
        if (ring_mutation_epoch == 0) {
            ring_mutation_epoch = 1;
        }
    }

    // Rebuild the host ring mirror from the GPU ring (only the cross window is GPU-
    // resident; older entries keep their last host copy). Needed before checkpoint
    // save when D2D writes bypassed the host ring.
    void sync_cpu_ring_from_gpu() {
        if (!cpu_ring_stale || !gpu_ring_handle) {
            return;
        }
        // only the tokens D2D-written since the last sync are stale on the host —
        // older window entries kept their host copies (delta read, not full window)
        const int entries = std::min({ring_filled, ctx_window, staged_since_sync});
        if (entries <= 0) {
            cpu_ring_stale = false;
            staged_since_sync = 0;
            return;
        }
        sync_tmp.resize((size_t)entries * n_embd);
        const int gpu_pos = ((ring_write_pos - entries) % ctx_window + ctx_window) % ctx_window;
        for (int l = 0; l < n_target_layers; ++l) {
            if (!llama_dflash_cross_ring_gpu_read(gpu_ring_handle, l, gpu_pos, sync_tmp.data(), entries, n_embd)) {
                return; // proc unavailable — leave stale host data rather than corrupt it
            }
            for (int t = 0; t < entries; ++t) {
                int cpu_slot = (ring_write_pos - entries + t + RING_SIZE) % RING_SIZE;
                memcpy(ring_buf[l].data() + (size_t)cpu_slot * n_embd,
                       sync_tmp.data() + (size_t)t * n_embd,
                       n_embd * sizeof(float));
            }
        }
        cpu_ring_stale = false;
        staged_since_sync = 0;
    }

    // called after initial prefill — grab all hidden states
    void capture_target_hiddens() {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;

        int64_t n_tokens = captured_n_tokens();
        if (n_tokens <= 0) return;

        // only keep last RING_SIZE tokens if prompt exceeds ring capacity
        int start_offset = std::max(0, (int)n_tokens - RING_SIZE);
        int to_store = (int)n_tokens - start_offset;

        ring_write_pos = 0;
        ring_filled = 0;
        crosskv_projected_len = 0; // ring reset invalidates all cached projections
        ring_write(to_store, start_offset);
        committed_len = (int)n_tokens;
    }

    // called after each verification decode — append only the accepted tokens' hidden states
    void append_target_hiddens(int n_accepted) {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || n_accepted <= 0) {
            return;
        }

        ring_write(n_accepted);
        committed_len += n_accepted;
    }
};


struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    // fork: current implementation (for single-seq mode, used by server per-slot)
    common_speculative_impl * curr_impl = nullptr;

    std::vector<double> synth_probs;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "draft-dflash";
        case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "draft-dspark";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        case COMMON_SPECULATIVE_TYPE_COPYSPEC:      return "copyspec";
        case COMMON_SPECULATIVE_TYPE_RECYCLE:       return "recycle";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

std::vector<common_speculative_type> common_speculative_types_from_gguf(const std::string & path) {
    struct gguf_init_params gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ nullptr,
    };

    gguf_context_ptr gguf_ctx(gguf_init_from_file(path.c_str(), gguf_params));
    if (!gguf_ctx) {
        return {};
    }

    const int64_t arch_id = gguf_find_key(gguf_ctx.get(), "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(gguf_ctx.get(), arch_id) != GGUF_TYPE_STRING) {
        return {};
    }

    const std::string arch = gguf_get_val_str(gguf_ctx.get(), arch_id);
    if (arch != "dflash") {
        const uint32_t block_count = gguf_get_val_u32(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), (arch + ".block_count").c_str()));

        if (gguf_find_tensor(gguf_ctx.get(), ("blk." + std::to_string(block_count - 1) + ".nextn.eh_proj.weight").c_str()) >= 0) {
            return { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        }

        return {};
    }

    // the Markov head distinguishes draft-dspark from draft-dflash
    const auto type = gguf_find_tensor(gguf_ctx.get(), "markov_w1.weight") >= 0
                    ? COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK
                    : COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

    SPC_INF("auto-detected speculative type '%s' from the draft model metadata\n", common_speculative_type_to_str(type).c_str());

    return { type };
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_SUFFIX:
            case COMMON_SPECULATIVE_TYPE_COPYSPEC:
            case COMMON_SPECULATIVE_TYPE_RECYCLE:
            case COMMON_SPECULATIVE_TYPE_DFLASH:
                // fork types: handled by the fork per-slot init path, not this multi-seq API
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

int32_t common_speculative_n_max(const common_speculative * spec) {
    int32_t n_max = 0;

    if (spec == nullptr) {
        return n_max;
    }

    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, std::max(0, impl->n_max));
    }

    return n_max;
}

std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max) {
    const bool has_length = spec->synth_len != -1.0;
    const bool has_rates  = !spec->synth_rates.empty();

    if (!has_length && !has_rates) {
        return {};
    }
    if (has_length && has_rates) {
        throw std::invalid_argument("synthetic acceptance length and rates are mutually exclusive");
    }

    if (n_max <= 0) {
        throw std::invalid_argument("synthetic acceptance requires at least one speculative token");
    }

    if (has_rates) {
        const auto & rates = spec->synth_rates;
        if (rates.size() != (size_t) n_max) {
            throw std::invalid_argument(string_format(
                    "synthetic acceptance rates must contain %d values, got %zu", n_max, rates.size()));
        }

        for (size_t i = 0; i < rates.size(); ++i) {
            if (!std::isfinite(rates[i]) || rates[i] < 0.0 || rates[i] > 1.0) {
                throw std::invalid_argument("synthetic acceptance rates must be finite and within [0, 1]");
            }
            if (i > 0 && rates[i] > rates[i - 1]) {
                throw std::invalid_argument("synthetic acceptance rates must be monotonically non-increasing");
            }
        }

        return rates;
    }

    const double length = spec->synth_len;
    const double length_max = (double) n_max + 1.0;
    if (!std::isfinite(length) || length < 1.0 || length > length_max) {
        throw std::invalid_argument(string_format(
                "synthetic acceptance length must be finite and within [1, %.0f]", length_max));
    }

    double p = 0.0;
    if (length == length_max) {
        p = 1.0;
    } else if (length > 1.0) {
        double p_min = 0.0;
        double p_max = 1.0;
        for (int i = 0; i < 32; ++i) {
            const double p_mid = 0.5 * (p_min + p_max);
            double sum = 0.0;
            double term = p_mid;
            for (int32_t j = 0; j < n_max; ++j) {
                sum += term;
                term *= p_mid;
            }

            if (sum < length - 1.0) {
                p_min = p_mid;
            } else {
                p_max = p_mid;
            }
        }
        p = 0.5 * (p_min + p_max);
    }

    std::vector<double> rates;
    rates.reserve(n_max);
    double rate = p;
    for (int32_t i = 0; i < n_max; ++i) {
        rates.push_back(rate);
        rate *= p;
    }

    return rates;
}

const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec) {
    GGML_ASSERT(spec);
    return spec->synth_probs;
}

common_params common_base_params_to_speculative(const common_params & params) {
    const bool has_draft = params.speculative.has_dft();

    const auto & params_spec = params.speculative.draft;
    common_params result = params;

    result.embedding    = false;
    result.pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (has_draft) {
        result.devices               = params_spec.devices;
        result.model                 = params_spec.mparams;
        result.n_gpu_layers          = params_spec.n_gpu_layers;
        result.tensor_buft_overrides = params_spec.tensor_buft_overrides;

        if (params_spec.cpuparams.n_threads > 0) {
            result.cpuparams.n_threads          = params_spec.cpuparams.n_threads;
            result.cpuparams.n_threads_explicit = params_spec.cpuparams.n_threads_explicit;
        }
        if (params_spec.cpuparams_batch.n_threads > 0) {
            result.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            result.cpuparams_batch.n_threads_explicit =
                params_spec.cpuparams_batch.n_threads_explicit;
        }
    }

    result.cache_type_k  = params_spec.cache_type_k;
    result.cache_type_v  = params_spec.cache_type_v;
    // Drafter caches are small and ephemeral — never arm dynamic VBR for them. The
    // wholesale copy above inherits the base params' default-on VBR flags, and a second
    // dynamic-VBR context trips the one-marker-per-process co-tenancy guard, failing
    // draft-context creation outright (observed with the Muse-Glimmer day-0 drafter).
    // Static -ctkd/-ctvd types still apply verbatim.
    result.vbr_cache_type_k         = false;
    result.vbr_cache_type_v         = false;
    result.vbr_budget_explicit      = false;
    result.vbr_min_bits_explicit    = false;
    result.vbr_vram_budget_explicit = false;
    result.vbr_policy_explicit      = false;
    result.n_outputs_max = params.n_parallel;
    result.n_outputs_max_per_seq = 1;

    // dflash/dspark decode the whole noise block in a single pass and sample every block position on the backend
    // TODO: refactor such properties to be announced by the speculative types
    //       something like `struct common_speculative_type_props common_speculative_type_get_props(...);`
    const bool has_block_draft = std::any_of(
        params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });
    if (has_block_draft) {
        // per-seq output positions: DFlash decodes anchor + n_max masks (n_max + 1); DSpark n_max -> +1 covers both
        const int32_t per_seq = std::max(1, params_spec.n_max + 1);
        result.n_outputs_max = params.n_parallel * per_seq;
        if (params_spec.backend_sampling) {
            result.n_outputs_max_per_seq = per_seq;
        }
    }

    return result;
}

common_speculative_mtp_context_params common_speculative_mtp_context_params_resolve(
        uint32_t target_n_ctx_seq,
        int32_t explicit_draft_n_ctx,
        uint32_t requested_n_seq_max,
        bool requested_kv_unified) {
    if (explicit_draft_n_ctx > 0) {
        return { (uint32_t) explicit_draft_n_ctx, requested_n_seq_max, requested_kv_unified };
    }

    return { target_n_ctx_seq, requested_n_seq_max, true };
}

bool common_speculative_mtp_context_available(const common_params_speculative & params) {
    if (params.has_non_mtp_model_drafter()) {
        return params.draft.ctx_mtp != nullptr;
    }

    return params.draft.ctx_mtp != nullptr || params.draft.ctx_dft != nullptr;
}

struct common_speculative_init_result::impl {
    impl() = default;
    ~impl() = default;

    // note: the order in which model, context, etc. are declared matters because their destructors will be called bottom-to-top
    llama_model_ptr   model;
    llama_context_ptr context;
    llama_context_ptr context_mtp;
};

common_speculative_init_result::common_speculative_init_result(
    common_params & params,
      llama_model * model_tgt,
    llama_context * ctx_tgt) :
    pimpl(new impl{}) {
    const bool has_draft = params.speculative.has_dft();
    const bool spec_mtp = std::find(params.speculative.types.begin(),
                                    params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();
    const bool combined_external_and_mtp = has_draft && spec_mtp &&
                                           params.speculative.has_non_mtp_model_drafter();
    const bool external_mtp_sidecar = has_draft && spec_mtp && !combined_external_and_mtp;

    auto mparams = common_model_params_to_llama(params);
    auto cparams = common_context_params_to_llama(params);

    // Non-MTP drafters retain the established full target-context geometry.
    cparams.n_ctx = llama_n_ctx(ctx_tgt);

    // note: for small models maybe we can set this to the maximum possible draft from all speculative types
    //       the extra memory for small models is likely negligible?
    cparams.n_rs_seq  = 0;
    cparams.ctx_other = ctx_tgt;

    auto cparams_mtp = cparams;
    const auto mtp_context = common_speculative_mtp_context_params_resolve(
        llama_n_ctx_seq(ctx_tgt), params.speculative.draft.n_ctx,
        cparams_mtp.n_seq_max,
        cparams_mtp.kv_unified);
    cparams_mtp.n_ctx      = mtp_context.n_ctx;
    cparams_mtp.n_seq_max  = mtp_context.n_seq_max;
    cparams_mtp.kv_unified = mtp_context.kv_unified;
    cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;

    std::string model_path;
    if (has_draft) {
        model_path = params.speculative.draft.mparams.path;
        LOG_INF("%s: loading draft model '%s'\n", __func__, model_path.c_str());

        llama_model * model_dft = llama_model_load_from_file(model_path.c_str(), mparams);
        if (model_dft == NULL) {
            LOG_ERR("%s: failed to load draft model, '%s'\n", __func__, model_path.c_str());
            return;
        }

        pimpl->model.reset(model_dft);

        llama_context * ctx_dft = llama_init_from_model(
                model_dft, external_mtp_sidecar ? cparams_mtp : cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create draft context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);

        if (combined_external_and_mtp) {
            LOG_INF("%s: creating native MTP context against the target model '%s'\n",
                    __func__, params.model.path.c_str());

            llama_context * ctx_mtp = llama_init_from_model(model_tgt, cparams_mtp);
            if (ctx_mtp == nullptr) {
                LOG_WRN("%s: target model has no native MTP layers, skipping draft-mtp\n", __func__);
                params.speculative.types.erase(
                        std::remove(params.speculative.types.begin(), params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP),
                        params.speculative.types.end());
            } else {
                pimpl->context_mtp.reset(ctx_mtp);
            }
        }
    } else if (spec_mtp) {
        model_path = params.model.path;

        LOG_INF("%s: creating MTP draft context against the target model '%s'\n", __func__, model_path.c_str());

        llama_context * ctx_dft = llama_init_from_model(model_tgt, cparams_mtp);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s: failed to create MTP context\n", __func__);
            return;
        }

        pimpl->context.reset(ctx_dft);
    }
}

common_speculative_init_result::~common_speculative_init_result() = default;

llama_model * common_speculative_init_result::model() {
    return pimpl->model.get();
}

llama_context * common_speculative_init_result::context() {
    return pimpl->context.get();
}

llama_context * common_speculative_init_result::context_mtp() {
    return pimpl->context_mtp.get();
}

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt) {
    return std::make_unique<common_speculative_init_result>(params, model_tgt, ctx_tgt);
}

void common_speculative_select_dflash2(common_params_speculative & params) {
    params.types.erase(
            std::remove(params.types.begin(), params.types.end(), COMMON_SPECULATIVE_TYPE_NONE),
            params.types.end());
    for (auto & type : params.types) {
        if (type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            type = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
        }
    }
    if (!params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) {
        params.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
    }
}

void common_speculative_resolve_draft_model_type(
        common_params_speculative & params,
        const llama_model *         model_dft) {
    if (model_dft && llama_model_dflash2_has_selector(model_dft)) {
        common_speculative_select_dflash2(params);
    }
}

common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft) {
    const int64_t per_seq = 1 + (int64_t) std::max(0, n_draft);
    const int64_t total   = (int64_t) n_parallel * per_seq;

    return {
        /* .total   = */ (int32_t) std::min<int64_t>(n_batch, total),
        /* .per_seq = */ (int32_t) std::min<int64_t>(n_batch, per_seq),
    };
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    common_speculative_resolve_draft_model_type(
            params, params.draft.ctx_dft ? llama_get_model(params.draft.ctx_dft) : nullptr);

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        auto add_config_if_enabled = [&](common_speculative_type type, bool available = true) {
            if (available && (enabled_configs & (1u << type))) {
                configs.emplace_back(type, params);
            }
        };

        // when adding a new type - update here the logic above
        // SUFFIX/RECYCLE/legacy DFLASH remain per-slot. CopySpec is also hosted
        // here when paired with shared multi-seq DFlash2 so both implementations
        // have one owner and one per-sequence acceptance lifecycle.
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 15);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        const bool has_dflash2 = params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
        // Unlike legacy DFlash, keep this explicit for DFlash2. Copy-heavy
        // prompts can win substantially, but CopySpec's extra extensions can
        // perturb an already strong DFlash2 cycle on ordinary generated code.
        if (has_dflash2 && params.has_type(COMMON_SPECULATIVE_TYPE_COPYSPEC)) {
            configs.emplace_back(COMMON_SPECULATIVE_TYPE_COPYSPEC, params);
        }
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);

        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
                common_speculative_mtp_context_available(params));
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, params.draft.ctx_dft != nullptr);
        add_config_if_enabled(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, params.draft.ctx_dft != nullptr);
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_dflash>(
                        config.params, n_seq, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_COPYSPEC: {
                auto impl = std::make_unique<common_speculative_impl_copyspec>(
                        config.type, n_seq, config.params.copyspec_gamma, config.params.draft.n_max);
                impl->has_model_drafter = true;
                impls.push_back(std::move(impl));
                SPC_INF("copyspec speculative decoding (gamma=%d)\n",
                        config.params.copyspec_gamma);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        SPC_TRC("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    common_speculative_ptr result(new common_speculative {
        /* .dparams     = */ common_speculative_draft_params_vec(n_seq),
        /* .impls       = */ std::move(impls),
        /* .impl_last   = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .curr_impl   = */ nullptr,
        /* .synth_probs = */ {},
    });

    const int32_t n_max_configured = common_speculative_n_max(&params);
    const int32_t n_max_effective  = common_speculative_n_max(result.get());
    const auto rates = common_speculative_synth_rates_resolve(&params, n_max_effective);

    std::vector<std::string> rates_str;
    rates_str.reserve(rates.size());
    result->synth_probs.reserve(rates.size());
    double rate_prev = 1.0;
    double acceptance_length = 1.0;
    for (const double rate : rates) {
        result->synth_probs.push_back(rate_prev > 0.0 ? rate / rate_prev : 0.0);
        rates_str.push_back(string_format("%.6g", rate));
        rate_prev = rate;
        acceptance_length += rate;
    }
    if (!result->synth_probs.empty()) {
        SPC_WRN("%s", "synthetic speculative acceptance is enabled for benchmarking; generated output is not valid\n");
        if (n_max_effective != n_max_configured) {
            SPC_WRN("synthetic acceptance draft limit was reduced from %d to %d by the initialized speculative implementations\n",
                    n_max_configured, n_max_effective);
        }
        SPC_INF("synthetic acceptance: n_max = %zu, mean length = %.6f, rates = [%s]\n",
                rates.size(), acceptance_length, string_join(rates_str, ", ").c_str());
    }

    return result.release();
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }
    for (const auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }
    return false;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }
    for (const auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }
    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        SPC_DBG("truncating draft to %d tokens\n", dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n",
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // A model draft can end on a sequence that appears verbatim in the
    // existing context. Let CopySpec extend that draft without changing the
    // implementation that owns acceptance (and, for DFlash2, its exact-q
    // prefix). This is the multi-sequence equivalent of the legacy per-slot
    // composition below.
    common_speculative_impl_copyspec * copyspec = nullptr;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
            copyspec = static_cast<common_speculative_impl_copyspec *>(impl.get());
            break;
        }
    }
    if (copyspec) {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];
            if (spec->impl_last[seq_id] == copyspec || !dp.result || dp.result->empty() ||
                    !dp.prompt || dp.n_max <= (int32_t) dp.result->size()) {
                continue;
            }
            const size_t n_before = dp.result->size();
            copyspec->extend(seq_id, *dp.prompt, dp.id_last, *dp.result, dp.n_max);
            if (dp.result->size() > n_before) {
                copyspec->n_gen_drafts++;
                copyspec->n_gen_tokens += dp.result->size() - n_before;
            }
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (impl == nullptr) {
        GGML_ASSERT(n_accepted == 0);
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);

        if (impl->n_acc_tokens_per_pos.size() < n_accepted) {
            impl->n_acc_tokens_per_pos.resize(n_accepted, 0);
        }

        for (size_t i = 0; i < n_accepted; ++i) {
            impl->n_acc_tokens_per_pos[i]++;
        }

        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

// TODO: support the case of more than one speculative implementations having a state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->get_state(seq_id, data)) {
            return true;
        }
    }

    return false;
}

void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->set_state(seq_id, data);
    }
}

void common_speculative_sequence_transition(
        common_speculative * spec,
        llama_seq_id         seq_id,
        common_speculative_sequence_event event) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        impl->sequence_transition(seq_id, event);
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        std::string str_stats;
        if (impl->n_call_accept > 0) {
            const double mean =
                1.0 + (double) impl->n_acc_tokens / (double) impl->n_call_accept;
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(3);
            for (size_t i = 0; i < impl->n_acc_tokens_per_pos.size(); ++i) {
                if (i > 0) {
                    tmp << ", ";
                }
                tmp << (double) impl->n_acc_tokens_per_pos[i] / (double) impl->n_call_accept;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << mean;
            str_stats = ", #mean acc len = " + oss.str() + ", #acc rate/pos = (" + tmp.str() + ")";
        }

        SPC_TRC("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_stats.c_str(),
                str_perf.c_str());
    }
}

// ============================================================================
// Fork-specific init and dispatch functions
// ============================================================================

llama_context * common_speculative_create_ctx_dft(const common_params_speculative & params, int dflash_n_slots) {
    if (!params.model_dft) {
        return nullptr;
    }
    llama_context_params cparams_dft = params.cparams_dft;
    cparams_dft.dflash_n_slots = dflash_n_slots;
    llama_context * ctx_dft = llama_init_from_model(params.model_dft, cparams_dft);
    if (ctx_dft == nullptr) {
        LOG_ERR("%s", "failed to create draft context\n");
        return nullptr;
    }
    if (params.draft_topk > 1) {
        llama_set_dflash_topk(ctx_dft, params.draft_topk);
        LOG_INF("dflash: top-K=%d enabled for tree branching\n", params.draft_topk);
    }
    if (params.sample_temp > 0.0f) {
        llama_set_dflash_sample_temp(ctx_dft, params.sample_temp);
    }

    // the fork drafter graphs gate their in-graph argmax/top-K sampling tail on this
    // flag — enable it here (the historical default) and let the warmup below verify
    // that the sched actually runs it on a GPU backend
    llama_set_dflash_argmax(ctx_dft, true);

    // warmup the draft context
    {
        const llama_vocab * vocab_dft = llama_model_get_vocab(llama_get_model(ctx_dft));

        llama_token bos = llama_vocab_bos(vocab_dft);
        llama_token eos = llama_vocab_eos(vocab_dft);

        llama_token tmp[2];
        int n_tmp = 0;
        if (bos != LLAMA_TOKEN_NULL) { tmp[n_tmp++] = bos; }
        if (eos != LLAMA_TOKEN_NULL) { tmp[n_tmp++] = eos; }
        if (n_tmp == 0) { tmp[n_tmp++] = 0; }

        llama_set_warmup(ctx_dft, true);
        int ret = llama_decode(ctx_dft, llama_batch_get_one(tmp, n_tmp));
        if (ret != 0) {
            LOG_WRN("%s: draft warmup decode failed: %d (non-fatal)\n", __func__, ret);
        } else if (llama_get_logits_argmax(ctx_dft) && !llama_get_logits_argmax_gpu(ctx_dft)) {
            // the sched placed the sampling tail on the CPU backend (e.g. -ngld 0):
            // only the GPU argmax kernels implement the extended top-K ids/log-probs
            // layout, so the in-graph results would be uninitialized garbage —
            // disable the tail (dropping the warmup's stale results with it) and
            // sample on the host from raw logits instead
            llama_set_dflash_argmax(ctx_dft, false);
            LOG_INF("%s: draft sampling on host (drafter sampling tail on CPU)\n", __func__);
        }

        llama_memory_t mem_dft = llama_get_memory(ctx_dft);
        if (mem_dft) {
            llama_memory_clear(mem_dft, true);
        }
        llama_synchronize(ctx_dft);
        llama_perf_context_reset(ctx_dft);
        llama_set_warmup(ctx_dft, false);

        LOG_INF("%s: draft model warmup complete\n", __func__);
    }

    return ctx_dft;
}

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_dft_shared) {
    const bool owns_ctx_dft = (ctx_dft_shared == nullptr);
    llama_context * ctx_dft = ctx_dft_shared;
    // Only DFlash consumes this per-slot draft context (see the config loop below). Creating it
    // for other separate-model drafters is redundant AND crashes: a bare MTP-head draft model has
    // no trunk weights, so building its (DEFAULT-typed) full-trunk graph derefs a null wqkv in
    // graph_reserve. draft-mtp speculates through the member MTP context (ctx_type == MTP) wired up
    // at load, not this one, so skip the creation unless DFlash will actually use it.
    if (ctx_dft == nullptr && params.model_dft && params.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
        ctx_dft = common_speculative_create_ctx_dft(params);
    }

    std::vector<common_speculative_config> configs = {};
    {
        bool has_suffix   = params.has_type(COMMON_SPECULATIVE_TYPE_SUFFIX);
        bool has_copyspec = params.has_type(COMMON_SPECULATIVE_TYPE_COPYSPEC) &&
            !params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
        bool has_recycle  = params.has_type(COMMON_SPECULATIVE_TYPE_RECYCLE);
        bool has_dflash   = params.has_type(COMMON_SPECULATIVE_TYPE_DFLASH);

        if (has_copyspec) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_COPYSPEC, params));
        }
        if (has_recycle) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_RECYCLE, params));
        }
        if (has_suffix) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_SUFFIX, params));
        }
        if (has_dflash) {
            if (!has_copyspec && env_on("GGML_DFLASH_COPYSPEC")) {
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_COPYSPEC, params));
            }
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
        }
    }

    const uint32_t n_seq = 1;
    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_INF("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                GGML_ASSERT(ctx_dft != nullptr);
                impls.push_back(std::make_unique<common_speculative_impl_dflash>(
                    config.type, n_seq, ctx_tgt, ctx_dft, params.model_dft,
                    params.n_max, owns_ctx_dft, params.p_min));
                if (owns_ctx_dft) {
                    ctx_dft = nullptr;
                }
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                impls.push_back(std::make_unique<common_speculative_impl_suffix>(
                    config.type, n_seq,
                    config.params.suffix_max_depth,
                    config.params.n_max,
                    config.params.suffix_spec_factor,
                    config.params.suffix_spec_offset,
                    config.params.suffix_min_prob));
                LOG_INF("%s: suffix tree speculative decoding (max_depth=%d, factor=%.1f, min_prob=%.2f)\n",
                    __func__, config.params.suffix_max_depth,
                    config.params.suffix_spec_factor, config.params.suffix_min_prob);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_COPYSPEC: {
                impls.push_back(std::make_unique<common_speculative_impl_copyspec>(
                    config.type, n_seq, config.params.copyspec_gamma, config.params.n_max));
                LOG_INF("%s: copyspec speculative decoding (gamma=%d)\n",
                    __func__, config.params.copyspec_gamma);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_RECYCLE: {
                impls.push_back(std::make_unique<common_speculative_impl_recycle>(
                    config.type, n_seq, config.params.recycle_k, config.params.n_max));
                LOG_INF("%s: token recycling speculative decoding (k=%d)\n",
                    __func__, config.params.recycle_k);
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        // Shared draft implementations are initialized once above the slots. An empty
        // per-slot set is therefore normal when every requested helper is shared too.
        return nullptr;
    }

    // if a model-based drafter exists, tell CopySpec to only fire as primary for long matches
    bool has_model_impl = false;
    for (auto & impl : impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH ||
            impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE ||
            impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) {
            has_model_impl = true;
            break;
        }
    }
    if (has_model_impl) {
        for (auto & impl : impls) {
            if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
                static_cast<common_speculative_impl_copyspec *>(impl.get())->has_model_drafter = true;
            }
        }
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .curr_impl = */ nullptr,
        /* .synth_probs = */ {},
    };

    return result;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(0, prompt);
        impl->n_call_begin++;
    }
}

void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            static_cast<common_speculative_impl_dflash *>(impl.get())->set_seq_id(seq_id);
        }
    }
}

void common_speculative_set_rng_seed(
        common_speculative * spec,
        llama_seq_id         seq_id,
        uint32_t             seed) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) {
            static_cast<common_speculative_impl_draft_dflash *>(impl.get())->set_rng_seed(seq_id, seed);
        }
    }
}

llama_tokens common_speculative_draft(
        common_speculative              * spec,
        const common_params_speculative & params,
        const llama_tokens              & prompt_tgt,
        llama_token                       id_last,
        std::vector<float>              * draft_log_probs,
        llama_pos                         n_past_override) {
    llama_tokens result;

    if (spec == nullptr) {
        return result;
    }

    spec->curr_impl = nullptr;
    for (auto & impl : spec->impls) {
        impl->last_draft_model_decode_succeeded = false;
    }

    // set up dparams for seq 0
    auto & dp = spec->dparams[0];
    dp.drafting = true;
    dp.n_max    = params.n_max;
    // M-RoPE: actual positions may exceed text token count due to image spatial dims
    dp.n_past   = n_past_override >= 0 ? n_past_override : (llama_pos)prompt_tgt.size();
    dp.id_last  = id_last;
    dp.prompt   = &prompt_tgt;
    dp.result   = &result;

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(spec->dparams);
            impl->n_call_draft++;
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get();
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break;
        }
    }

    dp.drafting = false;

    // try extension impls (e.g. CopySpec appending suffix matches after DFlash draft)
    if (!result.empty()) {
        for (auto & impl : spec->impls) {
            if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
                const size_t pre = result.size();
                auto * cs = static_cast<common_speculative_impl_copyspec *>(impl.get());
                cs->extend(0, prompt_tgt, id_last, result, params.n_max);
                if (result.size() > pre) {
                    LOG_DBG("%s: extended draft by %zu tokens (%s)\n", __func__,
                        result.size() - pre, common_speculative_type_to_str(impl->type).c_str());
                }
            }
        }
    }

    GGML_UNUSED(draft_log_probs);
    return result;
}

void common_speculative_draft_batch(
        std::vector<common_speculative *> & specs,
        llama_context                     * ctx_dft,
        const common_params_speculative   & params,
        const std::vector<llama_token>    & id_last_per_spec,
        std::vector<llama_tokens>         & result_per_spec) {
    const int n_specs = (int) specs.size();
    for (auto * spec : specs) {
        if (spec == nullptr) {
            continue;
        }
        spec->curr_impl = nullptr;
        for (auto & impl : spec->impls) {
            impl->last_draft_model_decode_succeeded = false;
        }
    }
    result_per_spec.clear();
    result_per_spec.resize(n_specs);

    if (n_specs == 0 || !ctx_dft) {
        return;
    }

    const llama_model * model_dft  = llama_get_model(ctx_dft);
    const int block_size           = llama_model_dflash_block_size(model_dft);
    const int n_draft              = std::min(block_size - 1, params.n_max);
    const int batch_len            = n_draft + 1;
    const llama_token mask_tok     = (llama_token) llama_model_dflash_mask_token_id(model_dft);

    const int64_t t0 = ggml_time_us();

    struct ready_slot {
        common_speculative_impl * impl;
        int           cross_len;
        llama_seq_id  seq_id;
        int           spec_idx;
    };
    std::vector<ready_slot> ready;
    ready.reserve(n_specs);

    for (int s = 0; s < n_specs; s++) {
        for (auto & impl : specs[s]->impls) {
            if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
                continue;
            }

            auto * dfl = static_cast<common_speculative_impl_dflash *>(impl.get());
            const int cross_len = dfl->prepare_batch_draft(ctx_dft);
            if (cross_len < 0) {
                break;
            }

            ready.push_back({ impl.get(), cross_len, dfl->seq_id, s });
            break;
        }
    }

    if (ready.empty()) {
        return;
    }

    const int n_ready = (int) ready.size();

    llama_set_dflash_n_slots(ctx_dft, n_ready);

    const int64_t t1 = ggml_time_us();

    llama_batch batch = llama_batch_init(n_ready * batch_len, 0, 1);

    for (const auto & rs : ready) {
        common_batch_add(batch, id_last_per_spec[rs.spec_idx], rs.cross_len, { rs.seq_id }, true);
        for (int i = 1; i < batch_len; i++) {
            common_batch_add(batch, mask_tok, rs.cross_len + i, { rs.seq_id }, true);
        }
    }

    const int ret = llama_decode(ctx_dft, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        LOG_ERR("dflash batch: decode failed with %d\n", ret);
        return;
    }

    for (const auto & rs : ready) {
        rs.impl->last_draft_model_decode_succeeded = true;
    }

    const int64_t t2 = ggml_time_us();

    int32_t * argmax       = llama_get_logits_argmax(ctx_dft);
    float   * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
    const int K_flat       = llama_get_logits_argmax_k(ctx_dft);
    for (int r = 0; r < n_ready; r++) {
        auto & rs     = ready[r];
        auto & result = result_per_spec[rs.spec_idx];
        const int offset = r * batch_len;

        if (argmax) {
            for (int i = 1; i < batch_len && (int) result.size() < n_draft; i++) {
                if (argmax_probs && params.p_min > 0.0f && i > 1) {
                    float log_prob = argmax_probs[(offset + i) * K_flat];
                    if (log_prob < logf(params.p_min)) {
                        break;
                    }
                }
                result.push_back((llama_token) argmax[(offset + i) * K_flat]);
            }
        } else {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
            for (int i = 1; i < batch_len && (int) result.size() < n_draft; i++) {
                float * logits = llama_get_logits_ith(ctx_dft, offset + i);
                if (!logits) {
                    break;
                }
                llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
                result.push_back(best);
            }
        }

        rs.impl->n_call_draft++;
        auto * dfl = static_cast<common_speculative_impl_dflash *>(rs.impl);
        dfl->n_draft_last = (int) result.size();
        if (!result.empty()) {
            rs.impl->n_gen_drafts++;
            rs.impl->n_gen_tokens += result.size();
            specs[rs.spec_idx]->curr_impl = rs.impl;
        }
    }

    const int64_t t3 = ggml_time_us();

    LOG_DBG("dflash batch draft (%d specs): prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
            n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3);
}

bool common_speculative_last_draft_model_decode_succeeded(const common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }
    return std::any_of(spec->impls.begin(), spec->impls.end(),
            [](const auto & impl) { return impl->last_draft_model_decode_succeeded; });
}

const common_speculative_proposal * common_speculative_get_proposal(
        const common_speculative * spec,
        llama_seq_id               seq_id) {
    if (!spec || seq_id < 0 || seq_id >= (llama_seq_id) spec->impl_last.size()) {
        return nullptr;
    }
    const auto * impl = spec->impl_last[seq_id];
    if (!impl || impl->type != COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) {
        return nullptr;
    }
    const auto * dfl = static_cast<const common_speculative_impl_draft_dflash *>(impl);
    const auto & proposal = dfl->proposals[seq_id];
    return proposal.exact_q ? &proposal : nullptr;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (spec == nullptr) {
        return;
    }

    common_speculative_impl * impl = spec->curr_impl;
    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }
        impl->accept(0, n_accepted, false);
        impl->n_call_accept++;
    }
}

void common_speculative_update_logits(
        common_speculative * spec,
        llama_seq_id         seq_id,
        llama_context      * ctx,
        const llama_tokens & batch_tokens,
        int                  n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            static_cast<common_speculative_impl_dflash *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
            static_cast<common_speculative_impl_copyspec *>(impl.get())->update_logits(seq_id, batch_tokens, n_accepted);
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_RECYCLE) {
            static_cast<common_speculative_impl_recycle *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        }
    }
}

void common_speculative_update_logits(
        common_speculative * spec,
        llama_context      * ctx,
        const llama_tokens & batch_tokens,
        int                  n_accepted) {
    common_speculative_update_logits(spec, 0, ctx, batch_tokens, n_accepted);
}

void common_speculative_rollback_dft(common_speculative * spec, llama_seq_id seq_id, llama_pos n_past, uint16_t n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            auto * mtp = static_cast<common_speculative_impl_draft_mtp *>(impl.get());
            auto * ctx_dft = mtp->params.ctx_dft;
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            mtp->accept(seq_id, n_accepted, false);
        }
    }
}

void common_speculative_flush_prefill(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            static_cast<common_speculative_impl_dflash *>(impl.get())->flush_prefill();
        }
    }
}

size_t common_speculative_ring_state_size(const common_speculative * spec) {
    if (spec == nullptr) return 0;
    size_t total = 0;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            total += static_cast<const common_speculative_impl_dflash *>(impl.get())->ring_state_size();
        }
    }
    return total;
}

void common_speculative_ring_state_save(const common_speculative * spec, uint8_t * buf, size_t size) {
    if (spec == nullptr) return;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            auto * dfl = static_cast<const common_speculative_impl_dflash *>(impl.get());
            size_t impl_size = dfl->ring_state_size();
            if (impl_size > 0 && impl_size <= size) {
                dfl->ring_state_save(buf, impl_size);
                buf += impl_size;
                size -= impl_size;
            }
        }
    }
}

bool common_speculative_ring_state_write(
        const common_speculative * spec,
        llama_io_write_i & output) {
    if (spec == nullptr) {
        return false;
    }
    size_t written = 0;
    for (const auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
            continue;
        }
        const auto * dfl = static_cast<
            const common_speculative_impl_dflash *>(impl.get());
        const size_t before = output.n_bytes();
        if (!dfl->ring_state_write(output) || output.n_bytes() < before) {
            return false;
        }
        written += output.n_bytes()-before;
    }
    return written != 0;
}

bool common_speculative_ring_state_load(common_speculative * spec, const uint8_t * buf, size_t size) {
    if (spec == nullptr) return false;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            if (static_cast<common_speculative_impl_dflash *>(impl.get())->ring_state_load(buf, size)) {
                return true;
            }
        }
    }
    return false;
}

bool common_speculative_ring_state_read(
        common_speculative * spec,
        llama_io_read_i & input,
        size_t size) {
    if (spec == nullptr || size == 0) {
        return false;
    }
    common_speculative_impl_dflash * selected = nullptr;
    for (auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
            continue;
        }
        // The contiguous load API has always selected one DFlash image. Keep
        // the streaming authority equally narrow instead of ambiguously
        // partitioning a single companion across multiple implementations.
        if (selected != nullptr) {
            return false;
        }
        selected = static_cast<common_speculative_impl_dflash *>(impl.get());
    }
    return selected && selected->ring_state_read(input, size);
}

bool common_speculative_ring_state_get_currency(
        const common_speculative * spec,
        common_speculative_ring_state_currency & output) {
    output = {};
    output.terminal = -1;
    if (spec == nullptr) {
        return false;
    }
    const common_speculative_impl_dflash * selected = nullptr;
    for (const auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
            continue;
        }
        if (selected != nullptr) {
            return false;
        }
        selected = static_cast<const common_speculative_impl_dflash *>(
            impl.get());
    }
    if (!selected) {
        return false;
    }
    selected->ring_state_currency(output);
    return output.mutation_epoch != 0;
}

bool common_speculative_ring_state_matches_frontier(
        const common_speculative * spec,
        const uint8_t * buf,
        size_t size,
        llama_pos expected_terminal) {
    if (spec == nullptr || buf == nullptr || size == 0) {
        return false;
    }
    size_t matched = 0;
    for (const auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
            continue;
        }
        const auto * dfl = static_cast<
            const common_speculative_impl_dflash *>(impl.get());
        size_t impl_size = 0;
        if (!dfl->ring_state_serialized_size(
                buf+matched, size-matched, impl_size) ||
            impl_size == 0 ||
            !dfl->ring_state_matches_frontier(
                buf+matched, impl_size, expected_terminal)) {
            return false;
        }
        matched += impl_size;
    }
    return matched != 0 && matched == size;
}

bool common_speculative_ring_state_terminal(
        const common_speculative * spec,
        llama_pos & terminal) {
    terminal = -1;
    if (spec == nullptr) {
        return false;
    }
    bool found = false;
    for (const auto & impl : spec->impls) {
        if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
            continue;
        }
        llama_pos one = -1;
        if (!static_cast<const common_speculative_impl_dflash *>(impl.get())->
                ring_state_terminal(one) ||
            (found && one != terminal)) {
            terminal = -1;
            return false;
        }
        terminal = one;
        found = true;
    }
    return found;
}

bool common_speculative_ring_state_serialized_terminal(
        const uint8_t * buf,
        size_t size,
        llama_pos & terminal) {
    terminal = -1;
    if (!buf || size < 6*sizeof(int32_t)) {
        return false;
    }
    int32_t committed = 0;
    std::memcpy(&committed, buf+2*sizeof(int32_t), sizeof(committed));
    if (committed <= 0) {
        return false;
    }
    terminal = llama_pos(committed-1);
    return true;
}

bool common_speculative_ring_state_empty(const common_speculative * spec) {
    if (spec == nullptr) return true;
    for (const auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH &&
            !static_cast<const common_speculative_impl_dflash *>(impl.get())->
                ring_state_empty()) {
            return false;
        }
    }
    return true;
}

void common_speculative_ring_state_reset(common_speculative * spec) {
    if (spec == nullptr) return;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            static_cast<common_speculative_impl_dflash *>(impl.get())->
                ring_state_reset();
        }
    }
}

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }
    if (params.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
        return params.n_max;
    }
    return common_speculative_n_max(&params);
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }
    if (params.has_type(COMMON_SPECULATIVE_TYPE_DFLASH)) {
        return params.n_min;
    }
    if (params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) ||
        params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) ||
        params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) ||
        params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
        params.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
        return params.draft.n_min;
    }
    if (params.has_type(COMMON_SPECULATIVE_TYPE_NGRAM_MOD)) {
        return params.ngram_mod.n_min;
    }
    return 0;
}
