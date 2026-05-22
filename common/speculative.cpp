#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_pre_norm / llama_get_embeddings_pre_norm_ith (used by MTP)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cinttypes>
#include <queue>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
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

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
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
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
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

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted) = 0;

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_pre_norm() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

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
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

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

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

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
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

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
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
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
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
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

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    //common_params_speculative_eagle3 params;

    common_speculative_impl_draft_eagle3(const common_params_speculative & /*params*/, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        // TODO: implement
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_state_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    int32_t n_embd = 0;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    common_speculative_state_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd(llama_get_model(ctx_dft));

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 1; // TODO: re-enable top_k == 10 and utilize `p_min` spec param
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        llama_set_embeddings_pre_norm(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_pre_norm(ctx_dft, true, /*masked*/ true);

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);
    }

    ~common_speculative_state_draft_mtp() override {
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
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d — "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
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

        // truncate ctx_dft to the start of this batch — draft() may have advanced
        // ctx_dft past these positions, and M-RoPE requires monotonic positions
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
            if (i_batch_beg[s] >= 0) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), s, batch_in.pos[i_batch_beg[s]], -1);
            }
        }

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

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
            const float * h_tgt = llama_get_embeddings_pre_norm(ctx_tgt);
            std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));

            //{
            //    // string with seq_ids in the batch
            //    std::stringstream ss;
            //    for (int i = 0; i < n_tokens; ++i) {
            //        ss << batch_in.seq_id[i][0] << ",";
            //    }
            //    LOG_WRN("%s: batch_in.seq_id = %s\n", __func__, ss.str().c_str());
            //}
        }

        // fill the pending embeddings from a previous run
        auto set_h = [&](int idx, const float * h_row) {
            std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
        }

        const int32_t rc = llama_decode(ctx_dft, batch);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
            return false;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_pre_norm_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
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

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

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
                h_row = llama_get_embeddings_pre_norm_ith(ctx_dft, i_batch);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
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

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_pre_norm() const override {
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
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config) {}

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

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_params_speculative & params,
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
        , params(params.ngram_map_k) {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }
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

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
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

        // length of the last drafted n‑gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                this->params.n_match, mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
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
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

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

        // store length of drafted n‑gram for later acceptance analysis
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

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.5) {
                sinfo.n_low++;
                if (sinfo.n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, sinfo.n_low);
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

    bool need_embd() const override {
        return false;
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
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
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
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
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

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

// ============================================================================
// Fork-specific speculative implementations
// ============================================================================

// (fork classes will be inserted here via sed)
// Fork-specific speculative decoding classes, ported to common_speculative_impl base.
// Insert after the last upstream impl (ngram_cache) and before `struct common_speculative`.

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
        : common_speculative_impl(type, n_seq)
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

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
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

    // hash of gamma-length window -> position after the window in the prompt
    std::unordered_multimap<uint64_t, int32_t> index;
    llama_tokens prompt_tokens;

    int32_t original_prompt_size = 0;
    bool has_model_drafter = false; // true when paired with DFlash/draft (apply primary threshold)

    common_speculative_impl_copyspec(common_speculative_type type, uint32_t n_seq, int32_t gamma)
        : common_speculative_impl(type, n_seq)
        , gamma(gamma)
    {}

    static uint64_t hash_window(const llama_token * tokens, int32_t len) {
        uint64_t h = FNV_OFFSET;
        for (int32_t i = 0; i < len; i++) {
            h ^= (uint64_t)(uint32_t)tokens[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        index.clear();
        prompt_tokens = prompt;
        original_prompt_size = (int32_t)prompt.size();
        if ((int32_t)prompt.size() <= gamma) {
            return;
        }
        for (int32_t i = 0; i <= (int32_t)prompt.size() - gamma; i++) {
            uint64_t h = hash_window(prompt.data() + i, gamma);
            index.emplace(h, i + gamma);
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
            auto range = index.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                int32_t pos = it->second;
                // verify hash match (collision check)
                if (pos < gamma || pos > (int32_t)prompt_tokens.size()) {
                    continue;
                }
                bool match = true;
                for (int32_t j = 0; j < gamma; j++) {
                    if (prompt_tokens[pos - gamma + j] != window[j]) {
                        match = false;
                        break;
                    }
                }
                if (!match) {
                    continue;
                }
                int32_t avail = (int32_t)prompt_tokens.size() - pos;
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
                const int32_t avail_in_orig = std::max(0, original_prompt_size - best_pos);
                if (avail_in_orig < 2 * n_max_eff) {
                    continue;
                }
            }

            const int32_t draft_len = std::min(n_max_eff, best_avail);
            for (int32_t i = 0; i < draft_len; i++) {
                result.push_back(prompt_tokens[best_pos + i]);
            }
        }
    }

    // Extend an existing draft by looking for suffix matches at the end of (prompt + draft)
    void extend(
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            int32_t n_max_ext) {
        if (result.empty() || index.empty()) {
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

        auto range = index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            int32_t pos = it->second;
            if (pos < gamma || pos > (int32_t)prompt_tokens.size()) {
                continue;
            }
            bool match = true;
            for (int32_t j = 0; j < gamma; j++) {
                if (prompt_tokens[pos - gamma + j] != window[j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            int32_t avail = std::min(max_ext, (int32_t)prompt_tokens.size() - pos);
            if (avail > best_len) {
                best_len = avail;
                best_pos = pos;
            }
        }

        if (best_pos < 0) {
            return;
        }

        for (int32_t i = 0; i < best_len; i++) {
            result.push_back(prompt_tokens[best_pos + i]);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
    }

    // incrementally extend index with accepted tokens
    void update_logits(llama_context * /*ctx*/, const llama_tokens & batch_tokens, int n_accepted) {
        // batch_tokens = [id_last, draft0, draft1, ...], n_accepted of which were accepted
        for (int i = 0; i < n_accepted && i < (int)batch_tokens.size(); i++) {
            prompt_tokens.push_back(batch_tokens[i]);
            // add new gamma-length window ending at this position
            if ((int32_t)prompt_tokens.size() >= gamma) {
                int32_t start = (int32_t)prompt_tokens.size() - gamma;
                uint64_t h = hash_window(prompt_tokens.data() + start, gamma);
                index.emplace(h, (int32_t)prompt_tokens.size());
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

    common_speculative_impl_recycle(common_speculative_type type, uint32_t n_seq, int32_t k)
        : common_speculative_impl(type, n_seq)
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

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
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
    bool prefill_flushed = false; // true if flush_prefill() was called during this request

    // Interleaved cross-attention buffer — rebuilt from ring on each draft call
    // Only holds ctx_window tokens worth of data
    std::vector<float> cross_buf;

    // A2: sliding window limit for drafter context (0 = unlimited)
    static constexpr int ctx_window = LLAMA_DFLASH_PER_SLOT_CTX;

    // GPU cross-attention ring (nullptr = CPU fallback)
    void * gpu_ring_handle = nullptr;

    // Adaptive draft length tracking
    int n_low_accept = 0;
    int n_draft_last = 0;
    int adaptive_n_draft = -1; // -1 = use default

    // build interleaved cross-attention data from ring buffer (GPU or CPU path)
    int build_cross_data(llama_context * ctx) {
        if (gpu_ring_handle) {
            int gpu_write_pos = ring_write_pos % ctx_window;
            int gpu_filled = std::min(ring_filled, ctx_window);
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
            bool            owns_ctx_dft_ = true,
            float           p_min_ = 0.0f)
        : common_speculative_impl(type, n_seq)
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

        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n_tokens <= 0) return;

        if (!prefill_flushed) {
            // first flush for this request — reset ring
            ring_write_pos = 0;
            ring_filled = 0;
            committed_len = 0;
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

    void ring_state_save(uint8_t * buf, size_t size) const {
        int n_entries = std::min(ring_filled, RING_SIZE);
        size_t expected = 6 * sizeof(int32_t) +
                          (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
        if (size < expected) return;

        int32_t * hdr = (int32_t *)buf;
        hdr[0] = ring_write_pos;
        hdr[1] = ring_filled;
        hdr[2] = committed_len;
        hdr[3] = n_target_layers;
        hdr[4] = n_embd;
        hdr[5] = n_entries;

        uint8_t * dst = buf + 6 * sizeof(int32_t);
        size_t layer_bytes = (size_t)n_entries * n_embd * sizeof(float);

        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(dst, ring_buf[l].data(), layer_bytes);
            dst += layer_bytes;
        }
    }

    bool ring_state_load(const uint8_t * buf, size_t size) {
        if (size < 6 * sizeof(int32_t)) return false;

        const int32_t * hdr = (const int32_t *)buf;
        int saved_write_pos = hdr[0];
        int saved_filled    = hdr[1];
        int saved_committed = hdr[2];
        int saved_layers    = hdr[3];
        int saved_embd      = hdr[4];
        int saved_entries    = hdr[5];

        if (saved_layers != n_target_layers || saved_embd != n_embd) {
            LOG_WRN("dflash: ring state mismatch: layers %d/%d, embd %d/%d\n",
                    saved_layers, n_target_layers, saved_embd, n_embd);
            return false;
        }

        if (saved_write_pos < 0 || saved_write_pos >= RING_SIZE ||
            saved_filled < 0 || saved_entries < 0 || saved_entries > RING_SIZE) {
            LOG_WRN("dflash: ring state corrupt: write_pos=%d, filled=%d, entries=%d\n",
                    saved_write_pos, saved_filled, saved_entries);
            return false;
        }

        size_t layer_bytes = (size_t)saved_entries * n_embd * sizeof(float);
        if (size < 6 * sizeof(int32_t) + layer_bytes * n_target_layers) return false;

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

        return true;
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

            const int64_t t3 = ggml_time_us();

            // read argmax tokens for positions 1..batch_len-1 (skip position 0 = staged_first)
            {
                int32_t * argmax = llama_get_logits_argmax(ctx_dft);
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

    void accept(llama_seq_id /*seq_id*/, uint16_t n_accepted) override {
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
    void ring_write(int n_tokens, int src_offset = 0) {
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
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
        ring_write_pos = (ring_write_pos + n_tokens) % RING_SIZE;
        ring_filled = std::min(ring_filled + n_tokens, RING_SIZE);
    }

    // called after initial prefill — grab all hidden states
    void capture_target_hiddens() {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;

        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n_tokens <= 0) return;

        // only keep last RING_SIZE tokens if prompt exceeds ring capacity
        int start_offset = std::max(0, (int)n_tokens - RING_SIZE);
        int to_store = (int)n_tokens - start_offset;

        ring_write_pos = 0;
        ring_filled = 0;
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

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_model_path = !params.draft.mparams.path.empty();

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 13);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            if (!has_draft_model_path) {
                LOG_WRN("%s: draft model is not specified - cannot use 'draft' type\n", __func__);
                has_draft_simple = false;
            }
        } else if (has_draft_model_path && !has_mtp && !has_draft_eagle3) {
            LOG_WRN("%s: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it\n", __func__);
            has_draft_simple = true;
        }

        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(config.type).c_str());
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
                impls.push_back(std::make_unique<common_speculative_state_draft_mtp>(config.params, n_seq));
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
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            config.params, get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
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
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
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

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_pre_norm(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_pre_norm()) {
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

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
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

    if (!impl) {
        return;
    }

    // TODO: currently only the implementation that generated the draft is used to accept it
    //       however, some implementations (such as MTP) need to also "see" the accepted tokens
    //       extend `common_speculative_impl::accept()` with an extra argument `bool is_other` to
    //       inform the implementation if the accepted tokens are from another implementation and
    //       pass the accepted tokens to all remaining implementations using `is_other == true`
    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted);
        impl->n_call_accept++;
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

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
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
    if (ctx_dft == nullptr && params.model_dft) {
        ctx_dft = common_speculative_create_ctx_dft(params);
    }

    auto has_type = [&](common_speculative_type t) {
        return std::find(params.types.begin(), params.types.end(), t) != params.types.end();
    };

    std::vector<common_speculative_config> configs = {};
    {
        bool has_suffix   = has_type(COMMON_SPECULATIVE_TYPE_SUFFIX);
        bool has_copyspec = has_type(COMMON_SPECULATIVE_TYPE_COPYSPEC);
        bool has_recycle  = has_type(COMMON_SPECULATIVE_TYPE_RECYCLE);
        bool has_dflash   = has_type(COMMON_SPECULATIVE_TYPE_DFLASH);

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
            if (!has_copyspec) {
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
                    owns_ctx_dft, params.p_min));
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
                    config.type, n_seq, config.params.copyspec_gamma));
                LOG_INF("%s: copyspec speculative decoding (gamma=%d)\n",
                    __func__, config.params.copyspec_gamma);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_RECYCLE: {
                impls.push_back(std::make_unique<common_speculative_impl_recycle>(
                    config.type, n_seq, config.params.recycle_k));
                LOG_INF("%s: token recycling speculative decoding (k=%d)\n",
                    __func__, config.params.recycle_k);
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
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

    // set up dparams for seq 0
    auto & dp = spec->dparams[0];
    dp.drafting = true;
    dp.n_max    = params.n_max;
    // Use explicit n_past when provided (needed for vision tokens where
    // actual positions exceed text token count due to M-RoPE spatial dims).
    // Fall back to prompt_tgt.size() for text-only prompts.
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
                cs->extend(prompt_tgt, id_last, result, params.n_max);
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

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0 || spec == nullptr) {
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
        impl->accept(0, n_accepted);
        impl->n_call_accept++;
    }
}

void common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            static_cast<common_speculative_impl_dflash *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
            static_cast<common_speculative_impl_copyspec *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_RECYCLE) {
            static_cast<common_speculative_impl_recycle *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        }
    }
}

void common_speculative_rollback_dft(common_speculative * spec, llama_seq_id seq_id, llama_pos n_past, uint16_t n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            auto * mtp = static_cast<common_speculative_state_draft_mtp *>(impl.get());
            auto * ctx_dft = mtp->params.ctx_dft;
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            mtp->accept(seq_id, n_accepted);
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

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }
    GGML_UNUSED(params);
    return params.n_max;
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }
    return params.n_min;
}
