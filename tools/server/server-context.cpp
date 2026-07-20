#include "server-context.h"
#include "server-chat.h"
#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-schema.h"
#include "server-stream.h"

#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "fit.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <cinttypes>
#include <exception>
#include <memory>
#include <random>
#include <filesystem>
#include <string>
#include <utility>
#include <fstream>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

// The dynamic VBR runtime controller flips KV tensor types in place as the context fills; state
// save/restore, context checkpoints and cache reuse all assume a fixed cache layout and would
// restore/reuse bytes under the wrong tier. Gate them off whenever the controller can arm.
// (This replaces the old VBR_STAGE2A env gate — Stage2A itself is gone.)
static bool server_vbr_dynamic_active(const common_params & params) {
    return params.vbr_dynamic();
}

// defined near get_model_info(); shared by the /props and /models responses
static json server_vbr_meta_json(const server_context_meta * meta);

static uint32_t server_n_outputs_max(const common_params & params) {
    const uint32_t n_batch  = params.n_batch;

    if (params.embedding ||
            (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED && params.pooling_type != LLAMA_POOLING_TYPE_NONE)) {
        return n_batch;
    }

    // the fork DFlash impl reports need_embd() on generating slots, so every prompt
    // chunk on a DFlash slot decodes with output-all embeddings. The cap must cover a
    // full batch or output_reserve aborts on the first prompt chunk longer than the
    // per-seq floor (~64 tokens) — e.g. any follow-up turn that re-processes context.
    // DFlash may only be detected from the draft GGUF after the target context is
    // created (same ordering problem as the verify floor below), so gate on has_dft()
    // too. output_reserve grows its buffers lazily, so the higher cap costs nothing
    // until a batch actually requests that many outputs (the pre-cap reservation).
    if (params.speculative.has_type(COMMON_SPECULATIVE_TYPE_DFLASH) ||
        params.speculative.has_dft()) {
        return n_batch;
    }

    const uint32_t n_outputs_per_seq = 1 + common_speculative_n_max(&params.speculative);

    const uint64_t n_outputs = (uint64_t) params.n_parallel * n_outputs_per_seq;

    return std::max<uint32_t>(1, std::min<uint64_t>(n_batch, n_outputs));
}

// state diagram: https://github.com/ggml-org/llama.cpp/pull/9283
enum slot_state {
    SLOT_STATE_IDLE,
    SLOT_STATE_WAIT_OTHER, // after assigning a task, but waiting for parent slot to process prompt
    SLOT_STATE_STARTED,    // after assigning a task and about to process prompt
    SLOT_STATE_PROCESSING_PROMPT,
    SLOT_STATE_DONE_PROMPT,
    SLOT_STATE_GENERATING,
};

struct server_slot; // forward declaration

struct server_batch {
    llama_batch batch;
    bool batch_rendered = false;

    struct token {
        int32_t id_slot;
        llama_token token;
        llama_pos pos;
        bool output;
    };
    std::vector<token> tokens;
    int32_t n_tokens_alloc = 0;

    // track if given slot can be batched with slots already in the batch
    server_slot * slot_batched = nullptr;

    float  alora_scale       = -1.0f;
    size_t alora_disabled_id = 0;

    server_batch() {
        batch.token = nullptr; // sentinel: uninitialized batch
    }

    ~server_batch() {
        if (batch.token != nullptr) {
            llama_batch_free(batch);
        }
    }

    void init(int32_t n_tokens_alloc) {
        this->n_tokens_alloc = n_tokens_alloc;
        batch = llama_batch_init(n_tokens_alloc, 0, 1);
        tokens.reserve(n_tokens_alloc);
    }

    bool add(int32_t id_slot, llama_token token, llama_pos pos, bool output) {
        GGML_ASSERT(batch.token != nullptr);
        if ((int32_t)tokens.size() >= n_tokens_alloc) {
            return false;
        }
        tokens.push_back({ id_slot, token, pos, output });
        return true;
    }

    void clear() {
        tokens.clear();
        common_batch_clear(batch);
        slot_batched      = nullptr;
        alora_scale       = -1.0f;
        alora_disabled_id = 0;
        batch_rendered    = false;
    }

    int32_t size() const {
        return (int32_t)tokens.size();
    }

    void set_output(int32_t idx, bool output) {
        GGML_ASSERT(idx >= 0 && idx < (int32_t)tokens.size());
        tokens[idx].output = output;
    }

    void render() {
        GGML_ASSERT(batch.token != nullptr);
        common_batch_clear(batch);
        for (int32_t i = 0; i < size(); i++) {
            const auto & t = tokens[i];
            common_batch_add(batch, t.token, t.pos, { t.id_slot }, t.output);
        }
        batch_rendered = true;
    }

    llama_batch get_view(int32_t off, int32_t n_tokens) const {
        GGML_ASSERT(batch.token != nullptr);
        GGML_ASSERT(batch_rendered);
        GGML_ASSERT(off >= 0 && off < size());
        GGML_ASSERT(n_tokens > 0 && off + n_tokens <= size());

        llama_batch view = {
            n_tokens,
            batch.token    + off,
            nullptr,
            batch.pos      + off,
            batch.n_seq_id + off,
            batch.seq_id   + off,
            batch.logits   + off,
        };

        return view;
    }
};

struct server_slot {
    int id;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    // multimodal
    mtmd_context * mctx = nullptr;
    mtmd::batch_ptr mbatch = nullptr;

    // diffusion self-speculation
    bool                 diff_self_spec     = false;
    int32_t              diff_draft_length  = 4;
    llama_token          diff_mask_token_id = LLAMA_TOKEN_NULL;
    llama_token          diff_think_open_id = LLAMA_TOKEN_NULL;
    llama_token          diff_think_close_id = LLAMA_TOKEN_NULL;
    std::vector<float>   diff_prev_logits;
    std::vector<llama_token> diff_prev_assistant_tokens;

    // speculative decoding
    common_speculative_ptr spec;
    common_speculative * spec_shared = nullptr; // non-owning

    llama_tokens spec_draft;
    llama_tokens spec_prompt;
    std::vector<int32_t> spec_i_batch;
    common_prompt_checkpoint spec_ckpt;

    // TODO: move members that belong to the task (such as `generated_text`, `has_new_line`) to task_results_state
    //       see https://github.com/ggml-org/llama.cpp/pull/18283#issuecomment-3710175837
    std::unique_ptr<const server_task> task;
    std::unique_ptr<const server_task> task_prev; // used for debugging

    // used to determine the slot that has been used the longest
    int64_t t_last_used = -1;

    // generation props
    int32_t n_ctx       = 0;  // context size per slot
    int32_t n_keep      = 0;
    int32_t n_decoded   = 0;
    int32_t n_remaining = -1;
    int32_t i_batch     = -1;

    int32_t n_prompt_tokens_cache     = 0;
    int32_t n_prompt_tokens_processed = 0;

    size_t last_nl_pos = 0;

    std::string  generated_text;
    std::string  debug_generated_text;
    llama_tokens generated_tokens;

    std::vector<completion_token_output> generated_token_probs;

    bool has_next_token = true;
    bool has_new_line   = false;
    bool truncated      = false;

    stop_type stop;

    std::string stopping_word;

    // state
    slot_state state = SLOT_STATE_IDLE;

    server_prompt prompt;

    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        GGML_ASSERT(prompt.data.size() == 0);

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)\n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }

    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache\n");
        }

        return res;
    }

    void prompt_clear(bool allow_processing) {
        if (!allow_processing) {
            GGML_ASSERT(!is_processing());
        }

        SLT_TRC(*this, "clearing prompt with %zu tokens\n", prompt.tokens.size());

        common_context_seq_rm(ctx_tgt, id, -1, -1);
        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, id, -1, -1);
        }

        prompt.tokens.clear();
    }

    std::vector<common_adapter_lora_info> lora;
    int32_t alora_invocation_start = -1;

    // sampling
    json json_schema;

    common_sampler_ptr smpl;

    llama_token sampled; // in speculative mode, this is the last accepted token

    // stats
    size_t n_sent_text = 0; // number of sent text character

    // TODO @ngxson : move all metrics to a sub-struct for clarity
    int64_t t_start_process_prompt;
    int64_t t_start_generation;
    int64_t t_print_last = 0;
    int32_t n_decoded_last = 0;

    double t_prompt_processing = 0.0; // ms
    double t_token_generation = 0.0;  // ms

    std::function<void(int /* id_slot */)> callback_on_release;

    // Speculative decoding stats
    int32_t n_draft_total = 0;      // Total draft tokens generated
    int32_t n_draft_accepted = 0;   // Draft tokens actually accepted
    int32_t n_draft_verif_steps = 0; // Total draft token verification steps by the target model
    std::vector<int32_t> n_accepted_per_pos; // Accepted tokens per draft position

    // Hybrid model: recurrent state backup for speculative decoding
    bool has_draft_backup = false;
    llama_seq_id seq_id_backup = -1;
    int  n_tokens_before_draft = 0; // prompt token count before draft tokens were added

    void reset() {
        SLT_DBG(*this, "%s", "\n");

        n_prompt_tokens_cache = 0;

        last_nl_pos    = 0;
        generated_text = "";
        has_new_line   = false;
        truncated      = false;
        stop           = STOP_TYPE_NONE;
        stopping_word  = "";
        n_sent_text    = 0;

        if (can_speculate()) {
            spec_draft.clear();
            spec_i_batch.clear();
            spec_ckpt.clear();
        }
        generated_tokens.clear();
        generated_token_probs.clear();
        json_schema = json();

        // clear speculative decoding stats
        n_draft_total = 0;
        n_draft_accepted = 0;
        n_draft_verif_steps = 0;
        n_accepted_per_pos.clear();
        has_draft_backup = false;
        seq_id_backup = -1;
        n_tokens_before_draft = 0;

        task_prev = std::move(task);
        task.reset();

        llama_set_sampler(ctx_tgt, id, nullptr);

        // clear alora start
        alora_invocation_start = -1;

        // clear multimodal state
        mbatch.reset();
    }

    void init_sampler() const {
        common_sampler_reset(smpl.get());

        if (!task->need_sampling()) {
            return;
        }

        const int64_t t_start = ggml_time_us();

        int n_text = 0;

        for (int i = 0; i < (int) prompt.tokens.size(); i++) {
            const llama_token id = prompt.tokens[i];

            if (id != LLAMA_TOKEN_NULL) {
                common_sampler_accept(smpl.get(), id, false);
                n_text++;
            }
        }

        SLT_TRC(*this, "init sampler, took %0.2f ms, tokens: text = %d, total = %d\n",
                (ggml_time_us() - t_start) / 1000.0, n_text, (int) prompt.tokens.size());
    }

    bool need_embd() const {
        GGML_ASSERT(task);
        return task->need_embd() || (get_spec() && common_speculative_need_embd(get_spec()));
    }

    bool need_embd_nextn() const {
        GGML_ASSERT(task);
        return get_spec() && common_speculative_need_embd_nextn(get_spec());
    }

    // if the context does not have a memory module then all embeddings have to be computed within a single ubatch
    // also we cannot split if the pooling would require any past tokens
    // (MTP supports splitting — uses task->need_embd() not need_embd())
    bool can_split() const {
        GGML_ASSERT(task);

        return
            !task->need_embd() ||
            (llama_get_memory(ctx_tgt) && llama_pooling_type(ctx_tgt) == LLAMA_POOLING_TYPE_LAST);
    }

    bool can_batch_with(server_slot & other_slot) const {
        GGML_ASSERT(task);

        return task->type == other_slot.task->type && are_lora_equal(lora, other_slot.lora);
    }

    bool has_budget(const common_params & global_params) {
        GGML_ASSERT(task);

        if (task->params.n_predict == -1 && global_params.n_predict == -1) {
            return true; // limitless
        }

        n_remaining = -1;

        if (task->params.n_predict != -1) {
            n_remaining = task->params.n_predict - n_decoded;
        } else if (global_params.n_predict != -1) {
            n_remaining = global_params.n_predict - n_decoded;
        }

        return n_remaining > 0; // no budget
    }

    bool is_processing() const {
        return state != SLOT_STATE_IDLE;
    }

    bool can_speculate() const {
        return spec || spec_shared;
    }

    common_speculative * get_spec() const {
        return spec ? spec.get() : spec_shared;
    }

    void add_token(const completion_token_output & token) {
        if (!is_processing()) {
            SLT_WRN(*this, "%s", "slot is not processing\n");
            return;
        }

        generated_token_probs.push_back(token);
    }

    int get_n_draft_max() const {
        GGML_ASSERT(task);

        if (!can_speculate()) {
            return 0;
        }

        if (common_sampler_grammar_is_active(smpl.get())) {
            return 0;
        }

        const int n_draft_min = common_speculative_n_min(get_spec(), task->params.speculative);


        // determine the max draft that fits the current slot state
        // note: slot.prompt is not yet expanded with the `id` token sampled above
        //       also, need to leave space for 1 extra token to allow context shifts
        int n_draft_max = n_ctx - prompt.n_tokens() - 2;

        if (n_remaining > 0) {
            n_draft_max = std::min(n_draft_max, n_remaining - 1);
        }

        SLT_DBG(*this, "max possible draft: %d\n", n_draft_max);

        return n_draft_max;
    }

    // add sampled token of this slot to the batch, optionally add the speculative draft tokens if any
    void handle_last_sampled_token(server_batch & batch) {
        bool add_ok = true;
        if (spec_draft.empty()) {
            // no speculative decoding
            i_batch = batch.size();

            add_ok &= batch.add(id, sampled, prompt.tokens.pos_next(), true);

            SLT_DBG(*this, "slot decode token, id=%d, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                    sampled, n_ctx, prompt.n_tokens(), truncated);
        } else {
            SLT_DBG(*this, "generate_draft: id=%d, #tokens=%zu, #draft=%zu, pos_next=%d\n",
                    sampled, prompt.tokens.size(), spec_draft.size(), prompt.tokens.pos_next());

            GGML_ASSERT(spec_i_batch.empty());

            spec_i_batch.push_back(batch.size());
            for (size_t i = 0; i < spec_draft.size(); i++) {
                spec_i_batch.push_back(batch.size() + i + 1);
            }

            auto pos0 = prompt.tokens.pos_next();

            add_ok &= batch.add(id, sampled, pos0++, true);
            for (auto token : spec_draft) {
                add_ok &= batch.add(this->id, token, pos0++, true);
            }
        }

        GGML_ASSERT(add_ok && "batch must be large enough to hold the sampled and draft tokens");

        prompt.tokens.push_back(sampled);
        prompt.tokens.insert(spec_draft);
    }

    void release() {
        if (is_processing()) {
            GGML_ASSERT(task);

            SLT_INF(*this, "stop processing: n_tokens = %d, truncated = %d\n", prompt.n_tokens(), truncated);

            t_last_used        =  ggml_time_us();
            t_token_generation = (ggml_time_us() - t_start_generation) / 1e3;

            state = SLOT_STATE_IDLE;

            // clean up speculative backup sequence to avoid orphaned KV cells
            if (has_draft_backup && seq_id_backup >= 0) {
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id_backup, -1, -1);
            }

            // do not keep context of the child slots - the parent's context is enough
            if (task->is_child()) {
                prompt_clear(false);
            }

            reset();

            callback_on_release(id);
        }
    }

    result_timings get_timings() const {
        result_timings timings;
        timings.cache_n = n_prompt_tokens_cache;

        timings.prompt_n            = n_prompt_tokens_processed;
        timings.prompt_ms           = t_prompt_processing;
        timings.prompt_per_token_ms = t_prompt_processing / n_prompt_tokens_processed;
        timings.prompt_per_second   = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        timings.predicted_n            = n_decoded;
        timings.predicted_ms           = t_token_generation;
        timings.predicted_per_token_ms = t_token_generation / n_decoded;
        timings.predicted_per_second   = 1e3 / t_token_generation * n_decoded;

        // Add speculative metrics
        if (n_draft_total > 0) {
            timings.draft_n          = n_draft_total;
            timings.draft_n_accepted = n_draft_accepted;
        }

        // live effective KV bits/value (moves under dynamic VBR as tiers degrade/reset)
        if (ctx_tgt != nullptr) {
            timings.kv_bpv = llama_memory_kv_bpv(llama_get_memory(ctx_tgt));
        }

        return timings;
    }

    size_t find_stopping_strings(const std::string & text, const size_t last_token_size, bool is_full_stop) {
        GGML_ASSERT(task);

        size_t stop_pos = std::string::npos;

        for (const std::string & word : task->params.antiprompt) {
            size_t pos;

            if (is_full_stop) {
                const size_t tmp      = word.size() + last_token_size;
                const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

                pos = text.find(word, from_pos);
            } else {
                // otherwise, partial stop
                pos = string_find_partial_stop(text, word);
            }

            if (pos != std::string::npos && (stop_pos == std::string::npos || pos < stop_pos)) {
                if (is_full_stop) {
                    stop           = STOP_TYPE_WORD;
                    stopping_word  = word;
                    has_next_token = false;
                }
                stop_pos = pos;
            }
        }

        return stop_pos;
    }

    void print_timings_tg() {
        if (n_decoded < 100) {
            return;
        }

        const int64_t t_now = ggml_time_us();

        if (t_now - t_print_last < 3*1000*1000) {
            return;
        }

        const double n_gen_second     = 1e3 / (t_token_generation)   * (n_decoded);
        const double n_gen_second_win = 1e6 / (t_now - t_print_last) * (n_decoded - n_decoded_last);

        t_print_last = t_now;
        n_decoded_last = n_decoded;

        SLT_INF(*this, "n_decoded = %6d, tg = %6.2f t/s, tg_3s = %6.2f t/s\n", n_decoded, n_gen_second, n_gen_second_win);
    }

    void print_timings_pp() const {
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;
        const double f_progress = (float) prompt.n_tokens() / task->n_tokens();

        if (t_prompt_processing < 3000.0) {
            return;
        }

        SLT_INF(*this, "prompt processing, n_tokens = %6d, progress = %.2f, t = %6.2f s / %.2f tokens per second\n",
                n_prompt_tokens_processed, f_progress, t_prompt_processing / 1e3, n_prompt_second);
    }

    void print_timings() const {
        const double t_prompt        =       t_prompt_processing / n_prompt_tokens_processed;
        const double n_prompt_second = 1e3 / t_prompt_processing * n_prompt_tokens_processed;

        const double t_gen        =       t_token_generation / n_decoded;
        const double n_gen_second = 1e3 / t_token_generation * n_decoded;

        SLT_INF(*this,
                "prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_prompt_processing, n_prompt_tokens_processed, t_prompt, n_prompt_second);

        SLT_INF(*this,
                "       eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                t_token_generation, n_decoded, t_gen, n_gen_second);

        SLT_INF(*this,
                "      total time = %10.2f ms / %5d tokens\n",
                t_prompt_processing + t_token_generation, n_prompt_tokens_processed + n_decoded);

        SLT_INF(*this,
                "   graphs reused = %10d\n",
                llama_perf_context(ctx_tgt).n_reused);

        if (n_draft_total > 0) {
            const float  draft_ratio  = (float) n_draft_accepted / n_draft_total;
            const double mean_acc_len = n_draft_verif_steps > 0 ? 1.0 + (double) n_draft_accepted / (double) n_draft_verif_steps : 1.0;

            std::string acceptance_rates_per_pos;
            if (n_draft_verif_steps > 0) {
                for (size_t i = 0; i < n_accepted_per_pos.size(); ++i) {
                    if (i > 0) {
                        acceptance_rates_per_pos += ", ";
                    }
                    acceptance_rates_per_pos += string_format("%.3f", (double) n_accepted_per_pos[i] / (double) n_draft_verif_steps);
                }
            }

            SLT_INF(*this,
                    "draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f\n",
                    draft_ratio, n_draft_accepted, n_draft_total, mean_acc_len);
            SLT_TRC(*this,
                    "     acc per pos = (%s)\n", acceptance_rates_per_pos.c_str());
        }

        common_speculative_print_stats(spec.get());
    }

    json to_json(bool only_metrics = false) const {
        json res;

        res = {
            {"id",            id},
            {"n_ctx",         n_ctx},
            {"speculative",   can_speculate()},
            {"is_processing", is_processing()},
        };

        // live effective KV bits/value (moves under dynamic VBR); pollable via GET /slots
        if (ctx_tgt != nullptr) {
            const double kv_bpv = llama_memory_kv_bpv(llama_get_memory(ctx_tgt));
            if (kv_bpv >= 0.0) {
                res["kv_bpv"] = kv_bpv;
            }
        }

        const auto & ptask = task ? task : task_prev;

        if (ptask) {
            res["id_task"] = ptask->id;
            res["n_prompt_tokens"]           = (int32_t) prompt.tokens.size();
            res["n_prompt_tokens_processed"] = n_prompt_tokens_processed;
            res["n_prompt_tokens_cache"]     = n_prompt_tokens_cache;
            res["params"] = ptask->params.to_json(only_metrics);
            res["next_token"] = {
                {
                    {"has_next_token", has_next_token},
                    {"has_new_line",   has_new_line},
                    {"n_remain",       n_remaining},
                    {"n_decoded",      n_decoded},
                }
            };

            if (!only_metrics) {
                res["prompt"] = ptask->tokens.detokenize(ctx_tgt, true);
                res["generated"] = generated_text.empty() ? debug_generated_text : generated_text;
            }
        }

        return res;
    }

    void copy_state_to(server_slot & other) const {
        GGML_ASSERT(state == SLOT_STATE_DONE_PROMPT);

        common_context_seq_rm(ctx_tgt, other.id,     -1, -1);
        common_context_seq_cp(ctx_tgt, id, other.id, -1, -1);

        if (ctx_dft) {
            common_context_seq_rm(ctx_dft, other.id,     -1, -1);
            common_context_seq_cp(ctx_dft, id, other.id, -1, -1);
        }

        other.n_decoded   = n_decoded;
        other.n_remaining = n_remaining;
        other.i_batch     = i_batch;

        other.t_start_process_prompt    = t_start_process_prompt;
        other.t_prompt_processing       = t_prompt_processing;
        other.n_prompt_tokens_cache     = n_prompt_tokens_cache;
        other.n_prompt_tokens_processed = n_prompt_tokens_processed;

        other.prompt = prompt.clone();
        other.init_sampler();
    }

    // returns 0 on success
    // caller need to update prompt.tokens after a successful call to keep track of the processing progress
    int process_mtmd_chunk(size_t idx, size_t & n_tokens_out) {
        GGML_ASSERT(mctx);
        const auto & input_tokens = task->tokens;
        const auto & chunk = input_tokens.find_chunk(idx);
        int32_t res = 0;

        auto try_decode = [&]() -> int32_t {
            if (mbatch) {
                float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk.get());
                if (embd) {
                    void * cb_data = spec.get();
                    static auto cb = [](llama_batch batch, void * user_data) {
                        common_speculative * spec = static_cast<common_speculative *>(user_data);
                        if (!common_speculative_process(spec, batch)) {
                            return 1;
                        }
                        return 0;
                    };

                    llama_pos new_n_past; // unused for now
                    res = mtmd_helper_decode_image_chunk(
                        mctx,
                        ctx_tgt,
                        chunk.get(),
                        embd,
                        prompt.tokens.pos_next(),
                        id,
                        llama_n_batch(ctx_tgt),
                        &new_n_past,
                        cb,
                        cb_data
                    );
                    if (res != 0) {
                        SLT_ERR(*this, "failed to decode mtmd chunk, idx = %zu, res = %d\n", idx, res);
                        return -1;
                    }
                    n_tokens_out = mtmd_input_chunk_get_n_tokens(chunk.get());
                    return 0; // success
                }
            }
            return 1; // (non-error) need to create & encode batch
        };

        // if the batch is already exist, try searching & encode
        res = try_decode();
        if (res == 0) {
            return 0;
        }
        if (res < 0) {
            // fatal error
            return res;
        }

        // otherwise, the batch is either uninitialized or is used up
        // we need to create & encode a new batch
        mbatch.reset(mtmd_batch_init(mctx));
        res = mtmd_batch_add_chunk(mbatch.get(), chunk.get());
        GGML_ASSERT(res == 0); // we should never have an empty batch

        // try batching as much as possible
        int n_added = 1;
        size_t idx_cur = idx;
        while (res == 0) {
            auto [next_chunk, next_idx] = input_tokens.find_next_media_chunk(idx_cur);
            if (next_chunk == nullptr) {
                break;
            }
            res = mtmd_batch_add_chunk(mbatch.get(), next_chunk->get());
            n_added += (res == 0 ? 1 : 0);
            idx_cur = next_idx;
            SLT_DBG(*this, "try adding media chunk idx = %zu to batch, res = %d\n", next_idx, res);
            // if res != 0, batch is full or chunk is not compatible -> this loop breaks
        }

        // TODO @ngxson : move this log line to debug when it become more stable
        SLT_TRC(*this, "encoding mtmd batch from idx = %zu, n_chunks = %d\n", idx, n_added);

        res = mtmd_batch_encode(mbatch.get());
        if (res != 0) {
            SLT_ERR(*this, "failed to encode mtmd batch for chunk idx = %zu, res = %d\n", idx, res);
            return -1;
        }

        return try_decode();
    }
};



//
// server_metrics
//

struct server_metrics {
    int64_t t_start = 0;

    uint64_t n_prompt_tokens_processed_total = 0;
    uint64_t t_prompt_processing_total       = 0;
    uint64_t n_tokens_predicted_total        = 0;
    uint64_t t_tokens_generation_total       = 0;

    uint64_t n_tokens_max = 0;

    uint64_t n_prompt_tokens_processed = 0;
    uint64_t t_prompt_processing       = 0;

    uint64_t n_tokens_predicted  = 0;
    uint64_t t_tokens_generation = 0;

    uint64_t n_decode_total     = 0;
    uint64_t n_busy_slots_total = 0;

    void init() {
        t_start = ggml_time_us();
    }

    void on_prompt_eval(const server_slot & slot) {
        n_prompt_tokens_processed_total += slot.n_prompt_tokens_processed;
        n_prompt_tokens_processed       += slot.n_prompt_tokens_processed;
        t_prompt_processing             += slot.t_prompt_processing;
        t_prompt_processing_total       += slot.t_prompt_processing;

        n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
    }

    void on_prediction(const server_slot & slot) {
        n_tokens_predicted_total   += slot.n_decoded;
        n_tokens_predicted         += slot.n_decoded;
        t_tokens_generation        += slot.t_token_generation;
        t_tokens_generation_total  += slot.t_token_generation;
    }

    void on_decoded(const std::vector<server_slot> & slots) {
        n_decode_total++;
        for (const auto & slot : slots) {
            if (slot.is_processing()) {
                n_busy_slots_total++;
            }
            n_tokens_max = std::max(n_tokens_max, (uint64_t) slot.prompt.n_tokens());
        }
    }

    void reset_bucket() {
        n_prompt_tokens_processed = 0;
        t_prompt_processing       = 0;
        n_tokens_predicted        = 0;
        t_tokens_generation       = 0;
    }
};


//
// server_context_impl (private implementation)
//

struct server_context_impl {
    friend struct server_context;

public:
    // only use these pointers outside of this class:
    //  - when not in sleeping state
    //  - and, with thread-safe APIs (e.g., tokenizer calls)
    llama_model * model_tgt = nullptr;

    mtmd_context * mctx = nullptr;
    const llama_vocab * vocab = nullptr;

    server_queue    queue_tasks;
    server_response queue_results;

    // note: chat_params must not be refreshed upon existing sleeping state
    server_chat_params chat_params;

    server_state_callback_t callback_state = [](server_state, json) -> void {};

    server_context_impl() {
        mtmd_helper_log_set(common_log_default_callback, nullptr);
    }

    ~server_context_impl() {
        if (!sleeping) {
            // destroy() is already called when entering sleeping state
            // we don't call it again here to avoid double free
            destroy();
        }
    }

private:
    // note: accessing these fields outside of this class is not thread-safe
    // use server_context methods instead

    common_params params_base;

    // note: keep these alive - they determine the lifetime of the model, context, etc.
    common_init_result_ptr llama_init;

    llama_context * ctx_tgt = nullptr;

    // DFlash: one drafter context shared across all slots'
    // common_speculative states (non-owning refs). Must outlive all specs — the
    // destroy() order below (specs first, then this) enforces that; when destroy()
    // isn't called explicitly, member-destructor order (reverse-declaration) frees
    // specs via `slots` before this unique_ptr runs.
    llama_context_ptr ctx_dft_shared;

    server_batch batch;

    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    common_context_seq_rm_type ctx_tgt_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type ctx_dft_seq_rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;

    common_speculative_ptr spec;

    bool add_bos_token = true;
    bool is_diffusion  = false;

    // hybrid/recurrent models need re-evaluation of accepted tokens after
    // rejecting draft tokens, because the recurrent state cannot be rolled back
    bool needs_reeval = false;
    int  n_parallel_user = 0;
    int  n_seq_max_full = 0;      // target n_seq_max after expansion (2*n_parallel_user)
    bool recurrent_expanded = true; // false = backup cells deferred, expand before first draft

    int32_t n_ctx; // total context for all clients / slots

    // set to llama_model_n_swa(model)
    // if swa_full is enabled, this is set to 0 to simulate a non-SWA model
    int32_t n_swa;

    // slots / clients
    std::vector<server_slot> slots;

    int trace = 0;
    int slots_debug = 0;
    int n_empty_consecutive = 0;

    std::unique_ptr<server_prompt_cache> prompt_cache;

    server_metrics metrics;

    json json_ui_settings = json::object();

    // Necessary similarity of prompt for slot selection
    float slot_prompt_similarity = 0.0f;

    std::string model_name; // name of the loaded model, to be used by API
    std::set<std::string> model_aliases; // additional names for the model
    std::set<std::string> model_tags;    // informational tags

    bool sleeping = false;

    // MTP↔mmproj GPU swap state
    bool mmproj_gpu_swap = false;
    bool mmproj_is_on_gpu = false;
    bool mtp_was_active_before_swap = false;

    int64_t t_last_load_progress_ms = 0;

    void destroy() {
        spec.reset();
        ctx_dft.reset();
        model_dft.reset();

        llama_init.reset();

        ctx_tgt = nullptr;
        model_tgt = nullptr;

        mtmd_free(mctx);
        mctx = nullptr;

        for (server_slot & slot : slots) {
            if (slot.can_speculate()) {
                slot.spec.reset();
            }
        }

        ctx_dft_shared.reset();

        // note: batch is a server_batch — its destructor frees the llama_batch
    }

    mtmd_context_params make_mmproj_params(bool use_gpu, mtmd_progress_callback progress_cb = nullptr, void * progress_ud = nullptr) const {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = use_gpu;
        mparams.print_timings    = false;
        mparams.n_threads        = params_base.cpuparams.n_threads;
        mparams.flash_attn_type  = params_base.flash_attn_type;
        mparams.warmup           = use_gpu; // only warmup when on GPU
        mparams.image_min_tokens = params_base.image_min_tokens;
        mparams.image_max_tokens = params_base.image_max_tokens;
        mparams.batch_max_tokens = params_base.mtmd_batch_max_tokens;
        mparams.media_marker     = get_media_marker();
        mparams.progress_callback           = progress_cb;
        mparams.progress_callback_user_data = progress_ud;
        return mparams;
    }

    void reload_mmproj(bool use_gpu) {
        mtmd_free(mctx);
        mctx = nullptr;
        auto mparams = make_mmproj_params(use_gpu);
        mctx = mtmd_init_from_file(params_base.mmproj.path.c_str(), model_tgt, mparams);
        for (server_slot & slot : slots) {
            slot.mctx = mctx;
        }
    }

    llama_context * create_mtp_context() {
        auto cparams = common_context_params_to_llama(params_base);
        cparams.ctx_type      = LLAMA_CONTEXT_TYPE_MTP;
        cparams.type_k        = params_base.speculative.draft.cache_type_k;
        cparams.type_v        = params_base.speculative.draft.cache_type_v;
        cparams.n_rs_seq      = 0;
        cparams.n_outputs_max = params_base.n_parallel;
        cparams.ctx_other     = ctx_tgt;
        return llama_init_from_model(model_tgt, cparams);
    }

    void swap_mtp_to_mmproj_gpu() {
        SRV_INF("%s", "swapping MTP out, loading mmproj to GPU...\n");
        int64_t t0 = ggml_time_us();

        for (server_slot & slot : slots) {
            slot.spec.reset();
            slot.spec_shared = nullptr;
        }
        spec.reset();

        mtp_was_active_before_swap = ctx_dft != nullptr;
        if (ctx_dft) {
            ctx_dft.reset();
            params_base.speculative.draft.ctx_dft = nullptr;
        }

        reload_mmproj(true);
        if (!mctx) {
            SRV_ERR("%s", "failed to load mmproj to GPU, falling back to CPU\n");
            reload_mmproj(false);
        } else {
            mmproj_is_on_gpu = true;
        }

        SRV_INF("swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
    }

    void swap_mmproj_to_mtp() {
        SRV_INF("%s", "unloading mmproj from GPU, restoring prior state...\n");
        int64_t t0 = ggml_time_us();

        mmproj_is_on_gpu = false;
        reload_mmproj(false);

        if (mtp_was_active_before_swap) {
            // Already had MTP before the swap — recreate it
            ctx_dft.reset(create_mtp_context());
            if (!ctx_dft) {
                SRV_ERR("%s", "failed to recreate MTP context after mmproj swap\n");
                return;
            }

            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();

            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to reinit speculative context: %s\n", e.what());
            }

            for (server_slot & slot : slots) {
                slot.ctx_dft = ctx_dft.get();
                if (spec) {
                    slot.spec_shared = spec.get();
                    common_speculative_set_seq_id(slot.get_spec(), slot.id);
                }
            }
        } else {
            // No MTP was active — mmproj stays on CPU until next image arrives
            SRV_INF("%s", "(no MTP to restore, mmproj will reload on next image)\n");
        }

        SRV_INF("swap done in %" PRId64 " ms\n", (ggml_time_us() - t0) / 1000);
    }

    void slot_save_and_clear(server_slot & slot) {
        if (slot.prompt.n_tokens() == 0) {
            return;
        }
        SLT_INF(slot, "%s", "saving idle slot to prompt cache\n");
        SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
        slot.prompt_save(*prompt_cache);
        slot.prompt_clear(false);
        prompt_cache->update();
    }

    // dynamic VBR: clear-only reclaim of idle slots (the prompt cache is disabled under the VBR
    // gates, so unlike cache_idle_slots there is nothing to save into — the cost is a re-prefill
    // if that conversation returns). Never touches processing slots or a slot an explicitly
    // pinned deferred task is waiting on. Returns the number of slots cleared.
    int vbr_clear_idle_slots(int except_id, const char * reason) {
        int cleared = 0;
        for (auto & s : slots) {
            if (s.id == except_id || s.is_processing() || s.prompt.n_tokens() == 0) {
                continue;
            }
            if (queue_tasks.has_deferred_for_slot(s.id)) {
                SLT_INF(s, "vbr reclaim (%s): kept — a deferred id_slot task pins this slot\n", reason);
                continue;
            }
            SLT_WRN(s, "vbr reclaim (%s): clearing %d cached tokens\n", reason, (int) s.prompt.n_tokens());
            s.prompt_clear(false);
            s.prompt.checkpoints.clear(); // host-RAM recurrent blobs would linger until slot reuse
            cleared++;
        }
        return cleared;
    }

    // reclaim-before-degrade: when the incoming work's projected footprint would push the
    // degrade ladder BELOW the quality floor, pay with idle caches instead of tiers. Above the
    // floor the trade inverts — a within-band degrade is near-lossless and cheaper than
    // destroying another conversation's re-prefillable cache, so caches are kept. Policy reads
    // deficit_raw only: page-exact and free-VRAM independent, so WHETHER a cache is erased
    // never depends on driver jitter or co-tenants (the cache's own live clamp still handles
    // those with tier degrades).
    void vbr_reclaim_before_degrade(int except_id, uint32_t n_tokens_extra, const char * reason) {
        if (!server_vbr_dynamic_active(params_base) || params_base.vbr_reclaim_floor_bpv <= 0.0f) {
            return;
        }
        const auto st = llama_memory_vbr_state(llama_get_memory(ctx_tgt), -1, n_tokens_extra);
        if (st.deficit_raw <= 0 || st.bpv_if_degraded >= (double) params_base.vbr_reclaim_floor_bpv) {
            return;
        }
        const int cleared = vbr_clear_idle_slots(except_id, reason);
        if (cleared > 0) {
            SRV_WRN("vbr reclaim (%s): cleared %d idle slot(s) — deficit %.2f MiB, degrading instead would land %.3f bpv < floor %.3f\n",
                    reason, cleared, st.deficit_raw / 1024.0 / 1024.0, st.bpv_if_degraded,
                    (double) params_base.vbr_reclaim_floor_bpv);
        }
    }

    // reset-on-low-LCP (dynamic VBR): tiers are per-tensor and promotion cannot cross the tap
    // boundary, so the ONLY full-quality recovery is the lossless empty-cache reset. When a
    // DEGRADED conversation keeps a token-trivial prefix anyway (rolling-window agent
    // harnesses rewrite mid-context every turn), trade the prefix for the reset: clear the
    // idle slots (recovery reclaim — deliberately NOT deficit-gated: budget slack is exactly
    // the state that starves the pressure path, and worthless idle caches are the only
    // obstacle to recovery that benefits every stream), and if the pool then holds nothing
    // but this slot's cells, drop n_past to 0 — the full re-prefill re-enters at the entry
    // tier, so turn-N cache quality equals turn-1. Returns the (possibly zeroed) n_past.
    int vbr_reset_on_low_lcp(server_slot & slot, int n_past) {
        if (!server_vbr_dynamic_active(params_base) || params_base.vbr_reset_keep_frac <= 0.0f) {
            return n_past;
        }
        const int n_prompt = slot.task->n_tokens();
        if (n_prompt <= 0 || (float) n_past >= params_base.vbr_reset_keep_frac * (float) n_prompt) {
            return n_past;
        }
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        auto st = llama_memory_vbr_state(mem, slot.id, 0);
        if (st.cursor < 2) {
            return n_past; // pristine or one transient step — a full re-prefill buys ~nothing
        }
        vbr_clear_idle_slots(slot.id, "reset-recovery");
        st = llama_memory_vbr_state(mem, slot.id, 0);
        if (st.used_cells_other > 0) {
            SLT_INF(slot, "vbr reset blocked: %u used cells belong to other sequences (pinned or processing slots)\n",
                    st.used_cells_other);
            return n_past;
        }
        SLT_WRN(slot, "vbr reset: cursor %d and only %d/%d prompt tokens reusable (< %.2f) — dropping the prefix; "
                "the full re-prefill re-enters at the entry tier\n",
                st.cursor, n_past, n_prompt, (double) params_base.vbr_reset_keep_frac);
        slot.prompt.checkpoints.clear(); // all invalidated by the full clear
        return 0;
    }

    void recurrent_shrink_for_prefill(const char * reason) {
        if (!recurrent_expanded || !needs_reeval || n_seq_max_full <= n_parallel_user) {
            return;
        }

        // DFlash: backup cells are ~1.5 MB each, but the shrink/expand cycle costs
        // ~100 ms of buffer realloc + CUDA-graph re-capture at the first draft of EVERY
        // request (measured on a 3090: -np 4 lost ~12% end-to-end on a 101-token reply).
        // The memory savings never outweigh that — keep the backup cells resident.
        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            return;
        }

        for (const server_slot & slot : slots) {
            if (slot.is_processing() || slot.has_draft_backup) {
                SRV_DBG("not shrinking recurrent state for prefill (%s): slot %d processing=%d has_backup=%d\n",
                        reason, slot.id, slot.is_processing(), slot.has_draft_backup);
                return;
            }
        }

        auto * mem = llama_get_memory(ctx_tgt);
        for (const server_slot & slot : slots) {
            const llama_seq_id seq_backup = slot.id + n_parallel_user;
            llama_memory_seq_rm(mem, seq_backup, -1, -1);
        }

        if (llama_memory_recurrent_shrink(mem, n_parallel_user)) {
            recurrent_expanded = false;
            SRV_INF("shrunk recurrent state to %d cells for prefill (%s, removed %d backup cells)\n",
                    n_parallel_user, reason, n_seq_max_full - n_parallel_user);
        } else {
            SRV_ERR("failed to shrink recurrent state to %d cells for prefill (%s)\n",
                    n_parallel_user, reason);
        }
    }

    void handle_sleeping_state(bool new_state) {
        GGML_ASSERT(sleeping != new_state);
        if (new_state) {
            SRV_INF("%s", "server is entering sleeping state\n");
            destroy();
        } else {
            SRV_INF("%s", "server is exiting sleeping state\n");
            if (!load_model(params_base)) {
                GGML_ABORT("failed to reload model after sleeping");
            }
        }
        sleeping = new_state;
    }

    struct load_progress_data {
        server_context_impl * ctx;
        std::string stage;
        std::vector<std::string> stages;
        int64_t t_last_load_progress_ms = 0;
        load_progress_data(server_context_impl * ctx, const std::string & stage) : ctx(ctx), stage(stage) {}
    };
    static bool load_progress_callback(float progress, void * user_data) {
        auto * d = static_cast<load_progress_data *>(user_data);
        GGML_ASSERT(d);
        // always emit the first and final sample; throttle the rest to one per 200ms
        {
            auto & t_last = d->t_last_load_progress_ms;
            const int64_t t_now = ggml_time_ms();
            const bool first = t_last == 0;
            const bool done  = progress >= 1.0f;
            const bool throttled = !first && !done && (t_now - t_last) < 200;
            if (throttled) {
                return true;
            }
            t_last = t_now;
        }
        if (d->ctx->callback_state) {
            d->ctx->callback_state(SERVER_STATE_LOADING, {
                {"stages", d->stages},
                {"current", d->stage},
                {"value", progress},
            });
        }
        return true;
    }

    // load the model and initialize llama_context
    // this may also be called to resume from sleeping state
    bool load_model(common_params & params) {
        load_progress_data load_progress_text  (this, "text_model");
        load_progress_data load_progress_mmproj(this, "mmproj_model");
        load_progress_data load_progress_spec  (this, "spec_model");

        const bool is_resume = sleeping;

        params_base = params;
        params_base.n_outputs_max = server_n_outputs_max(params_base);

        const bool has_mmproj = !params.mmproj.path.empty();
        const bool has_draft = params.speculative.has_dft();
        const bool spec_mtp = std::find(params_base.speculative.types.begin(),
                                        params_base.speculative.types.end(),
                                        COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params_base.speculative.types.end();
        const bool has_spec = has_draft || spec_mtp;

        if (callback_state) {
            std::vector<std::string> stages = {"text_model"};
            if (has_spec) {
                stages.push_back("spec_model");
            }
            if (has_mmproj) {
                stages.push_back("mmproj_model");
            }
            load_progress_text.stages   = stages;
            load_progress_mmproj.stages = stages;
            load_progress_spec.stages   = stages;

            // trigger 0% progress
            load_progress_callback(0.0f, &load_progress_text);
        }


        SRV_INF("loading model '%s'\n", params.model.get_name().c_str());
        SRV_TRC("local path '%s'\n", params.model.path.c_str());

        const std::string & mmproj_path = params_base.mmproj.path;

        // measure mmproj memory for auto-fit (upstream #21489)
        // Only reserve mmproj space when auto-fit is actively selecting context size.
        // When -c is explicit, the fitter doesn't run so there's no need to reserve.
        if (has_mmproj && params_base.fit_params && params_base.n_ctx == 0 && !params_base.mmproj_gpu_swap) {
            auto mparams_measure = make_mmproj_params(params_base.mmproj_use_gpu);
            int64_t t_start = ggml_time_us();
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams_measure);
            int64_t t_elapsed = ggml_time_us() - t_start;
            if (!mmproj_mem.empty()) {
                size_t total = 0;
                for (auto & [dev, size] : mmproj_mem) {
                    total += size;
                }
                SRV_TRC("[mtmd] estimated worst-case memory usage of mmproj is %.2f MiB (took %.2f ms)\n", total / (1024.0 * 1024.0), t_elapsed / 1000.0);
                GGML_ASSERT(!params_base.fit_params_target.empty());
                for (auto & [dev, size] : mmproj_mem) {
                    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                        if (ggml_backend_dev_get(i) == dev) {
                            if (i < params_base.fit_params_target.size()) {
                                SRV_DBG("[mtmd] adding %.2f MiB to fit_params_target for device %s\n", size / (1024.0 * 1024.0), ggml_backend_dev_name(dev));
                                params_base.fit_params_target[i] += size;
                            }
                            break;
                        }
                    }
                }
            } else {
                SRV_ERR("%s", "[mtmd] failed to get memory usage of mmproj\n");
            }
        }

        n_parallel_user = params_base.n_parallel;
        recurrent_expanded = true;
        // optionally reserve VRAM for the draft / MTP context before fitting the target model
        if (params_base.fit_params) {
            if (has_spec) {
                common_params params_dft = params_base;
                bool measure_model_bytes = true;

                if (has_draft) {
                    const auto & params_spec = params_base.speculative.draft;
                    params_dft.devices               = params_spec.devices;
                    params_dft.model                 = params_spec.mparams;
                    params_dft.n_gpu_layers          = params_spec.n_gpu_layers;
                    params_dft.cache_type_k          = params_spec.cache_type_k;
                    params_dft.cache_type_v          = params_spec.cache_type_v;
                    params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;
                } else {
                    // MTP draft context lives on the target model, only context+compute are new
                    measure_model_bytes = false;
                }

                params_dft.n_outputs_max = params_base.n_parallel;

                auto mparams_dft = common_model_params_to_llama(params_dft);
                auto cparams_dft = common_context_params_to_llama(params_dft);
                if (spec_mtp) {
                    cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;

                }
                cparams_dft.n_rs_seq = 0;

                std::vector<ggml_backend_dev_t> devs;
                uint32_t hp_ngl = 0;
                uint32_t hp_nct = 0;
                uint32_t hp_nex = 0;
                try {
                    auto dmd = common_get_device_memory_data(
                        params_dft.model.path.c_str(), &mparams_dft, &cparams_dft,
                        devs, hp_ngl, hp_nct, hp_nex, GGML_LOG_LEVEL_ERROR);

                    GGML_ASSERT(!params_base.fit_params_target.empty());
                    size_t total = 0;

                    std::vector<ggml_backend_dev_t> tgt_devices = params.devices;

                    if (tgt_devices.empty()) {
                        for(size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                           tgt_devices.push_back(ggml_backend_dev_get(i));
                        }
                    }

                    for (size_t j = 0; j < devs.size(); ++j) {
                        const size_t bytes = (measure_model_bytes ? dmd[j].model : 0) + dmd[j].context + dmd[j].compute;
                        total += bytes;
                        for (size_t i = 0; i < tgt_devices.size(); i++) {
                            if (tgt_devices[i] == devs[j]) {
                                if (bytes > params_base.fit_params_target[i]) {
                                    SRV_DBG("[spec] raising fit_params_target to %.2f MiB for device %s\n",
                                            bytes / (1024.0 * 1024.0), ggml_backend_dev_name(devs[j]));
                                    params_base.fit_params_target[i] = bytes;
                                }
                                break;
                            }
                        }
                    }
                    SRV_TRC("[spec] estimated memory usage of %s is %.2f MiB\n",
                            has_draft ? "draft model" : "MTP context",
                            total / (1024.0 * 1024.0));
                } catch (const std::exception & e) {
                    SRV_WRN("[spec] failed to measure %s memory: %s\n",
                            has_draft ? "draft model" : "MTP context", e.what());
                }
            }
        }

        // When mmproj GPU swap is active, run fitter before n_parallel doubling.
        // The doubled n_parallel inflates recurrent state estimates, causing the
        // fitter to think even minimum context doesn't fit.
        const bool has_mtp = params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        if (params_base.mmproj_gpu_swap && has_mtp && has_mmproj
                && params_base.fit_params && params_base.n_ctx == 0) {
            auto mparams_gpu = make_mmproj_params(true);
            auto mmproj_mem = mtmd_get_memory_usage(mmproj_path.c_str(), mparams_gpu);
            size_t mmproj_gpu_estimate = 0;
            for (auto & [dev, size] : mmproj_mem) {
                mmproj_gpu_estimate += size;
            }
            if (mmproj_gpu_estimate == 0) {
                std::error_code ec;
                const auto fsize = std::filesystem::file_size(mmproj_path, ec);
                if (!ec && fsize > 0) {
                    mmproj_gpu_estimate = fsize + 256ull * 1024 * 1024;
                }
            }
            for (auto & margin : params_base.fit_params_target) {
                if (margin < mmproj_gpu_estimate) {
                    margin = mmproj_gpu_estimate;
                }
            }

            auto mparams_fit = common_model_params_to_llama(params_base);
            auto cparams_fit = common_context_params_to_llama(params_base);

            SRV_INF("mmproj GPU swap: running auto-fit with n_parallel=%d, margin=%zu MiB\n",
                    params_base.n_parallel, params_base.fit_params_target[0] / (1024 * 1024));

            common_fit_params(params_base.model.path.c_str(), &mparams_fit, &cparams_fit,
                params_base.tensor_split,
                params_base.tensor_buft_overrides.data(),
                params_base.fit_params_target.data(),
                params_base.fit_params_min_ctx,
                params_base.verbosity >= 4 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR);

            if (cparams_fit.n_ctx > 0) {
                params_base.n_ctx = cparams_fit.n_ctx;
                SRV_INF("mmproj GPU swap: auto-fit chose n_ctx = %u\n", cparams_fit.n_ctx);
            }
            params_base.fit_params = false;
        }

        // Always enable kv_unified for single-slot servers — simplifies CUDA graph topology,
        // giving ~28% faster prompt eval even without speculative decoding.
        if (n_parallel_user == 1 && !params_base.kv_unified) {
            params_base.kv_unified = true;
            SRV_INF("%s", "auto-enabled kv-unified: single-slot server doesn't need separate KV stream\n");
        }

        // Double n_parallel only when actual speculative decoding is active
        // (draft model, MTP, or model-free self-speculation), not for phantom
        // --spec-type draft without -md. Model-free types verify their drafts through
        // the TARGET context too, so hybrid/recurrent targets need the same backup
        // sequence for partial-accept rollback: without the doubling n_seq_max stays at
        // n_parallel_user and llama_memory_recurrent::seq_cp silently no-ops on the
        // out-of-range backup seq — rollback then WIPES the recurrent state instead of
        // restoring it (#74: output stays plausible but wrong, degrading over time).
        if (params_base.speculative.has_dft() ||
            params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP) ||
            params_base.speculative.has_model_free_type()) {
            params_base.n_parallel = n_parallel_user * 2;
            n_seq_max_full = params_base.n_parallel;
            recurrent_expanded = false;

            // The backup sequences double n_seq_max. Without unified KV each sequence gets its
            // own n_ctx / n_seq_max stream, so the doubling silently HALVES every slot's usable
            // context on top of the user's n_parallel division (-np 2 + MTP -> n_ctx/4 per
            // slot). Unified KV shares one full-length cell pool across sequences instead —
            // same auto-enable precedent as the single-slot case above.
            if (!params_base.kv_unified) {
                params_base.kv_unified = true;
                SRV_INF("%s", "auto-enabled kv-unified: speculative decoding doubles n_seq_max, "
                        "which would halve the per-slot context with per-sequence KV streams\n");
            }

            // n_outputs_max was computed above (server_n_outputs_max) with the pre-doubling
            // n_parallel. The target context is created just below with the DOUBLED n_parallel,
            // so its n_seq_max grows accordingly and output_reserve(n_seq_max) needs the cap to
            // cover it. For spec types whose n_outputs_per_seq is 1 (notably the fork DFlash
            // type, which returns n_max==0), the original cap stayed at n_parallel_user and
            // tripped GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max) in output_reserve.
            // Recompute with the doubled n_parallel so the cap tracks the real n_seq_max.
            params_base.n_outputs_max = server_n_outputs_max(params_base);

            // The fork DFlash drafter verifies a full block (up to block_size tokens, typ. 16,
            // plus ddtree branches) against the TARGET per step, so the target's output_reserve
            // peaks well above n_seq_max. But common_speculative_n_max is 0 for the fork DFlash
            // type and block_size isn't known until the draft model loads (which happens after
            // this target context is created), so server_n_outputs_max undercounts here.
            // Reserve a generous per-sequence output budget as a floor. The cap only bounds
            // output_reserve's lazily-grown buffers and is clamped to n_batch — the same
            // ceiling a non-speculative context gets by default — so it costs nothing until
            // a batch actually requests that many outputs.
            constexpr int32_t DFLASH_VERIFY_OUTPUTS_PER_SEQ = 32; // ~2x block_size headroom
            const int32_t dflash_verify_floor = std::min<int32_t>(
                (int32_t) params_base.n_batch, (int32_t) n_seq_max_full * DFLASH_VERIFY_OUTPUTS_PER_SEQ);
            params_base.n_outputs_max = std::max<int32_t>(params_base.n_outputs_max, dflash_verify_floor);
        }

        // attach a progress callback
        {
            params_base.load_progress_callback = load_progress_callback;
            params_base.load_progress_callback_user_data = &load_progress_text;
        }

        llama_init = common_init_from_params(params_base);
        params_base.n_parallel = n_parallel_user;

        model_tgt = llama_init->model();
        ctx_tgt   = llama_init->context();

        if (model_tgt == nullptr) {
            SRV_ERR("failed to load model, '%s'\n", params_base.model.path.c_str());
            return false;
        }

        vocab = llama_model_get_vocab(model_tgt);

        needs_reeval = llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt);
        is_diffusion = llama_model_is_diffusion(model_tgt);

        dflash_tape_ok = llama_dflash_tape_replay_available(ctx_tgt);
        if (needs_reeval && !dflash_tape_ok && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            SRV_INF("%s", "DFlash rollback: GPU tape replay unavailable (recurrent states on "
                    "host or multiple devices) — partial accepts re-decode the accepted tokens\n");
        }

        if (is_diffusion) {
            SRV_INF("%s", "diffusion model detected — enabling self-speculation\n");
        }

        n_ctx = llama_n_ctx(ctx_tgt);

        add_bos_token = llama_vocab_get_add_bos(vocab);

        if (has_draft) {
            // TODO speculative: move to common/speculative.cpp?
            const auto & params_spec = params_base.speculative.draft;

            SRV_TRC("loading draft model '%s'\n", params_spec.mparams.path.c_str());

            auto params_dft = params_base;

            params_dft.n_parallel   = 1;
            params_dft.n_ctx        = params_spec.n_ctx == 0 ? llama_n_ctx_seq(ctx_tgt) : params_spec.n_ctx;
            params_dft.n_batch      = params_dft.n_ctx;
            params_dft.devices      = params_spec.devices;

            // SPLIT_MODE_TENSOR is target-only: drafter archs have no meta split rules, and the
            // hidden-state feed runs through host buffers anyway — run the drafter as a normal
            // (layer-split or pinned) model; pin with --spec-draft-device to keep it on one GPU
            if (params_dft.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
                params_dft.split_mode = LLAMA_SPLIT_MODE_LAYER;
                SRV_INF("%s", "tensor-split target: draft model falls back to layer split "
                        "(use --spec-draft-device to pin it to one GPU)\n");
            }
            params_dft.model        = params_spec.mparams;
            params_dft.n_gpu_layers = params_spec.n_gpu_layers;
            params_dft.cache_type_k = params_spec.cache_type_k;
            params_dft.cache_type_v = params_spec.cache_type_v;

            if (params_spec.cpuparams.n_threads > 0) {
                params_dft.cpuparams.n_threads       = params_spec.cpuparams.n_threads;
                params_dft.cpuparams_batch.n_threads = params_spec.cpuparams_batch.n_threads;
            }

            params_dft.tensor_buft_overrides = params_spec.tensor_buft_overrides;

            auto mparams_dft = common_model_params_to_llama(params_dft);

            // progress callback
            mparams_dft.progress_callback           = load_progress_callback;
            mparams_dft.progress_callback_user_data = &load_progress_spec;

            model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
            if (model_dft == nullptr) {
                SRV_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
                return false;
            }

            // Auto-detect DFlash from drafter model architecture
            if (llama_model_dflash_block_size(model_dft.get()) > 0 &&
                params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
                params_base.speculative.set_type(COMMON_SPECULATIVE_TYPE_DFLASH);
                SRV_INF("auto-detected DFlash drafter (block_size=%d)\n",
                        llama_model_dflash_block_size(model_dft.get()));
            }

            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                const int block_size = llama_model_dflash_block_size(model_dft.get());
                params_dft.n_ubatch = LLAMA_DFLASH_MAX_SLOTS * block_size;
                params_dft.n_parallel = std::max(1,
                    std::min(params_base.speculative.dflash_max_slots, params_base.n_parallel));

                // --spec-dflash-default leaves draft-max at -1 = auto: the drafter emits
                // at most block_size - 1 tokens per step and the full depth strictly wins
                // (EXP-37i depth sweep) — resolve it here so slot task defaults see it
                if (params_base.speculative.n_max < 0) {
                    params_base.speculative.n_max = block_size > 1 ? block_size - 1 : 12;
                    SRV_INF("draft-max auto (DFlash): %d (drafter block_size %d)\n",
                            params_base.speculative.n_max, block_size);
                }
            }

            params_base.speculative.model_dft = model_dft.get();
            params_base.speculative.cparams_dft = common_context_params_to_llama(params_dft);
            // share buffers with the target context (upstream #24922 family)
            params_base.speculative.cparams_dft.ctx_other = ctx_tgt;

            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                llama_model_share_tensors(model_dft.get(), llama_get_model(ctx_tgt));
            }

            // Upstream MTP: create draft context from target model's MTP heads
            if (spec_mtp && params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH) {
                auto cparams = common_context_params_to_llama(params_dft);
                cparams.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
                cparams.n_rs_seq  = 0;
                cparams.ctx_other = ctx_tgt;
                ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));
                if (ctx_dft == nullptr) {
                    SRV_ERR("%s", "failed to create draft context\n");
                    return false;
                }
                ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
                params_base.speculative.draft.ctx_tgt = ctx_tgt;
                params_base.speculative.draft.ctx_dft = ctx_dft.get();
            }
        } else if (params_base.speculative.has_type(COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) {
            // no new model load, so we simply report 0.0 and 1.0 progress
            load_progress_callback(0.0f, &load_progress_spec);

            SRV_INF("creating MTP draft context against the target model '%s'\n",
                    params_base.model.path.c_str());

            ctx_dft.reset(create_mtp_context());
            if (ctx_dft == nullptr) {
                SRV_ERR("%s", "failed to create MTP context\n");
                return false;
            }

            params_base.speculative.draft.ctx_tgt = ctx_tgt;
            params_base.speculative.draft.ctx_dft = ctx_dft.get();

            load_progress_callback(1.0f, &load_progress_spec);
        }

        if (has_mmproj) {
            if (callback_state) {
                callback_state(SERVER_STATE_LOADING, {{"stage", "mmproj_model"}});
            }

            if (!is_resume) {
                mtmd_helper_log_set(common_log_default_callback, nullptr);
            }

            mmproj_gpu_swap = params_base.mmproj_gpu_swap && !model_dft;
            // Note: ctx_dft is NOT required here. Without MTP, the swap simply
            // keeps mmproj on CPU until an image arrives, loads it to GPU for
            // encoding, then moves it back. No MTP state to swap out.

            const bool use_gpu = mmproj_gpu_swap ? false : params_base.mmproj_use_gpu;
            auto mparams = make_mmproj_params(use_gpu, load_progress_callback, &load_progress_mmproj);
            if (!mmproj_gpu_swap) {
                mparams.warmup = params_base.warmup;
            }

            mctx = mtmd_init_from_file(mmproj_path.c_str(), model_tgt, mparams);
            if (mctx == nullptr) {
                SRV_ERR("failed to load multimodal model, '%s'\n", mmproj_path.c_str());
                return false;
            }

            if (mmproj_gpu_swap) {
                SRV_INF("loaded multimodal model on CPU (GPU swap enabled), '%s'\n", mmproj_path.c_str());
            } else {
                SRV_INF("loaded multimodal model, '%s'\n", mmproj_path.c_str());
            }

            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by multimodal, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by multimodal, it will be disabled");
            }

        }

        if (!llama_memory_can_shift(llama_get_memory(ctx_tgt))) {
            if (params_base.ctx_shift) {
                params_base.ctx_shift = false;
                SRV_WRN("%s\n", "ctx_shift is not supported by this context, it will be disabled");
            }

            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by this context, it will be disabled");
            }
        }

        if (llama_model_n_swa(model_tgt) == 0) {
            if (params_base.swa_full) {
                params_base.swa_full = false;
                SRV_WRN("%s\n", "swa_full is not supported by this model, it will be disabled");
            }
        }

        n_swa = params_base.swa_full ? 0 : llama_model_n_swa(model_tgt);

        // Necessary similarity of prompt for slot selection
        slot_prompt_similarity = params_base.slot_prompt_similarity;

        const int n_ctx_train = llama_model_n_ctx_train(model_tgt);

        int n_ctx_slot = llama_n_ctx_seq(ctx_tgt);
        if (n_ctx_slot > n_ctx_train) {
            SRV_WRN("the slot context (%d) exceeds the training context of the model (%d) - capping\n", n_ctx_slot, n_ctx_train);
            n_ctx_slot = n_ctx_train;
        }

        slots.clear();

        ctx_tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            SRV_WRN("%s", "speculative decoding not supported by this context\n");
        }

        if (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL) {
            SRV_TRC("%s", "speculative decoding will use checkpoints\n");
        }

        const bool can_spec = (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO);

        // DFlash multi-slot: --dflash-max-slots caps how many server slots keep DFlash state;
        // slots above the cap fall back to non-speculative decode (slot.spec stays null). The
        // matching tape/hidden buffers are allocated after the per-slot init loop (set_dflash_capture
        // runs inside common_speculative_init for slot 0, so dflash_capture must exist first).
        int dflash_slots_cap = 0;
        if (can_spec && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
            dflash_slots_cap = std::max(1, std::min(params_base.speculative.dflash_max_slots, params_base.n_parallel));
            if (dflash_slots_cap < params_base.n_parallel) {
                SRV_INF("DFlash enabled for slots 0..%d; slots %d+ will use non-speculative decode\n",
                        dflash_slots_cap - 1, dflash_slots_cap);
            } else {
                SRV_INF("DFlash enabled for all %d slots\n", dflash_slots_cap);
            }

            // Create the shared DFlash drafter context once;
            // every slot's common_speculative gets a non-owning reference to it.
            // dflash_slots_cap is passed at init so the initial graph reserve sizes
            // the compute buffer for the requested width — single-slot servers stay
            // narrow (cheap), multi-slot servers get a compute buffer big enough for
            // the batched cross-attention. Runtime widening past this cap requires a
            // larger compute buffer than is available.
            ctx_dft_shared.reset(common_speculative_create_ctx_dft(params_base.speculative, dflash_slots_cap));
            if (!ctx_dft_shared) {
                SRV_ERR("%s", "failed to create shared DFlash drafter context\n");
                return false;
            }
        }

        // setup slots
        SRV_INF("initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'\n",
                params_base.n_parallel, n_ctx_slot, params_base.kv_unified ? "true" : "false");

        // initialize slots
        for (int i = 0; i < params_base.n_parallel; i++) {
            slots.emplace_back();
        }

        // try speculative decoding (upstream shared spec — not used by fork types which init per-slot)
        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO && !dflash_slots_cap) {
            try {
                spec.reset(common_speculative_init(params_base.speculative, params_base.n_parallel));
            } catch (const std::exception & e) {
                SRV_ERR("failed to initialize speculative decoding context: %s\n", e.what());
            }
        }

        if (ctx_dft) {
            ctx_dft_seq_rm_type = common_context_can_seq_rm(ctx_dft.get());
        }

        if (spec) {
            SRV_TRC("%s", "speculative decoding context initialized\n");
        } else {
            ctx_dft.reset();
        }

        for (int i = 0; i < params_base.n_parallel; i++) {
            server_slot & slot = slots[i];

            slot.id      = i;
            slot.ctx_tgt = ctx_tgt;
            slot.ctx_dft = ctx_dft.get();
            slot.n_ctx   = n_ctx_slot;

            slot.mctx                   = mctx;
            slot.prompt.tokens.has_mtmd = mctx != nullptr;

            const bool slot_can_spec = can_spec &&
                (params_base.speculative.type() != COMMON_SPECULATIVE_TYPE_DFLASH || i < dflash_slots_cap);

            if (is_diffusion) {
                slot.diff_self_spec = true;
                slot.diff_mask_token_id = llama_vocab_mask(vocab);
                slot.diff_draft_length = 4;
                auto think_open  = common_tokenize(vocab, "<think>",  false, true);
                auto think_close = common_tokenize(vocab, "</think>", false, true);
                if (think_open.size()  == 1) slot.diff_think_open_id  = think_open[0];
                if (think_close.size() == 1) slot.diff_think_close_id = think_close[0];
                SLT_INF(slot, "diffusion self-spec enabled, mask_id=%d, draft_length=%d, think_suppress=%d/%d\n",
                        slot.diff_mask_token_id, slot.diff_draft_length, slot.diff_think_open_id, slot.diff_think_close_id);
            } else if (slot_can_spec) {
                slot.spec.reset(common_speculative_init(params_base.speculative, slot.ctx_tgt, ctx_dft_shared.get()));
                if (!slot.spec && spec) {
                    slot.spec_shared = spec.get();
                }
                if (slot.can_speculate()) {
                    common_speculative_set_seq_id(slot.get_spec(), slot.id);
                    SLT_INF(slot, "%s", "speculative decoding context initialized\n");
                }
            }

            SLT_TRC(slot, "new slot, n_ctx = %d\n", slot.n_ctx);

            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };

            slot.reset();
        }

        // safety net (#74): hybrid/recurrent partial-accept rollback requires one backup
        // sequence per user slot. If a future speculative type slips past the n_parallel
        // doubling above, refuse to speculate instead of silently corrupting the
        // recurrent state on the first partially-accepted draft.
        if (needs_reeval && ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
            llama_n_seq_max(ctx_tgt) < (uint32_t) (2 * n_parallel_user)) {
            bool any_spec = spec != nullptr;
            for (auto & slot : slots) {
                any_spec = any_spec || slot.can_speculate();
                slot.spec.reset();
                slot.spec_shared = nullptr;
            }
            spec.reset();
            if (any_spec) {
                SRV_ERR("speculative decoding disabled: hybrid/recurrent rollback needs n_seq_max >= %d, context has %u\n",
                        2 * n_parallel_user, llama_n_seq_max(ctx_tgt));
            }
        }

        {
            const char * LLAMA_TRACE = getenv("LLAMA_TRACE");
            trace = LLAMA_TRACE ? atoi(LLAMA_TRACE) : 0;

            if (trace) {
                SRV_WRN("LLAMA_TRACE = %d\n", trace);
            }
        }

        // Allocate DFlash per-slot tape + hidden buffers now that common_speculative_init
        // (run for slot 0 above) has created dflash_capture on the target context.
        // Runs even when !dflash_tape_ok: the call also sizes the per-slot hidden-capture
        // buffers, which every DFlash target needs (the unused tape itself is small).
        if (dflash_slots_cap > 0) {
            llama_dflash_allocate_slots(ctx_tgt, dflash_slots_cap);
        }

        // Under --split-mode tensor the load-time capability probe was predictive; now
        // that capture setup exists the tape allocation (with its shard-consistency
        // check) has run, so re-probe for the authoritative answer.
        if (dflash_tape_ok && !llama_dflash_tape_replay_available(ctx_tgt)) {
            dflash_tape_ok = false;
            SRV_INF("%s", "DFlash rollback: GPU tape rejected after allocation (shard "
                    "mismatch) — partial accepts re-decode the accepted tokens\n");
        }

        // DFlash + hybrid target: expand the recurrent state to its full backup-cell count
        // NOW instead of at the first draft — the expand costs ~100 ms of buffer realloc +
        // CUDA-graph re-capture, which otherwise lands on the first request's latency.
        // (recurrent_shrink_for_prefill never shrinks DFlash, so this is a one-time cost.)
        if (needs_reeval && can_spec &&
            params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH &&
            ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS &&
            n_seq_max_full > n_parallel_user && !recurrent_expanded) {
            auto * mem = llama_get_memory(ctx_tgt);
            if (llama_memory_recurrent_expand(mem, n_seq_max_full)) {
                recurrent_expanded = true;
                SRV_INF("pre-expanded recurrent state to %d cells for speculative backup\n", n_seq_max_full);
            }
        }

        {
            const char * LLAMA_SERVER_SLOTS_DEBUG = getenv("LLAMA_SERVER_SLOTS_DEBUG");
            slots_debug = LLAMA_SERVER_SLOTS_DEBUG ? atoi(LLAMA_SERVER_SLOTS_DEBUG) : 0;

            if (slots_debug) {
                SRV_WRN("LLAMA_SERVER_SLOTS_DEBUG = %d\n", slots_debug);
            }
        }

        // the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens
        // note that n_batch can be > n_ctx (e.g. for non-causal attention models such as BERT where the KV cache is not used)
        {
            const int32_t n_batch = llama_n_batch(ctx_tgt);
            batch.init(std::max(n_batch, params_base.n_parallel));
        }

        if (server_vbr_dynamic_active(params_base)) {
            // Disabled: these mechanisms serialize/shift the ATTENTION KV, whose tensor tiers flip
            // in place at runtime under the dynamic VBR controller (a FLAGS_NONE state restore
            // would land bytes under the wrong tier or past the VMM watermark; cache_reuse needs
            // shifts, and get_can_shift() is false under VMM anyway).
            if (params_base.cache_ram_mib != 0) {
                params_base.cache_ram_mib = 0;
                SRV_WRN("%s\n", "prompt cache state storage is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
            if (params_base.n_cache_reuse) {
                params_base.n_cache_reuse = 0;
                SRV_WRN("%s\n", "cache_reuse is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
            if (!params_base.slot_save_path.empty()) {
                // llama_state_seq_save_file carries full tier-typed attention KV: a save taken
                // after any degrade is refused at the lib level, and even entry-tier saves stop
                // restoring once the target cache has degraded — predictably off beats flaky
                params_base.slot_save_path.clear();
                SRV_WRN("%s\n", "slot save/restore (--slot-save-path) is not supported by dynamic VBR (KV tiers change at runtime), it will be disabled");
            }
            // Context checkpoints (PARTIAL_ONLY):
            //   hybrid models  — routed to the recurrent state only (attention KV skipped, see
            //     llama_memory_hybrid::state_write): tier-agnostic and load-bearing for prompt
            //     rewind, so they stay ENABLED;
            //   iSWA models    — llama_kv_cache_iswa::state_write DOES serialize the SWA
            //     attention KV under PARTIAL_ONLY; once a tier flips every restore would fail
            //     (clean fallback, but the checkpoints are dead weight) — disable them.
            if (params_base.n_ctx_checkpoints > 0 &&
                llama_model_n_swa(model_tgt) > 0 && !llama_model_is_hybrid(model_tgt)) {
                params_base.n_ctx_checkpoints = 0;
                SRV_WRN("%s\n", "context checkpoints are not supported by dynamic VBR on SWA models (the SWA attention KV is part of the checkpoint), they will be disabled");
            }
        }

        if (params_base.cache_ram_mib != 0) {
            if (params_base.cache_ram_mib < 0) {
                SRV_TRC("prompt cache is enabled, size limit: %s\n", "no limit");
            } else {
                SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", params_base.cache_ram_mib);
            }
            SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

            prompt_cache = std::make_unique<server_prompt_cache>(params_base.cache_ram_mib, n_ctx);
        } else {
            SRV_TRC("%s", "prompt cache is disabled - use `--cache-ram N` to enable it\n");
        }
        SRV_TRC("%s", "for more info see https://github.com/ggml-org/llama.cpp/pull/16391\n");

        if (params_base.n_ctx_checkpoints > 0) {
            SRV_TRC("context checkpoints enabled, max = %d, min spacing = %d\n",
                    params_base.n_ctx_checkpoints, params_base.checkpoint_min_step);
        } else {
            SRV_TRC("%s", "context checkpoints disabled\n");
        }

        if (!params_base.model_alias.empty()) {
            // backward compat: use first alias as model name
            model_name = *params_base.model_alias.begin();
        } else if (!params_base.model.get_name().empty()) {
            model_name = params_base.model.get_name();
        } else {
            // fallback: derive model name from file name
            auto model_path = std::filesystem::path(params_base.model.path);
            model_name = model_path.filename().string();
        }

        model_aliases = params_base.model_alias;
        model_tags    = params_base.model_tags;

        // propagate new defaults back to caller
        params = params_base;

        if (!is_resume) {
            return init();
        }

        if (callback_state) {
            callback_state(SERVER_STATE_READY, {});
        }

        return true;
    }

    // unlike load_model(), this is only called once during initialization
    bool init() {
        GGML_ASSERT(ctx_tgt   != nullptr);
        GGML_ASSERT(model_tgt != nullptr);

        GGML_ASSERT(!sleeping);

        // wiring up server queues
        queue_tasks.on_new_task([this](server_task && task) {
            process_single_task(std::move(task));
        });
        queue_tasks.on_update_slots([this]() {
            update_slots();
        });
        queue_tasks.on_sleeping_state([this](bool sleeping) {
            handle_sleeping_state(sleeping);
        });

        metrics.init();

        if (params_base.cache_idle_slots) {
            if (params_base.cache_ram_mib == 0) {
                SRV_WRN("%s", "--cache-idle-slots requires --cache-ram, disabling\n");
                params_base.cache_idle_slots = false;
            } else {
                if (params_base.kv_unified) {
                    SRV_TRC("%s", "idle slots will be saved to prompt cache and cleared upon starting a new task\n");
                } else {
                    // without a unified KV cache, clearing a slot frees no reusable room, so we only
                    // publish a RAM-cache copy of idle slots (their KV stays in VRAM) [TAG_IDLE_SLOT_CLEAR]
                    SRV_TRC("%s", "idle slots will be saved to prompt cache upon starting a new task\n");
                }
                SRV_DBG("%s", "__TEST_TAG_CACHE_IDLE_SLOTS_ENABLED__\n");
            }
        }

        {
            const std::string & cfg = params_base.ui_config_json;
            if (!cfg.empty()) {
                try {
                    json json_settings = json::parse(cfg);
                    json_ui_settings = json_settings;
                } catch (const std::exception & e) {
                    SRV_ERR("%s: failed to parse UI config: %s\n", __func__, e.what());
                    return false;
                }
            }
        }

        // populate chat template params
        {
            common_chat_templates_ptr chat_templates;

            try {
                chat_templates = common_chat_templates_init(model_tgt, params_base.chat_template);

                SRV_TRC("%s: chat template, example_format: '%s'\n", __func__,
                    common_chat_format_example(chat_templates.get(), params_base.use_jinja, params_base.default_template_kwargs).c_str());

            } catch (const std::exception & e) {
                SRV_ERR("%s: chat template parsing error: %s\n", __func__, e.what());
                SRV_ERR("%s: please consider disabling jinja via --no-jinja, or use a custom chat template via --chat-template\n", __func__);
                SRV_ERR("%s: for example: --no-jinja --chat-template chatml\n", __func__);
                return false;
            }

            // thinking is enabled if:
            // 1. It's not explicitly disabled via --reasoning off
            // 2. The chat template supports it
            const bool template_supports_thinking = params_base.use_jinja && common_chat_templates_support_enable_thinking(chat_templates.get());
            const bool enable_thinking = params_base.enable_reasoning != 0 && template_supports_thinking;
            SRV_TRC("%s: chat template, thinking = %d\n", __func__, enable_thinking);

            // IMPORTANT: chat_params is reused across sleeping / resuming states,
            //            never store llama_context/llama_model pointers in chat_params,
            //            as they may be invalidated after sleeping
            chat_params = {
                /* use_jinja             */ params_base.use_jinja,
                /* prefill_assistant     */ params_base.prefill_assistant,
                /* reasoning_format      */ params_base.reasoning_format,
                /* chat_template_kwargs  */ params_base.default_template_kwargs,
                /* tmpls                 */ std::move(chat_templates),
                /* allow_image           */ mctx ? mtmd_support_vision(mctx) : false,
                /* allow_audio           */ mctx ? mtmd_support_audio (mctx) : false,
                /* allow_video           */ mctx ? mtmd_helper_support_video(mctx) : false,
                /* enable_thinking       */ enable_thinking,
                /* reasoning_budget      */ params_base.sampling.reasoning_budget_tokens,
                /* reasoning_budget_msg  */ params_base.sampling.reasoning_budget_message,
                /* media_path            */ params_base.media_path,
                /* force_pure_content    */ params_base.force_pure_content_parser
            };

            {
                auto caps = common_chat_templates_get_caps(chat_params.tmpls.get());
                auto it = params_base.default_template_kwargs.find("preserve_reasoning");
                bool supported = caps.at("supports_preserve_reasoning");
                bool enabled = it != params_base.default_template_kwargs.end();
                if (supported && !enabled) {
                    SRV_INF("%s", "chat template supports preserving reasoning, consider enabling it via --reasoning-preserve\n");
                }
                if (!supported && enabled) {
                    SRV_WRN("%s", "chat template does NOT support preserving reasoning, --reasoning-preserve has no effect\n");
                }
            }
        }

        // Shrink recurrent state to free backup cells during prefill.
        // Must happen after init-time decodes (common_speculative_is_compat, warmup)
        // so the scheduler's CUDA graph state isn't stale.
        if (!recurrent_expanded && needs_reeval) {
            if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH && n_parallel_user <= 2) {
                recurrent_expanded = true;
            } else {
                auto * mem = llama_get_memory(ctx_tgt);
                if (llama_memory_recurrent_shrink(mem, n_parallel_user)) {
                    SRV_INF("shrunk recurrent state to %d cells for prefill (deferred %d backup cells)\n",
                            n_parallel_user, n_seq_max_full - n_parallel_user);
                }
            }
        }

        return true;
    }

    server_slot * get_slot_by_id(int id_slot) {
        // note: allow id_slot to be out of bounds (wrap around)
        id_slot = id_slot % slots.size();

        for (server_slot & slot : slots) {
            if (slot.id == id_slot) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_slot_by_cmpl_id(const std::string & cmpl_id) {
        if (cmpl_id.empty()) {
            return nullptr;
        }

        for (server_slot & slot : slots) {
            if (slot.is_processing() && slot.task && slot.task->params.oaicompat_cmpl_id == cmpl_id) {
                return &slot;
            }
        }

        return nullptr;
    }

    server_slot * get_available_slot(const server_task & task) {
        server_slot * ret = nullptr;

        bool update_cache = false;

        // best similarity seen even BELOW the threshold — feeds the vbr route-home tier and the
        // LRU log line (a hopping conversation was undiagnosable: "selected by LRU" never said
        // how close the rejected candidates came)
        float sim_best_any = 0;

        // if a specific slot is requested, use it (still goes through cache update logic below)
        if (task.id_slot != -1) {
            ret = get_slot_by_id(task.id_slot);
            if (ret) {
                SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
            }
        }

        // find the slot that has at least n% prompt similarity
        if (slot_prompt_similarity != 0.0f) {
            float sim_best = 0;

            for (server_slot & slot : slots) {
                if (task.id_slot != -1 && slot.id != task.id_slot) {
                    continue;
                }

                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                // skip the slot if it does not contains cached tokens
                if (tokens.empty()) {
                    continue;
                }

                // fraction of the Longest Common Prefix length with respect to the input prompt length
                const float sim_cur = float(tokens.get_common_prefix(task.tokens)) / task.tokens.size();

                sim_best_any = std::max(sim_best_any, sim_cur);

                // select the current slot if the criteria match
                if (sim_cur > sim_best && sim_cur > slot_prompt_similarity) {
                    sim_best = sim_cur;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                const float f_keep = (sim_best*task.tokens.size()) / ret->prompt.tokens.size();

                if (task.id_slot == -1) {
                    SLT_INF(*ret, "selected slot by LCP similarity, sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                            sim_best, slot_prompt_similarity, f_keep);
                }

                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // dynamic VBR: route a returning conversation HOME before the LRU tier colonizes a fresh
        // slot. In the unified pool an idle slot's cells are live cost for everyone (mapped pages,
        // attention span, the one shared quality budget), and the LRU tier below ranks never-used
        // slots FIRST — a rewritten-context prompt that fails the similarity threshold would stack
        // slot after slot while its own history rots elsewhere. Any nonzero LCP (a shared system
        // prompt suffices) identifies the home slot; genuinely-new streams (zero LCP everywhere)
        // still take the LRU spread below.
        if (ret == nullptr && server_vbr_dynamic_active(params_base)) {
            size_t lcp_best = 0;

            for (server_slot & slot : slots) {
                if (slot.is_processing()) {
                    continue;
                }

                const auto & tokens = slot.prompt.tokens;

                if (tokens.empty()) {
                    continue;
                }

                const size_t lcp = tokens.get_common_prefix(task.tokens);
                if (lcp > lcp_best) {
                    lcp_best = lcp;

                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by route-home (vbr), lcp = %zu tokens (sim %.3f <= %.3f thold)\n",
                        lcp_best, (double) lcp_best / task.tokens.size(), slot_prompt_similarity);

                update_cache = true;
            }
        }

        // find the slot that has been least recently used
        // prefer spec-capable (DFlash) slots so requests get speculative decoding
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // strongly prefer spec-capable slots: pick a spec slot over a non-spec
                // slot regardless of LRU, then use LRU within the same capability tier
                const bool curr_spec = ret && ret->can_speculate();
                const bool slot_spec = slot.can_speculate();
                if (!ret ||
                    (slot_spec && !curr_spec) ||
                    (slot_spec == curr_spec && slot.t_last_used < t_last)) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }

            if (ret != nullptr) {
                SLT_INF(*ret, "selected slot by LRU, t_last = %" PRId64 " (best rejected sim = %.3f)\n",
                        t_last, sim_best_any);

                update_cache = true;
            }
        }

        if (ret) {
            recurrent_shrink_for_prefill("before prompt cache save/load");

            // note: prompt_save() itself is a no-op when the slot's context is empty

            update_cache = update_cache && prompt_cache;

            // cache prompts only for completion tasks
            update_cache = update_cache && task.type == SERVER_TASK_TYPE_COMPLETION;

            if (update_cache) {
                SRV_TRC("%s", "updating prompt cache\n");

                const int64_t t_start = ggml_time_us();

                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {
                    ret->prompt_clear(false);
                }

                prompt_cache->update();

                SRV_TRC("prompt cache update took %.2f ms\n", (ggml_time_us() - t_start) / 1000.0);
            }
        }

        return ret;
    }

    // return true if at least one slot has been cleared
    // TODO: improve logic
    //       - smarter decision which slot to clear (LRU or longest prompt?)
    //       - move slot to level 2 cache instead of removing?
    //       - instead of purging, try to store and resume later?
    bool try_clear_idle_slots() {
        bool res = false;

        if (!params_base.kv_unified) {
            return res;
        }

        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens\n", slot.id, slot.prompt.tokens.size());

                slot.prompt_clear(false);

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
    }

    std::vector<common_adapter_lora_info> construct_lora_list(const std::map<int, float> & config) const {
        std::vector<common_adapter_lora_info> output = params_base.lora_adapters; // copy
        for (size_t i = 0; i < output.size(); ++i) {
            auto it = config.find(i);
            if (it != config.end()) {
                output[i].scale = it->second;
            } else {
                output[i].scale = 0.0f;
            }
        }
        return output;
    }

    bool launch_slot_with_task(server_slot & slot, server_task && task) {
        // process per-request lora adapters
        if (!task.params.lora.empty()) {
            auto task_loras = construct_lora_list(task.params.lora);
            if (!are_lora_equal(task_loras, slot.lora)) {
                // if lora has changed, check to see if the cache should be cleared
                if (lora_should_clear_cache(slot.lora, task_loras)) {
                    SLT_TRC(slot, "clearing cache for lora change. %zu loras -> %zu loras\n", slot.lora.size(), task.params.lora.size());
                    slot.prompt.tokens.clear();
                } else {
                    SLT_TRC(slot, "keeping cache for alora. %zu target loras\n", task_loras.size());
                }
                slot.lora = task_loras;
            }
        } else {
            slot.lora = params_base.lora_adapters;
        }

        // if using alora, make sure it's only a single one requested and active
        size_t alora_invocation_start = task.tokens.size();
        if (lora_all_alora(slot.lora)) {
            const auto & enabled_ids = lora_get_enabled_ids(slot.lora);
            // TODO: This will error out if a user requests two aloras, but only
            // provides the activation string for one. We could, instead search
            // for all requested alora activation strings and then either keep
            // only the last one, or reject if multiple are found.
            if (enabled_ids.size() != 1) {
                send_error(task, "Cannot run multiple aLoRAs in a single request", ERROR_TYPE_INVALID_REQUEST);
                return false;
            }
            const auto & lora = slot.lora[enabled_ids[0]].ptr;

            // get the pointer and count for the invocation tokens
            const uint64_t      n_invocation_tokens = llama_adapter_get_alora_n_invocation_tokens(lora);
            const llama_token * invocation_tokens   = llama_adapter_get_alora_invocation_tokens  (lora);

            // scan backwards through the prompt tokens to find the last
            // occurrence of the invocation sequence
            int match_idx = static_cast<int>(n_invocation_tokens) - 1;
            for (int i = task.tokens.size() - 1; i >= 0; --i) {
                // the token in this position matches the next token to find in
                // the invocation sequence
                if (task.tokens[i] == invocation_tokens[match_idx]) {
                    // if it's a full match, we've found the start
                    if (match_idx == 0) {
                        alora_invocation_start = i;
                        break;
                    }
                    // otherwise, check the next token in the sequence
                    --match_idx;
                } else {
                    // no match in this position, so start looking over again
                    match_idx = static_cast<int>(n_invocation_tokens) - 1;
                }
            }

            // if the activation string is not found, disable the alora
            if (alora_invocation_start == task.tokens.size()) {
                SLT_DBG(slot, "alora %zu requested, but not found. deactivating\n", enabled_ids[0]);
                slot.lora[enabled_ids[0]].scale = 0.0f;
            } else {
                SLT_DBG(slot, "alora %zu activated starting at %zu\n", enabled_ids[0], alora_invocation_start);
                slot.alora_invocation_start = alora_invocation_start;
            }
        }

        if (!task.tokens.validate(ctx_tgt)) {
            send_error(task, "Prompt contains invalid tokens", ERROR_TYPE_INVALID_REQUEST);
            return false;
        }

        SLT_DBG(slot, "launching slot : %s\n", safe_json_to_str(slot.to_json()).c_str());

        // initialize samplers
        if (task.need_sampling()) {
            try {
                slot.smpl.reset(common_sampler_init(model_tgt, task.params.sampling));
            } catch (std::exception & e) {
                std::string err_msg = std::string("Failed to initialize samplers: ") + e.what();
                send_error(task, err_msg, ERROR_TYPE_INVALID_REQUEST);
                return false;
            }

            const bool need_pre_sample_logits = task.params.sampling.n_probs > 0 && !task.params.post_sampling_probs;

            bool backend_sampling = true;

            backend_sampling &= task.params.sampling.backend_sampling;

            // TODO: speculative decoding requires multiple samples per batch - not supported yet
            backend_sampling &= !(slot.can_speculate());

            // diffusion self-spec needs raw logits for acceptance checking
            backend_sampling &= !slot.diff_self_spec;

            // TODO: getting pre sampling logits is not yet supported with backend sampling
            backend_sampling &= !need_pre_sample_logits;

            // TODO: tmp until backend sampling is fully implemented
            if (backend_sampling) {
                llama_set_sampler(ctx_tgt, slot.id, common_sampler_get(slot.smpl.get()));
            } else {
                llama_set_sampler(ctx_tgt, slot.id, nullptr);
            }

            SLT_TRC(slot, "sampler chain: %s\n", common_sampler_print(slot.smpl.get()).c_str());
            SLT_TRC(slot, "sampler params: \n%s\n", task.params.sampling.print().c_str());
        } else {
            slot.smpl.reset();
        }

        if (slot.diff_self_spec && task.tokens.size() > 0) {
            // Work at the token level to avoid detokenize→retokenize roundtrip issues.
            llama_tokens toks(task.tokens.get_tokens());
            const llama_token think_open  = slot.diff_think_open_id;
            const llama_token think_close = slot.diff_think_close_id;
            auto nl_vec = common_tokenize(vocab, "\n", false, false);
            const llama_token nl_tok = nl_vec.empty() ? LLAMA_TOKEN_NULL : nl_vec[0];

            // Strip ALL closed <think>...</think> blocks (compact or newlined).
            // The Jinja template injects them into every assistant message but they
            // cause verbatim repetition in multi-turn. We re-add the correct form at the end.
            if (think_open != LLAMA_TOKEN_NULL && think_close != LLAMA_TOKEN_NULL) {
                llama_tokens fixed;
                fixed.reserve(toks.size());
                for (size_t ti = 0; ti < toks.size(); ti++) {
                    if (toks[ti] == think_open) {
                        size_t end = ti + 1;
                        if (end < toks.size() && nl_tok != LLAMA_TOKEN_NULL && toks[end] == nl_tok) end++;
                        if (end < toks.size() && toks[end] == think_close) {
                            end++;
                            if (end < toks.size() && nl_tok != LLAMA_TOKEN_NULL && toks[end] == nl_tok) end++;
                            ti = end - 1; // skip the entire closed think block
                            continue;
                        }
                    }
                    fixed.push_back(toks[ti]);
                }
                toks = std::move(fixed);
            }

            // Truncate looped content from previous assistant responses in history.
            // Once a response loops, the contaminated text poisons all future turns.
            {
                const llama_token im_start = 10, im_end = 11;
                const llama_token tok_ass = 1503, tok_ist = 19464;

                // Find all completed assistant blocks (not the final generation block)
                std::vector<std::pair<int,int>> asst_ranges; // (content_start, im_end_pos)
                for (int i = 0; i + 3 < (int)toks.size(); i++) {
                    if (toks[i] == im_start && toks[i+1] == tok_ass && toks[i+2] == tok_ist) {
                        int content_start = i + 4; // skip: im_start, ass, istant, \n
                        if (content_start >= (int)toks.size()) break;
                        int end_pos = content_start;
                        while (end_pos < (int)toks.size() && toks[end_pos] != im_end) end_pos++;
                        if (end_pos < (int)toks.size()) { // only completed blocks (have im_end)
                            asst_ranges.push_back({content_start, end_pos});
                        }
                    }
                }

                int total_removed = 0;
                for (auto it = asst_ranges.rbegin(); it != asst_ranges.rend(); ++it) {
                    auto [cs, ep] = *it;
                    int len = ep - cs;
                    if (len < 16) continue;

                    int truncate_at = -1;
                    for (int pos = cs + 16; pos <= ep; pos++) {
                        int n = pos - cs;
                        for (int period = 4; period <= std::min(32, n/2); period++) {
                            bool match = true;
                            for (int j = 0; j < period; j++) {
                                if (toks[pos - 1 - j] != toks[pos - 1 - j - period]) {
                                    match = false;
                                    break;
                                }
                            }
                            if (match) {
                                truncate_at = pos - 2 * period;
                                goto found_loop;
                            }
                        }
                    }
                    found_loop:
                    if (truncate_at > cs && truncate_at < ep) {
                        int remove_count = ep - truncate_at;
                        toks.erase(toks.begin() + truncate_at, toks.begin() + ep);
                        total_removed += remove_count;
                    }
                }
                if (total_removed > 0) {
                    SLT_INF(slot, "diff: stripped %d looped tokens from %d assistant blocks in prompt\n",
                            total_removed, (int)asst_ranges.size());
                }
            }

            // Strip trailing open <think> not followed by </think>
            {
                int think_pos = -1;
                for (int ti = (int)toks.size() - 1; ti >= 0; ti--) {
                    if (toks[ti] == think_open) { think_pos = ti; break; }
                }
                if (think_pos >= 0) {
                    bool has_close = false;
                    for (int ti = think_pos + 1; ti < (int)toks.size(); ti++) {
                        if (toks[ti] == think_close) { has_close = true; break; }
                    }
                    if (!has_close) {
                        toks.resize(think_pos);
                    }
                }
            }

            // Ensure prompt ends with <think>\n</think>\n
            {
                int n = (int)toks.size();
                bool has_think_block = (n >= 4 && nl_tok != LLAMA_TOKEN_NULL &&
                    toks[n-4] == think_open && toks[n-3] == nl_tok &&
                    toks[n-2] == think_close && toks[n-1] == nl_tok);
                if (!has_think_block && nl_tok != LLAMA_TOKEN_NULL) {
                    toks.push_back(think_open);
                    toks.push_back(nl_tok);
                    toks.push_back(think_close);
                    toks.push_back(nl_tok);
                }
            }

            task.tokens = server_tokens(toks, task.tokens.has_mtmd);
        }

        slot.task = std::make_unique<const server_task>(std::move(task));

        slot.state = slot.task->is_child()
            ? SLOT_STATE_WAIT_OTHER // wait for the parent to process prompt
            : SLOT_STATE_STARTED;

        // reset server kill-switch counter
        n_empty_consecutive = 0;

        SLT_INF(slot, "processing task, is_child = %d\n", slot.task->is_child());
        return true;
    }

    bool process_token(completion_token_output & result, server_slot & slot) {
        // remember which tokens were sampled - used for repetition penalties during sampling
        const std::string token_str = result.text_to_send;
        slot.sampled = result.tok;

        slot.generated_text += token_str;
        if (slot.task->params.return_tokens) {
            slot.generated_tokens.push_back(result.tok);
        }
        slot.has_next_token = true;

        // check if there is incomplete UTF-8 character at the end
        bool incomplete = validate_utf8(slot.generated_text) < slot.generated_text.size();

        // search stop word and delete it
        if (!incomplete) {
            size_t pos = std::min(slot.n_sent_text, slot.generated_text.size());

            const std::string str_test = slot.generated_text.substr(pos);
            bool send_text = true;

            size_t stop_pos = slot.find_stopping_strings(str_test, token_str.size(), true);
            if (stop_pos != std::string::npos) {
                slot.generated_text.erase(
                    slot.generated_text.begin() + pos + stop_pos,
                    slot.generated_text.end());
                pos = std::min(slot.n_sent_text, slot.generated_text.size());
            } else if (slot.has_next_token && !llama_vocab_is_eog(vocab, result.tok) ) {
                stop_pos = slot.find_stopping_strings(str_test, token_str.size(), false);
                send_text = stop_pos == std::string::npos;
            }

            // check if there is any token to predict
            if (send_text) {
                // no send the stop word in the response
                result.text_to_send = slot.generated_text.substr(pos, std::string::npos);
                slot.n_sent_text += result.text_to_send.size();
                // add the token to slot queue and cache
            } else {
                result.text_to_send = "";
            }

            slot.add_token(result);
            if (slot.task->params.stream) {
                send_partial_response(slot, result, false);
            }
        }

        if (incomplete) {
            slot.has_next_token = true;
        }

        // if context shifting is disabled, make sure that we don't run out of context
        if (!params_base.ctx_shift && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
            slot.truncated      = true;
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped due to running out of context capacity, prompt.n_tokens() = %d, task.n_tokens = %d, n_decoded = %d, n_ctx = %d\n",
                    slot.prompt.n_tokens(), slot.task->n_tokens(), slot.n_decoded, slot.n_ctx);
        }

        // check the limits
        if (slot.n_decoded > 0 && slot.has_next_token && !slot.has_budget(params_base)) {
            slot.stop           = STOP_TYPE_LIMIT;
            slot.has_next_token = false;

            SLT_DBG(slot, "stopped by limit, n_decoded = %d, n_predict = %d\n", slot.n_decoded, slot.task->params.n_predict);
        }

        if (slot.has_new_line) {
            // require that each new line has a whitespace prefix (i.e. indentation) of at least slot.params.n_indent
            if (slot.task->params.n_indent > 0) {
                // check the current indentation
                // TODO: improve by not doing it more than once for each new line
                if (slot.last_nl_pos > 0) {
                    size_t pos = slot.last_nl_pos;

                    int n_indent = 0;
                    while (pos < slot.generated_text.size() && (slot.generated_text[pos] == ' ' || slot.generated_text[pos] == '\t')) {
                        n_indent++;
                        pos++;
                    }

                    if (pos < slot.generated_text.size() && n_indent < slot.task->params.n_indent) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;

                        // cut the last line
                        slot.generated_text.erase(pos, std::string::npos);

                        SLT_DBG(slot, "stopped by indentation limit, n_decoded = %d, n_indent = %d\n", slot.n_decoded, n_indent);
                    }
                }

                // find the next new line
                {
                    const size_t pos = slot.generated_text.find('\n', slot.last_nl_pos);

                    if (pos != std::string::npos) {
                        slot.last_nl_pos = pos + 1;
                    }
                }
            }
        }

        // check if there is a new line in the generated text
        if (result.text_to_send.find('\n') != std::string::npos) {
            slot.has_new_line = true;

            // if we have seen a new line, we stop after a certain time limit, but only upon another new line
            if (slot.task->params.t_max_predict_ms > 0 && (ggml_time_us() - slot.t_start_generation > 1000.0f*slot.task->params.t_max_predict_ms)) {
                slot.stop           = STOP_TYPE_LIMIT;
                slot.has_next_token = false;

                SLT_DBG(slot, "stopped by time limit, n_decoded = %d, t_max_predict_ms = %d ms\n", slot.n_decoded, (int) slot.task->params.t_max_predict_ms);
            }
        }

        if (llama_vocab_is_eog(vocab, result.tok)) {
            slot.stop           = STOP_TYPE_EOS;
            slot.has_next_token = false;

            SLT_DBG(slot, "%s", "stopped by EOS\n");
        }

        SLT_DBG(slot, "n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n", slot.n_decoded, slot.n_remaining, result.tok, token_str.c_str());

        return slot.has_next_token; // continue
    }

    void populate_token_probs(const server_slot & slot, completion_token_output & result, bool post_sampling, bool special, int idx) const {
        const size_t n_probs_request = slot.task->params.sampling.n_probs;

        if (post_sampling) {
            const auto * cur_p = common_sampler_get_candidates(slot.smpl.get(), true);
            const size_t max_probs = cur_p->size;
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                if (cur_p->data[i].id == result.tok) {
                    result.prob = cur_p->data[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                // Some samplers do return 0.0 probabilities, others don't.
                // Filter 0.0 probailities, to ensure the behavior is consistent.
                if (cur_p->data[i].p == 0.0) {
                    break;
                }

                result.probs.push_back({
                    cur_p->data[i].id,
                    common_token_to_piece(ctx_tgt, cur_p->data[i].id, special),
                    cur_p->data[i].p
                });
            }
        } else {
            std::vector<llama_token_data> cur = get_token_probabilities(ctx_tgt, idx, n_probs_request);
            const size_t max_probs = cur.size();
            const size_t n_probs = std::min(max_probs, n_probs_request);

            // set probability for sampled token
            for (size_t i = 0; i < max_probs; i++) {
                // set probability for sampled token
                if (cur[i].id == result.tok) {
                    result.prob = cur[i].p;
                    break;
                }
            }

            // set probability for top n_probs tokens
            result.probs.reserve(n_probs);
            for (size_t i = 0; i < n_probs; i++) {
                result.probs.push_back({
                    cur[i].id,
                    common_token_to_piece(ctx_tgt, cur[i].id, special),
                    cur[i].p
                });
            }
        }
    }

    void send_error(const server_task & task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(task.id, error, type);
    }

    void send_error(const server_slot & slot, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER) {
        send_error(slot.task->id, error, type, slot.task->n_tokens(), slot.n_ctx);
    }

    void send_error(const int id_task, const std::string & error, const enum error_type type = ERROR_TYPE_SERVER, const int32_t n_prompt_tokens = 0, const int32_t n_ctx = 0) {
        SRV_ERR("task id = %d, error: %s\n", id_task, error.c_str());

        if (type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
            GGML_ASSERT(n_ctx > 0 && n_prompt_tokens > 0);
        }

        auto res = std::make_unique<server_task_result_error>();
        res->id              = id_task;
        res->err_type        = type;
        res->err_msg         = error;
        res->n_prompt_tokens = n_prompt_tokens;
        res->n_ctx           = n_ctx;

        queue_results.send(std::move(res));
    }

    // if multimodal is enabled, send an error and return false
    bool check_no_mtmd(const int id_task) {
        if (mctx) {
            send_error(id_task, "This feature is not supported by multimodal", ERROR_TYPE_NOT_SUPPORTED);
            return false;
        }
        return true;
    }

    void send_partial_response(server_slot & slot, const completion_token_output & tkn, bool is_progress, bool is_begin = false) {
        auto res = std::make_unique<server_task_result_cmpl_partial>();

        res->id    = slot.task->id;
        res->index = slot.task->index;

        if (is_progress) {
            res->is_progress        = true;
            res->progress.total     = slot.task->n_tokens();
            res->progress.cache     = slot.n_prompt_tokens_cache;
            res->progress.processed = slot.prompt.tokens.size();
            res->progress.time_ms   = (ggml_time_us() - slot.t_start_process_prompt) / 1000;
        }
        if (is_begin) {
            res->is_begin = true;
        } else {
            res->content = tkn.text_to_send;
            res->tokens  = { tkn.tok };
        }

        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            res->prob_output = tkn; // copy the token probs
        }

        // populate timings if this is final response or timings_per_token is enabled
        if (slot.stop != STOP_TYPE_NONE || slot.task->params.timings_per_token) {
            res->timings = slot.get_timings();
        }

        queue_results.send(std::move(res));
    }

    void send_final_response(server_slot & slot) {
        auto res = std::make_unique<server_task_result_cmpl_final>();

        res->id      = slot.task->id;
        res->id_slot = slot.id;

        res->index = slot.task->index;

        // keep copy of last generated text for debugging purposes
        if (slots_debug) {
            slot.debug_generated_text = slot.generated_text;
        }

        // in stream mode, content and tokens are already in last partial chunk
        if (slot.task->params.stream) {
            res->content     = "";
            res->tokens      = llama_tokens{};
        } else {
            res->content     = std::move(slot.generated_text);
            res->tokens      = std::move(slot.generated_tokens);
        }
        res->timings         = slot.get_timings();
        res->prompt          = slot.task->tokens.detokenize(ctx_tgt, true);
        res->response_fields = std::move(slot.task->params.response_fields);

        res->truncated             = slot.truncated;
        res->n_decoded             = slot.n_decoded;
        res->n_prompt_tokens       = slot.task->n_tokens();
        res->n_prompt_tokens_cache = slot.n_prompt_tokens_cache;
        res->n_tokens_cached       = slot.prompt.n_tokens();
        res->has_new_line          = slot.has_new_line;
        res->stopping_word         = slot.stopping_word;
        res->stop                  = slot.stop;
        res->post_sampling_probs   = slot.task->params.post_sampling_probs;

        res->verbose           = slot.task->params.verbose;
        res->stream            = slot.task->params.stream;
        res->include_usage     = slot.task->params.include_usage;
        res->res_type          = slot.task->params.res_type;
        res->oaicompat_model   = slot.task->params.oaicompat_model;
        res->oaicompat_cmpl_id = slot.task->params.oaicompat_cmpl_id;

        // populate res.probs_output
        if (slot.task->params.sampling.n_probs > 0) {
            if (!slot.task->params.stream && slot.stop == STOP_TYPE_WORD) {
                const llama_tokens stop_word_toks = common_tokenize(ctx_tgt, slot.stopping_word, false);

                size_t safe_offset = std::min(slot.generated_token_probs.size(), stop_word_toks.size());
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end() - safe_offset);
            } else {
                res->probs_output = std::vector<completion_token_output>(
                        slot.generated_token_probs.begin(),
                        slot.generated_token_probs.end());
            }
        }

        res->generation_params = slot.task->params; // copy the parameters

        queue_results.send(std::move(res));
    }

    void send_embedding(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_embd>();
        res->id        = slot.task->id;
        res->index     = slot.task->index;
        res->n_tokens  = slot.task->n_tokens();
        res->res_type  = slot.task->params.res_type;

        const int n_embd_out = llama_model_n_embd_out(model_tgt);

        std::vector<float> embd_res(n_embd_out, 0.0f);

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = nullptr;
            if (llama_pooling_type(slot.ctx_tgt) == LLAMA_POOLING_TYPE_NONE) {
                embd = llama_get_embeddings_ith(slot.ctx_tgt, i);
            } else {
                embd = llama_get_embeddings_seq(slot.ctx_tgt, batch.seq_id[i][0]);
            }

            if (embd == nullptr) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->embedding.push_back(std::vector<float>(n_embd_out, 0.0f));
                continue;
            }

            // normalize only when there is pooling
            if (llama_pooling_type(slot.ctx_tgt) != LLAMA_POOLING_TYPE_NONE) {
                common_embd_normalize(embd, embd_res.data(), n_embd_out, slot.task->params.embd_normalize);
                res->embedding.push_back(embd_res);
                break;
            }

            res->embedding.emplace_back(embd, embd + n_embd_out);
        }

        SLT_DBG(slot, "%s", "sending embeddings\n");

        queue_results.send(std::move(res));
    }

    void send_rerank(const server_slot & slot, const llama_batch & batch) {
        auto res = std::make_unique<server_task_result_rerank>();
        res->id       = slot.task->id;
        res->index    = slot.task->index;
        res->n_tokens = slot.task->n_tokens();

        for (int i = 0; i < batch.n_tokens; ++i) {
            if (!batch.logits[i] || batch.seq_id[i][0] != slot.id) {
                continue;
            }

            const float * embd = llama_get_embeddings_seq(ctx_tgt, batch.seq_id[i][0]);
            if (embd == NULL) {
                embd = llama_get_embeddings_ith(ctx_tgt, i);
            }

            if (embd == NULL) {
                SLT_ERR(slot, "failed to get embeddings, token = %d, seq_id = %d\n", batch.token[i], batch.seq_id[i][0]);

                res->score = -1e6;
                continue;
            }

            res->score = embd[0];
        }

        SLT_DBG(slot, "sending rerank result, res.score = %f\n", res->score);

        queue_results.send(std::move(res));
    }

    //
    // Functions to process the task
    //

    // tokenize the input if it's set by CLI, return false on error
    bool tokenize_cli_input(server_task & task) {
        try {
            auto & prompt = task.cli_prompt;
            if (mctx != nullptr) {
                task.tokens = process_mtmd_prompt(mctx, prompt, task.cli_files);
            } else {
                task.tokens = std::move(tokenize_input_prompts(vocab, mctx, prompt, true, true)[0]);
            }
            task.cli_prompt.clear();
            task.cli_files.clear();
        } catch (const std::exception & e) {
            send_error(task, std::string("Failed to format input: ") + e.what(), ERROR_TYPE_INVALID_REQUEST);
            return false;
        }
        return true;
    }

    std::vector<server_slot *> get_free_slots(size_t n_slots_needed, int exclude_id_slot) {
        std::vector<server_slot *> free_slots;
        for (auto & slot : slots) {
            if (!slot.is_processing() && slot.id != exclude_id_slot) {
                free_slots.push_back(&slot);
            }
            if (free_slots.size() >= n_slots_needed) {
                break;
            }
        }
        return free_slots;
    }

    // launch multiple slots for parent + child tasks
    bool launch_slots_with_parent_task(server_slot & parent_slot, std::vector<server_slot *> & child_slots, server_task && parent_task) {
        GGML_ASSERT(!parent_slot.is_processing());
        GGML_ASSERT(parent_task.is_parent());
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());

        int id_parent = parent_task.id;

        SRV_TRC("launching slots for parent task id_task = %d with %zu child tasks\n", id_parent, parent_task.child_tasks.size());

        // to be called in case of failure to release all launched slots
        auto release_slots = [this, id_parent]() {
            for (auto & slot : slots) {
                if (slot.is_processing() && (
                        slot.task->id == id_parent ||
                        slot.task->id_parent == id_parent
                )) {
                    slot.release();
                }
            }
        };

        // launch all child tasks first
        size_t idx = 0;
        GGML_ASSERT(child_slots.size() == parent_task.child_tasks.size());
        for (auto * slot : child_slots) {
            int id_child = parent_task.child_tasks[idx].id;
            if (!launch_slot_with_task(*slot, std::move(parent_task.child_tasks[idx]))) {
                SRV_ERR("failed to launch slot with child task, id_task = %d\n", id_child);
                release_slots();
                return false;
            }
            idx++;
        }

        // finally, launch the parent task
        if (!launch_slot_with_task(parent_slot, std::move(parent_task))) {
            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_parent);
            release_slots();
            return false;
        }

        return true;
    }

    // n_tokens_cur: the number of tokens added to the batch for the current slot
    void create_checkpoint(server_slot & slot, const int64_t n_tokens_cur, llama_pos pos_min, llama_pos pos_max) {
        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
            // make room for the new checkpoint, if needed
            const auto & cur = slot.prompt.checkpoints.front();

            SLT_WRN(slot, "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);

            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
        }

        auto & cur = slot.prompt.checkpoints.emplace_back();

        // [TAG_CHECKPOINTS_FIX_POS_MIN]
        // TODO: here we incorrectly deterimne that the saved checkpoint data covers the [pos_min, pos_max] range
        //       this is not true for SWA models: https://github.com/ggml-org/llama.cpp/pull/24411#issuecomment-4677983225
        cur.update_pos(slot.prompt.n_tokens() - n_tokens_cur, pos_min, pos_max);

        cur.update_tgt(ctx_tgt,       slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        cur.update_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        // stash the draft's speculative state with the checkpoint
        common_speculative_get_state(spec.get(), slot.id, cur.data_spec);

        SLT_TRC(slot,
                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024 / 1024);
    }

    void process_single_task(server_task && task) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                {
                    // special case: if input is provided via CLI, tokenize it first
                    // otherwise, no need to tokenize as it's already done inside the HTTP thread
                    if (task.cli) {
                        if (!tokenize_cli_input(task)) {
                            break;
                        }
                    }

                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);

                    //
                    // slot scheduling logic
                    //

                    if (slot == nullptr) {
                        // if no slot is available, we defer this task for processing later
                        SRV_DBG("no slot is available, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", id_task);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    if (task.is_parent()) {
                        // try getting free slots for all child tasks
                        size_t n_child_tasks = task.child_tasks.size();
                        std::vector<server_slot *> child_slots = get_free_slots(n_child_tasks, slot->id);
                        if (child_slots.size() < n_child_tasks) {
                            SRV_DBG("not enough free slots for child tasks, n_free = %zu, n_children = %zu, defer task, id_task = %d\n", child_slots.size(), n_child_tasks, id_task);
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                        if (!launch_slots_with_parent_task(*slot, child_slots, std::move(task))) {
                            SRV_ERR("failed to launch slot with parent task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    } else {
                        // Unified KV: check if launching this task would overflow the shared cell pool.
                        // Use max(current, planned) since a just-launched slot hasn't filled yet.
                        if (params_base.kv_unified && task.n_tokens() > 0) {
                            int64_t cells_committed = 0;
                            for (const auto & s : slots) {
                                if (s.is_processing() && s.task) {
                                    cells_committed += std::max((int64_t) s.prompt.n_tokens(), (int64_t) s.task->n_tokens());
                                }
                            }
                            const int64_t cells_available = (int64_t) slot->n_ctx - cells_committed;
                            if (cells_available < (int64_t) task.n_tokens()) {
                                SRV_DBG("defer task %d: needs %d tokens but only %" PRId64 " cells available (%" PRId64 " committed by active slots)\n",
                                        id_task, task.n_tokens(), cells_available, cells_committed);
                                queue_tasks.defer(std::move(task));
                                break;
                            }
                        }

                        // dynamic VBR: if this launch's projected footprint would degrade the
                        // pool below the quality floor, clear idle caches first. The probe is
                        // sized by the post-LCP SUFFIX — the common prefix refills in place with
                        // zero cell growth, and probing the full prompt evicted other clients'
                        // caches for growth that was never going to happen.
                        if (server_vbr_dynamic_active(params_base) && task.n_tokens() > 0) {
                            const size_t lcp = slot->prompt.tokens.get_common_prefix(task.tokens);
                            vbr_reclaim_before_degrade(slot->id, (uint32_t) (task.n_tokens() - lcp), "launch");
                        }

                        if (!launch_slot_with_task(*slot, std::move(task))) {
                            SRV_ERR("failed to launch slot with task, id_task = %d\n", id_task);
                            break; // drop the task
                        }
                    }

                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache\n");

                                if (slot.prompt_save(*prompt_cache)) {
                                    SLT_DBG(slot, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\n");
                                    prompt_cache->update();
                                }

                                if (params_base.kv_unified) {
                                    // [TAG_IDLE_SLOT_CLEAR]
                                    slot.prompt_clear(false);
                                }
                            }
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CANCEL:
                {
                    // release slot linked with the task id
                    for (auto & slot : slots) {
                        if (slot.task && slot.task->id == task.id_target) {
                            slot.release();
                            break;
                        }
                    }
                } break;
            case SERVER_TASK_TYPE_CONTROL:
                {
                    auto res = std::make_unique<server_task_result_control>();
                    res->id = task.id;

                    server_slot * slot = get_slot_by_cmpl_id(task.params.control_cmpl_id);
                    if (slot == nullptr) {
                        SRV_WRN("control %s on unknown completion id=%s, no live slot\n",
                                task.params.control_action.c_str(), task.params.control_cmpl_id.c_str());
                        res->success = false;
                        res->message = "no active completion for this id";
                        queue_results.send(std::move(res));
                        break;
                    }

                    if (task.params.control_action == "reasoning_end") {
                        // the budget sampler only exists when reasoning control was armed
                        if (!slot->task->params.sampling.reasoning_control) {
                            res->success = false;
                            res->message = "reasoning control not enabled for this completion";
                            queue_results.send(std::move(res));
                            break;
                        }
                        // act on the live slot mid generation, never defer
                        common_sampler_reasoning_budget_force(slot->smpl.get());
                        res->success = true;
                    } else {
                        res->success = false;
                        res->message = "unknown control action";
                    }

                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                {
                    // do nothing
                } break;
            case SERVER_TASK_TYPE_METRICS:
                {
                    json slots_data = json::array();

                    int n_idle_slots       = 0;
                    int n_processing_slots = 0;

                    for (server_slot & slot : slots) {
                        json slot_data = slot.to_json(slots_debug == 0);

                        if (slot.is_processing()) {
                            n_processing_slots++;
                        } else {
                            n_idle_slots++;
                        }

                        slots_data.push_back(slot_data);
                    }
                    SRV_DBG("n_idle_slots = %d, n_processing_slots = %d\n", n_idle_slots, n_processing_slots);

                    auto res = std::make_unique<server_task_result_metrics>();
                    res->id                  = task.id;
                    res->slots_data          = std::move(slots_data);
                    res->n_idle_slots        = n_idle_slots;
                    res->n_processing_slots  = n_processing_slots;
                    res->n_tasks_deferred    = queue_tasks.queue_tasks_deferred_size();
                    res->t_start             = metrics.t_start;

                    res->n_prompt_tokens_processed_total = metrics.n_prompt_tokens_processed_total;
                    res->t_prompt_processing_total       = metrics.t_prompt_processing_total;
                    res->n_tokens_predicted_total        = metrics.n_tokens_predicted_total;
                    res->t_tokens_generation_total       = metrics.t_tokens_generation_total;

                    res->n_tokens_max = metrics.n_tokens_max;

                    res->n_prompt_tokens_processed = metrics.n_prompt_tokens_processed;
                    res->t_prompt_processing       = metrics.t_prompt_processing;
                    res->n_tokens_predicted        = metrics.n_tokens_predicted;
                    res->t_tokens_generation       = metrics.t_tokens_generation;

                    res->n_decode_total          = metrics.n_decode_total;
                    res->n_busy_slots_total      = metrics.n_busy_slots_total;

                    if (task.metrics_reset_bucket) {
                        metrics.reset_bucket();
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_SAVE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }

                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const size_t token_count = slot->prompt.tokens.size();
                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    const llama_tokens & tokens = slot->prompt.tokens.get_tokens();
                    const size_t nwrite = llama_state_seq_save_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), token_count);

                    const int64_t t_end = ggml_time_us();
                    const double t_save_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = true;
                    res->n_tokens = token_count;
                    res->n_bytes  = nwrite;
                    res->t_ms     = t_save_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_RESTORE:
                {
                    if (!check_no_mtmd(task.id)) break;
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    const int64_t t_start = ggml_time_us();

                    std::string filename = task.slot_action.filename;
                    std::string filepath = task.slot_action.filepath;

                    llama_tokens tokens;
                    tokens.resize(slot->n_ctx);
                    size_t token_count = 0;
                    size_t nread = llama_state_seq_load_file(ctx_tgt, filepath.c_str(), slot->id, tokens.data(), tokens.size(), &token_count);
                    if (nread == 0) {
                        slot->prompt.tokens.clear(); // KV may already been invalidated?
                        send_error(task, "Unable to restore slot, no available space in KV cache or invalid slot save file", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    tokens.resize(token_count);
                    slot->prompt.tokens.clear();
                    slot->prompt.tokens.insert(tokens);

                    const int64_t t_end = ggml_time_us();
                    const double t_restore_ms = (t_end - t_start) / 1000.0;

                    auto res = std::make_unique<server_task_result_slot_save_load>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->filename = filename;
                    res->is_save  = false;
                    res->n_tokens = token_count;
                    res->n_bytes  = nread;
                    res->t_ms     = t_restore_ms;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SLOT_ERASE:
                {
                    if (!check_no_mtmd(task.id)) {
                        break;
                    }
                    const int id_slot = task.slot_action.id_slot;
                    server_slot * slot = get_slot_by_id(id_slot);
                    if (slot == nullptr) {
                        send_error(task, "Invalid slot ID", ERROR_TYPE_INVALID_REQUEST);
                        break;
                    }
                    if (slot->is_processing()) {
                        // if requested slot is unavailable, we defer this task for processing later
                        SRV_DBG("requested slot is unavailable, defer task, id_task = %d\n", task.id);
                        queue_tasks.defer(std::move(task));
                        break;
                    }

                    // Erase token cache
                    const size_t n_erased = slot->prompt.tokens.size();

                    slot->prompt_clear(false);

                    auto res = std::make_unique<server_task_result_slot_erase>();
                    res->id       = task.id;
                    res->id_slot  = id_slot;
                    res->n_erased = n_erased;
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_GET_LORA:
                {
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    auto & loras = params_base.lora_adapters;
                    auto res = std::make_unique<server_task_result_get_lora>();
                    res->id = task.id;
                    for (size_t i = 0; i < loras.size(); ++i) {
                        auto & lora = loras[i];
                        std::string alora_invocation_string = "";
                        const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
                        llama_tokens alora_invocation_tokens;
                        if (n_alora_tokens) {
                            const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                            for (uint64_t j = 0; j < n_alora_tokens; ++j) {
                                alora_invocation_string += common_token_to_piece(vocab, alora_tokens[j]);
                                alora_invocation_tokens.push_back(alora_tokens[j]);
                            }
                        }
                        res->loras.push_back(server_task_result_get_lora::lora{
                            lora,
                            alora_invocation_string,
                            alora_invocation_tokens,
                        });
                    }
                    queue_results.send(std::move(res));
                } break;
            case SERVER_TASK_TYPE_SET_LORA:
                {
                    auto new_loras = construct_lora_list(task.set_lora);
                    // logging
                    for (size_t i = 0; i < new_loras.size(); ++i) {
                        SRV_TRC("set lora adapter idx=%zu scale=%f\n", i, new_loras[i].scale);
                    }
                    // TODO @ngxson : make lora_adapters a dedicated member of server_context
                    params_base.lora_adapters = new_loras;
                    auto res = std::make_unique<server_task_result_apply_lora>();
                    res->id = task.id;
                    queue_results.send(std::move(res));
                } break;
        }
    }

    void iterate(std::vector<server_slot> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(slot);
            } catch (const std::exception & e) {
                SLT_ERR(slot, "got exception: %s\n", e.what());
                send_error(slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    void iterate(std::vector<server_slot *> & slots, std::function<void(server_slot &)> callback) {
        for (auto & slot : slots) {
            try {
                callback(*slot);
            } catch (const std::exception & e) {
                SLT_ERR(*slot, "got exception: %s\n", e.what());
                send_error(*slot, std::string("got exception: ") + e.what(), ERROR_TYPE_SERVER);
                slot->release();
            }
        }
    }

    void abort_all_slots(const std::string & reason) {
        for (auto & slot : slots) {
            if (slot.is_processing()) {
                send_error(slot, reason, ERROR_TYPE_SERVER);
                slot.release();
            }
        }
    }

    // @ngxson : for debugging only
    int64_t t_pre_decode  = 0;
    int64_t t_decode      = 0;
    int64_t t_post_decode = 0;
    int64_t t_sampl       = 0;
    int64_t n_pre_decode  = 0;
    int64_t n_decode      = 0;
    int64_t n_post_decode = 0;
    int64_t n_sampl       = 0;

    // fork: spec-cycle instrumentation + cross-phase batch state.
    // Set in pre_decode(), consumed by decode()/post_decode()/post_cycle() —
    // the update_slots() split turned the old monolithic-loop locals into members.
    int64_t t_cycle_start   = 0;
    int64_t t_draft_total   = 0;
    int64_t t_verify_total  = 0;
    int64_t t_accept_total  = 0;
    int     n_slots_drafted = 0;

    // TG tokens in the current batch — pure-verify batches allow multi-seq batching
    int32_t n_tg_tokens = 0;

    // DFlash tape recording armed for this cycle (turned off in post_cycle())
    bool dflash_tape_active = false;
    // target can replay the tape losslessly on GPU after a partial accept; when false,
    // no tape is recorded and rollback re-decodes the accepted tokens instead
    bool dflash_tape_ok = false;
    // pure-TG batch → multi-seq ubatch allowed (force_split_seq restored in post_cycle())
    bool can_batch_multiseq = false;
// #define DEBUG_TIMINGS
#ifdef DEBUG_TIMINGS
    struct scoped_timer {
        int64_t & t;
        int64_t & n;
        int64_t t_start;
        scoped_timer(int64_t & t_, int64_t & n_) : t(t_), n(n_) {
            t_start = ggml_time_us();
        }
        ~scoped_timer() {
            t += ggml_time_us() - t_start;
            n++;
        }
    };
#else
    struct scoped_timer {
        scoped_timer(int64_t &, int64_t &) {}
        ~scoped_timer() {}
    };
#endif

    void update_slots() {
#ifdef DEBUG_TIMINGS
        static int64_t t_prev = 0;
        int64_t t_start = ggml_time_us();
        if (t_start - t_prev > 5 * 1000 * 1000) { // every 5 seconds
            t_prev = t_start;
            SRV_INF("n_pre_decode      = %" PRId64 "\n", n_pre_decode);
            SRV_INF("avg t_pre_decode  = %f ms\n", (double) t_pre_decode / n_pre_decode / 1000.0);
            SRV_INF("avg t_decode      = %f ms\n", (double) t_decode / n_decode / 1000.0);
            SRV_INF("avg t_post_decode = %f ms\n", (double) t_post_decode / n_post_decode / 1000.0);
            SRV_INF("avg t_sampl       = %f ms\n", (double) t_sampl / n_sampl / 1000.0);
        }
#endif

        // check if all slots are idle
        {
            bool all_idle = true;

            for (auto & slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                SRV_TRC("%s", "all slots are idle\n");
                return; // skip further processing

            } else {
                SRV_DBG("%s", "posting NEXT_RESPONSE\n");

                server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
                task.id = queue_tasks.get_new_id();
                queue_tasks.post(std::move(task));
            }
        }

        try {
            scoped_timer t(t_pre_decode, n_pre_decode);
            pre_decode();
            batch.render();
        } catch (const std::exception & e) {
            SRV_ERR("pre_decode() failed: %s\n", e.what());
            abort_all_slots("pre_decode() failed: " + std::string(e.what()));
        }

        // fork: the chunk loop below never runs for an empty batch, so the empty-batch
        // warn/abort lives here — with the diffusion exemption (diffusion slots contribute
        // no main-batch tokens; they decode their own batches in post_cycle())
        if (batch.size() == 0) {
            const bool has_diff_gen = std::any_of(slots.begin(), slots.end(),
                [](const server_slot & s) { return s.diff_self_spec && s.state == SLOT_STATE_GENERATING; });
            if (!has_diff_gen) {
                SRV_WRN("%s", "no tokens to decode\n");
                if (++n_empty_consecutive > 3) {
                    GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
                }
            }
        } else {
            n_empty_consecutive = 0;
        }

        llama_batch batch_view;
        int32_t off_next = 0;
        int32_t n_batch = llama_n_batch(ctx_tgt);
        for (int32_t off = 0; off < batch.size(); off = off_next) {
            const int32_t n_tokens = std::min(n_batch, batch.size() - off);
            try {
                scoped_timer t(t_decode, n_decode);
                // TODO @ngxson : maybe handle n_batch == 1 here instead of inside decode()

                batch_view = batch.get_view(off, n_tokens);
                bool ok = decode(n_batch, off, batch_view);
#ifdef DEBUG_TIMINGS
                llama_synchronize(ctx_tgt);
#endif

                if (ok) {
                    // move the head of the batch forward with the number of tokens we just processed
                    off_next = off + n_tokens;

                    // on successful decode, restore the original batch size
                    n_batch = llama_n_batch(ctx_tgt);
                } else {
                    // try again with the updated n_batch
                    continue;
                }
            } catch (const std::exception & e) {
                SRV_ERR("decode() failed: %s\n", e.what());
                abort_all_slots("decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

            try {
                scoped_timer t(t_post_decode, n_post_decode);
                post_decode(n_tokens, off, batch_view);
            } catch (const std::exception & e) {
                SRV_ERR("post_decode() failed: %s\n", e.what());
                abort_all_slots("post_decode() failed: " + std::string(e.what()));
                break; // stop any further processing
            }

        }

        // fork: once-per-cycle tail — diffusion self-speculation, spec-cycle report,
        // DFlash tape-off, force_split_seq restore
        try {
            post_cycle();
        } catch (const std::exception & e) {
            SRV_ERR("post_cycle() failed: %s\n", e.what());
            abort_all_slots("post_cycle() failed: " + std::string(e.what()));
        }
    }

    void pre_decode() {
        // apply context-shift if needed
        // TODO: simplify and improve
        iterate(slots, [&](server_slot & slot) {
            if (slot.state == SLOT_STATE_GENERATING && slot.prompt.n_tokens() + 1 >= slot.n_ctx) {
                if (!params_base.ctx_shift) {
                    // this check is redundant (for good)
                    // we should never get here, because generation should already stopped in process_token()
                    send_error(slot, "context shift is disabled", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                if (mctx) {
                    // we should never reach this because params_base.ctx_shift is automatically disabled if mmproj is loaded
                    // we don't support ctx_shift because an image chunk may contains multiple tokens
                    GGML_ABORT("not supported by multimodal");
                }

                if (slot.task->is_parent() || slot.task->is_child()) {
                    send_error(slot, "context shift cannot be used for shared prompt", ERROR_TYPE_SERVER);
                    slot.release();
                    return;
                }

                // Shift context
                int n_keep = slot.task->params.n_keep < 0 ? slot.task->n_tokens() : slot.task->params.n_keep;

                if (add_bos_token) {
                    n_keep += 1;
                }

                n_keep = std::min(slot.n_ctx - 4, n_keep);

                const int n_left    = slot.prompt.n_tokens() - n_keep;
                int       n_discard = slot.task->params.n_discard ? slot.task->params.n_discard : (n_left / 2);

                // ref: https://github.com/ggml-org/llama.cpp/pull/24786
                n_discard = std::clamp(n_discard, 0, std::max(0, n_left - 1));

                SLT_WRN(slot, "slot context shift, n_keep = %d, n_left = %d, n_discard = %d\n", n_keep, n_left, n_discard);

                common_context_seq_rm (ctx_tgt, slot.id, n_keep            , n_keep + n_discard);
                common_context_seq_add(ctx_tgt, slot.id, n_keep + n_discard, slot.prompt.n_tokens(), -n_discard);

                if (ctx_dft) {
                    common_context_seq_rm (ctx_dft.get(), slot.id, n_keep            , n_keep + n_discard);
                    common_context_seq_add(ctx_dft.get(), slot.id, n_keep + n_discard, slot.prompt.tokens.pos_next(), -n_discard);
                }

                // add generated tokens to cache
                // ref: https://github.com/ggml-org/llama.cpp/pull/16818#discussion_r2473269481
                {
                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                    llama_tokens new_tokens = slot.prompt.tokens.get_tokens(); // copy
                    for (size_t i = n_keep + n_discard; i < new_tokens.size(); i++) {
                        new_tokens[i - n_discard] = new_tokens[i];
                    }

                    new_tokens.resize(slot.prompt.tokens.size() - n_discard);

                    slot.prompt.tokens.clear();
                    slot.prompt.tokens.insert(new_tokens);
                }

                slot.truncated = true;
            }
        });

        // start populating the batch for this iteration
        batch.clear();

        // track if given slot can be batched with slots already in the batch
        auto & slot_batched = batch.slot_batched;

        // fork: reset the spec-cycle instrumentation (reported in post_cycle())
        t_cycle_start   = ggml_time_us();
        t_draft_total   = 0;
        t_verify_total  = 0;
        t_accept_total  = 0;
        n_slots_drafted = 0;

        std::vector<llama_tokens> batched_drafts(slots.size());
        if (ctx_dft_shared) {
            int n_drafting = 0;
            for (const auto & slot : slots) {
                if (slot.state == SLOT_STATE_GENERATING && slot.can_speculate() && slot.get_n_draft_max() > 0) {
                    n_drafting++;
                }
            }
            llama_set_dflash_n_slots(ctx_dft_shared.get(), std::max(1, n_drafting));

            if (n_drafting >= 2 && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                std::vector<common_speculative *> batch_specs;
                std::vector<llama_token>          batch_id_lasts;
                std::vector<int>                  batch_slot_ids;

                for (auto & slot : slots) {
                    if (slot.state == SLOT_STATE_GENERATING && slot.can_speculate() && slot.get_n_draft_max() > 0) {
                        batch_specs.push_back(slot.get_spec());
                        batch_id_lasts.push_back(slot.sampled);
                        batch_slot_ids.push_back(slot.id);
                    }
                }

                std::vector<llama_tokens> batch_results;
                const int64_t t_batch_start = ggml_time_us();
                common_speculative_draft_batch(
                        batch_specs, ctx_dft_shared.get(),
                        params_base.speculative, batch_id_lasts, batch_results);
                t_draft_total = ggml_time_us() - t_batch_start;

                for (size_t i = 0; i < batch_slot_ids.size(); i++) {
                    batched_drafts[batch_slot_ids[i]] = std::move(batch_results[i]);
                }
            }
        }

        // first, add sampled tokens from any ongoing sequences (and draft per slot)
        iterate(slots, [&](server_slot & slot) {
            if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.diff_self_spec) {
                return; // diffusion slots handled separately after main decode
            }

            // check if we can batch this slot with the previous one
            if (!slot_batched) {
                slot_batched = &slot;
            } else if (!slot_batched->can_batch_with(slot)) {
                return;
            }

            const int n_draft_max = slot.get_n_draft_max();
            if (n_draft_max > 0) {
                const int64_t t_draft_slot_start = ggml_time_us();

                llama_tokens draft;
                if (!batched_drafts[slot.id].empty()) {
                    draft = std::move(batched_drafts[slot.id]);
                } else {
                    const llama_tokens & cached_text_tokens = slot.prompt.tokens.get_text_tokens();
                    const auto & params_spec = slot.task->params.speculative;
                    const llama_pos n_past = slot.prompt.tokens.pos_next();
                    draft = common_speculative_draft(slot.get_spec(), params_spec, cached_text_tokens, slot.sampled, nullptr, n_past);
                }

                if (draft.size() > (size_t) n_draft_max) {
                    SLT_WRN(slot, "draft size %d exceeds max %d, truncating\n", (int) draft.size(), n_draft_max);
                    draft.resize(n_draft_max);
                }

                slot.n_tokens_before_draft = slot.prompt.n_tokens();

                slot.spec_i_batch.push_back(batch.size());
                batch.add(slot.id, slot.sampled, slot.prompt.tokens.pos_next(), true);
                slot.prompt.tokens.push_back(slot.sampled);

                // an empty draft (e.g. ngram-mod with no index match) must take the
                // no-speculation path: the else-branch would leave spec_draft empty while
                // spec_i_batch holds one index, so the accept loop (gated on
                // !spec_draft.empty()) never consumes it and never samples a token this
                // cycle — the stale index leaks into the next cycle, re-decoding
                // slot.sampled every step until a real draft breaks the
                // idxs.size() == draft.size() + 1 invariant (#74)
                if (draft.empty() || slot.task->params.speculative.n_min > (int) draft.size()) {
                    SLT_DBG(slot, "ignoring small draft: %d < %d\n", (int) draft.size(), slot.task->params.speculative.n_min);
                    slot.i_batch = slot.spec_i_batch[0];
                    slot.spec_draft.clear();
                    slot.spec_i_batch.clear();
                } else {
                    slot.n_draft_total += draft.size();

                    // keep the spec checkpoint's position bookkeeping current (upstream #24536 family);
                    // the fork accept loop rolls back via backup seqs / seq_rm, so only the cheap
                    // position update is taken here (no target/draft state snapshot)
                    slot.spec_ckpt.update_pos(
                            slot.n_tokens_before_draft,
                            llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id),
                            llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id));

                    if (needs_reeval) {
                        if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                            llama_tape_replay_sync(ctx_tgt);
                            const int n_batch_tokens = 1 + (int) draft.size();
                            std::vector<int32_t> linear_parents(n_batch_tokens);
                            linear_parents[0] = -1;
                            for (int i = 1; i < n_batch_tokens; i++) {
                                linear_parents[i] = i - 1;
                            }
                            llama_set_tree_parent_ids(ctx_tgt, linear_parents.data(), n_batch_tokens);
                        }

                        // RS contexts handle rollback internally via seq_rm snapshots
                        if (ctx_tgt_seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_RS) {
                            if (!recurrent_expanded) {
                                auto * mem = llama_get_memory(ctx_tgt);
                                if (llama_memory_recurrent_expand(mem, n_seq_max_full)) {
                                    SRV_INF("expanded recurrent state to %d cells for speculative backup\n", n_seq_max_full);
                                } else {
                                    SRV_ERR("failed to expand recurrent state to %d cells\n", n_seq_max_full);
                                }
                                recurrent_expanded = true;
                            }
                            const llama_seq_id seq_backup = slot.id + n_parallel_user;
                            auto * mem = llama_get_memory(ctx_tgt);
                            llama_memory_seq_rm(mem, seq_backup, -1, -1);
                            llama_memory_seq_cp(mem, slot.id, seq_backup, -1, -1);
                            slot.has_draft_backup = true;
                            slot.seq_id_backup = seq_backup;
                        }
                    }

                    for (size_t i = 0; i < draft.size(); i++) {
                        slot.spec_i_batch.push_back(batch.size());
                        batch.add(slot.id, draft[i], slot.prompt.tokens.pos_next(), true);
                        slot.prompt.tokens.push_back(draft[i]);
                    }
                    slot.spec_draft = std::move(draft);
                }
                t_draft_total += ggml_time_us() - t_draft_slot_start;
                n_slots_drafted++;
            } else {
                slot.i_batch = batch.size();

                batch.add(slot.id, slot.sampled, slot.prompt.tokens.pos_next(), true);

                slot.prompt.tokens.push_back(slot.sampled);

                SLT_DBG(slot, "slot decode token, n_ctx = %d, n_tokens = %d, truncated = %d\n",
                        slot.n_ctx, slot.prompt.n_tokens(), slot.truncated);
            }
        });

        // process in chunks of params.n_batch
        int32_t n_batch  = llama_n_batch(ctx_tgt);
        int32_t n_ubatch = llama_n_ubatch(ctx_tgt);

        // track how many TG tokens are in the batch vs total, to detect
        // pure-verify batches where multi-seq batching is safe.
        n_tg_tokens = batch.size();

        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // next, batch any pending prompts without exceeding n_batch
        if (params_base.cont_batching || batch.size() == 0) {
            bool add_ok = true; // false means the batch is full, skip remaining slots

            iterate(slots, [&](server_slot & slot) {
                if (!add_ok || batch.size() >= n_batch) {
                    return; // batch is full, skip remaining slots
                }

                if (!slot.is_processing()) {
                    return;
                }

                // check if we can batch this slot with the previous one
                if (slot_batched && !slot_batched->can_batch_with(slot)) {
                    return;
                }

                // check if this is a child slot
                if (slot.state == SLOT_STATE_WAIT_OTHER) {
                    SLT_DBG(slot, "%s", "waiting for parent slot to complete\n");
                    return;
                }

                // this slot still has a prompt to be processed
                if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_STARTED) {
                    const auto & input_tokens = slot.task->tokens;

                    // used to determine the number of tokens added to the batch for the current slot
                    const auto n_tokens_prev = batch.size();

                    // TODO: maybe move branch to outside of this loop in the future
                    if (slot.state == SLOT_STATE_STARTED) {
                        slot.t_start_process_prompt = ggml_time_us();
                        slot.t_start_generation = 0;

                        slot.state = SLOT_STATE_PROCESSING_PROMPT;

                        SLT_TRC(slot, "new prompt, n_ctx_slot = %d, n_keep = %d, task.n_tokens = %d\n",
                                slot.n_ctx, slot.task->params.n_keep, slot.task->n_tokens());

                        // print prompt tokens (for debugging)
                        /*if (1) {
                            // first 16 tokens (avoid flooding logs)
                            for (int i = 0; i < std::min<int>(16, input_tokens.size()); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        } else {
                            // all
                            for (int i = 0; i < (int) input_tokens.size(); i++) {
                                SLT_DBG(slot, "prompt token %3d: %6d '%s'\n", i, input_tokens[i], common_token_to_piece(ctx_tgt, input_tokens[i]).c_str());
                            }
                        }*/

                        // keep track how many tokens we can reuse from the previous state
                        int n_past = 0;

                        // empty prompt passed -> release the slot and send empty response
                        if (input_tokens.empty()) {
                            SLT_WRN(slot, "%s", "empty prompt - releasing slot\n");

                            slot.print_timings();
                            send_final_response(slot);
                            slot.release();

                            return;
                        }

                        // TODO: support memory-less logits computation
                        if (slot.task->need_logits() && !llama_get_memory(ctx_tgt)) {
                            send_error(slot, "the current context does not logits computation. skipping", ERROR_TYPE_SERVER);
                            slot.release();
                            return;
                        }

                        if (!slot.can_split()) {
                            if (slot.task->n_tokens() > n_ubatch) {
                                send_error(slot,
                                           string_format(
                                               "input (%d tokens) is too large to process. increase the physical batch "
                                               "size (current batch size: %d)",
                                               slot.task->n_tokens(), n_ubatch),
                                           ERROR_TYPE_SERVER);
                                slot.release();
                                return;
                            }

                            if (slot.task->n_tokens() > slot.n_ctx) {
                                send_error(
                                    slot,
                                    string_format(
                                        "input (%d tokens) is larger than the max context size (%d tokens). skipping",
                                        slot.task->n_tokens(), slot.n_ctx),
                                    ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }
                        } else {
                            if (slot.task->n_tokens() >= slot.n_ctx) {
                                send_error(slot,
                                           string_format("request (%d tokens) exceeds the available context size (%d "
                                                         "tokens), try increasing it",
                                                         slot.task->n_tokens(), slot.n_ctx),
                                           ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                                slot.release();
                                return;
                            }

                            if (slot.task->params.cache_prompt) {
                                // reuse any previously computed tokens that are common with the new prompt
                                n_past = slot.prompt.tokens.get_common_prefix(input_tokens);

                                // if there is an alora invoked, don't cache after the invocation start
                                if (slot.alora_invocation_start > 0) {
                                    SLT_DBG(slot, "only caching to alora invocation start (n_past = %d, alora_invocation_start = %d)\n", n_past, slot.alora_invocation_start);
                                    n_past = std::min(n_past, slot.alora_invocation_start - 1);
                                }

                                const auto n_cache_reuse = slot.task->params.n_cache_reuse;

                                const bool can_cache_reuse =
                                    llama_memory_can_shift(llama_get_memory(ctx_tgt)) &&
                                    !slot.prompt.tokens.has_mtmd;

                                if (!can_cache_reuse && n_cache_reuse > 0) {
                                    SLT_WRN(slot, "cache reuse is not supported - ignoring n_cache_reuse = %d\n", n_cache_reuse);
                                }

                                // reuse chunks from the cached prompt by shifting their KV cache in the new position
                                if (can_cache_reuse && n_cache_reuse > 0) {
                                    GGML_ASSERT(!slot.prompt.tokens.has_mtmd);

                                    size_t head_c = n_past; // cache
                                    size_t head_p = n_past; // current prompt

                                    if (mctx) {
                                        // we should never reach this
                                        GGML_ABORT("not supported by multimodal");
                                    }

                                    SLT_DBG(slot, "trying to reuse chunks with size > %d, n_past = %d\n", n_cache_reuse, n_past);

                                    while (head_c < slot.prompt.tokens.size() &&
                                           head_p < input_tokens.size()) {

                                        size_t n_match = 0;
                                        while (head_c + n_match < slot.prompt.tokens.size() &&
                                               head_p + n_match < input_tokens.size()       &&
                                               slot.prompt.tokens[head_c + n_match] == input_tokens[head_p + n_match]) {
                                            n_match++;
                                        }

                                        if (n_match >= (size_t) n_cache_reuse) {
                                            SLT_TRC(slot, "reusing chunk with size %zu, shifting KV cache [%zu, %zu) -> [%zu, %zu)\n", n_match, head_c, head_c + n_match, head_p, head_p + n_match);
                                            //for (size_t i = head_p; i < head_p + n_match; i++) {
                                            //    SLT_DBG(slot, "cache token %3zu: %6d '%s'\n", i, prompt_tokens[i], common_token_to_piece(ctx_tgt, prompt_tokens[i]).c_str());
                                            //}

                                            const int64_t kv_shift = (int64_t) head_p - (int64_t) head_c;

                                            common_context_seq_rm (ctx_tgt, slot.id, head_p, head_c);
                                            common_context_seq_add(ctx_tgt, slot.id, head_c, head_c + n_match, kv_shift);

                                            if (ctx_dft) {
                                                common_context_seq_rm (ctx_dft.get(), slot.id, head_p, head_c);
                                                common_context_seq_add(ctx_dft.get(), slot.id, head_c, head_c + n_match, kv_shift);
                                            }

                                            for (size_t i = 0; i < n_match; i++) {
                                                slot.prompt.tokens.set_token(head_p + i, slot.prompt.tokens[head_c + i]);
                                                n_past++;
                                            }

                                            head_c += n_match;
                                            head_p += n_match;
                                        } else {
                                            head_c += 1;
                                        }
                                    }

                                    SLT_DBG(slot, "after context reuse, new n_past = %d\n", n_past);
                                }
                            } else {
                                // if we don't cache the prompt, we have to remove all previous tokens
                                n_past = 0;
                            }

                            llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);

                            // ref: https://github.com/ggml-org/llama.cpp/pull/24110
                            const bool has_new_tokens = (n_past < slot.task->n_tokens());

                            // the largest pos_min required for a checkpoint to be useful
                            const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

                            if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
                                const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                                if (pos_min == -1) {
                                    SLT_ERR(slot, "n_past = %d, slot.prompt.tokens.size() = %d, seq_id = %d, pos_min = %d\n", n_past, (int) slot.prompt.tokens.size(), slot.id, pos_min);
                                    GGML_ABORT("pos_min == -1, but n_past > 0 - should not happen: https://github.com/ggml-org/llama.cpp/pull/13833#discussion_r2116181237");
                                }

                                // when the prompt prefix does not match, print the tokens around the mismatch
                                // this is useful for debugging prompt caching
                                if (slots_debug) {
                                    const int np0 = std::max<int>(n_past - 4, 0);
                                    const int np1 = std::min<int>(n_past + 6, std::min(slot.prompt.tokens.size(), slot.task->tokens.size()));

                                    std::stringstream ss0;
                                    std::stringstream ss1;

                                    std::stringstream st0;
                                    std::stringstream st1;

                                    ss0 << "old: ... ";
                                    ss1 << "new: ... ";

                                    for (int i = np0; i < np1; i++) {
                                        if (i == n_past) {
                                            ss0 << " | ";
                                            ss1 << " | ";
                                        }

                                        {
                                            const auto token = slot.prompt.tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss0 << piece;
                                            st0 << std::setw(8) << token;
                                        }

                                        {
                                            const auto token = slot.task->tokens[i];
                                            const auto piece = token != LLAMA_TOKEN_NULL ? common_token_to_piece(ctx_tgt, token) : "[mtmd]";
                                            ss1 << piece;
                                            st1 << std::setw(8) << token;
                                        }
                                    }

                                    SLT_WRN(slot, "%s\n", ss0.str().c_str());
                                    SLT_WRN(slot, "%s\n", ss1.str().c_str());

                                    SLT_WRN(slot, "%s\n", st0.str().c_str());
                                    SLT_WRN(slot, "%s\n", st1.str().c_str());
                                }

                                if (pos_min >= pos_min_thold) {
                                    // search for a context checkpoint
                                    const auto it = std::find_if(
                                        slot.prompt.checkpoints.rbegin(),
                                        slot.prompt.checkpoints.rend(),
                                        [&](const auto & cur) {
                                            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
                                            SLT_TRC(slot, "checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
                                            // for hybrid/recurrent models (DeltaNet, Mamba), pos_min always equals
                                            // the full sequence length, so the SWA-based pos_min check always fails.
                                            // use pos_max <= pos_next instead to find the most recent valid checkpoint.
                                            if (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) {
                                                return cur.pos_max <= pos_next;
                                            }
                                            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
                                            if (cur.pos_max > pos_next) {
                                                return false;
                                            }
                                            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
                                        }
                                    );

                                    bool do_reset = it == slot.prompt.checkpoints.rend();

                                    if (!do_reset) {
                                        // restore the context checkpoint
                                        // note: keep the raw size-checked restore (instead of it->load_tgt which aborts on
                                        //       mismatch) so a failed restore falls back to a full re-process (do_reset)
                                        const size_t checkpoint_size = it->data_tgt.size();
                                        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, it->data_tgt.data(), checkpoint_size, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                        if (n != checkpoint_size) {
                                            SLT_ERR(slot, "failed to restore context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, (float) checkpoint_size / 1024 / 1024);
                                            do_reset = true;
                                        } else {
                                            // restore the draft-side state, if any (draft-model checkpoints)
                                            it->load_dft(ctx_dft.get(), slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                                            // restore the drafter's speculative state (per-slot spec or shared)
                                            common_speculative_set_state(slot.get_spec(), slot.id, it->data_spec);

                                            pos_next = std::min(pos_next, std::max(it->pos_min + 1, it->pos_max));
                                            n_past = std::min(slot.prompt.tokens.size_up_to_pos(pos_next), (size_t) it->n_tokens);

                                            if (slot.can_speculate() && !it->ring_data.empty()) {
                                                common_speculative_ring_state_load(slot.get_spec(), it->ring_data.data(), it->ring_data.size());
                                            }

                                            SLT_WRN(slot, "restored context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_past = %d, size = %.3f MiB)\n", it->pos_min, it->pos_max, it->n_tokens, n_past, (float) checkpoint_size / 1024 / 1024);
                                        }
                                    }

                                    if (do_reset) {
                                        SLT_TRC(slot, "forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent memory, see %s)\n",
                                                "https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055");
                                        pos_next = 0;
                                        n_past = 0;
                                    }
                                }
                            }

                            {
                                // erase any checkpoints with pos_max > pos_next
                                for (auto it = slot.prompt.checkpoints.begin(); it != slot.prompt.checkpoints.end();) {
                                    const auto & cur = *it;
                                    if (cur.pos_max > pos_next) {
                                        SLT_TRC(slot, "erased invalidated context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64 ", n_swa = %d, pos_next = %d, size = %.3f MiB)\n", cur.pos_min, cur.pos_max, cur.n_tokens, n_swa, pos_next, (float) cur.size() / 1024 / 1024);
                                        it = slot.prompt.checkpoints.erase(it);
                                    } else {
                                        ++it;
                                    }
                                }
                            }
                        }

                        // dynamic VBR: trade a token-trivial reusable prefix for the lossless
                        // full reset (n_past -> 0 makes the seq_rm below a full clear; the
                        // cache empties and the next prepare restores every entry tier)
                        n_past = vbr_reset_on_low_lcp(slot, n_past);

                        // [TAG_PROMPT_LOGITS]
                        if (n_past == slot.task->n_tokens() && n_past > 0) {
                            SLT_WRN(slot, "need to evaluate at least 1 token for each active slot (n_past = %d, task.n_tokens() = %d)\n", n_past, slot.task->n_tokens());
                            n_past--;
                            SLT_WRN(slot, "n_past was set to %d\n", n_past);
                        }

                        slot.n_prompt_tokens_cache = n_past;
                        slot.n_prompt_tokens_processed = 0;

                        slot.prompt.tokens.keep_first(n_past);

                        // this is to signal the client that the request has started processing
                        if (slot.task->params.stream) {
                            if (slot.task->params.return_progress) {
                                // send initial 0% progress update if needed
                                send_partial_response(slot, {}, true);
                            } else {
                                // otherwise, for streaming without progress, signal HTTP to send the headers (i.e. 200 status)
                                send_partial_response(slot, {}, false, true);
                            }
                        }
                    } // end of SLOT_STATE_STARTED

                    if (!slot.can_split()) {
                        // cannot fit the prompt in the current batch - will try next iter
                        if (batch.size() + slot.task->n_tokens() > n_batch) {
                            return;
                        }
                    }

                    const int64_t t_now = ggml_time_us();
                    slot.t_prompt_processing = (t_now - slot.t_start_process_prompt) / 1e3;
                    slot.print_timings_pp();

                    // truncate any tokens that are beyond n_past for this slot
                    const llama_pos p0 = slot.prompt.tokens.pos_next();

                    SLT_TRC(slot, "cached n_tokens = %d, memory_seq_rm [%d, end)\n", slot.prompt.n_tokens(), p0);

                    common_context_seq_rm(ctx_tgt, slot.id, p0, -1);
                    if (ctx_dft) {
                        common_context_seq_rm(ctx_dft.get(), slot.id, p0, -1);
                    }

                    // If using an alora, there may be uncached tokens that come
                    // before the invocation sequence. When this happens, the
                    // tokens before the invocation sequence need to be
                    // processed without the adapter in a separate batch, then
                    // the adapter needs to be enabled for the remaining tokens.
                    if (lora_all_alora(slot.lora) && slot.alora_invocation_start - 1 > slot.prompt.n_tokens()) {
                        SLT_DBG(slot, "processing pre-alora tokens without the adapter (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                        const auto & enabled_loras = lora_get_enabled_ids(slot.lora);
                        GGML_ASSERT(enabled_loras.size() == 1);
                        alora_scale = slot.lora[enabled_loras[0]].scale;
                        slot.lora[enabled_loras[0]].scale = 0.0f;
                        alora_disabled_id = enabled_loras[0];
                    }

                    bool do_checkpoint = params_base.n_ctx_checkpoints > 0;

                    // make checkpoints only for completion tasks
                    do_checkpoint = do_checkpoint && slot.task->type == SERVER_TASK_TYPE_COMPLETION;

                    // make a checkpoint of the parts of the memory that cannot be rolled back.
                    // checkpoints are created only if:
                    // - the model does not support partial sequence removal
                    // - the model uses SWA (and we are not using `swa_full`)
                    do_checkpoint = do_checkpoint && (
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                            ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
                            n_swa > 0 ||
                            llama_model_is_hybrid(model_tgt));

                    bool has_mtmd = false;

                    // swap MTP out of VRAM so mmproj can use GPU for image encoding
                    const bool needs_mmproj_swap = mmproj_gpu_swap && !mmproj_is_on_gpu
                        && slot.prompt.n_tokens() < slot.task->n_tokens()
                        && input_tokens[slot.prompt.n_tokens()] == LLAMA_TOKEN_NULL;

                    if (needs_mmproj_swap) {
                        swap_mtp_to_mmproj_gpu();
                    }

                    // check if we should process the image
                    while (true) {
                        auto cur_token_idx = slot.prompt.n_tokens();
                        if (
                            cur_token_idx >= slot.task->n_tokens() ||
                            input_tokens[cur_token_idx] != LLAMA_TOKEN_NULL // encountered a text token
                        ) {
                            break;
                        }

                        // process the image
                        size_t n_tokens_out = 0;
                        int32_t res = slot.process_mtmd_chunk(cur_token_idx, n_tokens_out);
                        if (res != 0) {
                            SLT_ERR(slot, "failed to process image, res = %d\n", res);
                            send_error(slot, "failed to process image", ERROR_TYPE_SERVER);
                            slot.release();
                            continue;
                        }

                        slot.n_prompt_tokens_processed += n_tokens_out;

                        // add the image chunk to cache
                        {
                            const auto & chunk = input_tokens.find_chunk(cur_token_idx);
                            slot.prompt.tokens.push_back(chunk.get()); // copy
                        }

                        has_mtmd = true;
                    }

                    if (needs_mmproj_swap && mmproj_is_on_gpu) {
                        swap_mmproj_to_mtp();
                    }

                    const auto & spans = slot.task->params.message_spans;
                    const auto last_user_pos = spans.last_user_message_pos();

                    // add prompt tokens for processing in the current batch
                    while (slot.prompt.n_tokens() < slot.task->n_tokens() && batch.size() < n_batch) {
                        // get next token to process
                        llama_token cur_tok = input_tokens[slot.prompt.n_tokens()];
                        if (cur_tok == LLAMA_TOKEN_NULL) {
                            break; // end of text chunk
                        }

                        // if this is an alora request with pre-invocation
                        // tokens that are not cached, we need to stop filling
                        // this batch at those pre-invocation tokens.
                        if (alora_scale > 0 && slot.prompt.n_tokens() == slot.alora_invocation_start - 1) {
                            SLT_DBG(slot, "stop prompt batch filling at (n_tokens = %d, alora_invocation_start = %d)\n", slot.prompt.n_tokens(), slot.alora_invocation_start);
                            break;
                        }

                        // embedding requires all tokens in the batch to be output;
                        // MTP also wants logits at every prompt position so the
                        // streaming hook can mirror t_h_nextn into ctx_dft.
                        add_ok &= batch.add(slot.id,
                            cur_tok,
                            slot.prompt.tokens.pos_next(),
                            slot.need_embd());
                        slot.prompt.tokens.push_back(cur_tok);

                        slot.n_prompt_tokens_processed++;

                        // stop the prompt batch exactly before a user message
                        if (spans.is_user_start(slot.prompt.n_tokens())) {
                            break;
                        }

                        // process the last few tokens of the prompt separately in order to allow for a checkpoint to be created.
                        // create checkpoints that many tokens before the end of the prompt:
                        //  - 4 + n_ubatch
                        //  - 4
                        // ref: https://github.com/ggml-org/llama.cpp/pull/20288
                        if (do_checkpoint) {
                            static const int checkpoint_offsets[] = {4 + n_ubatch, 4};

                            bool should_break = false;
                            for (int offset : checkpoint_offsets) {
                                const int n_last = std::min(n_batch, offset);
                                if (slot.task->n_tokens() == slot.prompt.n_tokens() + n_last) {
                                    should_break = true;
                                    break;
                                }
                            }
                            if (should_break) {
                                break;
                            }
                        }
                    }

                    // the number of tokens added to the batch for the current slot
                    const auto n_tokens_cur = batch.size() - n_tokens_prev;

                    const auto n_tokens_start = slot.prompt.n_tokens() - n_tokens_cur;

                    const bool near_prompt_end = slot.task->n_tokens() < slot.prompt.n_tokens() + n_ubatch;

                    const bool is_user_start = spans.is_user_start(n_tokens_start);
                    const bool is_last_user_message = n_tokens_start == last_user_pos;

                    // entire prompt has been processed
                    if (slot.prompt.n_tokens() == slot.task->n_tokens()) {
                        slot.state = SLOT_STATE_DONE_PROMPT;

                        GGML_ASSERT(batch.size() > 0);

                        // extract the logits only for the last token
                        batch.set_output(batch.size() - 1, true);

                        slot.n_decoded = 0;
                        slot.i_batch   = batch.size() - 1;

                        slot.init_sampler();
                    } else {
                        // skip ordinary mid-prompt checkpoints, unless the batch starts a user
                        // message or we are near the end of the prompt
                        if (!is_user_start && !near_prompt_end) {
                            do_checkpoint = false;
                        }
                    }

                    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
                    const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);

                    // no need for empty or small checkpoints
                    // for hybrid/recurrent models, lower the checkpoint threshold so short prompts also get checkpointed
                    const int checkpoint_min_tokens = (llama_model_is_recurrent(model_tgt) || llama_model_is_hybrid(model_tgt)) ? 4 : 64;
                    do_checkpoint = do_checkpoint && (pos_min >= 0 && slot.prompt.n_tokens() >= checkpoint_min_tokens);

                    // nothing to checkpoint yet
                    // TODO: is this check needed?
                    if (do_checkpoint && pos_min < 0) {
                        do_checkpoint = false;
                    }

                    // do not checkpoint after mtmd chunks
                    do_checkpoint = do_checkpoint && !has_mtmd;

                    // no need to create checkpoints that are too close together, unless it's the last user message
                    do_checkpoint = do_checkpoint && (slot.prompt.checkpoints.empty() || is_last_user_message || n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
                    SLT_DBG(slot, "main/do_checkpoint = %s, pos_min = %d, pos_max = %d\n", do_checkpoint ? "yes" : "no", pos_min, pos_max);

                    // note: we create the checkpoint before calling llama_decode(), so the current batch is not
                    //       yet processed and therefore it is not part of the checkpoint.
                    if (do_checkpoint) {
                        while (slot.prompt.checkpoints.size() >= (size_t) params_base.n_ctx_checkpoints) {
                            // make room for the new checkpoint, if needed
                            const auto & cur = slot.prompt.checkpoints.front();

                            SLT_WRN(slot,
                                    "erasing old context checkpoint (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                    ", size = %.3f MiB)\n",
                                    cur.pos_min, cur.pos_max, cur.n_tokens, (float) cur.data_tgt.size() / 1024 / 1024);

                            slot.prompt.checkpoints.erase(slot.prompt.checkpoints.begin());
                        }

                        const size_t checkpoint_size =
                            llama_state_seq_get_size_ext(ctx_tgt, slot.id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        auto & cur = slot.prompt.checkpoints.emplace_back();
                        cur.pos_min  = pos_min;
                        cur.pos_max  = pos_max;
                        cur.n_tokens = slot.prompt.n_tokens() - n_tokens_cur;
                        cur.data_tgt.resize(checkpoint_size);

                        llama_state_seq_get_data_ext(ctx_tgt, cur.data_tgt.data(), checkpoint_size, slot.id,
                                                     LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                        // save DFlash ring buffer alongside the recurrent state checkpoint
                        if (slot.can_speculate()) {
                            size_t ring_size = common_speculative_ring_state_size(slot.get_spec());
                            if (ring_size > 0) {
                                cur.ring_data.resize(ring_size);
                                common_speculative_ring_state_save(slot.get_spec(), cur.ring_data.data(), ring_size);
                            }
                        }

                        SLT_WRN(slot,
                                "created context checkpoint %d of %d (pos_min = %d, pos_max = %d, n_tokens = %" PRId64
                                ", size = %.3f MiB)\n",
                                (int) slot.prompt.checkpoints.size(), params_base.n_ctx_checkpoints, cur.pos_min,
                                cur.pos_max, cur.n_tokens, (float) cur.size() / 1024.0 / 1024.0);
                    }
                }

                if (!slot_batched) {
                    slot_batched = &slot;
                }
            });
        }

        // DFlash: enable tape recording if any slot has draft backup (needs tape replay for rollback);
        // turned off in post_cycle() so recording stays active across ALL sub-batches.
        // Only when GPU tape replay is available (see llama_dflash_tape_replay_available) —
        // otherwise rollback re-decodes the accepted tokens (partial-accept branch below).
        dflash_tape_active = needs_reeval
            && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH
            && dflash_tape_ok
            && std::any_of(slots.begin(), slots.end(), [](const server_slot & s) { return s.has_draft_backup; });
        if (dflash_tape_active) {
            llama_set_tape_recording(ctx_tgt, true);
        }

        // allow multi-seq batching when the batch is pure TG (no prompt tokens).
        // This lets concurrent slots' verify tokens be processed in a single
        // multi-seq ubatch instead of N sequential per-seq ubatches.
        // (force_split_seq is restored in post_cycle())
        can_batch_multiseq = (n_tg_tokens == batch.size() && n_tg_tokens > 0
            && params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH);
        if (can_batch_multiseq) {
            llama_set_force_split_seq(ctx_tgt, false);
        }

        // dynamic VBR: mid-decode pressure — generation can cross a page/budget boundary long
        // after launch. Same floor-gated reclaim before the cache pays with tiers (idle slots
        // only; every generating slot is processing and untouchable).
        if (batch.size() > 0) {
            vbr_reclaim_before_degrade(-1, (uint32_t) batch.size(), "decode");
        }
    }

    // returns true = success ; false = retry with smaller batch size
    // throw std::runtime_error on fatal error
    bool decode(int32_t & n_batch, int32_t off, llama_batch & batch_view) {
        SRV_DBG("n_batch (effective) = %d, off = %d\n", n_batch, off);

        auto & slot_batched      = batch.slot_batched;
        auto & alora_scale       = batch.alora_scale;
        auto & alora_disabled_id = batch.alora_disabled_id;

        // TODO @ngxson : alora handling is too messy, need to refactor it to be more clear and maintainable
        if (slot_batched) {
            // apply lora, only need to do it once per batch
            common_set_adapter_lora(ctx_tgt, slot_batched->lora);

            // if the lora is temporarily disabled for an alora, re-enable it
            // for next time
            if (alora_scale > 0.0f) {
                SRV_DBG("re-enabling alora with scale %f\n", alora_scale);
                slot_batched->lora[alora_disabled_id].scale = alora_scale;
            }

            llama_set_embeddings(ctx_tgt, slot_batched->need_embd());
        }

        if (batch.size() == 0) {
            // note: normally unreachable — update_slots() skips decode() for an empty batch and
            //       handles the empty-batch warn/abort (with the diffusion exemption) itself
            SRV_WRN("%s", "no tokens to decode\n");

            if (++n_empty_consecutive > 3) {
                GGML_ABORT("fatal error - please provide logs and repro in %s\n", "https://github.com/ggml-org/llama.cpp/pull/20277");
            }

            return true; // nothing to decode
        } else {
            n_empty_consecutive = 0;
        }

        // Ensure causal attention is on for prompt eval (diffusion draft toggles it off)
        llama_set_causal_attn(ctx_tgt, true);

        const int64_t t_verify_start = ggml_time_us();
        const int ret = llama_decode(ctx_tgt, batch_view);
        const int64_t t_verify_elapsed = ggml_time_us() - t_verify_start;
        t_verify_total += t_verify_elapsed;
        SRV_DBG("  verify ubatch: %d tok, %.1fms (%.2fms/tok)\n",
                batch_view.n_tokens, t_verify_elapsed / 1e3, t_verify_elapsed / 1e3 / std::max(1, batch_view.n_tokens));

        metrics.on_decoded(slots);

        if (ret != 0) {
            {
                std::string err;

                if (n_batch == 1 && ret == 1) {
                    // TODO: try to terminate only the largest active slot/sequence and continue with the rest
                    //       need to remove the tokens from the current batch too
                    err = "Context size has been exceeded.";
                }

                if (ret == -1) {
                    err = "Invalid input batch.";
                }

                if (ret < -1) {
                    // TODO: update slot state based on llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
                    err = "Compute error.";
                }

                // TODO: handle ret == 2 (abort) when we start aborting

                if (!err.empty()) {
                    SRV_ERR("%s off = %d, n_batch = %d, ret = %d\n", err.c_str(), off, n_batch, ret);

                    for (auto & slot : slots) {
                        if (slot.is_processing()) {
                            send_error(slot, err);
                            slot.release();

                            // note: it's complicated to keep track of how much of the current batch has been
                            //       processed before the error occurred, so we simply clear the entire context
                            slot.prompt_clear(false);
                        }
                    }

                    // stop, do not retry with smaller batch size
                    throw std::runtime_error(err);
                }
            }

            // retry with half the batch size to try to find a free slot in the KV cache
            if (!try_clear_idle_slots()) {
                n_batch /= 2;
            }

            SRV_WRN("failed to find free space in the KV cache, retrying with smaller batch size, off = %d, n_batch = %d, ret = %d\n", off, n_batch, ret);

            return false; // retry with the updated n_batch
        }

        // TODO: avoid restoring the draft context and re-evaluating the drafted tokens when not needed [TAG_SPEC_AVOID_DRAFT_REEVAL]
        //       for now, always re-evaluate for simplicity
        //       ref: https://github.com/ggml-org/llama.cpp/pull/22728#issuecomment-4400925384
        if (!common_speculative_process(spec.get(), batch_view)) {
            SRV_ERR("%s", "failed to process speculative batch\n");

            // TODO: handle error
            throw std::runtime_error("failed to process speculative batch");
        }

        // DFlash: flush captured hidden states into the ring buffer before
        // the next llama_decode resets the capture buffer. This lets
        // checkpoint-split prefill preserve all hidden states incrementally.
        for (auto & slot : slots) {
            if ((slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) && slot.can_speculate()) {
                common_speculative_flush_prefill(slot.get_spec());
            }
        }

        // handle `n_cmpl > 1` tasks - when the main prompt is processed, activate all child tasks too
        for (auto & slot : slots) {
            if (slot.state == SLOT_STATE_DONE_PROMPT && slot.task->is_parent()) {
                std::vector<server_slot *> children;
                for (auto & other : slots) {
                    if (other.state == SLOT_STATE_WAIT_OTHER && slot.task->id == other.task->id_parent) {
                        children.push_back(&other);
                    }
                }

                // all children slots should already launched by launch_slots_with_parent_task()
                // copy state to the child slots
                for (auto & child : children) {
                    SLT_TRC(slot, " - copying state to child %d\n", child->id);

                    GGML_ASSERT(child->state == SLOT_STATE_WAIT_OTHER);

                    slot.copy_state_to(*child);
                    child->state = SLOT_STATE_DONE_PROMPT;
                }
            }
        }

        return true;
    }

    void post_decode(int32_t n_batch_tokens, int32_t off, llama_batch & batch_view) {
        // for checking if a given batch index is inside batch_view
        auto is_inside_view = [&](int32_t idx) {
            return idx >= off && idx < off + n_batch_tokens;
        };

        // TODO @ngxson : it's tricky to make sub-batch compatible with common_sampler_sample_and_accept_n,
        // so for now we will throw an error in this case: https://github.com/ggml-org/llama.cpp/issues/24840
        iterate(slots, [&](server_slot & slot) {
            for (auto & i : slot.spec_i_batch) {
                if (!is_inside_view(i)) {
                    throw std::runtime_error(string_format("speculative batch index %d is not inside the current sub-batch [%d, %d)", i, off, off + n_batch_tokens));
                }
            }
        });

        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        iterate(slots, [&](server_slot & slot) {
            // optionally send prompt processing progress
            if (slot.state == SLOT_STATE_PROCESSING_PROMPT || slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    send_partial_response(slot, {}, true);
                }
            }

            if (!is_inside_view(slot.i_batch)) {
                // the required token not in this sub-batch, skip
                return;
            }

            if (slot.state == SLOT_STATE_DONE_PROMPT) {
                if (slot.task->type == SERVER_TASK_TYPE_EMBEDDING) {
                    // prompt evaluated for embedding
                    send_embedding(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                if (slot.task->type == SERVER_TASK_TYPE_RERANK) {
                    send_rerank(slot, batch_view);
                    slot.release();
                    slot.i_batch = -1;
                    return;
                }

                GGML_ASSERT(slot.task->need_sampling());

                // prompt evaluated for next-token prediction
                slot.state = SLOT_STATE_GENERATING;

                if (slot.diff_self_spec) {
                    float * logits = llama_get_logits(ctx_tgt);
                    int32_t nv = llama_vocab_n_tokens(vocab);
                    slot.diff_prev_logits.resize(nv);
                    std::memcpy(slot.diff_prev_logits.data(), logits, nv * sizeof(float));

                    // Extract previous assistant response tokens for repetition penalty.
                    // Find all <|im_start|>assistant blocks, take the second-to-last
                    // (the last is the current generation position).
                    slot.diff_prev_assistant_tokens.clear();
                    {
                        const auto & toks = slot.prompt.tokens.get_text_tokens();
                        const llama_token im_start = 10;  // <|im_start|>
                        const llama_token im_end   = 11;  // <|im_end|>
                        const llama_token tok_ass  = 1503; // "ass"
                        const llama_token tok_ist  = 19464; // "istant"
                        int n = (int)toks.size();

                        // Find all assistant block start positions
                        std::vector<int> asst_starts;
                        for (int j = 0; j + 2 < n; j++) {
                            if (toks[j] == im_start && toks[j+1] == tok_ass && toks[j+2] == tok_ist) {
                                asst_starts.push_back(j);
                            }
                        }

                        // If there are 2+ assistant blocks, the second-to-last has
                        // the previous response
                        if (asst_starts.size() >= 2) {
                            int prev_start = asst_starts[asst_starts.size() - 2];
                            int content_start = prev_start + 4; // <|im_start|> ass istant \n
                            // Find <|im_end|> after content
                            int end = content_start;
                            while (end < n && toks[end] != im_end) end++;
                            for (int j = content_start; j < end; j++) {
                                slot.diff_prev_assistant_tokens.push_back(toks[j]);
                            }
                        }
                    }

                    SLT_DBG(slot, "diff prefill: n_prompt=%d, prev_asst_tokens=%d\n",
                            slot.prompt.n_tokens(), (int)slot.diff_prev_assistant_tokens.size());
                    slot.t_start_generation = ggml_time_us();
                    slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                    metrics.on_prompt_eval(slot);

                    slot.i_batch = -1;
                    return;
                }

                if (slot.can_speculate()) {
                    if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                        llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    }
                    if (slot.spec) {
                        // fork per-slot speculative state
                        common_speculative_begin(slot.get_spec(), slot.prompt.tokens.get_text_tokens());
                    } else if (spec) {
                        // upstream shared multi-seq speculative state
                        common_speculative_begin(spec.get(), slot.id, slot.prompt.tokens.get_text_tokens());
                    }
                }
            } else if (slot.state != SLOT_STATE_GENERATING) {
                return;
            }

            if (slot.can_speculate() && !slot.spec_draft.empty()) {
                return; // sample using speculative decoding
            }

            // shifted according to the current sub-batch
            const int tok_idx = slot.i_batch - off;

            llama_token id;
            {
                scoped_timer timer(t_sampl, n_sampl);
                id = common_sampler_sample(slot.smpl.get(), slot.ctx_tgt, tok_idx);
            }

            slot.i_batch = -1;

            common_sampler_accept(slot.smpl.get(), id, true);

            // update DFlash hidden state ring buffer with the decoded token's hidden states.
            // Skip on the first sample after prompt: common_speculative_begin() above already
            // populated the ring with all prefill hiddens. The capture buffer at this point
            // still holds prefill hiddens (no new decode happened), so ring_write(1) here would
            // append a stale duplicate at the position that should later hold `id`'s hidden —
            // silently corrupting the drafter's cross-attention context on every subsequent
            // verify. Fires correctly on the fallback non-spec path during generation
            // (draft too small → single-token decode), where slot.sampled was just decoded.
            if (slot.can_speculate() && slot.n_decoded > 0) {
                if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                }
                llama_tokens batch_tokens = { id };
                common_speculative_update_logits(slot.get_spec(), ctx_tgt, batch_tokens, 1);
            }

            // here we have synchronized the llama_context (due to the sampling above), so we can do time measurement
            const int64_t t_now = ggml_time_us();

            slot.n_decoded += 1;

            if (slot.n_decoded == 1) {
                slot.t_start_generation = t_now;
                slot.t_print_last = t_now;
                slot.n_decoded_last = 0;
                slot.t_prompt_processing = (slot.t_start_generation - slot.t_start_process_prompt) / 1e3;
                metrics.on_prompt_eval(slot);
            }

            slot.t_token_generation = std::max<int64_t>(1, t_now - slot.t_start_generation) / 1e3;

            completion_token_output result;
            result.tok          = id;
            result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
            result.prob         = 1.0f; // TODO: set it here instead of doing inside populate_token_probs

            if (slot.task->params.sampling.n_probs > 0) {
                populate_token_probs(slot, result, slot.task->params.post_sampling_probs, params_base.special, tok_idx);
            }

            if (!process_token(result, slot)) {
                // release slot because of stop condition
                slot.print_timings();
                send_final_response(slot);
                metrics.on_prediction(slot);
                slot.release();

                return;
            }

            slot.print_timings_tg();
        });

        // speculative decoding - main model sample and accept
        const int64_t t_accept_start = ggml_time_us();
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.can_speculate() || slot.spec_draft.empty()) {
                continue;
            }

            // save the original draft size
            const size_t n_draft = slot.spec_draft.size();

            // defensive (#74): never feed a desynced spec state into
            // common_sampler_sample_and_accept_n (it hard-asserts on
            // idxs.size() == draft.size() + 1). The batch build site always appends
            // the current cycle's (n_draft + 1) contiguous indices LAST, so any
            // excess can only be a stale prefix leaked from an earlier cycle.
            // Unreachable with the empty-draft gate in place — this converts any
            // future recurrence from a server abort into a loud warning.
            if (slot.spec_i_batch.size() != n_draft + 1) {
                SLT_WRN(slot, "spec state desync (spec_i_batch = %zu, spec_draft = %zu) - recovering\n",
                        slot.spec_i_batch.size(), n_draft);

                if (slot.spec_i_batch.size() > n_draft + 1) {
                    // drop the stale prefix, keep the current cycle's indices
                    slot.spec_i_batch.erase(slot.spec_i_batch.begin(), slot.spec_i_batch.end() - (n_draft + 1));
                } else {
                    // should be unreachable: verify only as many draft tokens as
                    // there are logit indices (n_draft keeps the original size so
                    // the prompt rollback below stays consistent)
                    slot.spec_draft.resize(slot.spec_i_batch.empty() ? 0 : slot.spec_i_batch.size() - 1);
                    if (slot.spec_i_batch.empty()) {
                        slot.spec_i_batch.push_back(batch.size() > 0 ? batch.size() - 1 : 0);
                    }
                }
            }

            // the accepted tokens from the speculation
            const auto ids = common_sampler_sample_and_accept_n(slot.smpl.get(), ctx_tgt, slot.spec_i_batch, slot.spec_draft);

            // update DFlash hidden state ring + CopySpec prompt window with accepted tokens.
            // Must run BEFORE rollback (matches speculative-simple ordering) and BEFORE clearing
            // slot.spec_draft so batch_tokens reflects the full verification batch [id_last, drafts].
            {
                if (params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                }
                llama_tokens batch_tokens;
                batch_tokens.push_back(slot.sampled);
                batch_tokens.insert(batch_tokens.end(), slot.spec_draft.begin(), slot.spec_draft.end());
                common_speculative_update_logits(slot.get_spec(), ctx_tgt, batch_tokens, (int) ids.size());
            }

            slot.spec_i_batch.clear();
            slot.spec_draft.clear();

            const int64_t t_current = ggml_time_us();

            slot.t_token_generation = std::max<int64_t>(1, t_current - slot.t_start_generation) / 1e3;

            // update how many tokens out of those tested were accepted
            slot.n_draft_accepted += ids.size() - 1;
            slot.n_draft_verif_steps += 1;

            // per-position acceptance histogram (/metrics)
            if (slot.n_accepted_per_pos.empty()) {
                slot.n_accepted_per_pos.resize(common_speculative_n_max(&params_base.speculative), 0);
            }
            for (size_t i = 0; i < ids.size() - 1 && i < slot.n_accepted_per_pos.size(); ++i) {
                slot.n_accepted_per_pos[i]++;
            }

            // notify the shared (upstream) speculative state, if this slot uses it
            if (!slot.spec && spec) {
                common_speculative_accept(spec.get(), slot.id, ids.size() - 1);
            }

            // add accepted tokens to the prompt
            slot.prompt.tokens.keep_first(slot.prompt.n_tokens() - n_draft);
            slot.prompt.tokens.insert(llama_tokens(ids.begin(), ids.end() - 1));

            if (slot.has_draft_backup) {
                const llama_seq_id seq_backup = slot.seq_id_backup;
                const bool all_accepted = (ids.size() == n_draft + 1);

                const bool is_dflash = params_base.speculative.type() == COMMON_SPECULATIVE_TYPE_DFLASH;
                if (is_dflash) {
                    llama_dflash_set_active_slot(ctx_tgt, slot.id);
                    llama_clear_tree_parent_ids(ctx_tgt);
                }

                if (is_dflash && !all_accepted && dflash_tape_ok) {
                    // lossless GPU tape replay of the accepted tokens' GDN state updates
                    llama_dflash_rollback(ctx_tgt, slot.id, seq_backup, slot.n_tokens_before_draft, (int) ids.size());
                } else {
                    // no tape recorded (see dflash_tape_active) or non-DFlash drafter:
                    // restore from backup and re-decode the accepted tokens (exact, replay-free)
                    auto * mem = llama_get_memory(ctx_tgt);

                    if (all_accepted) {
                        llama_memory_seq_rm(mem, seq_backup, -1, -1);
                        llama_memory_seq_rm(mem, slot.id, slot.prompt.tokens.pos_next(), -1);
                    } else {
                        const int n_past_before = slot.n_tokens_before_draft;

                        llama_memory_seq_rm(mem, slot.id, n_past_before, -1);
                        llama_memory_seq_cp(mem, seq_backup, slot.id, -1, -1);
                        llama_memory_seq_rm(mem, seq_backup, -1, -1);

                        const int n_reeval = slot.prompt.n_tokens() - n_past_before;
                        if (n_reeval > 0) {
                            llama_batch batch_reeval = llama_batch_init(n_reeval, 0, 1);
                            const auto & toks = slot.prompt.tokens.get_text_tokens();
                            for (int j = n_past_before; j < slot.prompt.n_tokens(); ++j) {
                                common_batch_add(batch_reeval, toks[j], j, { slot.id }, false);
                            }
                            llama_decode(ctx_tgt, batch_reeval);
                            llama_batch_free(batch_reeval);
                        }
                    }
                }

                slot.has_draft_backup = false;
                slot.seq_id_backup = -1;
            } else {
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), slot.id, slot.prompt.tokens.pos_next(), -1);
            }

            common_speculative_rollback_dft(slot.get_spec(), slot.id, slot.prompt.n_tokens(), (uint16_t)(ids.size() - 1));

            for (size_t i = 0; i < ids.size(); ++i) {
                completion_token_output result;

                result.tok          = ids[i];
                result.text_to_send = common_token_to_piece(slot.ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                result.prob         = 1.0f; // set later

                // TODO: set result.probs

                slot.n_decoded += 1;

                if (!process_token(result, slot)) {
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();

                    break;
                }
            }

            slot.print_timings_tg();

            SLT_DBG(slot, "accepted %d/%d draft tokens, new n_tokens = %d\n", (int) ids.size() - 1, (int) n_draft, slot.prompt.n_tokens());
        }
        t_accept_total += ggml_time_us() - t_accept_start;
    }

    // fork: once-per-update_slots() work that must run after ALL sub-batches:
    // diffusion self-speculation, the spec-cycle report, DFlash tape-off and
    // force_split_seq restore (kept out of post_decode(), which runs per sub-batch)
    void post_cycle() {
        auto accept_special_token = [&](server_slot & slot, llama_token token) {
            return params_base.special ||
                slot.task->params.sampling.preserved_tokens.find(token) != slot.task->params.sampling.preserved_tokens.end();
        };

        // diffusion self-speculation — runs independently of the main batch
        for (auto & slot : slots) {
            if (!slot.diff_self_spec || slot.state != SLOT_STATE_GENERATING) {
                continue;
            }

            const int32_t n_vocab = llama_vocab_n_tokens(vocab);
            llama_memory_t mem = llama_get_memory(ctx_tgt);
            const int32_t k = slot.diff_draft_length;

            const float temperature = slot.task->params.sampling.temp;
            std::mt19937 rng(slot.task->params.sampling.seed != LLAMA_DEFAULT_SEED
                ? slot.task->params.sampling.seed : std::random_device{}());
            std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

            auto argmax = [&](const float * l) -> llama_token {
                return (llama_token)(std::max_element(l, l + n_vocab) - l);
            };

            auto suppress_think = [&](float * l) {
                if (slot.diff_think_open_id  != LLAMA_TOKEN_NULL) l[slot.diff_think_open_id]  = -INFINITY;
                if (slot.diff_think_close_id != LLAMA_TOKEN_NULL) l[slot.diff_think_close_id] = -INFINITY;
            };

            // Temperature sampling: softmax(logits/T) → categorical sample
            auto sample_temp = [&](const float * logits) -> llama_token {
                float max_val = *std::max_element(logits, logits + n_vocab);
                float sum = 0;
                std::vector<float> probs(n_vocab);
                for (int32_t v = 0; v < n_vocab; v++) {
                    probs[v] = std::exp((logits[v] - max_val) / temperature);
                    sum += probs[v];
                }
                float r = uniform01(rng) * sum;
                float cumsum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    cumsum += probs[v];
                    if (r <= cumsum) return (llama_token)v;
                }
                return (llama_token)(n_vocab - 1);
            };

            auto sample_adjusted = [&](const float * p_logits, const float * q_logits) -> llama_token {
                float p_max = *std::max_element(p_logits, p_logits + n_vocab);
                float q_max = *std::max_element(q_logits, q_logits + n_vocab);
                float p_sum = 0, q_sum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    p_sum += std::exp((p_logits[v] - p_max) / temperature);
                    q_sum += std::exp((q_logits[v] - q_max) / temperature);
                }
                std::vector<float> adj(n_vocab);
                float sum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    float pv = std::exp((p_logits[v] - p_max) / temperature) / p_sum;
                    float qv = std::exp((q_logits[v] - q_max) / temperature) / q_sum;
                    adj[v] = std::max(0.0f, pv - qv);
                    sum += adj[v];
                }
                if (sum <= 0) return sample_temp(p_logits);
                float r = uniform01(rng) * sum;
                float cumsum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    cumsum += adj[v];
                    if (r <= cumsum) return (llama_token)v;
                }
                return (llama_token)(n_vocab - 1);
            };

            // Acceptance probability: min(1, p(x)/q(x)) where p=target, q=draft
            auto accept_prob = [&](const float * p_logits, const float * q_logits, llama_token tok) -> float {
                float p_max = *std::max_element(p_logits, p_logits + n_vocab);
                float q_max = *std::max_element(q_logits, q_logits + n_vocab);
                float p_sum = 0, q_sum = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    p_sum += std::exp((p_logits[v] - p_max) / temperature);
                    q_sum += std::exp((q_logits[v] - q_max) / temperature);
                }
                float p_tok = std::exp((p_logits[tok] - p_max) / temperature) / p_sum;
                float q_tok = std::exp((q_logits[tok] - q_max) / temperature) / q_sum;
                return std::min(1.0f, p_tok / std::max(1e-10f, q_tok));
            };

            // Cross-turn: penalize tokens from previous assistant response
            std::unordered_map<llama_token, int> prev_asst_freq;
            for (auto tok : slot.diff_prev_assistant_tokens) {
                prev_asst_freq[tok]++;
            }
            std::vector<llama_token> gen_history;

            const float rep_penalty = 1.3f;
            auto apply_rep_penalty = [&](float * l) {
                for (const auto & [tok, freq] : prev_asst_freq) {
                    if (l[tok] > 0) l[tok] /= rep_penalty;
                    else             l[tok] *= rep_penalty;
                }
            };
            auto detect_loop = [&]() -> bool {
                int n = (int)gen_history.size();

                // Short-period exact match (periods 4-32)
                if (n >= 16) {
                    for (int period = 4; period <= std::min(32, n/2); period++) {
                        bool match = true;
                        for (int i = 0; i < period; i++) {
                            if (gen_history[n - 1 - i] != gen_history[n - 1 - i - period]) {
                                match = false;
                                break;
                            }
                        }
                        if (match) {
                            SLT_INF(slot, "diff loop detected (period=%d), stopping generation\n", period);
                            return true;
                        }
                    }
                }

                // Long-period structural loop: any 6-gram appearing 6+ times in last 300 tokens
                const int struct_window = std::min(n, 300);
                if (struct_window >= 60) {
                    int start = n - struct_window;
                    std::unordered_map<uint64_t, int> ng6_freq;
                    for (int i = start; i <= n - 6; i++) {
                        uint64_t h = 0;
                        for (int j = 0; j < 6; j++) {
                            h = h * 100003ULL + (uint64_t)gen_history[i + j];
                        }
                        ng6_freq[h]++;
                    }
                    for (const auto & [h, freq] : ng6_freq) {
                        if (freq >= 6) {
                            SLT_INF(slot, "diff structural loop detected (6-gram freq=%d), stopping generation\n", freq);
                            return true;
                        }
                    }
                }
                return false;
            };

            std::vector<llama_token> draft(k);
            const bool use_temp = temperature > 0;
            std::vector<float> saved_draft_logits;
            if (use_temp) {
                saved_draft_logits.resize(k * n_vocab);
            }
            bool stopped = false;
            int64_t t_draft_us = 0, t_verify_us = 0, t_bonus_us = 0, t_loop_us = 0, n_cycles = 0;

            while (!stopped) {
                const int64_t t_loop_start = ggml_time_us();
                const int32_t committed = slot.prompt.n_tokens();
                const int32_t draft_len = std::min(k, (int32_t)(slot.n_ctx - committed - 1));
                if (draft_len <= 0) {
                    slot.truncated      = true;
                    slot.stop           = STOP_TYPE_LIMIT;
                    slot.has_next_token = false;
                    slot.print_timings();
                    send_final_response(slot);
                    metrics.on_prediction(slot);
                    slot.release();
                    break;
                }

                // DRAFT: bidirectional — single pass, all masks denoised in parallel
                llama_set_causal_attn(ctx_tgt, false);
                {
                    llama_batch bd = llama_batch_init(draft_len, 0, 1);
                    bd.n_tokens = draft_len;
                    for (int32_t j = 0; j < draft_len; j++) {
                        bd.token[j]     = slot.diff_mask_token_id;
                        bd.pos[j]       = committed + j;
                        bd.n_seq_id[j]  = 1;
                        bd.seq_id[j][0] = slot.id;
                        bd.logits[j]    = 1;
                    }
                    int64_t t0 = ggml_time_us();
                    int ret = llama_decode(ctx_tgt, bd);
                    t_draft_us += ggml_time_us() - t0;
                    llama_batch_free(bd);
                    if (ret != 0) {
                        SRV_ERR("diffusion draft decode failed, ret = %d\n", ret);
                        stopped = true;
                        break;
                    }
                }

                {
                    float * logits = llama_get_logits(ctx_tgt);
                    for (int32_t j = 0; j < draft_len; j++) {
                        suppress_think(logits + j * n_vocab);
                        apply_rep_penalty(logits + j * n_vocab);
                    }
                    if (use_temp) {
                        std::memcpy(saved_draft_logits.data(), logits, draft_len * n_vocab * sizeof(float));
                        for (int32_t j = 0; j < draft_len; j++) {
                            draft[j] = sample_temp(logits + j * n_vocab);
                        }
                    } else {
                        for (int32_t j = 0; j < draft_len; j++) {
                            draft[j] = argmax(logits + j * n_vocab);
                        }
                    }
                }

                llama_memory_seq_rm(mem, slot.id, committed, committed + draft_len);

                // VERIFY: causal — single pass with draft tokens
                llama_set_causal_attn(ctx_tgt, true);
                {
                    llama_batch bv = llama_batch_init(draft_len, 0, 1);
                    bv.n_tokens = draft_len;
                    for (int32_t j = 0; j < draft_len; j++) {
                        bv.token[j]     = draft[j];
                        bv.pos[j]       = committed + j;
                        bv.n_seq_id[j]  = 1;
                        bv.seq_id[j][0] = slot.id;
                        bv.logits[j]    = 1;
                    }
                    int64_t t1 = ggml_time_us();
                    int ret = llama_decode(ctx_tgt, bv);
                    t_verify_us += ggml_time_us() - t1;
                    llama_batch_free(bv);
                    if (ret != 0) {
                        SRV_ERR("diffusion verify decode failed, ret = %d\n", ret);
                        stopped = true;
                        break;
                    }
                }

                float * verify_logits = llama_get_logits(ctx_tgt);
                suppress_think(slot.diff_prev_logits.data());
                apply_rep_penalty(slot.diff_prev_logits.data());
                for (int32_t j = 0; j < draft_len; j++) {
                    suppress_think(verify_logits + j * n_vocab);
                    apply_rep_penalty(verify_logits + j * n_vocab);
                }

                int32_t n_accept = 0;
                llama_token bonus;

                if (use_temp) {
                    // Rejection sampling: accept draft[j] with prob min(1, p(d)/q(d))
                    float ap = accept_prob(slot.diff_prev_logits.data(), saved_draft_logits.data(), draft[0]);
                    if (uniform01(rng) < ap) {
                        n_accept = 1;
                        for (int32_t j = 0; j < draft_len - 1; j++) {
                            ap = accept_prob(verify_logits + j * n_vocab,
                                           saved_draft_logits.data() + (j + 1) * n_vocab, draft[j + 1]);
                            if (uniform01(rng) < ap) {
                                n_accept++;
                            } else {
                                break;
                            }
                        }
                    }
                    if (n_accept == draft_len) {
                        bonus = sample_temp(verify_logits + (draft_len - 1) * n_vocab);
                    } else if (n_accept == 0) {
                        bonus = sample_adjusted(slot.diff_prev_logits.data(), saved_draft_logits.data());
                    } else {
                        bonus = sample_adjusted(verify_logits + (n_accept - 1) * n_vocab,
                                              saved_draft_logits.data() + n_accept * n_vocab);
                    }
                } else {
                    if (argmax(slot.diff_prev_logits.data()) == draft[0]) {
                        n_accept = 1;
                        for (int32_t j = 0; j < draft_len - 1; j++) {
                            if (argmax(verify_logits + j * n_vocab) == draft[j + 1]) {
                                n_accept++;
                            } else {
                                break;
                            }
                        }
                    }
                    bonus = (n_accept == 0)
                        ? argmax(slot.diff_prev_logits.data())
                        : argmax(verify_logits + (n_accept - 1) * n_vocab);
                }

                if (n_accept < draft_len) {
                    llama_memory_seq_rm(mem, slot.id, committed + n_accept, committed + draft_len);
                }

                // BONUS: decode one token — produces new prev_logits for next cycle
                {
                    llama_batch bb = llama_batch_init(1, 0, 1);
                    bb.n_tokens     = 1;
                    bb.token[0]     = bonus;
                    bb.pos[0]       = committed + n_accept;
                    bb.n_seq_id[0]  = 1;
                    bb.seq_id[0][0] = slot.id;
                    bb.logits[0]    = 1;
                    int64_t t2 = ggml_time_us();
                    int ret = llama_decode(ctx_tgt, bb);
                    t_bonus_us += ggml_time_us() - t2;
                    llama_batch_free(bb);
                    if (ret != 0) {
                        SRV_ERR("diffusion bonus decode failed, ret = %d\n", ret);
                        stopped = true;
                        break;
                    }
                    std::memcpy(slot.diff_prev_logits.data(), llama_get_logits(ctx_tgt), n_vocab * sizeof(float));
                    suppress_think(slot.diff_prev_logits.data());
                }

                n_cycles++;

                // output accepted draft tokens + bonus
                for (int32_t j = 0; j < n_accept && !stopped; j++) {
                    completion_token_output result;
                    result.tok          = draft[j];
                    result.text_to_send = common_token_to_piece(ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f;
                    slot.n_decoded += 1;
                    slot.prompt.tokens.push_back(draft[j]);
                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();
                        stopped = true;
                    }
                }
                if (!stopped) {
                    completion_token_output result;
                    result.tok          = bonus;
                    result.text_to_send = common_token_to_piece(ctx_tgt, result.tok, accept_special_token(slot, result.tok));
                    result.prob         = 1.0f;
                    slot.n_decoded += 1;
                    slot.prompt.tokens.push_back(bonus);
                    if (!process_token(result, slot)) {
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();
                        stopped = true;
                    }
                }

                if (!stopped) {
                    for (int32_t j = 0; j < n_accept; j++) {
                        gen_history.push_back(draft[j]);
                    }
                    gen_history.push_back(bonus);
                    if (detect_loop()) {
                        slot.stop           = STOP_TYPE_LIMIT;
                        slot.has_next_token = false;
                        slot.print_timings();
                        send_final_response(slot);
                        metrics.on_prediction(slot);
                        slot.release();
                        stopped = true;
                    }

                    slot.n_draft_accepted += n_accept;
                    slot.t_token_generation = std::max<int64_t>(1, ggml_time_us() - slot.t_start_generation) / 1e3;
                    t_loop_us += ggml_time_us() - t_loop_start;
                    SLT_DBG(slot, "diff cycle %lld: %d/%d accepted, bonus=%d, pos=%d\n",
                            (long long)n_cycles, n_accept, draft_len, bonus, slot.prompt.n_tokens());
                }
            }

            if (n_cycles > 0) {
                int64_t t_decode_us = t_draft_us + t_verify_us + t_bonus_us;
                int64_t t_overhead_us = t_loop_us - t_decode_us;
                SLT_INF(slot, "diff stats: %lld cycles, draft=%.1fms verify=%.1fms bonus=%.1fms overhead=%.1fms (%.1f/%.1f/%.1f/%.1f ms/cycle)\n",
                        (long long)n_cycles,
                        t_draft_us/1e3, t_verify_us/1e3, t_bonus_us/1e3, t_overhead_us/1e3,
                        t_draft_us/1e3/n_cycles, t_verify_us/1e3/n_cycles, t_bonus_us/1e3/n_cycles, t_overhead_us/1e3/n_cycles);
            }

            llama_set_causal_attn(ctx_tgt, true);
        }

        // --- profiling: log per-cycle breakdown ---
        if (n_slots_drafted > 0) {
            const int64_t t_cycle_total = ggml_time_us() - t_cycle_start;
            const int64_t t_other = t_cycle_total - t_draft_total - t_verify_total - t_accept_total;
            SRV_DBG("spec cycle (%d slots): draft=%.1fms verify=%.1fms accept=%.1fms other=%.1fms total=%.1fms\n",
                    n_slots_drafted,
                    t_draft_total / 1e3, t_verify_total / 1e3, t_accept_total / 1e3,
                    t_other / 1e3, t_cycle_total / 1e3);
        }

        // turn off DFlash tape recording after all sub-batches — was turned on
        // before the sub-batch for loop. Placing it outside the loop (vs inside,
        // after the first decode) keeps recording active across all sub-batches,
        // which matters when multiple slots share one pass and the combined
        // verify batch spans more than one ubatch.
        if (dflash_tape_active) {
            llama_set_tape_recording(ctx_tgt, false);
        }

        // restore force_split_seq for the next cycle (prompt batches need it)
        if (can_batch_multiseq) {
            llama_set_force_split_seq(ctx_tgt, true);
        }

        SRV_DBG("%s", "run slots completed\n");
    }

    int get_slot_n_ctx() {
        return slots.back().n_ctx;
    }

    server_response_reader get_response_reader() {
        return server_response_reader(queue_tasks, queue_results, HTTP_POLLING_SECONDS);
    }
};

//
// server_context (public API)
//

server_context::server_context() : impl(new server_context_impl()) {}
server_context::~server_context() = default;

bool server_context::load_model(common_params & params) {
    return impl->load_model(params);
}

void server_context::start_loop() {
    auto & params = impl->params_base;
    impl->queue_tasks.start_loop(params.sleep_idle_seconds * 1000);
}

void server_context::terminate() {
    impl->queue_tasks.terminate();
}

llama_context * server_context::get_llama_context() const {
    return impl->ctx_tgt;
}

server_response_reader server_context::get_response_reader() {
    return impl->get_response_reader();
}

server_context_meta server_context::get_meta() const {
    auto bos_id = llama_vocab_bos(impl->vocab);
    auto eos_id = llama_vocab_eos(impl->vocab);
    auto bos_token_str = bos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, bos_id, true) : "";
    auto eos_token_str = eos_id != LLAMA_TOKEN_NULL ? common_token_to_piece(impl->ctx_tgt, eos_id, true) : "";

    const char * ftype_name = llama_ftype_name(llama_model_ftype(impl->model_tgt));

    return server_context_meta {
        /* build_info             */ std::string(llama_build_info()),
        /* model_name             */ impl->model_name,
        /* model_aliases          */ impl->model_aliases,
        /* model_tags             */ impl->model_tags,
        /* model_path             */ impl->params_base.model.path,
        /* has_mtmd               */ impl->mctx != nullptr,
        /* has_inp_image          */ impl->chat_params.allow_image,
        /* has_inp_audio          */ impl->chat_params.allow_audio,
        /* has_inp_video          */ impl->chat_params.allow_video,
        /* json_ui_settings       */ impl->json_ui_settings,
        /* slot_n_ctx             */ impl->get_slot_n_ctx(),
        /* vbr_enabled            */ impl->params_base.vbr_enabled(),
        /* vbr_dynamic            */ impl->params_base.vbr_dynamic(),
        /* vbr_type_k             */ impl->params_base.vbr_cache_type_k,
        /* vbr_type_v             */ impl->params_base.vbr_cache_type_v,
        /* vbr_min_bits           */ impl->params_base.vbr_min_bits_value,
        /* vbr_capacity_bits      */ impl->params_base.vbr_capacity_bits,
        /* vbr_selected_bpv       */ impl->params_base.vbr_selected_bpv,
        /* vbr_selected_kld       */ impl->params_base.vbr_selected_kld,
        /* vbr_vram_budget_bytes  */ impl->params_base.vbr_vram_budget_bytes,
        /* vbr_selected_family    */ impl->params_base.vbr_selected_family,
        /* vbr_selected_policy    */ impl->params_base.vbr_selected_policy,
        /* vbr_selected_schedule  */ impl->params_base.vbr_selected_schedule,
        /* pooling_type           */ llama_pooling_type(impl->ctx_tgt),

        /* chat_params            */ impl->chat_params,
        /* chat_template_caps     */ common_chat_templates_get_caps(impl->chat_params.tmpls.get()),

        /* bos_token_str          */ bos_token_str,
        /* eos_token_str          */ eos_token_str,
        /* fim_pre_token          */ llama_vocab_fim_pre(impl->vocab),
        /* fim_sub_token          */ llama_vocab_fim_suf(impl->vocab),
        /* fim_mid_token          */ llama_vocab_fim_mid(impl->vocab),
        /* fim_pad_token          */ llama_vocab_fim_pad(impl->vocab),
        /* fim_rep_token          */ llama_vocab_fim_rep(impl->vocab),
        /* fim_sep_token          */ llama_vocab_fim_sep(impl->vocab),

        /* logit_bias_eog         */ impl->params_base.sampling.logit_bias_eog,

        /* model_vocab_type       */ llama_vocab_type(impl->vocab),
        /* model_vocab_n_tokens   */ llama_vocab_n_tokens(impl->vocab),
        /* model_n_ctx_train      */ llama_model_n_ctx_train(impl->model_tgt),
        /* model_n_embd_inp       */ llama_model_n_embd(impl->model_tgt),
        /* model_n_params         */ llama_model_n_params(impl->model_tgt),
        /* model_size             */ llama_model_size(impl->model_tgt),
        /* model_ftype            */ ftype_name,
    };
}



// generator-like API for HTTP response generation
// may have bypass_sleep = true if the task does not use ctx_server
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_queue & queue_tasks, server_response & queue_results, int sleep_idle_seconds, bool bypass_sleep = false)
            : rd(queue_tasks, queue_results, HTTP_POLLING_SECONDS) {
        // fast path in case sleeping is disabled
        bypass_sleep |= sleep_idle_seconds < 0;
        if (!bypass_sleep) {
            queue_tasks.wait_until_no_sleep();
        }
    }
    ~server_res_generator() override {
        // cleanup() must run while rd is still alive (rd is destroyed after this body returns)
        if (spipe) {
            spipe->cleanup();
        }
    }
    void stop() override {
        rd.stop();
    }
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

void server_context::set_state_callback(server_state_callback_t callback) {
    impl->callback_state = std::move(callback);
    impl->queue_tasks.on_sleeping_state([this](bool sleeping) {
        if (sleeping) {
            impl->callback_state(SERVER_STATE_SLEEPING, {});
        }
        // for sleeping == false, event is emitted by load_model()
    });
}

//
// server_routes
//

std::unique_ptr<server_res_generator> server_routes::handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type) {
    GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

    auto res = create_response();
    auto completion_id = gen_chatcmplid();
    auto & rd = res->rd;
    auto & params = this->params;

    int32_t sse_ping_interval = params.sse_ping_interval;

    try {
        std::vector<server_task> tasks;

        const auto & prompt = data.at("prompt");
        // TODO: this log can become very long, put it behind a flag or think about a more compact format
        //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

        if (!params.path_prompts_log_dir.empty()) {
            const auto file_path = std::filesystem::path(params.path_prompts_log_dir) / string_format("%012" PRId64 ".txt", ggml_time_ms());
            std::ofstream f(file_path);
            if (f) {
                f << (prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());
            } else {
                SRV_ERR("failed to create %s\n", file_path.string().c_str());
            }
        }

        // process prompt
        std::vector<server_tokens> inputs;

        if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
            // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
            inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
        } else {
            // Everything else, including multimodal completions.
            inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        }

        // tasks.reserve(inputs.size()); // TODO: this is inaccurate due to child tasks

        // message delimiters for checkpointing
        auto delimiters = common_chat_msg_delimiters_parse(json_value(data, "message_delimiters", json::array()));
        delimiters.tokenize(ctx_server.vocab);

        for (size_t i = 0; i < inputs.size(); i++) {
            server_task task = server_task(type);

            task.id = rd.get_new_id();

            task.tokens = std::move(inputs[i]);
            task.params = server_schema::eval_llama_cmpl_schema(
                    ctx_server.vocab,
                    params,
                    meta->slot_n_ctx,
                    meta->logit_bias_eog,
                    data);

            task.params.message_spans = task.tokens.find_message_spans(delimiters);

            task.id_slot = json_value(data, "id_slot", -1);
            sse_ping_interval = task.params.sse_ping_interval;

            // OAI-compat
            task.params.res_type          = res_type;
            task.params.oaicompat_cmpl_id = completion_id;
            task.params.oaicompat_model   = meta->model_name;

            // prepare child tasks
            if (task.params.n_cmpl > 1) {
                int n_children = task.params.n_cmpl - 1;
                for (int j = 0; j < n_children; j++) {
                    task.add_child(task.id, rd.get_new_id());
                }
            }

            tasks.push_back(std::move(task));
        }

        rd.post_tasks(std::move(tasks));
    } catch (const std::exception & e) {
        res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool stream = json_value(data, "stream", false);

    if (!stream) {
        // non-stream, wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            json arr = json::array();
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                arr.push_back(res->to_json());
            }
            GGML_ASSERT(!arr.empty() && "empty results");
            if (arr.size() == 1) {
                // if single request, return single object instead of array
                res->ok(arr[0]);
            } else if (res_type == TASK_RESPONSE_TYPE_OAI_CHAT || res_type == TASK_RESPONSE_TYPE_OAI_CMPL) {
                // if multiple results in OAI format, we need to re-format them
                json & choices = arr[0]["choices"];
                for (size_t i = 1; i < arr.size(); i++) {
                    choices.push_back(std::move(arr[i]["choices"][0]));
                }
                res->ok(arr[0]);
            } else {
                // multi-results, non-OAI compat
                res->ok(arr);
            }
        }
    } else {
        // in streaming mode, the first error must be treated as non-stream response
        // this is to match the OAI API behavior
        // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
        auto first_result = rd.next(req.should_stop);
        if (first_result == nullptr) {
            GGML_ASSERT(req.should_stop());
            return res; // connection is closed
        }

        if (first_result->is_error()) {
            res->error(first_result->to_json());
            return res;
        }

        GGML_ASSERT(
            dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr ||
            dynamic_cast<server_task_result_cmpl_final*>  (first_result.get()) != nullptr
        );

        // next responses are streamed
        // to be sent immediately
        json first_result_json = first_result->to_json();
        if (first_result_json == nullptr) {
            res->data = ""; // simply send HTTP headers and status code
        } else if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
            res->data = format_anthropic_sse(first_result_json);
        } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
            res->data = format_oai_resp_sse(first_result_json);
        } else {
            res->data = format_oai_sse(first_result_json);
        }
        res->status = 200;
        res->content_type = "text/event-stream";
        res->next = [res_this = res.get(), res_type, sse_ping_interval, &req](std::string & output) -> bool {
            static auto format_error = [](task_response_type res_type, const json & res_json) {
                if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                    return format_anthropic_sse({
                        {"event", "error"},
                        {"data", res_json},
                    });
                } else {
                    return format_oai_sse(json {{ "error", res_json }});
                }
            };

            auto effective_should_stop = stream_aware_should_stop(res_this, req.should_stop);

            try {
                if (effective_should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    switch (res_type) {
                        case TASK_RESPONSE_TYPE_NONE:
                        case TASK_RESPONSE_TYPE_OAI_RESP:
                        case TASK_RESPONSE_TYPE_ANTHROPIC:
                            output = "";
                            break;

                        default:
                            output = "data: [DONE]\n\n";
                            break;
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                bool timeout = false;
                int64_t start_time = ggml_time_ms();
                auto result = rd.next([&timeout, &start_time, sse_ping_interval, &effective_should_stop]() {
                    if (effective_should_stop()) {
                        return true; // should_stop condition met
                    } else if (sse_ping_interval > 0 && ggml_time_ms() - start_time > (int64_t)sse_ping_interval * 1000) {
                        timeout = true;
                        return true; // timeout
                    }
                    return false;
                });

                if (timeout) {
                    // some clients may time out (e.g. undici) will time out if no data is received for a while, so we need to send a ping to keep the connection alive
                    SRV_DBG("%s", "sending SSE ping\n");
                    output = ":\n\n";
                    return true;
                }

                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    GGML_ASSERT(effective_should_stop());
                    return false; // should_stop condition met
                }

                // send the results
                if (result->is_error()) {
                    json res_json = result->to_json();
                    output = format_error(res_type, res_json);
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    json res_json = result->to_json();
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else if (res_type == TASK_RESPONSE_TYPE_OAI_RESP) {
                        output = format_oai_resp_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;

            } catch (const std::exception & e) {
                json error_json = format_error_response(e.what(), ERROR_TYPE_SERVER);
                output = format_error(res_type, error_json);

                // terminate on exception
                return false;
            }
        };
    }

    // attach a producer pipe to the response when X-Conversation-Id is present.
    // the pipe mirrors SSE chunks into the ring buffer and wires up the cancel hook.
    stream_session_attach_pipe(*res, req.headers);

    return res;
}

std::unique_ptr<server_res_generator> server_routes::create_response(bool bypass_sleep) {
    return std::make_unique<server_res_generator>(queue_tasks, queue_results, params.sleep_idle_seconds, bypass_sleep);
}

server_routes::server_routes(const common_params & params, server_context & ctx_server)
        : params(params),
          ctx_server(*ctx_server.impl),
          queue_tasks(ctx_server.impl->queue_tasks),
          queue_results(ctx_server.impl->queue_results) {
    init_routes();
}

void server_routes::init_routes() {
    // IMPORTANT: all lambda functions must start with create_response()
    // this is to ensure that the server_res_generator can handle sleeping case correctly

    this->get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        res->ok({{"status", "ok"}});
        return res;
    };

    this->get_metrics = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            },{
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / std::max((float) res_task->n_decode_total, 1.f)}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->status = 200;
        res->data = prometheus.str();
        return res;
    };

    this->get_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = res->rd.get_new_id();
            res->rd.post_task(std::move(task), true); // high-priority task
        }

        // get the result
        auto result = res->rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto * res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    this->post_slots = [this](const server_http_req & req) {
        auto res = create_response();
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");

        int id_slot;
        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        }
        if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        }
        if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        }

        res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
        return res;
    };

    this->get_props = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        task_params tparams;
        tparams.sampling = params.sampling;
        json default_generation_settings_for_props = json {
            { "params", tparams.to_json(true) },
            { "n_ctx",  meta->slot_n_ctx },
        };

        std::string tmpl_default = common_chat_templates_source(meta->chat_params.tmpls.get(), "");
        std::string tmpl_tools   = common_chat_templates_source(meta->chat_params.tmpls.get(), "tool_use");

        json props = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 params.n_parallel },
            { "model_alias",                 meta->model_name },
            { "model_path",                  meta->model_path },
            { "vbr",                         server_vbr_meta_json(meta.get()) },
            { "modalities",                  json {
                {"vision", meta->has_inp_image},
                {"video",  meta->has_inp_video},
                {"audio",  meta->has_inp_audio},
            } },
            { "media_marker",                get_media_marker() },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "ui",                          params.ui },
            { "ui_settings",                 meta->json_ui_settings },
            { "chat_template",               tmpl_default },
            { "chat_template_caps",          meta->chat_template_caps },
            { "bos_token",                   meta->bos_token_str },
            { "eos_token",                   meta->eos_token_str },
            { "build_info",                  meta->build_info },
            { "is_sleeping",                 queue_tasks.is_sleeping() },
            { "cors_proxy_enabled",          params.ui_mcp_proxy },
        };
        if (params.use_jinja) {
            if (!tmpl_tools.empty()) {
                props["chat_template_tool_use"] = tmpl_tools;
            }
        }
        res->ok(props);
        return res;
    };

    this->post_props = [this](const server_http_req &) {
        auto res = create_response();
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    this->post_infill = [this](const server_http_req & req) {
        auto res = create_response();
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            params.n_batch,
            params.n_predict,
            meta->slot_n_ctx,
            params.spm_infill,
            tokenized_prompts[0].get_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    this->post_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_NONE);
    };

    this->post_completions_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    this->post_chat_completions = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_chat_completions_tok = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    this->post_control = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        const std::string cmpl_id = json_value(body, "id", std::string());
        const std::string action  = json_value(body, "action", std::string());
        if (cmpl_id.empty()) {
            res->error(format_error_response("missing completion id", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (action != "reasoning_end") {
            res->error(format_error_response("unknown control action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_CONTROL);
            task.id              = rd.get_new_id();
            task.params.control_cmpl_id = cmpl_id;
            task.params.control_action  = action;
            rd.post_task(std::move(task));
        }

        auto result = rd.next(req.should_stop);
        if (!result) {
            GGML_ASSERT(req.should_stop());
            return res;
        }
        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }
        res->ok(result->to_json());
        return res;
    };

    this->post_responses_oai = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_responses_to_chatcmpl(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: OpenAI Responses -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_responses_tok_oai = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_OAI_RESP);
    };

    this->post_transcriptions_oai = [this](const server_http_req & req) {
        auto res = create_response();

        if (!meta->has_mtmd || !meta->chat_params.allow_audio) {
            res->error(format_error_response("The current model does not support audio input.", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::vector<raw_buffer> files;
        json body = convert_transcriptions_to_chatcmpl(
            json::parse(req.body),
            meta->chat_params.tmpls.get(),
            req.files,
            files);
        SRV_DBG("%s\n", "Request converted: OpenAI Transcriptions -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_OAI_ASR);
    };

    this->post_anthropic_messages = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files;
        json body = server_chat_convert_anthropic_to_oai(json::parse(req.body));
        SRV_DBG("%s\n", "Request converted: Anthropic -> OpenAI Chat Completions");
        SRV_DBG("converted request: %s\n", body.dump().c_str());
        json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        return handle_completions_impl(
            req,
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    this->post_anthropic_count_tokens = [this](const server_http_req & req) {
        return handle_count_tokens(ctx_server.vocab, ctx_server.mctx, req, TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    // same with handle_chat_completions, but without inference part
    this->post_apply_template = [this](const server_http_req & req) {
        auto res = create_response();
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    this->get_models = [this](const server_http_req &) {
        auto res = create_response(true);

        // this endpoint can be accessed during sleeping
        // the next LOC is to avoid someone accidentally use ctx_server
        bool ctx_server; // do NOT delete this line
        GGML_UNUSED(ctx_server);

        json models = {
            {"models", {
                {
                    {"name",  meta->model_name},
                    {"model", meta->model_name},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", meta->has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                get_model_info(),
            }}
        };

        res->ok(models);
        return res;
    };

    this->post_tokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.vocab, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    this->post_detokenize = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.vocab, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    this->post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    this->post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    this->post_rerank = [this](const server_http_req & req) {
        auto res = create_response();
        if (!params.embedding || params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        auto & rd = res->rd;
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model_tgt, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = rd.get_new_id();
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            meta->model_name,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    this->get_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_GET_LORA);
            task.id = rd.get_new_id();
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_get_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

    this->post_lora_adapters = [this](const server_http_req & req) {
        auto res = create_response();
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        auto & rd = res->rd;
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = rd.get_new_id();
            task.set_lora = parse_lora_request(body);
            rd.post_task(std::move(task));
        }

        // get the result
        auto result = rd.next(req.should_stop);
        if (!result) {
            // connection was closed
            GGML_ASSERT(req.should_stop());
            return res;
        }

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };
}

// the /props and /models "vbr" object — built in ONE place. selected_kld/schedule are
// static-ladder measurements: null under the dynamic controller (like realized_bpv), not 0.
static json server_vbr_meta_json(const server_context_meta * meta) {
    return json {
        {"enabled",           meta->vbr_enabled},
        {"dynamic",           meta->vbr_dynamic},
        {"type_k",            meta->vbr_type_k},
        {"type_v",            meta->vbr_type_v},
        {"floor_bpv",         meta->vbr_min_bits},
        {"capacity_floor_bpv", meta->vbr_capacity_bits},
        // realized bits/value is a fixed number only for static schedules; under the
        // dynamic runtime controller it varies with occupancy — null, not a fiction
        {"realized_bpv",      meta->vbr_dynamic ? json() : json(meta->vbr_capacity_bits)},
        {"selected_family",   meta->vbr_selected_family},
        {"selected_policy",   meta->vbr_selected_policy},
        {"selected_bpv",      meta->vbr_dynamic ? json() : json(meta->vbr_selected_bpv)},
        {"selected_kld",      meta->vbr_dynamic ? json() : json(meta->vbr_selected_kld)},
        {"selected_schedule", meta->vbr_dynamic ? json() : json(meta->vbr_selected_schedule)},
        {"vram_budget_bytes", meta->vbr_vram_budget_bytes},
    };
}

json server_routes::get_model_info() const {
    return json {
        {"id",       meta->model_name},
        {"aliases",  meta->model_aliases},
        {"tags",     meta->model_tags},
        {"object",   "model"},
        {"created",  std::time(0)},
        {"owned_by", "llamacpp"},
        {"meta",     {
            {"vocab_type",  meta->model_vocab_type},
            {"n_vocab",     meta->model_vocab_n_tokens},
            {"n_ctx",       meta->slot_n_ctx},
            {"n_ctx_train", meta->model_n_ctx_train},
            {"vbr",         server_vbr_meta_json(meta.get())},
            {"n_embd",      meta->model_n_embd_inp},
            {"n_params",    meta->model_n_params},
            {"size",        meta->model_size},
            {"ftype",       meta->model_ftype},
        }},
    };
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_save(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_restore(const server_http_req & req, int id_slot) {
    auto res = create_response();
    const json request_data = json::parse(req.body);
    std::string filename = request_data.at("filename");
    if (!fs_validate_filename(filename)) {
        res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }
    std::string filepath = params.slot_save_path + filename;

    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot  = id_slot;
        task.slot_action.filename = filename;
        task.slot_action.filepath = filepath;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_slots_erase(const server_http_req & req, int id_slot) {
    auto res = create_response();
    auto & rd = res->rd;
    {
        server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
        task.id = rd.get_new_id();
        task.slot_action.id_slot = id_slot;
        rd.post_task(std::move(task));
    }

    auto result = rd.next(req.should_stop);
    if (!result) {
        // connection was closed
        GGML_ASSERT(req.should_stop());
        return res;
    }

    if (result->is_error()) {
        res->error(result->to_json());
        return res;
    }

    GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
    res->ok(result->to_json());
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    if (!params.embedding) {
        res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
        return res;
    }

    if (res_type != TASK_RESPONSE_TYPE_NONE && meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
        res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    const json body = json::parse(req.body);

    // for the shape of input/content, see tokenize_input_prompts()
    json prompt;
    if (body.count("input") != 0) {
        prompt = body.at("input");
    } else if (body.contains("content")) {
        res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
        prompt = body.at("content");
    } else {
        res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
        return res;
    }

    bool use_base64 = false;
    if (body.count("encoding_format") != 0) {
        const std::string & format = body.at("encoding_format");
        if (format == "base64") {
            use_base64 = true;
        } else if (format != "float") {
            res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
    for (const auto & tokens : tokenized_prompts) {
        // this check is necessary for models that do not add BOS token to the input
        if (tokens.empty()) {
            res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    }

    int embd_normalize = params.embd_normalize;
    if (body.count("embd_normalize") != 0) {
        embd_normalize = body.at("embd_normalize");
        if (meta->pooling_type == LLAMA_POOLING_TYPE_NONE) {
            SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", meta->pooling_type);
        }
    }

    // create and queue the task
    json responses = json::array();
    auto & rd = res->rd;
    {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < tokenized_prompts.size(); i++) {
            server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

            task.id     = rd.get_new_id();
            task.tokens = std::move(tokenized_prompts[i]);

            // OAI-compat
            task.params.res_type = res_type;
            task.params.embd_normalize = embd_normalize;

            tasks.push_back(std::move(task));
        }
        rd.post_tasks(std::move(tasks));
    }

    // wait for the results
    auto all_results = rd.wait_for_all(req.should_stop);

    // collect results
    if (all_results.is_terminated) {
        return res; // connection is closed
    } else if (all_results.error) {
        res->error(all_results.error->to_json());
        return res;
    } else {
        for (auto & res : all_results.results) {
            GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
            responses.push_back(res->to_json());
        }
    }

    // write JSON response
    json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? format_embeddings_response_oaicompat(body, meta->model_name, responses, use_base64)
        : json(responses);
    res->ok(root);
    return res;
}

std::unique_ptr<server_res_generator> server_routes::handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const server_http_req & req, task_response_type res_type) {
    auto res = create_response();
    std::vector<raw_buffer> files;
    json body = json::parse(req.body);
    bool is_oai = false;

    switch (res_type) {
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            {
                is_oai = true;
            } break;
        case TASK_RESPONSE_TYPE_OAI_RESP:
            {
                is_oai = true;
                body = server_chat_convert_responses_to_chatcmpl(body);
            } break;
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            {
                body = server_chat_convert_anthropic_to_oai(body);
            } break;
        default:
            res->error(format_error_response("invalid res_type", ERROR_TYPE_INVALID_REQUEST));
            return res;
    }

    json body_parsed = oaicompat_chat_params_parse(
            body,
            meta->chat_params,
            files);
    json prompt = body_parsed.at("prompt");
    // SRV_DBG("prompt = %s\n", prompt.dump().c_str());

    // TODO @ngxson : refactor this code block, move this to server-common and reuse it in other places
    size_t n_tokens;
    if (mctx != nullptr) {
        if (!prompt.is_string()) {
            throw std::runtime_error("for mtmd, input prompt must be a string.");
        }
        n_tokens = process_mtmd_prompt(mctx, prompt.get<std::string>(), files, true).size();
    } else {
        n_tokens = tokenize_mixed(vocab, prompt, true, true).size();
    }

    json response = {{"input_tokens", static_cast<int64_t>(n_tokens)}};
    if (is_oai) {
        response["object"] = "response.input_tokens";
    }
    res->ok(response);
    return res;
}
