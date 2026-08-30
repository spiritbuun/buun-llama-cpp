#pragma once

#include "llama.h"

#include "common.h"

#include <string>
#include <vector>

// common_sampler extends llama_sampler with additional functionality:
//
//  - grammar support
//  - custom sampler logic based on the parameters
//  - history of the last accepted tokens
//  - performance metrics
//
// This goal is to have a common implementation of the sampling logic shared across the examples.
// For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
// complex (top-k, top-p, etc).
//
// Another example is related to the grammar. In general, the grammar constraints applied on the full
// vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
// token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
// grammar constraints are applied to the full vocabulary and the token is resampled.
//
// The common_sampler also maintains a container with the last accepted tokens. In the future, this can
// be moved into the core llama library.
//
// For convenience, the common_sampler also maintains a container with the current candidate tokens.
// This can be used to access the probabilities of the rest of the non-sampled tokens.
//
// TODO: measure grammar performance
//

struct common_sampler;

// llama_sampler API overloads

// note: can mutate params in some cases
struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params);

void common_sampler_free(struct common_sampler * gsmpl);

// if is_generated is true, the token is accepted by the sampling chain, the reasoning budget sampler, and the grammar sampler
void                    common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated);
void                    common_sampler_reset (struct common_sampler * gsmpl);
struct common_sampler * common_sampler_clone (struct common_sampler * gsmpl);
void                    common_sampler_copy  (const struct common_sampler * src, struct common_sampler * dst);

// arguments can be nullptr to skip printing
void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl);

// get the underlying llama_sampler_chain
struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl);

// extended sampling implementation:
//
// - set logits
// - apply the configured sampler chain
// - check if the token fits the grammar (if any)
// - if not: resample by first applying the grammar constraints and then sampling again (slower path)
//
// if grammar_first is true, the grammar is applied before the samplers (slower)
// useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
//
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first = false);

// Sample from one complete raw vocabulary-logit row without requiring a
// llama_context output slot. This is used by authenticated prompt-cache
// frontiers whose KV state and terminal logits were restored together.
llama_token common_sampler_sample_from_logits(
        struct common_sampler * gsmpl,
        const float * logits,
        size_t n_logits,
        bool grammar_first = false);

// True when the sampler initialized for this request is exactly equivalent to
// selecting the token with the largest unmodified model logit.
bool common_sampler_raw_argmax_exact(const struct common_sampler * gsmpl);

// generalized version of common_sampler_sample
//
// will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
// if the sampler disagrees at some point, we stop and return the accepted tokens up to now
//
//      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
//
// is equivalent to
//
//      common_sampler_sample(gsmpl, ctx, idx);
//      common_sampler_accept(gsmpl, token, true);
//
// requires: idxs.size() == draft.size() + 1
//
// returns at least 1 token, up to idxs.size()
//
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first = false);

// Advance sampler state from already-selected target tokens, stopping after
// the first disagreement with the draft. Requires sampled.size() ==
// draft.size() + 1 and returns at least one token.
std::vector<llama_token> common_sampler_accept_draft(
        struct common_sampler * gsmpl,
        const llama_tokens & sampled,
        const llama_tokens & draft);

// assume idxs == [ 0, 1, 2, ..., draft.size() ]
std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first = false);

// Distribution-preserving speculative verification for a sparse proposal q.
// The first q_covered rows describe draft[0..q_covered), each as top_k token
// IDs followed by its normalized probabilities. Returns false without touching
// sampler state when the configured target sampler is not supported.
bool common_sampler_sample_and_accept_n_q(
        struct common_sampler *      gsmpl,
        struct llama_context *       ctx,
        const std::vector<int> &     idxs,
        const llama_tokens &         draft,
        int32_t                      top_k,
        const std::vector<int32_t> & candidate_ids,
        const std::vector<float> &   q_rows,
        size_t                       q_covered,
        std::vector<llama_token> &   result);

// Pure distribution helpers used by sparse-q speculative verification. The
// proposal arrays contain one row of unique token IDs and non-negative masses;
// neither p nor q has to be pre-normalized.
double common_sampler_speculative_acceptance_probability(
        const llama_token_data_array * p,
        llama_token                    proposed,
        const int32_t *                q_ids,
        const float *                  q_probs,
        size_t                         q_size);

llama_token common_sampler_speculative_sample_residual(
        const llama_token_data_array * p,
        const int32_t *                q_ids,
        const float *                  q_probs,
        size_t                         q_size,
        double                         uniform_draw);

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl);

// returns true if grammar is actively constraining output (for lazy grammars, only after trigger fired)
bool common_sampler_grammar_is_active(const struct common_sampler * gsmpl);

// force the reasoning budget sampler (if any) to begin forcing its end sequence now.
bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl);

// helpers

// access the internal list of current candidate tokens
// if do_sort == true, the candidates are guaranteed to be sorted afterwards (in descending order of probability)
// the .sorted flag of the result indicates whether the returned candidates are sorted
llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort);

// get the last accepted token
llama_token common_sampler_last(const struct common_sampler * gsmpl);

// print the sampler chain into a string
std::string common_sampler_print(const struct common_sampler * gsmpl);

// get a string representation of the last accepted tokens
std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n);

char        common_sampler_type_to_chr(enum common_sampler_type cnstr);
std::string common_sampler_type_to_str(enum common_sampler_type cnstr);

std::vector<enum common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampler_type> common_sampler_types_from_chars(const std::string & chars);

llama_sampler * llama_sampler_init_llg(const llama_vocab * vocab,
                const char * grammar_kind, const char * grammar_data);

struct common_sampler_deleter {
    void operator()(common_sampler * s) { common_sampler_free(s); }
};

typedef std::unique_ptr<common_sampler, common_sampler_deleter> common_sampler_ptr;
