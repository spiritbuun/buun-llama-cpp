#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);

// Whether backend_apply() will run at least the first element of this initialized
// sampler chain. Later unsupported elements are allowed and run on the CPU.
bool llama_sampler_chain_has_backend_prefix(const llama_sampler * chain);

// Whether every sampler in the active backend prefix preserves an explicit
// candidate-token domain. Unknown and index-producing samplers fail closed so
// callers can fall back to dense full-vocabulary logits.
bool llama_sampler_chain_supports_candidates(const llama_sampler * chain);
