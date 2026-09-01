#pragma once

#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"

#include "json.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>

struct server_context_impl; // private implementation
class server_cache_control_authority;

enum class server_speculative_decode_terminal {
    success,
    preserve_hard_seal,
    ordinary_ret_error,
    retry,
    reset_committed_then_throw,
};

server_speculative_decode_terminal
server_speculative_decode_terminal_resolve(
    int decode_result,
    bool hard_seal_terminal,
    bool single_token_batch,
    bool selected_exception,
    bool speculative_ok) noexcept;

struct server_committed_decode_reset_test_result {
    bool processing_prompt_cleared = false;
    bool processing_family_cleared = false;
    bool idle_prompt_preserved = false;
};

server_committed_decode_reset_test_result
server_committed_decode_reset_for_test();

struct server_slot_frontier_logits_test_result {
    bool round_trip = false;
    bool primary_binding_mutation_refused = false;
    bool runtime_family_mutation_refused = false;
    bool model_family_mutation_refused = false;
    bool model_nonsemantic_variation_matches = false;
    bool unresolved_context_fallback_changes_identity = false;
    bool explicit_context_override_ignores_training_context = false;
    bool sequence_geometry_changes_identity = false;
    bool padding_equivalent_geometry_matches = false;
    bool control_content_mutation_refused = false;
    bool missing_family_receipt_disables_hot = false;
    bool adapter_mutation_refused = false;
    bool token_count_mutation_refused = false;
    bool next_position_mutation_refused = false;
    bool token_digest_mutation_refused = false;
    bool vocabulary_mutation_refused = false;
    bool logits_mutation_refused = false;
    bool serialized_payload_mutation_refused = false;
    bool nonfinite_logits_refused = false;
    bool torn_companion_refused = false;
    bool missing_companion_is_cold = false;
    bool destination_slot_rebound = false;
    bool destination_epoch_rebound = false;
    bool source_process_epoch_not_reused = false;
    bool exact_hit_skips_decode = false;
    bool missing_capability_replays = false;
    bool decode_failure_refuses_publication = false;
    bool decode_failure_clears_slot = false;
    bool rollback_decode_allows_cold_save = false;
    bool partial_decode_requires_reset = false;
    bool aligned_without_logits_allows_cold_save = false;
    bool missing_memory_requires_reset = false;
    bool multi_token_gap_requires_reset = false;
    bool consumed_logits_release_capacity = false;
};

server_slot_frontier_logits_test_result
server_slot_frontier_logits_for_test();

struct server_vbr_occupied_quarantine_reset_result {
    bool replay_preserved_prefix = false;
    bool replay_preserved_slot = false;
    bool quarantined = false;
    bool retained_prefix_zero = false;
    bool prompt_cleared = false;
    bool family_cleared = false;
};

server_vbr_occupied_quarantine_reset_result
server_vbr_occupied_quarantine_reset_for_test();

// TEST-ONLY door. It constructs the private server_slot, resolves a
// scheduler family token, exercises the real no-restore cache load, and then
// verifies that host/checkpoint carriers are sourced from that same slot.
struct server_cache_family_slot_round_trip_result {
    bool resolved = false;
    bool second_resolved = false;
    bool roles_distinct = false;
    bool host_roles_distinct = false;
    bool no_restore_resume = false;
    bool binding_intact = false;
    bool host_save_carries = false;
    bool checkpoint_carries = false;
};

server_cache_family_slot_round_trip_result
server_cache_family_slot_round_trip_for_test(
        server_cache_control_authority & authority,
        server_cache_control_token binding_token,
        server_cache_control_token second_binding_token = {});

struct server_rejected_prompt_preservation_result {
    bool rejected = false;
    bool prompt_preserved = false;
    bool checkpoints_preserved = false;
    bool retention_preserved = false;
    bool error_geometry_valid = false;
    bool oversized_child_rejected = false;
    bool selection_skipped = false;
};

server_rejected_prompt_preservation_result
server_rejected_prompt_preservation_for_test();

struct server_mmproj_lifecycle_test_result {
    bool null_binding_clears_views = false;
    bool restored_binding_updates_views = false;
    bool failed_recreation_stays_null = false;
    bool normal_restore_once = false;
    bool thrown_media_restore_once = false;
    bool thrown_callback_restore_once = false;
    bool throwing_restore_not_retried = false;
    bool incompatible_draft_disables_shift = false;
    bool incompatible_draft_not_shifted = false;
    bool compatible_draft_enables_shift = false;
    bool compatible_draft_shifted = false;
};

// TEST-ONLY model-free exercise of the production slot rebinder and exactly-once
// restoration guard. It uses opaque pointer identities but never dereferences
// them, so ownership transitions can be proved without loading a projector.
server_mmproj_lifecycle_test_result
server_mmproj_lifecycle_for_test();

struct server_vbr_retention_wiring_result {
    bool slot_metadata_wired = false;
    bool slot_lifecycle_absent = false;
    bool slot_lease_absent = false;
    bool prefix_tracking_enabled = false;
    bool authority_prefix_tracking_enabled = false;
    bool external_coverage_exact = false;
};

server_vbr_retention_wiring_result
server_vbr_retention_wiring_for_test();

struct server_vbr_reclaim_policy_result {
    bool learned_kept_hot = false;
    bool learned_removed_cold = false;
    bool stopped_at_sufficiency = false;
    bool fallback_removed_oldest = false;
    bool zero_yield_fell_back = false;
    bool automatic_cache_preserved_undurable = false;
    bool mixed_host_kept_hot = false;
    bool mixed_host_removed_cold = false;
    bool token_identity_distinguishes_attempt = false;
    bool successful_attempt_is_state_sealed = false;
    bool multi_fresh_pressure_isolated = false;
    bool unchanged_admission_refusal_is_suppressed = false;
    bool checkpoint_admission_refusals_are_independent = false;
    bool admission_refusal_reopens_on_currency_change = false;
    bool admission_refusal_reopens_at_lease_expiry = false;
};

server_vbr_reclaim_policy_result
server_vbr_reclaim_policy_for_test();

struct server_vbr_slot_selection_result {
    bool learned_selected_cold = false;
    bool learned_kept_hot = false;
    bool selection_was_pure = false;
    bool fixed_learned_selected_cold = false;
    bool fixed_learned_kept_hot = false;
    bool fixed_selection_was_pure = false;
    bool fixed_incomplete_used_lru = false;
    bool fixed_protected_fallback_was_safe = false;
    bool fixed_capability_tier_was_preserved = false;
    bool incomplete_used_lru = false;
    bool protected_fallback_was_safe = false;
    bool all_protected_has_no_target = false;
    bool empty_slot_was_preferred = false;
    bool capability_tier_was_preserved = false;
    bool exhausted_tier_used_alternate = false;
    bool weak_prefix_preserved_empty = false;
    bool weak_prefix_preserved_hot = false;
    bool stem_recovery_allows_selection = false;
    bool stem_recovery_not_proactive = false;
    bool undurable_filter_makes_progress = false;
    bool undurable_selection_makes_progress = false;
    bool full_rebind_clears_stem_authority = false;
};

server_vbr_slot_selection_result
server_vbr_slot_selection_for_test(
    server_cache_lease_fallback_provider * lease_fallback);

struct server_context_meta {
    std::string build_info;
    std::string model_name;
    std::set<std::string> model_aliases;
    std::set<std::string> model_tags;
    std::string model_path;
    bool has_mtmd;
    bool has_inp_image;
    bool has_inp_audio;
    bool has_inp_video;
    json json_ui_settings;
    int slot_n_ctx;
    bool vbr_enabled;
    bool vbr_dynamic;
    bool vbr_type_k;
    bool vbr_type_v;
    std::string vbr_entry_type_k;
    std::string vbr_entry_type_v;
    double vbr_min_bits;
    double vbr_capacity_bits;
    double vbr_selected_bpv;
    double vbr_selected_kld;
    uint64_t vbr_vram_budget_bytes;
    std::string vbr_selected_family;
    std::string vbr_selected_policy;
    std::string vbr_selected_schedule;
    enum llama_pooling_type pooling_type;

    // chat params
    server_chat_params & chat_params;
    std::map<std::string, bool> chat_template_caps;

    // tokens
    std::string bos_token_str;
    std::string eos_token_str;
    llama_token fim_pre_token;
    llama_token fim_sub_token;
    llama_token fim_mid_token;
    llama_token fim_pad_token;
    llama_token fim_rep_token;
    llama_token fim_sep_token;

    // sampling
    std::vector<llama_logit_bias> logit_bias_eog;

    // model meta
    enum llama_vocab_type model_vocab_type;
    int32_t model_vocab_n_tokens;
    int32_t model_n_ctx_train;
    int32_t model_n_embd_inp;
    uint64_t model_n_params;
    uint64_t model_size;
    std::string model_ftype;
};

enum server_state {
    SERVER_STATE_DOWNLOADING,
    SERVER_STATE_LOADING,
    SERVER_STATE_READY,
    SERVER_STATE_SLEEPING,
};

static std::string server_state_to_str(server_state state) {
    switch (state) {
        case SERVER_STATE_DOWNLOADING: return "downloading";
        case SERVER_STATE_LOADING:     return "loading";
        case SERVER_STATE_READY:       return "ready";
        case SERVER_STATE_SLEEPING:    return "sleeping";
        default: GGML_ASSERT(false && "invalid server_state");
    }
}

static server_state server_state_from_str(const std::string & str) {
    if (str == "downloading") return SERVER_STATE_DOWNLOADING;
    if (str == "loading")     return SERVER_STATE_LOADING;
    if (str == "ready")       return SERVER_STATE_READY;
    if (str == "sleeping")    return SERVER_STATE_SLEEPING;
    GGML_ASSERT(false && "invalid server_state string");
}

using server_state_callback_t = std::function<void(server_state, json /* payload */)>;

struct server_context {
    std::unique_ptr<server_context_impl> impl;

    server_context();
    ~server_context();

    // load the model and initialize llama_context
    // returns true on success
    bool load_model(common_params & params);

    // this function will block main thread until termination
    void start_loop();

    // terminate main loop (will unblock start_loop)
    void terminate();

    // get the underlaying llama_context, can return nullptr if sleeping
    // not thread-safe, should only be used from the main thread
    llama_context * get_llama_context() const;

    // get a new response reader, used by CLI application
    server_response_reader get_response_reader();

    // get server metadata (read-only), can only be called after load_model()
    // not thread-safe, should only be used from the main thread
    server_context_meta get_meta() const;

    // note: must be set before load_model() is called
    void set_state_callback(server_state_callback_t callback);
};


// forward declarations
struct server_res_generator;

struct server_routes {
    server_routes(const common_params & params, server_context & ctx_server);

    void init_routes();

    // note: this is not thread-safe and can only when ctx_http.is_ready is false
    void update_meta(const server_context & ctx_server) {
        this->meta = std::make_unique<server_context_meta>(ctx_server.get_meta());
    }

    // handlers using lambda function, so that they can capture `this` without `std::bind`
    // they won't be called until ctx_http.is_ready is set to true
    server_http_context::handler_t get_health;
    server_http_context::handler_t get_metrics;
    server_http_context::handler_t get_slots;
    server_http_context::handler_t post_slots;
    server_http_context::handler_t post_cache_plan;
    server_http_context::handler_t post_cache_control;
    server_http_context::handler_t get_props;
    server_http_context::handler_t post_props;
    server_http_context::handler_t post_infill;
    server_http_context::handler_t post_completions;
    server_http_context::handler_t post_completions_oai;
    server_http_context::handler_t post_chat_completions;
    server_http_context::handler_t post_chat_completions_tok;
    server_http_context::handler_t post_control;
    server_http_context::handler_t post_responses_oai;
    server_http_context::handler_t post_responses_tok_oai;
    server_http_context::handler_t post_transcriptions_oai;
    server_http_context::handler_t post_anthropic_messages;
    server_http_context::handler_t post_anthropic_count_tokens;
    server_http_context::handler_t post_apply_template;
    server_http_context::handler_t get_models;
    server_http_context::handler_t post_tokenize;
    server_http_context::handler_t post_detokenize;
    server_http_context::handler_t post_embeddings;
    server_http_context::handler_t post_embeddings_oai;
    server_http_context::handler_t post_rerank;
    server_http_context::handler_t get_lora_adapters;
    server_http_context::handler_t post_lora_adapters;

    // to be used in router mode
    json get_model_info() const;

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
            const server_http_req & req,
            server_task_type type,
            const json & data,
            const std::vector<raw_buffer> & files,
            task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_capture(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_slots_import(const server_http_req & req, int id_slot);
    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type);
    std::unique_ptr<server_res_generator> handle_count_tokens(const llama_vocab * vocab, mtmd_context * mctx, const mtmd_helper_init_opt & init_opt, const server_http_req & req, task_response_type res_type);

    // using unique_ptr to allow late initialization of const
    std::unique_ptr<const server_context_meta> meta;

    const common_params & params;
    server_context_impl & ctx_server;

    server_queue & queue_tasks;
    server_response & queue_results;
    std::unique_ptr<server_res_generator> create_response(bool bypass_sleep = false);

    // cached responses, to be used during sleep
    std::mutex     mutex_cache;
    json           cached_models  = nullptr;
    json           cached_props   = nullptr;
    server_metrics cached_metrics;
    // set when a scrape during sleep already reported the throughput buckets
    bool           should_reset_buckets = false;
    // call right before sleep to update the cached responses
    void update_cached_responses(bool is_sleeping);
};
