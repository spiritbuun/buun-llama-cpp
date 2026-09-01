#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;
class llama_io_write_i;
class llama_io_read_i;

// Select the shared DFlash driver while preserving composable extensions.
void common_speculative_select_dflash2(common_params_speculative & params);

// Select DFlash2 after the loaded draft model confirms selector support.
void common_speculative_resolve_draft_model_type(
        common_params_speculative & params,
        const llama_model *         model_dft);

struct common_speculative_proposal {
    llama_tokens selected;
    int32_t top_k = 0;
    llama_tokens candidate_ids; // [selected.size(), top_k]
    std::vector<float> q_rows;  // [selected.size(), top_k]
    size_t q_covered_tokens = 0;
    llama_seq_id seq_id = -1;
    bool exact_q = false;
};

// comma separated list the provided types
std::string common_speculative_type_name_str(const std::vector<enum common_speculative_type> & types);

// comma separated list of all types
const char * common_speculative_all_types_str();

// parse user provided types
std::vector<enum common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_speculative_type> common_speculative_types_from_gguf(const std::string & path);

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// return the max number of draft tokens based on the speculative parameters
int32_t common_speculative_n_max(const common_params_speculative * spec);

// return the max number of draft tokens from the initialized implementations
int32_t common_speculative_n_max(const common_speculative * spec);

// validate and resolve the unconditional synthetic acceptance rates
std::vector<double> common_speculative_synth_rates_resolve(const common_params_speculative * spec, int32_t n_max);

// return the conditional synthetic acceptance probabilities
const std::vector<double> & common_speculative_get_synth_probs(const common_speculative * spec);

common_params common_base_params_to_speculative(const common_params & params);

struct common_speculative_mtp_context_params {
    uint32_t n_ctx;
    uint32_t n_seq_max;
    bool kv_unified;
};

// Native and sidecar MTP contexts must expose the target's realized context
// width per sequence. An implicit draft context can do that without multiplying
// its KV allocation by the number of server slots by using unified KV. An
// explicit -cd remains an exact user override, including the requested KV
// topology.
common_speculative_mtp_context_params common_speculative_mtp_context_params_resolve(
        uint32_t target_n_ctx_seq,
        int32_t explicit_draft_n_ctx,
        uint32_t requested_n_seq_max,
        bool requested_kv_unified);

bool common_speculative_mtp_context_available(const common_params_speculative & params);

struct common_speculative_output_limits {
    int32_t total;
    int32_t per_seq;
};

// return the output limits needed for speculative decoding
common_speculative_output_limits common_speculative_get_output_limits(
        int32_t n_batch, int32_t n_parallel, int32_t n_draft);

common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq);

void common_speculative_free(common_speculative * spec);

struct common_speculative_draft_params {
    // this flag is used to chain the drafts through all the available implementations
    // after the first successful draft from an implementation, we set it
    //   to false to prevent further drafts for that sequence
    // at the end of the draft() call, all drafting flags will be reset to false
    bool drafting = false;

    // overrides individual configurations (-1 disabled)
    // can be used to constraint the max draft based on the remaining context size
    int32_t n_max = -1;

    llama_pos   n_past;
    llama_token id_last;

    // TODO: remove in the future by keeping track of the prompt from the _begin() call and the consecutive accept calls
    const llama_tokens * prompt;

    // the generated draft from the last _draft() call
    llama_tokens * result;
};

common_speculative_draft_params & common_speculative_get_draft_params(common_speculative * spec, llama_seq_id seq_id);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt);

// process the batch and update the internal state of the speculative context
//
// CALLING CONTRACT (every consumer — server AND examples — must follow this;
// the 2026-07 upstream sync silently broke speculative-simple by reordering it):
//   1. common_speculative_init() BEFORE the prompt is decoded on the target —
//      capture-based impls (DFlash) enable target hidden-state extraction here,
//      so a prompt decoded earlier is invisible to the drafter.
//   2. common_speculative_process(batch) after EVERY llama_decode on the target
//      (prompt prefill and verify steps alike) — capture-based impls gather the
//      extracted features here before the next decode overwrites them. No-op
//      for impls that don't need it; always safe to call.
//   3. DFlash-family drafter contexts must NOT be fed raw token batches
//      (classic lockstep drafter decodes) — their state is managed entirely via
//      begin()/process()/draft().
bool common_speculative_process(common_speculative * spec, const llama_batch & batch);

// Whether any configured implementation requires target embeddings.
bool common_speculative_need_embd(common_speculative * spec);
bool common_speculative_need_embd_nextn(common_speculative * spec);

// generate drafts for the sequences specified with `common_speculative_get_draft_params`
void common_speculative_draft(common_speculative * spec);

// informs the speculative context that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, llama_seq_id, uint16_t n_accepted);

// (optional) get/set internal state
bool common_speculative_get_state(common_speculative * spec, llama_seq_id seq_id, std::vector<uint8_t> & data);
void common_speculative_set_state(common_speculative * spec, llama_seq_id seq_id, const std::vector<uint8_t> & data);

enum class common_speculative_sequence_event : uint8_t {
    prompt_rewind = 0,          // retained draft prefix remains installed
    target_restored_without_draft, // target restored; draft must reconstruct
    draft_image_restored,       // complete target+draft sequence images restored
    composite_image_restored,   // draft plus typed accelerator state restored
    live_range_shift,           // live target/draft positions were renumbered
    target_replaced,            // target replaced; draft sequence was cleared
    full_clear,                 // target and draft sequences were cleared
};

// MTP's pending target-hidden row is process-local state: sequence images do
// not serialize it. Keep the validity decision separate from the row storage
// so a restored/cloned sequence cannot consume a row from the previous branch.
class common_speculative_mtp_carry_lifecycle {
public:
    enum class process_mode {
        retained_carry,
        cold_zero,
        target_only,
    };

    bool draft_ready() const noexcept;
    const float * draft_carry(const float * pending_h) const noexcept;
    process_mode target_process_mode(llama_pos first_position) const noexcept;

    void target_process_refreshed() noexcept;
    void target_process_skipped() noexcept;
    void sequence_transition(common_speculative_sequence_event event) noexcept;

private:
    bool ready = false;
};

enum class common_speculative_mtp_process_preflight {
    cold_or_retained,
    target_only,
};

// Resolve every active sequence before process() mutates draft memory. A
// single nonzero frontier without an authenticated carry makes the whole
// multi-sequence batch target-only so no sequence advances on partial state.
common_speculative_mtp_process_preflight
common_speculative_mtp_process_preflight_resolve(
    const std::vector<common_speculative_mtp_carry_lifecycle> & lifecycles,
    const std::vector<int32_t> & active_batch_beg,
    const llama_pos * positions) noexcept;

// One owner for external sequence lifecycle mutations. Implementations discard
// branch-local state and apply the event-specific memory/ring policy.
void common_speculative_sequence_transition(
        common_speculative * spec,
        llama_seq_id         seq_id,
        common_speculative_sequence_event event);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;

// fork: per-slot init (n_seq=1, shared drafter context)
llama_context * common_speculative_create_ctx_dft(const common_params_speculative & params, int dflash_n_slots = 1);
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_dft_shared = nullptr);

// fork: single-seq overloads
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// fork: DFlash slot routing
void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id);
void common_speculative_set_rng_seed(
        common_speculative * spec,
        llama_seq_id         seq_id,
        uint32_t             seed);

// fork: single-seq draft (returns tokens)
llama_tokens common_speculative_draft(
        common_speculative              * spec,
        const common_params_speculative & params,
        const llama_tokens              & prompt_tgt,
        llama_token                       id_last,
        std::vector<float>              * draft_log_probs = nullptr,
        llama_pos                         n_past_override = -1);

// fork: batched multi-slot DFlash drafting
void common_speculative_draft_batch(
        std::vector<common_speculative *> & specs,
        llama_context                     * ctx_dft,
        const common_params_speculative   & params,
        const std::vector<llama_token>    & id_last_per_spec,
        std::vector<llama_tokens>         & result_per_spec);

// True only when the immediately preceding single/batched draft call completed
// at least one decode on a linked draft-model context.
bool common_speculative_last_draft_model_decode_succeeded(const common_speculative * spec);

// Proposal distribution owned by the DFlash state that produced the most
// recent draft. Null means the current draft has no verified proposal payload.
const common_speculative_proposal * common_speculative_get_proposal(
        const common_speculative * spec,
        llama_seq_id               seq_id);

// fork: logit/state management
void   common_speculative_update_logits(common_speculative * spec, llama_seq_id seq_id, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted);
void   common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted);
void   common_speculative_flush_prefill(common_speculative * spec);

void   common_speculative_rollback_dft(common_speculative * spec, llama_seq_id seq_id, llama_pos n_past, uint16_t n_accepted);

// fork: DFlash ring buffer state save/load
size_t common_speculative_ring_state_size(const common_speculative * spec);
void   common_speculative_ring_state_save(const common_speculative * spec, uint8_t * buf, size_t size);
bool   common_speculative_ring_state_write(
        const common_speculative * spec,
        llama_io_write_i & output);
bool   common_speculative_ring_state_load(common_speculative * spec, const uint8_t * buf, size_t size);
// Restore a DFlash ring directly from a bounded reader. The reader is consumed
// exactly `size` bytes on success; implementations issue no read larger than
// 1 MiB so retained artifact chains do not need a contiguous materialization.
bool   common_speculative_ring_state_read(
        common_speculative * spec,
        llama_io_read_i & input,
        size_t size);

// Cheap post-install currency. A successful ring mutation changes the epoch;
// callers can therefore close a late publication gate without serializing the
// full ring again.
struct common_speculative_ring_state_currency {
    size_t serialized_bytes = 0;
    llama_pos terminal = -1;
    uint64_t mutation_epoch = 0;
};
bool   common_speculative_ring_state_get_currency(
        const common_speculative * spec,
        common_speculative_ring_state_currency & output);
// Authenticate a serialized DFlash ring against the exact committed prompt
// frontier before it is captured or installed.  The terminal is the logical
// position of the last committed token (committed_len - 1).
bool   common_speculative_ring_state_matches_frontier(
        const common_speculative * spec,
        const uint8_t * buf,
        size_t size,
        llama_pos expected_terminal);
bool   common_speculative_ring_state_terminal(
        const common_speculative * spec,
        llama_pos & terminal);
bool   common_speculative_ring_state_serialized_terminal(
        const uint8_t * buf,
        size_t size,
        llama_pos & terminal);
bool   common_speculative_ring_state_empty(const common_speculative * spec);
void   common_speculative_ring_state_reset(common_speculative * spec);

// fork: draft length params
int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);

struct common_speculative_init_result {
    common_speculative_init_result(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
    ~common_speculative_init_result();

    llama_model   * model();
    llama_context * context();
    llama_context * context_mtp();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_speculative_init_result_ptr = std::unique_ptr<common_speculative_init_result>;

common_speculative_init_result_ptr common_speculative_init_from_params(common_params & params, llama_model * model_tgt, llama_context * ctx_tgt);
