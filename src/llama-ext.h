#pragma once

// this is a staging header for new llama.cpp API
// breaking changes and C++ are allowed. everything here should be considered WIP
// try as much as possible to not include this header in the rest of the codebase

#include "llama.h"

#include <cstdint>
#include <map>

struct llama_dflash_proposal_view {
    const int32_t * candidate_ids = nullptr; // [blocks, steps, top_k]
    const float   * q_rows        = nullptr; // [blocks, steps, top_k]
    int32_t top_k   = 0;
    int32_t n_steps = 0;
    int32_t n_blocks = 0;
};

// Internal model classifier used to select the DFlash2 driving protocol.
extern "C" LLAMA_API bool llama_model_dflash2_has_selector(const struct llama_model * model);

// Returns the stochastic DFlash2 proposal rows produced by the most recent
// decode. The pointers remain valid until the next decode on this context.
LLAMA_API bool llama_get_dflash_proposal(
        struct llama_context * ctx,
        struct llama_dflash_proposal_view * view);

LLAMA_API void llama_set_dflash_proposal_uniforms(
        struct llama_context * ctx,
        llama_seq_id seq_id,
        const float * values,
        int32_t n);

// Internal speculative memory operations: real mutations with ordinary accounting, but no global
// checkpoint-lineage publication for a caller-proven disposable backup or rejected suffix.
LLAMA_API bool llama_memory_seq_rm_transient(
        llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
LLAMA_API bool llama_memory_seq_rm_attn_transient(
        llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
LLAMA_API bool llama_memory_try_seq_cp_transient(
        llama_memory_t mem, llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
        llama_pos p0, llama_pos p1);

// Reserve a new compute graph. It is valid until the next call to llama_graph_reserve.
LLAMA_API struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs);

// Get the default ggml_type for a given ftype.
LLAMA_API ggml_type llama_ftype_get_default_type(llama_ftype ftype);

struct quantize_state_impl;

LLAMA_API quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params);

LLAMA_API void llama_quant_free(quantize_state_impl * qs);

// Descriptor for constructing a mock model for quantization testing.
struct llama_quant_model_desc {
    const char * architecture;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_expert;
    uint32_t n_embd_head_k;
    uint32_t n_embd_head_v;
};

// Create a mock model from a metadata descriptor (for testing).
// The returned model must be freed with llama_model_free().
LLAMA_API llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc);

// Returns true if this tensor should be quantized (based on name, dims, params).
LLAMA_API bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor);

// Compute quantization type assignments for a list of tensors.
// All tensors should be quantizable (use llama_quant_tensor_allows_quantization to filter).
// result_types: caller-allocated array of n_tensors elements, filled with assigned types.
LLAMA_API void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors);

//
// device memory querying
//

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers
    // portion of `context` that does NOT scale with n_ctx (recurrent-state cache, sized by
    // n_seq_max). Included in `context`, never added separately. Budget formulas that exploit
    // the context term cancelling out of context-linear projections must still charge this part.
    size_t context_fixed = 0;
    // Portion of `context` whose representation/footprint is selected by Turbo/VBR. Auxiliary
    // fixed-layout caches can be context-linear without belonging to this subset.
    size_t context_vbr_managed = 0;

    size_t total() const {
        return model + context + compute;
    }
};

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

// TODO: convert to C-style data structure
using llama_memory_breakdown = std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data>;

LLAMA_API int32_t llama_model_n_expert (const struct llama_model * model);
LLAMA_API int32_t llama_model_n_devices(const struct llama_model * model);

// Copies the effective, non-cumulative model split weights in physical-device
// order. Passing nullptr/zero capacity returns the required element count.
// Unlike llama_model_params::tensor_split, automatic placement is resolved.
LLAMA_API size_t llama_model_resolved_tensor_split(
        const struct llama_model * model,
        float * weights,
        size_t capacity);

struct llama_moe_tensor_info {
    enum ggml_type type;
    size_t expert_size;
    int64_t n_input;
    int64_t n_output;
    int64_t n_expert;
    // Transformer block index parsed from the tensor name ("blk.<N>..."),
    // or -1 when the layer cannot be determined.
    int64_t layer;
};

LLAMA_API size_t llama_model_get_moe_tensor_info(
        const struct llama_model * model,
        struct llama_moe_tensor_info * info,
        size_t capacity);

// True when at least one routed-expert weight tensor is backed by host memory.
// This is intentionally about resolved placement, not architecture metadata.
LLAMA_API bool llama_model_has_host_moe_weights(const struct llama_model * model);

LLAMA_API ggml_backend_dev_t llama_model_get_device(const struct llama_model * model, int i);

LLAMA_API llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx);

// Resident cache-state allocation for one context, split into mutually-exclusive physical
// leaves: attention KV, live recurrent state, recurrent rollback planes, and speculative
// rolling-window tape. Keys are backend buffer types (one row per allocation). Recurrent
// buffers physically contain (1 + n_rs_seq) equal state planes; their allocation bytes are
// partitioned between the live plane and rollback planes without changing the measured total.
// The rolling-window field excludes the fixed speculative tape.
struct llama_live_memory_breakdown_data {
    size_t attention             = 0;
    size_t recurrent             = 0;
    size_t recurrent_rollback    = 0;
    size_t rolling_window_tape   = 0;
};

using llama_live_memory_breakdown =
    std::map<ggml_backend_buffer_type_t, llama_live_memory_breakdown_data>;

LLAMA_API llama_live_memory_breakdown llama_get_live_memory_breakdown(
        const struct llama_context * ctx);

// Per-token KV bits of the layout the --vbr-floor clamp lands on: walk the VBR degrade order
// from the given entry types (GGML_TYPE_COUNT = each tensor's current type) until the aggregate
// bits/value would cross floor_bpv (<= 0 = bottom-tier default; pass 1e30 for the un-walked
// layout cost of the given types). Returns 0 when the context has no VBR-capable cache.
// Works on no_alloc (fit dry-load) contexts — the fit uses it for floor-true capacity math.
LLAMA_API double llama_vbr_floor_bits_per_token(struct llama_context * ctx,
        enum ggml_type entry_k, enum ggml_type entry_v, double floor_bpv);

// Per-token bytes of the flash-attention f16 dequant scratch at the settled deep-fill tier state
// (see llama-memory.h memory_vbr_scratch_bytes_per_token). The fit charges this in its
// total-VRAM wall constraint only — it must NOT enter the KV budget solves (the scratch draws
// from the fit margin, not the budget). Works on no_alloc (fit dry-load) contexts.
LLAMA_API double llama_vbr_scratch_bytes_per_token(struct llama_context * ctx,
        enum ggml_type entry_k, enum ggml_type entry_v, double floor_bpv);

// co-tenancy plan hint: total bytes this process still intends to allocate on the device
// (PCI bus id per ggml_backend_dev_props.device_id). Set by the fit pass before load so a
// held demand can publish an honest joint cross-device estimate instead of drip-feeding
// per-failure asks (est_partial).
LLAMA_API void llama_vram_plan_hint(const char * device_id, uint64_t bytes);

// Composite application-load transaction. Nested component loaders cannot
// publish SATISFIED or abandon an outer load. The outermost end owns that result.
LLAMA_API void llama_vram_load_begin(bool application_owned_completion);
LLAMA_API void llama_vram_load_end(bool success);
LLAMA_API void llama_vram_plan_aux(const char * device_id, uint64_t bytes);
LLAMA_API void llama_vram_load_complete(void);

// co-tenancy: declare this process serviced (runs an idle tick — llama-server). Presence
// markers then advertise serviced:1, the qualifying signal for a co-loader's LONG patience.
LLAMA_API void llama_vram_mark_serviced(void);

// co-tenancy telemetry for /props//slots: all zeros = inert (single tenant, no ledger)
struct llama_vram_cotenancy_state {
    uint64_t grant_decrement; // unamortized bytes currently decremented from KV budgets
    uint32_t grants_active;   // live grant rows
    uint64_t shed_offer;      // published donation offer, summed over devices
    uint64_t grant_pending;   // granted-but-not-yet-flushed bytes
};
LLAMA_API struct llama_vram_cotenancy_state llama_vram_cotenancy(const struct llama_context * ctx);

// Set whether the context outputs nextn embeddings or not
// If masked == true,  output the embeddings only for the tokens with batch.logits != 0
// If masked == false, output the embeddings for all tokens in the batch regardless of batch.logits
LLAMA_API void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

// Select which appended NextN block the DECODER_MTP graph runs (offset past
// the trunk: il = n_layer() + offset). Used by the speculative NextN driver to
// chain multiple trained NextN heads. Default 0 (first head).
LLAMA_API void llama_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_nextn(struct llama_context * ctx);

// LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
LLAMA_API float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);

// Set whether the context outputs the input embeddings of a specific layer
LLAMA_API void llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);

// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
//

LLAMA_API int32_t llama_model_dflash_selector_top_k(const struct llama_model * model);

// returns pointer to the target-model layer indices
LLAMA_API const int32_t * llama_model_target_layer_ids  (const struct llama_model * model);
// returns the number of extracted layers from target model
LLAMA_API uint32_t        llama_model_target_layer_ids_n(const struct llama_model * model);

// retrieves the whole token embedding matrix in F32 format (n_embd * n_vocab)
// returns total number of elements or 0 on error
// if out is nullptr, returns the number of tokens without writing to out
// caller must allocate enough memory for out before calling
LLAMA_API uint32_t llama_model_get_tok_embd(const struct llama_model * model, float * out);
