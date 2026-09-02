#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct dflash_tape_gpu;

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    uint32_t n_outputs_max;   // max outputs supported by the context
    uint32_t n_outputs_max_per_seq;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

    enum llama_moe_cache_mode moe_cache_mode;
    size_t moe_cache_budget_mib;
    int32_t moe_cache_expert_parallel;
    std::string moe_cache_profile_path;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_nextn;        // also extract the hidden state before the final output norm
    bool embeddings_nextn_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool fused_lid;          // use fused lightning indexer
    bool auto_flid;
    bool fused_dsv4_hc_pre;
    bool fused_dsv4_hc_comb;
    bool fused_dsv4_hc_post;
    bool auto_fhc;
    bool no_perf;
    bool warmup;             // TODO: remove [TAG_LLAMA_GRAPH_NO_WARMUP]
    bool op_offload;
    bool kv_unified;
    bool logits_all;
    bool pipeline_parallel;
    bool vbr_dynamic;

    double vbr_min_bits = 0.0;
    uint64_t vbr_vram_budget_bytes = 0;
    uint64_t vbr_growth_headroom_bytes = 0;
    bool vbr_budget_explicit = false;
    bool vbr_min_bits_explicit = false;
    // mixed-config side pins, see llama.h vbr_pin_k
    bool vbr_pin_k = false;
    bool vbr_pin_v = false;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;

    // DFlash: target layer indices to capture hidden states from (empty = disabled)
    std::vector<int> dflash_capture_layers;

    // DFlash: drafter sampling temperature (0 = greedy argmax, >0 = Gumbel sampling)
    float dflash_sample_temp = 0.0f;

    // DFlash: top-K candidates per position (1 = argmax only, >1 = tree branching)
    int dflash_topk = 1;

    // Upstream block-diffusion drafter (arch "dflash"): build the in-graph top-K/argmax
    // tail (t_logits_argmax) on the decode graph so the draft loop reads K ids+logprobs
    // per position instead of the full-vocab logits. Opt-in from the speculative impl;
    // default off keeps raw-logits behavior for every other consumer.
    bool dflash_argmax = false;

    // DFlash target verification: append a raw greedy argmax to the ordinary
    // target graph. Unlike dflash_argmax, this is architecture-independent and
    // deliberately ignores the drafter's temperature/top-K controls.
    bool dflash_target_argmax = false;

    // DFlash2 target verification can prefer CUDA's MMA-backed quantized
    // matrix path for one exact width during an explicitly marked verify
    // submission. Zero keeps ordinary dispatch.
    int32_t dflash_target_mmq_batch = 0;

    // Upstream block-diffusion drafter: fused encoder+injection. Decode embd batches
    // carry raw concatenated target features (n_embd_inp_enc wide); the injection graph
    // applies fc + enc-norm itself, replacing the separate llama_encode round-trip.
    bool dflash_fused_inject = false;

    // Upstream drafter device-staged capture (TARGET context): when set, the target
    // graph copies each captured layer's input into this persistent [n_embd_enc, T]
    // tensor with an interleaved (strided) layout — the interleave happens inside the
    // D2D capture copies. Layer order in dflash_draft_stage_layers (k -> layer id).
    ggml_tensor *        dflash_draft_stage = nullptr;
    std::vector<int32_t> dflash_draft_stage_layers;

    // Upstream drafter staged injection (DRAFTER context): the fused inject graph
    // gathers its feature rows from this tensor (the target's stage) via a per-decode
    // row-index input instead of a host embd upload.
    ggml_tensor *        dflash_inject_stage = nullptr;
    std::vector<int32_t> dflash_inject_rows;

    // Upstream drafter single-graph fused cycle (DRAFTER context): when n_inject > 0 a
    // token-batch decode builds the fused graph — rows [0, n_inject) are staged KV
    // injections gathered from the carry tensor (dflash_inject_rows holds their carry
    // row indices), the remaining rows are the noise block. Constant n_inject keeps the
    // graph topology stable across generation cycles (gf_res_prev + CUDA graph reuse).
    ggml_tensor * dflash_oneg_stage    = nullptr;
    int32_t       dflash_oneg_n_inject = 0;

    // DFlash drafter: number of concurrent slots the batched drafter graph is reserved
    // for. ctx_len in the drafter graph = dflash_n_slots * LLAMA_DFLASH_PER_SLOT_CTX,
    // and drafter n_tokens reservation = dflash_n_slots * block_size. Set on the
    // drafter context (not the target) via llama_set_dflash_n_slots(). Default 1
    // (single-slot) so the drafter graph stays narrow when no batching is configured.
    // Capped at LLAMA_DFLASH_MAX_SLOTS.
    int dflash_n_slots = 1;

    // GPU-resident tape for DeltaNet rollback (graph writes directly, no eval callback sync).
    // tape_gpu is non-null when GPU tape is enabled (backward compat sentinel).
    dflash_tape_gpu * tape_gpu = nullptr;

    // Per-seq tape pointers for multi-seq verify batching.
    // tape_gpu_seqs[s] = tape for ubatch seq index s (0..tape_gpu_n_seqs-1).
    // Populated by the decode loop before each process_ubatch().
    dflash_tape_gpu * tape_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
    int tape_gpu_n_seqs = 0;
    // Minimal-F32 mode omits redundant post-conv K/V graph copies. QKV,
    // gate, and beta remain authoritative for exact reconstruction.

    // DFlash GPU capture staging: graph-embedded copies of each captured layer's l_out
    // into capture_stage[i] (one [n_embd, max_tokens] tensor per entry of
    // dflash_capture_layers, same order; capacity = the tensor's ne[1]). Non-null iff
    // staging covers the in-flight ubatch: the graph builder (llm_graph_context::cb)
    // then embeds the copies and the eval callback skips those layers entirely — no
    // per-layer graph chop, no device→host round-trip. The decode loop toggles this
    // per ubatch (single-seq, whole-batch-in-one-ubatch decodes only).
    ggml_tensor ** capture_stage = nullptr;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;

    llama_context * ctx_other;
};
