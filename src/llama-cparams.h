#pragma once

#include "llama.h"

#include <cstdint>
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
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    int32_t  nextn_layer_offset = 0;

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
    // mixed-config side pins, see llama.h vbr_pin_k
    bool vbr_pin_k = false;
    bool vbr_pin_v = false;

    std::vector<bool> embeddings_layer_inp; // [n_layer()] extract input embeddings for layer

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    // DFlash: target layer indices to capture hidden states from (empty = disabled)
    std::vector<int> dflash_capture_layers;

    // DFlash: drafter sampling temperature (0 = greedy argmax, >0 = Gumbel sampling)
    float dflash_sample_temp = 0.0f;

    // DFlash: top-K candidates per position (1 = argmax only, >1 = tree branching)
    int dflash_topk = 1;

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
