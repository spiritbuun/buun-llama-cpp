#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"
#include "llama-memory.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;
struct llama_memory_recurrent;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

// DFlash: hidden state buffer for captured layer activations
struct dflash_layer_hidden_buf {
    std::vector<float> data;
    int64_t n_embd = 0;
    int64_t n_tokens = 0;
};

// DFlash: tape recording data for one recurrent layer
struct dflash_tape_layer {
    std::vector<float> k;          // [S_k * H_k * n_tokens] after l2_norm
    std::vector<float> v;          // [S_v * H_v * n_tokens]
    std::vector<float> gate;       // [H_v * n_tokens] pre-exp
    std::vector<float> beta;       // [H_v * n_tokens] pre-sigmoid
    std::vector<float> qkv_mixed;  // [conv_channels * n_tokens * n_seqs] for conv state rebuild
    int64_t S_k = 0, H_k = 0, S_v = 0, H_v = 0;
    int64_t conv_channels = 0;
    int n_tokens = 0;
    // per-seq metadata for multi-seq verify QKV scatter
    int n_seqs = 1;
    llama_seq_id seq_ids[LLAMA_DFLASH_MAX_SLOTS] = {};
};

// GPU-resident tape: persistent tensors that the graph writes into directly (no eval callback sync)
struct dflash_tape_gpu_layer {
    ggml_tensor * k    = nullptr;  // [S_k, H_k, max_tokens]
    ggml_tensor * v    = nullptr;  // [S_v, H_v, max_tokens]
    ggml_tensor * gate = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * beta = nullptr;  // [1, H_v, max_tokens]
    ggml_tensor * qkv  = nullptr;  // [conv_channels, max_tokens] (conv rebuild staging; null = eval-callback capture)
};

struct dflash_tape_gpu {
    std::vector<dflash_tape_gpu_layer> layers;  // one per recurrent layer
    std::vector<int32_t> layer_ids;             // model layer indices → tape index mapping
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;               // owns the tensor descriptors
    int max_tokens = 0;                         // allocated capacity
    int n_tokens = 0;                           // actual tokens recorded this pass

    // qkv is graph-staged for this tape (single-seq staged decodes bypass the eval
    // callback entirely) — the one predicate every staging consumer must agree on
    bool qkv_staged() const {
        return !layers.empty() && layers[0].qkv != nullptr;
    }

    ~dflash_tape_gpu() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

enum dflash_tape_type {
    DFLASH_TAPE_K    = 0,
    DFLASH_TAPE_V    = 1,
    DFLASH_TAPE_GATE = 2,
    DFLASH_TAPE_BETA = 3,
    DFLASH_TAPE_QKV  = 4,
};

// DDTree: tree attention mask for verification
struct llama_tree_mask {
    bool active = false;
    int n_tree_tokens = 0;         // number of tree tokens (root + nodes)
    std::vector<uint8_t> visibility;  // [n² row-major] true = can attend
};

// DFlash: eval callback data for hidden state capture + tape recording
struct dflash_capture_data {
    // hidden state capture (for drafter conditioning)
    std::vector<int32_t> layer_ids;           // layer indices to capture
    std::vector<std::string> tensor_names;    // pre-formatted "l_out-{id}" names
    std::unordered_map<std::string, size_t> hidden_name_idx; // name → index for O(1) lookup
    // pointer to context's layer_hiddens (outer: per-slot, inner: per-captured-layer)
    std::vector<std::vector<dflash_layer_hidden_buf>> * hiddens;

    // tape recording (for DeltaNet state rollback)
    bool tape_enabled = false;
    std::vector<int32_t> recurrent_layer_ids;       // model layer indices that are DeltaNet
    std::unordered_map<std::string, std::pair<int, int>> tape_name_map;  // name → (layer_idx, type)
    std::vector<dflash_tape_layer> tape_layers;     // one per recurrent layer (CPU fallback)

    // GPU-resident tape: graph writes directly to these tensors (no eval callback sync).
    // One entry per slot for multi-slot DFlash (see --dflash-max-slots). For single-slot
    // (default), `tapes` has size 1 and `active_tape_idx` is always 0 — behavior is
    // byte-identical to the pre-multi-slot singleton.
    std::vector<std::unique_ptr<dflash_tape_gpu>> tapes;
    int active_tape_idx = 0;
    // set when the meta-path shard-consistency check rejected the tape (unusual
    // --tensor-split ratios) — capability then reports false instead of re-probing
    bool tape_meta_failed = false;

    // Active ubatch for the in-flight process_ubatch() call. The eval callback
    // reads ubatch->n_seqs_unq / ubatch->seq_id to route hidden-state captures
    // to layer_hiddens[seq] (per-token scatter under multi-seq ubatches).
    // ggml's scheduler serializes callbacks within a graph compute, so this
    // pointer is safe to read without synchronization.
    const llama_ubatch * ubatch = nullptr;

    // Reused scratch for the multi-seq scatter path (avoid per-ubatch alloc).
    std::vector<float> scatter_buf;

    // GPU capture staging: per-captured-layer [n_embd, stage_max_tokens] tensors the
    // graph copies l_out into directly (single-seq whole-batch decodes). While a staged
    // ubatch is in flight the eval callback skips l_out entirely — no graph chop, no
    // device→host gather. stage_n_tokens reports how many tokens the last staged
    // decode captured (0 = staging did not cover the last decode; read the host
    // buffers instead).
    std::vector<ggml_tensor *> stage_tensors;
    ggml_context * stage_ctx = nullptr;
    ggml_backend_buffer_t stage_buf = nullptr;
    int stage_max_tokens = 0;
    bool stage_enabled = false;  // consumer opted in (a D2D route out of staging exists)
    bool stage_active = false;
    int stage_n_tokens = 0;

    // tokens covered by the graph-staged qkv tape copies in the last tape-enabled decode
    // (0 = staging did not cover it; the eval-callback capture holds the data instead)
    int tape_stage_n_tokens = 0;

    dflash_tape_gpu * active_tape() const {
        return (active_tape_idx >= 0 && active_tape_idx < (int) tapes.size())
                   ? tapes[active_tape_idx].get()
                   : nullptr;
    }

    // true iff a staged decode fully covers every eval-callback ask (hiddens graph-staged,
    // GPU tape k/v/g/b graph-copied, qkv graph-staged) — the callback can go dormant for
    // this decode. Must mirror the ask-paths in dflash_eval_callback.
    bool eval_callback_dormant() const {
        if (!stage_active) {
            return false;
        }
        if (!tape_enabled) {
            return true;
        }
        dflash_tape_gpu * tg = active_tape();
        return tg && tg->qkv_staged();
    }

    std::vector<dflash_layer_hidden_buf> * slot_hiddens(int slot) const {
        if (!hiddens || slot < 0 || slot >= (int) hiddens->size()) {
            return nullptr;
        }
        return &(*hiddens)[slot];
    }

    std::vector<dflash_layer_hidden_buf> * active_slot_hiddens() const {
        return slot_hiddens(active_tape_idx);
    }

    // persistent GPU buffer for tape replay (avoids per-call alloc/free)
    ggml_backend_buffer_t replay_buf = nullptr;
    size_t replay_buf_size = 0;

    // S2: pre-allocated zeros buffer for Q input (avoids per-call alloc+zero)
    std::vector<float> replay_zeros;

    // async tape replay state (GDN launched, waiting for sync before conv rebuild)
    bool replay_pending = false;
    ggml_backend_t replay_gpu_backend = nullptr;  // meta backend under --split-mode tensor (sync fans out)
    ggml_context * replay_graph_ctx = nullptr;

    // per-device replay resources for the meta (tensor-split) path: graph contexts are
    // per-rollback (freed in tape_replay_sync); the intermediate buffers are persistent
    // grow-only scratch per simple device (same scheme as replay_buf above)
    std::vector<ggml_context *> replay_meta_ctxs;
    std::vector<ggml_backend_buffer_t> replay_meta_bufs;
    std::vector<size_t> replay_meta_buf_sizes;
    int replay_n_accepted = 0;
    int32_t replay_cell_idx = -1;
    llama_seq_id replay_seq_id = 0;
    llama_memory_recurrent * replay_mem_recurrent = nullptr;

    ~dflash_capture_data() {
        if (replay_graph_ctx) {
            ggml_free(replay_graph_ctx);
        }
        for (auto * ctx : replay_meta_ctxs) {
            ggml_free(ctx);
        }
        for (auto * buf : replay_meta_bufs) {
            ggml_backend_buffer_free(buf);
        }
        if (replay_buf) {
            ggml_backend_buffer_free(replay_buf);
        }
        if (stage_buf) {
            ggml_backend_buffer_free(stage_buf);
        }
        if (stage_ctx) {
            ggml_free(stage_ctx);
        }
    }
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    int32_t * get_logits_argmax();
    int32_t   get_logits_argmax_n();
    int32_t   get_logits_argmax_k();
    float   * get_logits_argmax_probs();

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    // async-copy enabled layer-input tensors (per cparams.output_layer_inp)
    // from backend into host-side embd_layer_inp buffers
    void extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens);

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    llama_memory_ptr memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state required by the nextn layers (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_nextn is enabled and the model graph
    // sets llm_graph_result::t_h_nextn
    buffer_view<float> embd_nextn = {nullptr, 0};

    // host buffers for output layer input embeddings, per layer
    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once    = false;
    bool warned_logits_all     = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused

public:
    // --- fork: DFlash / tree speculative decoding ---

    float * get_layer_hidden(int layer_idx);
    int64_t get_layer_hidden_n_tokens(int layer_idx) const;
    int64_t get_layer_hidden_n_embd(int layer_idx) const;
    int32_t get_n_layer_hiddens() const;

    void set_dflash_capture(const int32_t * layer_ids, int32_t n_layers);
    void allocate_capture_stage_gpu();
    void set_capture_stage_enabled(bool enabled);
    // returns tokens staged by the last staged decode (0 = none) and the device
    // pointer of the layer's staging data (shard 0 under --split-mode tensor)
    int32_t dflash_capture_stage_get(int32_t layer_idx, const void ** data);
    void set_dflash_sample_temp(float temp);
    void set_dflash_topk(int k);
    void set_dflash_n_slots(int n);

    void dflash_reset_hidden_capture();
    void dflash_ensure_recurrent_setup();

    void set_tape_recording(bool enable);
    void allocate_tape_gpu(int max_tokens) { allocate_tape_gpu(1, max_tokens); }
    void allocate_tape_gpu(int n_slots, int max_tokens);
    void tape_replay_meta(ggml_backend_t meta_backend, llama_memory_recurrent * mem_recurrent,
                          int32_t cell_idx, int n_accepted, llama_seq_id seq_id);
    void set_active_dflash_slot(int slot_idx);

    // first GPU/IGPU-typed backend, or nullptr (tensor-split's meta backend is neither)
    ggml_backend_t find_gpu_backend();
    ggml_backend_t find_meta_backend();

    bool tape_replay_available();

    void tape_replay(llama_seq_id seq_id, int n_accepted);
    void tape_replay_sync();
    void tape_replay_conv(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id = 0);
    void tape_replay_cpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted);

    void dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted);
    void dflash_prepare_branch(llama_seq_id seq_id, llama_seq_id seq_backup, int depth);

    void set_cross_data(const float * data, int64_t n_embd, int64_t n_tokens);
    void set_cross_data_seq(llama_seq_id seq_id, const float * data, int64_t n_embd, int64_t n_tokens);
    void * init_cross_ring_gpu(int n_layers, int n_embd, int ring_size);

    using set_tensor_d2d_fn_t = void (*)(void *, const void *, size_t, size_t);
    void set_cross_data_gpu(llama_seq_id seq_id, const void * d_staging, int cross_len,
                            int n_layers, int n_embd, set_tensor_d2d_fn_t fn_d2d);

    void set_tree_mask(const uint8_t * visibility, int n_tree_tokens);
    void clear_tree_mask();
    void set_tree_parent_ids(const int32_t * parents, int n_tokens);
    void clear_tree_parent_ids();
    void allocate_tree_buffers(int max_tree_tokens);
    void tree_rollback(int commit_n, const int32_t * parents);
    void set_tree_seq0_count(int n) { tree_bufs.n_seq0_tokens = n; }

    // fork data members
    std::vector<int32_t> logits_argmax_buf;
    std::vector<float>   logits_argmax_prob_buf;
    int32_t logits_argmax_count = 0;
    int32_t logits_argmax_k = 1;

    std::vector<std::vector<dflash_layer_hidden_buf>> layer_hiddens;
    std::unique_ptr<dflash_capture_data> dflash_capture;

    llama_tree_mask tree_mask;

    struct {
        bool active = false;
        bool disabled = false;
        int n_seq0_tokens = 0;
        int n_tokens = 0;
        std::vector<int32_t> parent_ids_cpu;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_context * ggml_ctx = nullptr;
        ggml_tensor * parent_ids_gpu = nullptr;
        std::vector<ggml_tensor *> ssm_intermediates;
        int max_tree_tokens = 0;
    } tree_bufs;
};
