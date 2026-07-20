#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

//
// llama_context
//

// thrown for the expected (non-fatal) "ctx_other not yet set" case during memory fitting;
// caught in llama_init_from_model and logged as a warning rather than an error (upstream PR #24590)
class llama_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

static bool turbo_vbr_layer_schedule_enabled() {
    const char * e = getenv("VBR_LAYER_SCHEDULE");
    return e && e[0];
}

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    cparams.embeddings_layer_inp.resize(hparams.n_layer(), false);
    embd_layer_inp.resize(hparams.n_layer());

    cparams.ctx_type     = params.ctx_type;
    cparams.pooling_type = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            throw llama_exception("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw llama_exception(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    cparams.dflash_n_slots = std::clamp(params.dflash_n_slots <= 0 ? 1 : params.dflash_n_slots,
                                        1, (int) LLAMA_DFLASH_MAX_SLOTS);

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = !params.no_fused_gdn;
    cparams.fused_gdn_ch = !params.no_fused_gdn;
    cparams.auto_fgdn    = !params.no_fused_gdn;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;
    cparams.logits_all = params.logits_all;
    cparams.vbr_dynamic = params.vbr_dynamic;
    cparams.vbr_min_bits = params.vbr_min_bits;
    cparams.vbr_vram_budget_bytes = params.vbr_vram_budget_bytes;
    cparams.vbr_growth_headroom_bytes = params.vbr_growth_headroom_bytes;
    cparams.vbr_budget_explicit = params.vbr_budget_explicit;
    cparams.vbr_pin_k = params.vbr_pin_k;
    cparams.vbr_pin_v = params.vbr_pin_v;

    // A shared-KV drafter (gemma4 assistant / weightless DFlash/Eagle3) views the target's
    // KV tensors — the target's VBR controller owns those, and the drafter's graphs follow
    // the owner's tier flips via the delegated tier epoch (llama_kv_cache::vbr_tier_epoch).
    // Running a second controller in the drafter would double-manage the same pool (and the
    // kv-cache ctor rejects an armed VBR on a share-linked cache), so disarm it here.
    if (cparams.ctx_other != nullptr &&
            (cparams.vbr_dynamic || cparams.vbr_vram_budget_bytes > 0 || cparams.vbr_min_bits > 0.0)) {
        LLAMA_LOG_INFO("%s: shared-KV drafter: VBR is managed by the target context — "
                "disarming the drafter's own VBR controller (shared layers follow the "
                "target's tier flips; the drafter's own layers stay at their static types)\n", __func__);
        cparams.vbr_dynamic              = false;
        cparams.vbr_min_bits             = 0.0;
        cparams.vbr_vram_budget_bytes    = 0;
        cparams.vbr_growth_headroom_bytes = 0;
        cparams.vbr_budget_explicit      = false;
        cparams.vbr_pin_k                = false;
        cparams.vbr_pin_v                = false;
    }

    // Dynamic VBR requires single-stream KV (the VMM pool + degrade controller are gated on
    // n_stream == 1). Force unified KV here — at context init, AFTER tools have applied their
    // post-parse n_parallel/n_seq_max mutations (perplexity, imatrix, batched-bench) — so the
    // controller cannot silently disarm while the logs advertise dynamic VBR.
    if (cparams.vbr_dynamic && cparams.n_seq_max > 1 && !cparams.kv_unified) {
        LLAMA_LOG_WARN("%s: dynamic VBR with n_seq_max = %u would split the KV per sequence and "
                "disarm the degrade controller — forcing unified KV\n", __func__, cparams.n_seq_max);
        cparams.kv_unified = true;
    }

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    if (cparams.vbr_dynamic || cparams.vbr_vram_budget_bytes > 0 || cparams.vbr_min_bits > 0.0) {
        LLAMA_LOG_INFO("%s: vbr           = %s, min_bits=%g, vram_budget=%" PRIu64 "\n",
                __func__,
                cparams.vbr_dynamic ? "dynamic" : "static",
                cparams.vbr_min_bits,
                cparams.vbr_vram_budget_bytes);
    }
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq      = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max = %u\n",   __func__, cparams.n_outputs_max);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        // turbo3/turbo4 KV cache stores data in FWHT-rotated space.
        // Q pre-rotation and V inverse rotation are only implemented in the Flash Attention path.
        // Without FA, attention computes dot(Q_unrotated, K_rotated) = garbage.
        // Must enable FA BEFORE sched_reserve() so the scheduler knows FA is required
        // and builds the graph plan with FA ops on GPU from the start.
        {
            const bool turbo_k = ggml_is_turbo_kv_type(params.type_k);
            const bool turbo_v = ggml_is_turbo_kv_type(params.type_v);
            const bool vbr_layer_schedule = turbo_vbr_layer_schedule_enabled();
            // dynamic VBR with the f16 entry tier: no turbo types at init, but later degrades
            // flip tensors to turbo (FA-only decode) and the VMM gate needs v_trans == false
            if (turbo_k || turbo_v || vbr_layer_schedule || params.vbr_dynamic) {
                if (!cparams.flash_attn) {
                    LLAMA_LOG_WARN("%s: turbo/VBR KV cache requires Flash Attention — enabling automatically\n", __func__);
                    cparams.flash_attn = true;
                }
                cparams.auto_fa = false;  // turbo/VBR requires FA — don't let sched_reserve override
            }
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        // Fused GDN kernels are only tested on NVIDIA CUDA. Disable on ROCm/MUSA/other.
        bool have_cuda_gpu = false;
        ggml_backend_dev_t gpu_dev = nullptr;
        if (ggml_backend_t gpu_backend = find_gpu_backend()) {
            gpu_dev = ggml_backend_get_device(gpu_backend);
        } else if (ggml_backend_t meta_backend = find_meta_backend()) {
            // --split-mode tensor: the compute device is the meta device, which
            // find_gpu_backend rejects by type — judge by its first simple device
            // (the meta backend has a dedicated GDN split handler, head-parallel)
            gpu_dev = ggml_backend_meta_dev_simple_dev(ggml_backend_get_device(meta_backend), 0);
        }
        if (gpu_dev) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(gpu_dev);
            const char * reg_name = ggml_backend_reg_name(reg);
            if (reg_name && (strncmp(reg_name, "CUDA", 4) == 0 || strncmp(reg_name, "ROCm", 4) == 0)) {
                // HIP builds register as "ROCm": the fused-GDN path compiles and runs there
                // (validated on RDNA3, buun-llama-cpp#69) — the CUDA-only name check was
                // silently dropping AMD to the decomposed ops (~9% generation speed)
                have_cuda_gpu = true;
            }
        }

        if (!have_cuda_gpu) {
            cparams.fused_gdn_ar = false;
            cparams.fused_gdn_ch = false;
            cparams.auto_fgdn    = false;
            LLAMA_LOG_INFO("%s: fused Gated Delta Net disabled (non-CUDA backend)\n", __func__);
        }
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails.
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }

    // reserve worst-case graph
    // when logits_all is false, reserve for n_seqs outputs only to save VRAM on big-vocab models
    const bool     reserve_all_outputs = cparams.logits_all || cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP || cparams.embeddings || cparams.pooling_type != LLAMA_POOLING_TYPE_NONE;
    const uint32_t n_outputs_pp = std::min(reserve_all_outputs ? n_tokens : n_seqs, cparams.n_outputs_max);

    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_outputs_pp, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const bool     reserve_all_outputs = cparams.logits_all || cparams.embeddings || cparams.pooling_type != LLAMA_POOLING_TYPE_NONE || cparams.ctx_type == LLAMA_CONTEXT_TYPE_MTP;
        const uint32_t n_outputs           = std::min(reserve_all_outputs ? n_tokens : n_seqs, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

int32_t * llama_context::get_logits_argmax() {
    synchronize();
    if (logits_argmax_buf.empty()) {
        return nullptr;
    }
    return logits_argmax_buf.data();
}

int32_t llama_context::get_logits_argmax_n() {
    return logits_argmax_count;
}

int32_t llama_context::get_logits_argmax_k() {
    return logits_argmax_k;
}

float * llama_context::get_logits_argmax_probs() {
    synchronize();
    if (logits_argmax_prob_buf.empty()) {
        return nullptr;
    }
    return logits_argmax_prob_buf.data();
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

// Readers return data from the active DFlash slot; multi-slot callers must
// call llama_dflash_set_active_slot() before reading.
float * llama_context::get_layer_hidden(int layer_idx) {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    if (!sh || layer_idx < 0 || layer_idx >= (int) sh->size()) {
        return nullptr;
    }
    return (*sh)[layer_idx].data.data();
}

int64_t llama_context::get_layer_hidden_n_tokens(int layer_idx) const {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    if (!sh || layer_idx < 0 || layer_idx >= (int) sh->size()) {
        return 0;
    }
    return (*sh)[layer_idx].n_tokens;
}

int64_t llama_context::get_layer_hidden_n_embd(int layer_idx) const {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    if (!sh || layer_idx < 0 || layer_idx >= (int) sh->size()) {
        return 0;
    }
    return (*sh)[layer_idx].n_embd;
}

int32_t llama_context::get_n_layer_hiddens() const {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    return sh ? (int32_t) sh->size() : 0;
}

// helper: read tensor data into a raw float pointer, handling non-contiguous views
static void dflash_read_tensor_to(struct ggml_tensor * t, float * dst, size_t n_floats) {
    if (ggml_is_contiguous(t)) {
        const size_t n_bytes = n_floats * sizeof(float);
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(dst, t->data, n_bytes);
        } else {
            ggml_backend_tensor_get(t, dst, 0, n_bytes);
        }
        return;
    }

    // non-contiguous view: read each innermost-contiguous slice separately
    // for 4D [ne0, ne1, ne2, ne3], ne0*ne1 is contiguous if nb[1]==ne[0]*elem_size
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const size_t esz = ggml_element_size(t);

    // find the largest contiguous inner chunk
    size_t contig_elems = ne0;
    if (t->nb[1] == ne0 * esz) {
        contig_elems = ne0 * ne1;
        if (t->nb[2] == ne0 * ne1 * esz) {
            contig_elems = ne0 * ne1 * ne2;
        }
    }

    size_t dst_off = 0;
    size_t n_chunks = n_floats / contig_elems;
    const size_t chunk_bytes = contig_elems * sizeof(float);

    for (size_t i = 0; i < n_chunks; ++i) {
        // compute source offset by iterating through outer dimensions
        size_t src_off = 0;
        size_t idx = i;
        if (contig_elems == (size_t)(ne0)) {
            int64_t i1 = idx % ne1; idx /= ne1;
            int64_t i2 = idx % ne2; idx /= ne2;
            int64_t i3 = idx;
            src_off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        } else if (contig_elems == (size_t)(ne0 * ne1)) {
            int64_t i2 = idx % ne2; idx /= ne2;
            int64_t i3 = idx;
            src_off = i2 * t->nb[2] + i3 * t->nb[3];
        } else {
            int64_t i3 = idx;
            src_off = i3 * t->nb[3];
        }

        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(dst + dst_off, (const char *)t->data + src_off, chunk_bytes);
        } else {
            ggml_backend_tensor_get(t, dst + dst_off, src_off, chunk_bytes);
        }
        dst_off += contig_elems;
    }
}

// helper: read tensor data to a float vector, handling non-contiguous views
static void dflash_read_tensor(struct ggml_tensor * t, std::vector<float> & dst, size_t n_floats) {
    dst.resize(n_floats);
    dflash_read_tensor_to(t, dst.data(), n_floats);
}

// DFlash eval callback: captures hidden state tensors + tape data during graph execution
// without modifying the compute graph (zero FP impact on model computation)
static bool dflash_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (dflash_capture_data *) user_data;
    const llama_ubatch * ub = cap->ubatch;
    const uint32_t n_seqs_unq = ub ? ub->n_seqs_unq : 0;

    auto h_it = cap->hidden_name_idx.find(t->name);

    if (ask) {
        if (h_it != cap->hidden_name_idx.end()) {
            // graph-embedded staging copies capture this ubatch — skipping here avoids
            // the per-layer graph chop + full-device sync the callback would force
            return !cap->stage_active;
        }
        if (cap->tape_enabled && cap->tape_name_map.count(t->name)) {
            if (cap->active_tape()) {
                // GPU tape: k/v/gate/beta captured by graph-embedded per-seq copies.
                // Only QKV mixed needs eval callback (for CPU-side conv state rebuild).
                // Multi-seq is supported — tensor stored with per-seq metadata.
                auto it = cap->tape_name_map.find(t->name);
                if (it == cap->tape_name_map.end() || it->second.second != DFLASH_TAPE_QKV) {
                    return false;
                }
                // Tensor split: QKV is graph-staged too (single-seq) — the callback's
                // meta get_tensor gather would misorder the segmented channels, and the
                // chop would force a per-layer all-device sync
                return !(cap->active_tape()->qkv_staged() && n_seqs_unq <= 1);
            }
            // CPU tape fallback: no multi-seq support
            if (n_seqs_unq > 1) {
                return false;
            }
            return true;
        }
        return false;
    }

    // ask=false: tensor data is ready, read it back. dflash_reset_hidden_capture()
    // (called at the top of decode()) zeroes buf.n_tokens for every slot before
    // the ubatch loop, so each slot's buffer accumulates only that slot's tokens
    // (in their ubatch order) across all ubatches in this llama_decode() call.
    if (h_it != cap->hidden_name_idx.end()) {
        const int64_t new_embd = t->ne[0];
        const int64_t new_n    = t->ne[1];
        const size_t  h_idx    = h_it->second;

        if (n_seqs_unq <= 1) {
            // single-seq fast path: route the whole tensor to one slot
            const int slot = ub ? ub->seq_id_unq[0] : -1;
            auto * sh = cap->slot_hiddens(slot);
            if (!sh) {
                return true; // no DFlash slot for this seq; skip capture
            }
            GGML_ASSERT(h_idx < sh->size());
            auto & buf = (*sh)[h_idx];
            buf.n_embd = new_embd;
            const size_t old_elems = (size_t) buf.n_tokens * (size_t) new_embd;
            const size_t add_elems = (size_t) new_n * (size_t) new_embd;
            buf.data.resize(old_elems + add_elems);
            dflash_read_tensor_to(t, buf.data.data() + old_elems, add_elems);
            buf.n_tokens += new_n;
            return true;
        }

        // multi-seq scatter: read full tensor once, count tokens per slot to
        // pre-reserve destination buffers, then append each token's hidden
        // vector to its owning slot's buffer in one pass.
        GGML_ASSERT(ub && (int64_t) ub->n_tokens == new_n);
        cap->scatter_buf.resize((size_t) new_embd * (size_t) new_n);
        dflash_read_tensor_to(t, cap->scatter_buf.data(), cap->scatter_buf.size());

        const int n_slots = cap->hiddens ? (int) cap->hiddens->size() : 0;
        for (uint32_t s = 0; s < n_seqs_unq; ++s) {
            const llama_seq_id seq = ub->seq_id_unq[s];
            if (seq < 0 || seq >= n_slots) continue;
            auto & slot_bufs = (*cap->hiddens)[seq];
            if (h_idx >= slot_bufs.size()) continue;
            auto & buf = slot_bufs[h_idx];
            buf.n_embd = new_embd;
            // Worst-case: all remaining tokens belong to this seq. Reserving
            // up to that bound costs at most one realloc per slot per ubatch
            // (vs one per token without reserve).
            buf.data.reserve((size_t) (buf.n_tokens + new_n) * (size_t) new_embd);
        }

        for (int64_t i = 0; i < new_n; ++i) {
            const llama_seq_id seq = ub->seq_id[i][0];
            if (seq < 0 || seq >= n_slots) continue;
            auto & slot_bufs = (*cap->hiddens)[seq];
            if (h_idx >= slot_bufs.size()) continue;
            auto & buf = slot_bufs[h_idx];
            const size_t old_elems = (size_t) buf.n_tokens * (size_t) new_embd;
            buf.data.resize(old_elems + (size_t) new_embd);
            std::memcpy(buf.data.data() + old_elems,
                        cap->scatter_buf.data() + (size_t) i * (size_t) new_embd,
                        (size_t) new_embd * sizeof(float));
            buf.n_tokens += 1;
        }
        return true;
    }

    // tape recording
    if (cap->tape_enabled) {
        auto it = cap->tape_name_map.find(t->name);
        if (it != cap->tape_name_map.end()) {
            int layer_idx = it->second.first;
            int type      = it->second.second;
            auto & tape   = cap->tape_layers[layer_idx];

            // when GPU tape is active, k/v/gate/beta are captured by graph-embedded copies
            // only qkv_mixed still needs eval callback (for CPU-side conv state rebuild)
            if (cap->active_tape() && type != DFLASH_TAPE_QKV) {
                return true; // skip — already on GPU
            }

            size_t n_elem = ggml_nelements(t);

            switch (type) {
                case DFLASH_TAPE_K:
                    tape.S_k = t->ne[0];
                    tape.H_k = t->ne[1];
                    tape.n_tokens = (int) t->ne[2];
                    dflash_read_tensor(t, tape.k, n_elem);
                    break;
                case DFLASH_TAPE_V:
                    tape.S_v = t->ne[0];
                    tape.H_v = t->ne[1];
                    dflash_read_tensor(t, tape.v, n_elem);
                    break;
                case DFLASH_TAPE_GATE:
                    dflash_read_tensor(t, tape.gate, n_elem);
                    break;
                case DFLASH_TAPE_BETA:
                    dflash_read_tensor(t, tape.beta, n_elem);
                    break;
                case DFLASH_TAPE_QKV:
                    tape.conv_channels = t->ne[0];
                    tape.n_tokens = (int) t->ne[1]; // tokens per seq (ne[1] of 3D [ch, n_seq_tokens, n_seqs])
                    if (ub && n_seqs_unq > 1) {
                        tape.n_seqs = std::min((int) n_seqs_unq, (int) LLAMA_DFLASH_MAX_SLOTS);
                        for (int s = 0; s < tape.n_seqs; ++s) {
                            tape.seq_ids[s] = ub->seq_id_unq[s];
                        }
                    } else {
                        tape.n_seqs = 1;
                        tape.seq_ids[0] = ub ? ub->seq_id_unq[0] : 0;
                    }
                    dflash_read_tensor(t, tape.qkv_mixed, n_elem);
                    break;
            }
            return true;
        }
    }

    return true;
}

void llama_context::set_dflash_sample_temp(float temp) {
    cparams.dflash_sample_temp = temp;
}

void llama_context::set_dflash_topk(int k) {
    cparams.dflash_topk = (k >= 1) ? k : 1;
    // invalidate graph cache since output tensor shape changes with K
    gf_res_prev->reset();
}

void llama_context::set_dflash_n_slots(int n) {
    const int clamped = std::max(1, std::min(n, (int) LLAMA_DFLASH_MAX_SLOTS));
    if (cparams.dflash_n_slots == clamped) {
        return;
    }
    cparams.dflash_n_slots = clamped;
    // drafter graph ctx_len depends on n_slots → force a fresh reserve on next decode
    sched_need_reserve = true;
    gf_res_prev->reset();
}

void llama_context::set_dflash_capture(const int32_t * layer_ids, int32_t n_layers) {
    // store layer IDs for the graph builder (still needed so qwen35.cpp knows which layers)
    cparams.dflash_capture_layers.clear();
    for (int32_t i = 0; i < n_layers; ++i) {
        cparams.dflash_capture_layers.push_back(layer_ids[i]);
    }

    // set up eval callback for zero-graph-modification capture
    dflash_capture = std::make_unique<dflash_capture_data>();
    dflash_capture->hiddens = &layer_hiddens;
    layer_hiddens.assign(1, std::vector<dflash_layer_hidden_buf>(n_layers));

    for (int32_t i = 0; i < n_layers; ++i) {
        dflash_capture->layer_ids.push_back(layer_ids[i]);
        std::string name = "l_out-" + std::to_string(layer_ids[i]);
        dflash_capture->hidden_name_idx[name] = i;
        dflash_capture->tensor_names.push_back(std::move(name));
    }

    // install our eval callback (replaces any existing one)
    cparams.cb_eval = dflash_eval_callback;
    cparams.cb_eval_user_data = dflash_capture.get();

    // GPU tape, eval callback hidden scatter, and QKV per-seq metadata
    // all support multi-seq ubatches. However, the server's
    // batch can mix prompt + TG tokens from different slots; split_equal
    // on such mixed batches produces incorrect ubatches. Expose the flag
    // so callers can toggle it off for verify-only decodes.
    if (memory) {
        memory->set_force_split_seq(true);
    }

    allocate_capture_stage_gpu();
}

// GPU capture staging: one [n_embd, LLAMA_DFLASH_MAX_VERIFY_TOKENS] tensor per captured
// layer, allocated on the GPU (or in a meta buffer under --split-mode tensor, where the
// split-state callback defaults unknown names to MIRRORED — the post-allreduce l_out is
// mirrored too, so the graph-embedded copy is a device-local write on every GPU and the
// consumer reads shard 0). Failure to allocate just leaves the eval-callback path active.
void llama_context::allocate_capture_stage_gpu() {
    if (!dflash_capture || dflash_capture->layer_ids.empty() || !dflash_capture->stage_tensors.empty()) {
        return;
    }

    ggml_backend_buffer_type_t buft = nullptr;
    if (ggml_backend_t gpu_backend = find_gpu_backend()) {
        buft = ggml_backend_get_default_buffer_type(gpu_backend);
    } else if (ggml_backend_t meta_backend = find_meta_backend()) {
        buft = ggml_backend_get_default_buffer_type(meta_backend);
    }
    if (!buft) {
        return; // CPU-only context: host capture is already free of device syncs
    }

    const int n_layers   = (int) dflash_capture->layer_ids.size();
    const int max_tokens = (int) LLAMA_DFLASH_MAX_VERIFY_TOKENS;
    const int64_t n_embd = model.hparams.n_embd;

    size_t ctx_mem = ggml_tensor_overhead() * (n_layers + 2);
    struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
    ggml_context * stage_ctx = ggml_init(ctx_params);

    dflash_capture->stage_tensors.reserve(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        ggml_tensor * t = ggml_new_tensor_2d(stage_ctx, GGML_TYPE_F32, n_embd, max_tokens);
        ggml_format_name(t, "dflash_stage-%d", i);
        dflash_capture->stage_tensors.push_back(t);
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(stage_ctx, buft);
    if (!buf) {
        LLAMA_LOG_WARN("%s: failed to allocate GPU capture staging — falling back to eval-callback capture\n", __func__);
        dflash_capture->stage_tensors.clear();
        ggml_free(stage_ctx);
        return;
    }

    dflash_capture->stage_ctx = stage_ctx;
    dflash_capture->stage_buf = buf;
    dflash_capture->stage_max_tokens = max_tokens;

    // cparams.capture_stage stays null until the decode loop marks a covered ubatch

    LLAMA_LOG_INFO("%s: allocated GPU capture staging: %d layers x %d tokens x %" PRId64 " embd (%.1f MB)\n",
        __func__, n_layers, max_tokens, n_embd, ggml_backend_buffer_get_size(buf) / (1024.0 * 1024.0));
}

void llama_context::set_capture_stage_enabled(bool enabled) {
    if (!dflash_capture) {
        return;
    }
    dflash_capture->stage_enabled = enabled;
}

int32_t llama_context::dflash_capture_stage_get(int32_t layer_idx, const void ** data) {
    if (!dflash_capture || dflash_capture->stage_n_tokens <= 0 ||
        layer_idx < 0 || layer_idx >= (int32_t) dflash_capture->stage_tensors.size()) {
        return 0;
    }
    ggml_tensor * t = dflash_capture->stage_tensors[layer_idx];
    if (t->buffer && ggml_backend_buffer_is_meta(t->buffer)) {
        ggml_tensor * shard = ggml_backend_meta_buffer_simple_tensor(t, 0);
        if (!shard || !shard->data) {
            return 0;
        }
        *data = shard->data;
    } else {
        *data = t->data;
    }
    return dflash_capture->stage_n_tokens;
}

void llama_context::dflash_reset_hidden_capture() {
    if (!dflash_capture) {
        return;
    }
    // reset every slot because a single decode() may hold ubatches for multiple slots
    for (auto & slot_bufs : layer_hiddens) {
        for (auto & buf : slot_bufs) {
            buf.n_tokens = 0;
        }
    }
    // The decode loop sets ubatch per iteration; null it here so a callback
    // that fires outside the loop can't read a stale pointer.
    dflash_capture->ubatch = nullptr;
    // Staging validity is per-decode: the decode loop re-arms it for staged ubatches.
    dflash_capture->stage_active = false;
    dflash_capture->stage_n_tokens = 0;
    dflash_capture->tape_stage_n_tokens = 0;
}

// idempotent: populates recurrent-layer ids + tape name map the first time it's called.
// Both set_tape_recording(true) and allocate_tape_gpu() fall through here so the setup
// order between them is flexible.
void llama_context::dflash_ensure_recurrent_setup() {
    if (!dflash_capture || !dflash_capture->recurrent_layer_ids.empty()) {
        return;
    }
    const auto & hparams = model.hparams;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (hparams.is_recr(il)) {
            int idx = (int) dflash_capture->recurrent_layer_ids.size();
            dflash_capture->recurrent_layer_ids.push_back(il);

            std::string il_str = std::to_string(il);
            dflash_capture->tape_name_map["k_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_K};
            dflash_capture->tape_name_map["v_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_V};
            dflash_capture->tape_name_map["gate-" + il_str]                   = {idx, DFLASH_TAPE_GATE};
            dflash_capture->tape_name_map["beta-" + il_str]                   = {idx, DFLASH_TAPE_BETA};
            dflash_capture->tape_name_map["linear_attn_qkv_mixed-" + il_str] = {idx, DFLASH_TAPE_QKV};
        }
    }
    dflash_capture->tape_layers.resize(dflash_capture->recurrent_layer_ids.size());
}

void llama_context::set_tape_recording(bool enable) {
    if (!dflash_capture) {
        return;
    }

    dflash_capture->tape_enabled = enable;

    if (enable) {
        dflash_ensure_recurrent_setup();
        if (dflash_capture->tapes.empty()) {
            allocate_tape_gpu(1, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
        }
    }

    // expose to graph builder via cparams — populate all tape pointers so graph
    // reservation accounts for worst-case per-seq copy ops.
    if (enable && !dflash_capture->tapes.empty()) {
        const int n_tapes = (int) dflash_capture->tapes.size();
        cparams.tape_gpu = dflash_capture->tapes[0].get();
        cparams.tape_gpu_n_seqs = n_tapes;
        for (int s = 0; s < n_tapes && s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = dflash_capture->tapes[s].get();
        }
        for (int s = n_tapes; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = nullptr;
        }
    } else {
        cparams.tape_gpu = nullptr;
        cparams.tape_gpu_n_seqs = 0;
        for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = nullptr;
        }
    }
}

static llama_memory_recurrent * get_recurrent_mem(llama_memory_t mem);
static bool dflash_states_on_one_device(const llama_hparams & hparams, llama_memory_recurrent * mem_recurrent);

void llama_context::allocate_tape_gpu(int n_slots, int max_tokens) {
    if (!dflash_capture) {
        return;
    }

    if (n_slots < 1) {
        n_slots = 1;
    }

    // Keep layer_hiddens outer dim in sync with the slot count regardless of
    // whether GPU tape gets allocated. Hidden-state capture is needed by every
    // DFlash-enabled context (target side); tape allocation only fires for
    // models with DeltaNet-style recurrent layers (drafter side).
    if (!layer_hiddens.empty() && (int) layer_hiddens.size() != n_slots) {
        const size_t n_capture_layers = layer_hiddens.front().size();
        layer_hiddens.resize(n_slots);
        for (auto & slot_bufs : layer_hiddens) {
            if (slot_bufs.size() != n_capture_layers) {
                slot_bufs.resize(n_capture_layers);
            }
        }
    }

    dflash_ensure_recurrent_setup();

    if (dflash_capture->recurrent_layer_ids.empty()) {
        return;
    }

    ggml_backend_t gpu_backend  = find_gpu_backend();
    ggml_backend_t meta_backend = gpu_backend ? nullptr : find_meta_backend();
    if (!gpu_backend && !meta_backend) {
        return; // no GPU, fall back to CPU tape via eval callback
    }

    // The meta tape split rules assume the fused-GDN k layout (H_k = n_group, not
    // repeated). With decomposed GDN under tensor split, skip the tape — the server
    // then rolls back via exact re-decode.
    if (meta_backend && !(cparams.fused_gdn_ar && cparams.fused_gdn_ch)) {
        if (dflash_capture) {
            dflash_capture->tape_meta_failed = true;
        }
        return;
    }

    // Once a GPU tape exists, the eval callback stops capturing k/v/gate/beta on the CPU —
    // so only allocate it when replay can actually use it (states on one non-host device,
    // or head-sharded behind one meta buffer where replay runs per simple device — see
    // llama_dflash_tape_replay_available). Otherwise fall back to the CPU tape, which
    // captures everything the CPU replay needs.
    {
        auto * mem_recurrent = get_recurrent_mem(memory.get());
        if (!mem_recurrent || (!meta_backend && !dflash_states_on_one_device(model.hparams, mem_recurrent))) {
            return;
        }
    }

    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const int n_rec = (int) rec_ids.size();

    // DeltaNet dimensions
    // k shape at capture: [ssm_d_state, H_k, n_tokens] where H_k depends on fused GDN
    // v/gate/beta shape: [S, H_v, n_tokens] or [1, H_v, n_tokens]
    const int64_t S = hparams.ssm_d_state;     // 256 for Qwen3.5-27B
    const int64_t H_v = hparams.ssm_dt_rank;   // 8 (num_v_heads)
    // when fused GDN is active, k is NOT repeated (kernel handles GQA internally)
    const int64_t H_k = (cparams.fused_gdn_ar && cparams.fused_gdn_ch)
                       ? (int64_t) hparams.ssm_n_group   // 1 (not repeated)
                       : H_v;                             // 8 (repeated)

    dflash_capture->tapes.clear();
    dflash_capture->tapes.reserve(n_slots);

    size_t total_size = 0;

    for (int slot = 0; slot < n_slots; ++slot) {
        // allocate ggml context for this slot's tensor descriptors (k/v/gate/beta + qkv staging)
        size_t ctx_mem = ggml_tensor_overhead() * (n_rec * 5 + 2);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * tape_ctx = ggml_init(ctx_params);

        auto tape = std::make_unique<dflash_tape_gpu>();
        tape->layers.resize(n_rec);
        tape->layer_ids = dflash_capture->recurrent_layer_ids;
        tape->max_tokens = max_tokens;
        tape->ctx = tape_ctx;

        for (int li = 0; li < n_rec; ++li) {
            auto & tl = tape->layers[li];
            const int il = rec_ids[li];
            tl.k    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_k, (int64_t)max_tokens);
            tl.v    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_v, (int64_t)max_tokens);
            tl.gate = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
            tl.beta = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
            // names drive the meta split-state rules (llama_meta_device_get_split_state):
            // shards must line up with the GDN input tensors the graph copies slice from
            ggml_format_name(tl.k,    "dflash_tape_k_l%d",   il);
            ggml_format_name(tl.v,    "dflash_tape_v_l%d",   il);
            ggml_format_name(tl.gate, "dflash_tape_g_l%d",   il);
            ggml_format_name(tl.beta, "dflash_tape_b_l%d",   il);
            if (meta_backend) {
                // qkv staging (meta only): the conv rebuild's eval-callback capture reads
                // the qkv compute tensor via meta get_tensor, whose INFERRED split state
                // can lose the channel-segment interleave (canonically misordered gather
                // → corrupted conv window). Stage it in a tape tensor instead — its
                // name-rule split is authoritative, so write and gather agree. Also
                // removes the per-recurrent-layer graph chop the callback forced.
                const int64_t conv_ch = (int64_t) hparams.n_embd_r() / (hparams.ssm_d_conv - 1);
                tl.qkv = ggml_new_tensor_2d(tape_ctx, GGML_TYPE_F32, conv_ch, (int64_t)max_tokens);
                ggml_format_name(tl.qkv, "dflash_tape_qkv_l%d", il);
            }
        }

        tape->buf = ggml_backend_alloc_ctx_tensors(tape_ctx, gpu_backend ? gpu_backend : meta_backend);

        if (!tape->buf) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU tape buffer for slot %d, falling back to CPU tape\n",
                __func__, slot);
            ggml_free(tape_ctx);
            dflash_capture->tapes.clear();
            return;
        }

        total_size += ggml_backend_buffer_get_size(tape->buf);
        dflash_capture->tapes.push_back(std::move(tape));
    }

    // Under tensor split, replay runs one graph per simple device over shard views. That
    // is only sound when the tape's per-device head shards line up with the state cache's:
    // device j's tape v/gate/beta heads must be exactly the heads whose S x S state blocks
    // live in device j's s_l shard. Verify once here; on mismatch (unusual --tensor-split
    // ratios can round shard boundaries differently) drop the tape — the server then uses
    // the exact re-decode rollback.
    if (meta_backend) {
        auto * mem_recurrent = get_recurrent_mem(memory.get());
        const size_t n_devs = ggml_backend_meta_n_backends(meta_backend);
        bool consistent = mem_recurrent != nullptr;
        for (size_t j = 0; consistent && j < n_devs; ++j) {
            for (int li = 0; consistent && li < n_rec; ++li) {
                const int il = rec_ids[li];
                ggml_tensor * v_shard = ggml_backend_meta_buffer_simple_tensor(dflash_capture->tapes[0]->layers[li].v, j);
                ggml_tensor * g_shard = ggml_backend_meta_buffer_simple_tensor(dflash_capture->tapes[0]->layers[li].gate, j);
                ggml_tensor * s_shard = ggml_backend_meta_buffer_simple_tensor(mem_recurrent->s_l[il], j);
                if (!v_shard || !g_shard || !s_shard ||
                    v_shard->ne[1] != g_shard->ne[1] ||
                    s_shard->ne[0] != S * S * v_shard->ne[1]) {
                    LLAMA_LOG_WARN("%s: tape/state shard mismatch (dev %zu, layer %d: tape H_v=%" PRId64 ", state n_embd=%" PRId64 ") — dropping GPU tape, rollback falls back to re-decode\n",
                        __func__, j, il, v_shard ? v_shard->ne[1] : -1, s_shard ? s_shard->ne[0] : -1);
                    consistent = false;
                }
            }
        }
        if (!consistent) {
            dflash_capture->tapes.clear();
            dflash_capture->tape_meta_failed = true;
            return;
        }
    }

    dflash_capture->active_tape_idx = 0;

    LLAMA_LOG_INFO("%s: allocated GPU tape buffers: %.1f MB total (%d slot%s, %d layers, %d max tokens)\n",
        __func__, total_size / (1024.0 * 1024.0), n_slots, n_slots == 1 ? "" : "s", n_rec, max_tokens);
}

void llama_context::set_active_dflash_slot(int slot_idx) {
    if (!dflash_capture || dflash_capture->tapes.empty()) {
        return;
    }
    if (slot_idx < 0 || slot_idx >= (int) dflash_capture->tapes.size()) {
        LLAMA_LOG_WARN("%s: slot %d out of range [0, %d) — ignoring\n",
            __func__, slot_idx, (int) dflash_capture->tapes.size());
        return;
    }
    if (slot_idx == dflash_capture->active_tape_idx) {
        return;
    }
    dflash_capture->active_tape_idx = slot_idx;
    cparams.tape_gpu = dflash_capture->active_tape();
    // sync per-seq array (single-seq mode for external callers)
    cparams.tape_gpu_seqs[0] = cparams.tape_gpu;
    cparams.tape_gpu_n_seqs = 1;
    // graph nodes hold references to the previous slot's tape tensors; invalidate
    // so the next decode rebuilds with the new slot's tensors.
    if (gf_res_prev) {
        gf_res_prev->reset();
    }
}

ggml_backend_t llama_context::find_gpu_backend() {
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_IGPU)) { // accept APU/IGPU (gfx1151 Strix Halo)
            return backend.get();
        }
    }
    return nullptr;
}

// --split-mode tensor: the compute backend is the meta backend (device type META,
// invisible to find_gpu_backend). Tape/replay code paths that can operate per-device
// use this to detect it.
ggml_backend_t llama_context::find_meta_backend() {
    for (auto & backend : backends) {
        if (ggml_backend_is_meta(backend.get())) {
            return backend.get();
        }
    }
    return nullptr;
}

// true iff every recurrent state buffer is resident on one non-host device — the
// precondition for the GPU tape-replay graph (views into s_l, one compute backend)
static bool dflash_states_on_one_device(const llama_hparams & hparams, llama_memory_recurrent * mem_recurrent) {
    ggml_backend_dev_t first_dev = nullptr;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (!hparams.is_recr(il)) {
            continue;
        }
        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        if (!s_tensor || !s_tensor->buffer) {
            continue;
        }
        if (ggml_backend_buffer_is_host(s_tensor->buffer)) {
            return false;
        }
        auto * buft = ggml_backend_buffer_get_type(s_tensor->buffer);
        auto * dev  = buft ? ggml_backend_buft_get_device(buft) : nullptr;
        if (dev) {
            if (!first_dev) {
                first_dev = dev;
            } else if (dev != first_dev) {
                return false;
            }
        }
    }
    return true;
}

bool llama_context::tape_replay_available() {
    auto * mem_recurrent = get_recurrent_mem(memory.get());
    if (!mem_recurrent) {
        return false;
    }

    if (find_gpu_backend()) {
        return dflash_states_on_one_device(model.hparams, mem_recurrent);
    }

    if (find_meta_backend()) {
        // tensor split: replay runs per simple device over shard views. Before capture
        // setup exists this is a predictive answer; once it does, the tape allocation
        // (with its shard-consistency check) is the authority — callers should re-probe
        // after speculative init.
        if (!dflash_capture) {
            return true;
        }
        if (dflash_capture->tape_meta_failed) {
            return false;
        }
        if (dflash_capture->tapes.empty()) {
            allocate_tape_gpu(1, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
        }
        return !dflash_capture->tapes.empty();
    }

    return false;
}

// Tensor-split GDN rollback: one small graph per simple device, entirely over that
// device's shards — tape k/v/gate/beta shards (written by the graph-embedded copies,
// sharded by the dflash_tape_* split rules) and the s_l state shard. GDN heads are
// independent, so no cross-device communication is needed; correctness of the
// tape-shard/state-shard head alignment is verified once in allocate_tape_gpu.
// Shared tail of a tape-replay layer graph: sigmoid(beta) → 4D state view at the cell →
// gated_delta_net (K=1; caller supplies Q with zero semantics — the attention output is
// discarded, only the state update matters) → extract the new state from the result
// (layout: [attn_output | new_state]) and copy it back over the cell. The state-view
// stride math must match the forward pass's ggml_reshape_4d layout (see qwen3next.cpp);
// a flat [S*S*H_v,1,1,1] view is rejected by the synced op's state-shape asserts.
// Used by both the single-backend and the per-device meta replay paths — keep the
// subtle offset/layout math in one place.
static void dflash_build_gdn_state_update(
        ggml_context * ctx, ggml_cgraph * graph,
        ggml_tensor * q_in, ggml_tensor * k_in, ggml_tensor * v_in,
        ggml_tensor * g_in, ggml_tensor * b_in,
        ggml_tensor * s_tensor, size_t s_byte_offset,
        int64_t S, int64_t H_v, int n_accepted) {
    const int64_t n_embd_s = S * S * H_v;
    const size_t  s_esz    = ggml_element_size(s_tensor);

    ggml_tensor * b_sigmoid = ggml_sigmoid(ctx, b_in);
    ggml_tensor * s_view = ggml_view_4d(ctx, s_tensor, S, S, H_v, (int64_t) 1,
        S * s_esz, S * S * s_esz, n_embd_s * s_esz, s_byte_offset);

    ggml_tensor * result = ggml_gated_delta_net(ctx, q_in, k_in, v_in, g_in, b_sigmoid, s_view, /*K=*/1);

    const size_t attn_bytes = (size_t) (S * H_v * n_accepted) * ggml_element_size(result);
    ggml_tensor * result_state = ggml_view_1d(ctx, result, n_embd_s, attn_bytes);
    ggml_tensor * s_write = ggml_view_1d(ctx, s_tensor, n_embd_s, s_byte_offset);

    ggml_build_forward_expand(graph, ggml_cpy(ctx, result_state, s_write));
}

void llama_context::tape_replay_meta(ggml_backend_t meta_backend, llama_memory_recurrent * mem_recurrent,
                                     int32_t cell_idx, int n_accepted, llama_seq_id seq_id) {
    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;

    dflash_tape_gpu * tgpu = nullptr;
    if (seq_id >= 0 && seq_id < (int) dflash_capture->tapes.size()) {
        tgpu = dflash_capture->tapes[seq_id].get();
    }
    if (!tgpu) {
        LLAMA_LOG_WARN("%s: no tape for seq %d — GDN replay skipped, recurrent state will be stale\n", __func__, seq_id);
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return;
    }

    // with graph-staged qkv the eval callback no longer sets per-layer token counts —
    // every recurrent layer of a staged decode covers the same tokens
    if (tgpu->qkv_staged() && dflash_capture->tape_stage_n_tokens > 0) {
        for (auto & tape : tape_layers) {
            tape.n_tokens = dflash_capture->tape_stage_n_tokens;
        }
    }

    const int64_t S = hparams.ssm_d_state;
    const size_t n_devs = ggml_backend_meta_n_backends(meta_backend);
    const int n_rec = (int) rec_ids.size();

    GGML_ASSERT(dflash_capture->replay_meta_ctxs.empty()); // tape_replay_sync ran (tape_replay entry syncs)
    dflash_capture->replay_meta_bufs.resize(n_devs, nullptr);
    dflash_capture->replay_meta_buf_sizes.resize(n_devs, 0);

    // tape index for each recurrent layer (device-invariant)
    std::vector<int> li_to_gpu(n_rec, -1);
    for (int li = 0; li < n_rec; ++li) {
        for (int i = 0; i < (int) tgpu->layer_ids.size(); ++i) {
            if (tgpu->layer_ids[i] == rec_ids[li]) { li_to_gpu[li] = i; break; }
        }
    }

    bool launched = false;
    for (size_t j = 0; j < n_devs; ++j) {
        ggml_backend_t simple_backend = ggml_backend_meta_simple_backend(meta_backend, j);

        // per layer: 4 tape views + q scale + b sigmoid + s view + GDN + result view +
        // s write + cpy = 11 graph nodes (views are ops); size with headroom
        size_t ctx_mem = ggml_tensor_overhead() * ((size_t) n_rec * 16 + 4) + ggml_graph_overhead_custom(n_rec * 14, false);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * ctx = ggml_init(ctx_params);
        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_rec * 14, false);

        int n_nodes = 0;
        for (int li = 0; li < n_rec; ++li) {
            const int il = rec_ids[li];
            auto & tape = tape_layers[li];
            // n_tokens comes from the qkv_mixed capture (or qkv staging) for this decode
            if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;

            const int gpu_li = li_to_gpu[li];
            if (gpu_li < 0) continue;
            auto & tl = tgpu->layers[gpu_li];

            ggml_tensor * k_shard = ggml_backend_meta_buffer_simple_tensor(tl.k, j);
            ggml_tensor * v_shard = ggml_backend_meta_buffer_simple_tensor(tl.v, j);
            ggml_tensor * g_shard = ggml_backend_meta_buffer_simple_tensor(tl.gate, j);
            ggml_tensor * b_shard = ggml_backend_meta_buffer_simple_tensor(tl.beta, j);
            ggml_tensor * s_shard = ggml_backend_meta_buffer_simple_tensor(mem_recurrent->s_l[il], j);
            if (!k_shard || !v_shard || !g_shard || !b_shard || !s_shard) continue;

            const int64_t H_k_j = k_shard->ne[1];
            const int64_t H_v_j = v_shard->ne[1];
            if (H_k_j <= 0 || H_v_j <= 0) continue; // this device holds no heads of this layer

            const int64_t n_embd_s_j = S * S * H_v_j;
            GGML_ASSERT(s_shard->ne[0] == n_embd_s_j); // verified at allocate_tape_gpu

            ggml_tensor * k_in = ggml_view_3d(ctx, k_shard, S, H_k_j, (int64_t) n_accepted,
                                              k_shard->nb[1], k_shard->nb[2], 0);
            ggml_tensor * v_in = ggml_view_3d(ctx, v_shard, S, H_v_j, (int64_t) n_accepted,
                                              v_shard->nb[1], v_shard->nb[2], 0);
            ggml_tensor * g_in = ggml_view_3d(ctx, g_shard, (int64_t) 1, H_v_j, (int64_t) n_accepted,
                                              g_shard->nb[1], g_shard->nb[2], 0);
            ggml_tensor * b_in = ggml_view_3d(ctx, b_shard, (int64_t) 1, H_v_j, (int64_t) n_accepted,
                                              b_shard->nb[1], b_shard->nb[2], 0);

            // Q: zeros of k's shape — produced in-graph, no host upload
            ggml_tensor * q_in = ggml_scale(ctx, k_in, 0.0f);

            dflash_build_gdn_state_update(ctx, graph, q_in, k_in, v_in, g_in, b_in,
                s_shard, (size_t) cell_idx * s_shard->nb[1], S, H_v_j, n_accepted);
            n_nodes++;
        }

        if (n_nodes == 0) {
            ggml_free(ctx);
            continue;
        }

        // allocate intermediates (scale/sigmoid/GDN results) in this device's persistent
        // grow-only scratch (same scheme as the single-backend path's replay_buf)
        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(simple_backend);
        const size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
        if (needed > dflash_capture->replay_meta_buf_sizes[j]) {
            if (dflash_capture->replay_meta_bufs[j]) {
                ggml_backend_buffer_free(dflash_capture->replay_meta_bufs[j]);
            }
            dflash_capture->replay_meta_bufs[j] = ggml_backend_buft_alloc_buffer(buft, needed);
            dflash_capture->replay_meta_buf_sizes[j] = dflash_capture->replay_meta_bufs[j]
                ? ggml_backend_buffer_get_size(dflash_capture->replay_meta_bufs[j]) : 0;
        }
        if (!dflash_capture->replay_meta_bufs[j]) {
            LLAMA_LOG_WARN("%s: failed to allocate replay buffer on device %zu — GDN replay incomplete, recurrent state will be stale\n", __func__, j);
            ggml_free(ctx);
            continue;
        }
        {
            struct ggml_tallocr talloc = ggml_tallocr_new(dflash_capture->replay_meta_bufs[j]);
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                if (t->data == nullptr && t->view_src == nullptr) {
                    ggml_tallocr_alloc(&talloc, t);
                } else if (t->view_src != nullptr && t->buffer == nullptr) {
                    ggml_backend_view_init(t);
                }
            }
        }

        ggml_backend_graph_compute_async(simple_backend, graph);

        dflash_capture->replay_meta_ctxs.push_back(ctx);
        launched = true;
    }

    if (!launched) {
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return;
    }

    // conv rebuild + pos advance deferred to tape_replay_sync()
    dflash_capture->replay_pending = true;
    dflash_capture->replay_gpu_backend = meta_backend; // synchronize() fans out to all simple backends
    dflash_capture->replay_n_accepted = n_accepted;
    dflash_capture->replay_cell_idx = cell_idx;
    dflash_capture->replay_seq_id = seq_id;
    dflash_capture->replay_mem_recurrent = mem_recurrent;
}

void llama_context::tape_replay(llama_seq_id seq_id, int n_accepted) {
    if (!dflash_capture || n_accepted <= 0) {
        return;
    }

    // ensure any previous async replay is complete before launching a new one
    tape_replay_sync();

    if (dflash_capture->tape_layers.empty()) {
        return;
    }

    auto * mem_recurrent = get_recurrent_mem(memory.get());
    if (!mem_recurrent) {
        LLAMA_LOG_WARN("%s: tape replay requires recurrent memory\n", __func__);
        return;
    }

    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;

    // find the tail cell for this seq_id
    int32_t cell_idx = -1;
    if (seq_id >= 0 && (uint32_t) seq_id < mem_recurrent->size) {
        int32_t tail = mem_recurrent->cells[seq_id].tail;
        if (tail >= 0) {
            cell_idx = tail;
        }
    }
    if (cell_idx < 0) {
        LLAMA_LOG_WARN("%s: no active cell for seq %d\n", __func__, seq_id);
        return;
    }

    const uint32_t n_embd_s = hparams.n_embd_s();

    // find a GPU backend for graph computation
    ggml_backend_t gpu_backend = find_gpu_backend();

    if (!gpu_backend) {
        // tensor split: replay per simple device over shard views (host reads of the
        // head-sharded meta tensors would be the #22 corruption class — never CPU here)
        if (ggml_backend_t meta_backend = find_meta_backend()) {
            if (dflash_capture->active_tape()) {
                tape_replay_meta(meta_backend, mem_recurrent, cell_idx, n_accepted, seq_id);
            } else {
                LLAMA_LOG_WARN("%s: tensor-split rollback without a GPU tape — GDN replay skipped, recurrent state will be stale\n", __func__);
                tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            }
            return;
        }
        tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return;
    }

    // partial offload: if any recurrent layer's state lives on CPU, fall back to CPU replay
    // (GPU graph uses DeviceToDevice copies that crash when the state buffer is host memory)
    // Also detect multi-device splits — if states span GPUs, fall back to CPU replay.
    if (!dflash_states_on_one_device(hparams, mem_recurrent)) {
        if (dflash_capture->active_tape()) {
            // unreachable by construction: allocate_tape_gpu only creates the GPU tape when
            // this same predicate holds, and states do not migrate afterwards. If it ever
            // fires, k/v/gate/beta live only in the GPU tape — there is nothing for
            // tape_replay_cpu to replay (see llama_dflash_tape_replay_available).
            LLAMA_LOG_WARN("%s: GPU tape active but states are host/multi-device — GDN replay skipped, recurrent state will be stale\n", __func__);
        } else {
            tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
        }
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return;
    }

    // GPU tape replay: build a ggml graph with GDN ops for all recurrent layers
    const int n_rec = (int) rec_ids.size();
    if (n_rec == 0) goto conv_rebuild;

    {
        // per layer: k_view + v_view + g_view + b_view + q + b_sigmoid + s_view + GDN + result_state + s_write + cpy = ~11 tensors
        size_t ctx_mem = ggml_tensor_overhead() * ((size_t)n_rec * 14 + 4) + ggml_graph_overhead_custom(n_rec * 12, false);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * ctx = ggml_init(ctx_params);

        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_rec * 12, false);

        struct replay_input {
            ggml_tensor * q;
            ggml_tensor * k;
            ggml_tensor * v;
            ggml_tensor * g;
            ggml_tensor * b;
            size_t tape_li;
            bool gpu_tape; // k/v/g/b are views into GPU tape (skip CPU upload)
        };
        std::vector<replay_input> inputs;
        inputs.reserve(n_rec);

        // look up GPU tape for this seq_id (graph-embedded copies wrote k/v/g/b here)
        dflash_tape_gpu * tgpu = nullptr;
        if (seq_id >= 0 && seq_id < (int) dflash_capture->tapes.size()) {
            tgpu = dflash_capture->tapes[seq_id].get();
        }

        for (int li = 0; li < n_rec; ++li) {
            int il = rec_ids[li];

            auto & tape = tape_layers[li];
            if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;

            // find this layer in GPU tape (if available)
            int gpu_li = -1;
            if (tgpu) {
                for (int i = 0; i < (int) tgpu->layer_ids.size(); ++i) {
                    if (tgpu->layer_ids[i] == il) { gpu_li = i; break; }
                }
            }

            int64_t S, H_k, H_v;
            ggml_tensor * k_in, * v_in, * g_in, * b_in;
            bool use_gpu_tape = (gpu_li >= 0);

            if (use_gpu_tape) {
                auto & tl = tgpu->layers[gpu_li];
                S   = tl.k->ne[0];
                H_k = tl.k->ne[1];
                H_v = tl.v->ne[1];
                // views into GPU tape buffers — already populated by graph-embedded copies
                k_in = ggml_view_3d(ctx, tl.k, S, H_k, (int64_t)n_accepted,
                                    tl.k->nb[1], tl.k->nb[2], 0);
                v_in = ggml_view_3d(ctx, tl.v, S, H_v, (int64_t)n_accepted,
                                    tl.v->nb[1], tl.v->nb[2], 0);
                g_in = ggml_view_3d(ctx, tl.gate, (int64_t)1, H_v, (int64_t)n_accepted,
                                    tl.gate->nb[1], tl.gate->nb[2], 0);
                b_in = ggml_view_3d(ctx, tl.beta, (int64_t)1, H_v, (int64_t)n_accepted,
                                    tl.beta->nb[1], tl.beta->nb[2], 0);
            } else {
                S   = tape.S_k;
                H_k = tape.H_k;
                H_v = tape.H_v;
                k_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t)n_accepted, (int64_t)1);
                v_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, (int64_t)n_accepted, (int64_t)1);
                g_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)n_accepted, (int64_t)1);
                b_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)n_accepted, (int64_t)1);
                ggml_set_input(k_in); ggml_set_input(v_in);
                ggml_set_input(g_in); ggml_set_input(b_in);
            }

            // Q: zeros (attention output discarded, only state update matters)
            ggml_tensor * q_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t)n_accepted, (int64_t)1);
            ggml_set_input(q_in);

            // one recurrent cell (n_embd_s) must be exactly the 4D state the shared
            // builder views — its read-back views rely on the same equality
            GGML_ASSERT((int64_t) n_embd_s == S * S * H_v);
            ggml_tensor * s_tensor = mem_recurrent->s_l[il];
            const size_t s_byte_offset = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);

            dflash_build_gdn_state_update(ctx, graph, q_in, k_in, v_in, g_in, b_in,
                s_tensor, s_byte_offset, S, H_v, n_accepted);

            inputs.push_back({ q_in, k_in, v_in, g_in, b_in, (size_t)li, use_gpu_tape });
        }

        if (inputs.empty()) {
            ggml_free(ctx);
            goto conv_rebuild;
        }

        // allocate non-view tensors on GPU (reuse persistent buffer)
        ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
        size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, gpu_buft);

        if (needed > dflash_capture->replay_buf_size) {
            if (dflash_capture->replay_buf) {
                ggml_backend_buffer_free(dflash_capture->replay_buf);
            }
            dflash_capture->replay_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
            dflash_capture->replay_buf_size = dflash_capture->replay_buf
                ? ggml_backend_buffer_get_size(dflash_capture->replay_buf) : 0;
        }

        if (!dflash_capture->replay_buf) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU buffer for tape replay, falling back to CPU\n", __func__);
            ggml_free(ctx);
            tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return;
        }

        // assign tensors within the persistent buffer
        {
            struct ggml_tallocr talloc = ggml_tallocr_new(dflash_capture->replay_buf);
            struct ggml_tensor * t = ggml_get_first_tensor(ctx);
            while (t) {
                if (t->data == nullptr && t->view_src == nullptr) {
                    ggml_tallocr_alloc(&talloc, t);
                } else if (t->view_src != nullptr && t->buffer == nullptr) {
                    ggml_backend_view_init(t);
                }
                t = ggml_get_next_tensor(ctx, t);
            }
        }

        // upload data for tensors that need it
        for (auto & inp : inputs) {
            // Q: always needs zeros
            {
                const int64_t S = inp.q->ne[0];
                const int64_t H = inp.q->ne[1];
                size_t q_size = (size_t)(S * H * n_accepted);
                if (dflash_capture->replay_zeros.size() < q_size) {
                    dflash_capture->replay_zeros.resize(q_size, 0.0f);
                }
                ggml_backend_tensor_set(inp.q, dflash_capture->replay_zeros.data(), 0, ggml_nbytes(inp.q));
            }

            if (!inp.gpu_tape) {
                auto & tape = tape_layers[inp.tape_li];
                const int64_t S   = tape.S_k;
                const int64_t H_k = tape.H_k;
                const int64_t H_v = tape.H_v;

                ggml_backend_tensor_set(inp.k, tape.k.data(), 0, S * H_k * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.v, tape.v.data(), 0, S * H_v * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.g, tape.gate.data(), 0, H_v * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.b, tape.beta.data(), 0, H_v * n_accepted * sizeof(float));
            }
        }

        // compute: launch GDN ops + state copies on GPU (async — overlap with next draft)
        ggml_backend_graph_compute_async(gpu_backend, graph);

        // save deferred state for async completion
        dflash_capture->replay_pending = true;
        dflash_capture->replay_gpu_backend = gpu_backend;
        dflash_capture->replay_graph_ctx = ctx; // freed in tape_replay_sync
        dflash_capture->replay_n_accepted = n_accepted;
        dflash_capture->replay_cell_idx = cell_idx;
        dflash_capture->replay_seq_id = seq_id;
        dflash_capture->replay_mem_recurrent = mem_recurrent;
        return; // conv rebuild deferred to tape_replay_sync()
    }

conv_rebuild:
    tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
}

void llama_context::tape_replay_conv(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id) {
    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;
    const uint32_t n_embd_r = hparams.n_embd_r();

    // rebuild conv state from qkv_mixed tape (small, CPU is fine)
    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers[li];

        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;
        if (tape.qkv_mixed.empty() || !mem_recurrent->r_l[il]) continue;

        // for multi-seq verify, QKV mixed has per-seq data packed
        // contiguously as [channels, n_seq_tokens, n_seqs]. Find offset.
        size_t qkv_seq_offset = 0;
        if (tape.n_seqs > 1) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                if (tape.seq_ids[s] == seq_id) { found = true; break; }
                qkv_seq_offset += (size_t) tape.n_tokens * (size_t) tape.conv_channels;
            }
            GGML_ASSERT(found && "tape_replay_conv: seq_id not found in tape");
        }

        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        const size_t r_offset = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);

        const int64_t conv_ch = tape.conv_channels;
        const int64_t conv_window = (int64_t)(n_embd_r / conv_ch); // kernel_size - 1

        std::vector<float> old_window(n_embd_r);
        ggml_backend_tensor_get(r_tensor, old_window.data(), r_offset, n_embd_r * sizeof(float));

        std::vector<float> new_conv(n_embd_r);
        for (int64_t w = 0; w < conv_window; ++w) {
            int src_pos = n_accepted + (int)w;
            for (int64_t ch = 0; ch < conv_ch; ++ch) {
                float val;
                if (src_pos < (int)conv_window) {
                    val = old_window[ch * conv_window + src_pos];
                } else {
                    val = tape.qkv_mixed[qkv_seq_offset + (src_pos - conv_window) * conv_ch + ch];
                }
                new_conv[ch * conv_window + w] = val;
            }
        }

        ggml_backend_tensor_set(r_tensor, new_conv.data(), r_offset, n_embd_r * sizeof(float));
    }

    mem_recurrent->cells[cell_idx].pos += n_accepted;
}

void llama_context::tape_replay_sync() {
    if (!dflash_capture || !dflash_capture->replay_pending) {
        return;
    }

    // wait for async GDN graph(s) to complete (a meta backend fans out to every device)
    ggml_backend_synchronize(dflash_capture->replay_gpu_backend);

    // free the graph context(s) — the meta scratch buffers are persistent (freed in dtor)
    ggml_free(dflash_capture->replay_graph_ctx);
    dflash_capture->replay_graph_ctx = nullptr;
    for (auto * ctx : dflash_capture->replay_meta_ctxs) {
        ggml_free(ctx);
    }
    dflash_capture->replay_meta_ctxs.clear();

    // qkv staged on GPU (tensor split): gather it for the host conv rebuild. The tape
    // tensor's name-rule split state makes this gather channel-order-correct — reading
    // the qkv compute tensor directly (old path) misordered the segment interleave.
    {
        dflash_tape_gpu * tg = nullptr;
        const llama_seq_id rsid = dflash_capture->replay_seq_id;
        if (rsid >= 0 && rsid < (llama_seq_id) dflash_capture->tapes.size()) {
            tg = dflash_capture->tapes[rsid].get();
        }
        const int n_tok = dflash_capture->tape_stage_n_tokens;
        if (tg && tg->qkv_staged() && n_tok > 0) {
            for (size_t li = 0; li < tg->layers.size(); ++li) {
                ggml_tensor * qkv = tg->layers[li].qkv;
                auto & tape = dflash_capture->tape_layers[li];
                const int64_t conv_ch = qkv->ne[0];
                tape.qkv_mixed.resize((size_t) conv_ch * n_tok);
                ggml_backend_tensor_get(qkv, tape.qkv_mixed.data(), 0, (size_t) n_tok * qkv->nb[1]);
                tape.conv_channels = conv_ch;
                tape.n_tokens = n_tok;
                tape.n_seqs = 1;
                tape.seq_ids[0] = rsid;
            }
        }
    }

    // finish conv rebuild + position advance
    tape_replay_conv(dflash_capture->replay_mem_recurrent,
                     dflash_capture->replay_cell_idx,
                     dflash_capture->replay_n_accepted,
                     dflash_capture->replay_seq_id);

    dflash_capture->replay_pending = false;
}

// CPU fallback for tape replay (used when no GPU backend available)
void llama_context::tape_replay_cpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;
    const uint32_t n_embd_s = hparams.n_embd_s();

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers[li];

        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;

        const int64_t S = tape.S_k;
        const int64_t H_k = tape.H_k;
        const int64_t H_v = tape.H_v;
        const int64_t head_ratio = H_v / H_k;

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        const size_t s_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
        std::vector<float> state(n_embd_s);
        ggml_backend_tensor_get(s_tensor, state.data(), s_offset, n_embd_s * sizeof(float));

        for (int tok = 0; tok < n_accepted; ++tok) {
            for (int64_t hv = 0; hv < H_v; ++hv) {
                int64_t hk = hv / head_ratio;
                float g_val = expf(tape.gate[tok * H_v + hv]);
                float b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv]));

                float * S_h = state.data() + hv * S * S;
                const float * k_t = tape.k.data() + tok * (S * H_k) + hk * S;
                const float * v_t = tape.v.data() + tok * (S * H_v) + hv * S;

                // kv = S^T @ k, delta = (v - g*kv) * beta, S = g*S + k⊗delta (fused)
                for (int64_t col = 0; col < S; ++col) {
                    float kv = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        kv += S_h[col * S + row] * k_t[row];
                    }
                    float delta_col = (v_t[col] - g_val * kv) * b_val;
                    for (int64_t row = 0; row < S; ++row) {
                        S_h[col * S + row] = g_val * S_h[col * S + row] + k_t[row] * delta_col;
                    }
                }
            }
        }

        ggml_backend_tensor_set(s_tensor, state.data(), s_offset, n_embd_s * sizeof(float));
    }
}

void llama_context::dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) {
        LLAMA_LOG_WARN("%s: dflash_rollback requires hybrid memory\n", __func__);
        return;
    }

    auto * mem_attn = mem_hybrid->get_mem_attn();
    auto * mem_recr = mem_hybrid->get_mem_recr();

    if (tree_bufs.n_tokens > 0) {
        // Tree mode: branch tokens may have polluted KV at accepted positions.
        // Remove ALL entries from n_past_before onwards and restore from backup.
        mem_attn->seq_rm(seq_id, n_past_before, -1);
        mem_attn->seq_cp(seq_backup, seq_id, n_past_before, -1);
        mem_attn->seq_rm(seq_backup, -1, -1);
    } else {
        // Flat mode: no duplicate entries at same position, safe to keep accepted KV
        int kv_keep_pos = n_past_before + n_accepted;
        mem_attn->seq_rm(seq_id, kv_keep_pos, -1);
        mem_attn->seq_rm(seq_backup, -1, -1);
    }

    // Recurrent state: restore from backup, then tape replay
    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp(seq_backup, seq_id, -1, -1);
    mem_recr->seq_rm(seq_backup, -1, -1);

    // Replay DeltaNet state updates for accepted tokens
    tape_replay(seq_id, n_accepted);
}

void llama_context::dflash_prepare_branch(llama_seq_id seq_id, llama_seq_id seq_backup, int depth) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) {
        LLAMA_LOG_WARN("%s: dflash_prepare_branch requires hybrid memory\n", __func__);
        return;
    }

    auto * mem_recr = mem_hybrid->get_mem_recr();

    // restore recurrent state from backup (keep backup intact for subsequent branches)
    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp(seq_backup, seq_id, -1, -1);

    // tape replay to get DeltaNet state after processing 'depth' tokens (root + main_path[1..depth-1])
    tape_replay(seq_id, depth);
}

// round up to next bucket: 16, 32, 64, 128, 256, 512, 1024, 2048, ...
static int64_t cross_bucket(int64_t n) {
    if (n <= 16) return 16;
    int64_t b = 1;
    while (b < n) b <<= 1;
    return b;
}

static int64_t dflash_max_cross_ctx() {
    static const int64_t max_ctx = [] {
        const char * e = getenv("GGML_DFLASH_MAX_CTX");
        return e ? (int64_t) atoi(e) : (int64_t) 4096;
    }();
    return max_ctx;
}

void llama_context::set_cross_data(const float * data, int64_t n_embd, int64_t n_tokens) {
    const int64_t max_ctx = dflash_max_cross_ctx();
    const int64_t capped = (max_ctx > 0 && n_tokens > max_ctx) ? max_ctx : n_tokens;
    const int64_t bucket = cross_bucket(capped);

    if (cross.n_enc != bucket) {
        sched_need_reserve = true;
    }
    cross.n_embd    = n_embd;
    cross.n_enc     = bucket;
    cross.n_enc_real = n_tokens;  // actual full data length (for windowing in set_input)
    cross.v_embd.resize(n_embd * n_tokens);
    if (data) {
        memcpy(cross.v_embd.data(), data, n_embd * n_tokens * sizeof(float));
    }
}

// Per-seq cross data stash for multi-slot DFlash
void llama_context::set_cross_data_seq(llama_seq_id seq_id, const float * data, int64_t n_embd, int64_t n_tokens) {
    if (seq_id < 0) {
        set_cross_data(data, n_embd, n_tokens);
        return;
    }

    // Also update the single-slot v_embd — sequential (non-batched) draft() calls
    // read from v_embd directly, and the graph's set_input single-slot path uses it.
    set_cross_data(data, n_embd, n_tokens);

    auto & entry = cross.v_embd_per_seq[seq_id];
    entry.n_enc      = cross.n_enc;
    entry.n_enc_real = n_tokens;
    entry.v_embd.resize(n_embd * n_tokens);
    if (data) {
        memcpy(entry.v_embd.data(), data, n_embd * n_tokens * sizeof(float));
    }
}

void llama_context::set_cross_data_gpu(
        llama_seq_id seq_id, const void * d_staging, int cross_len,
        int n_layers, int n_embd_layer, set_tensor_d2d_fn_t fn_d2d) {
    int64_t n_target_features = (int64_t)n_layers * n_embd_layer;

    const int64_t max_ctx = dflash_max_cross_ctx();
    const int64_t capped = (max_ctx > 0 && cross_len > max_ctx) ? max_ctx : cross_len;
    const int64_t bucket = cross_bucket(capped);

    if (cross.n_enc != bucket) {
        sched_need_reserve = true;
    }
    cross.n_embd     = n_target_features;
    cross.n_enc      = bucket;
    cross.n_enc_real = cross_len;
    cross.v_embd_gpu = d_staging;
    cross.v_embd_gpu_n_enc_real = cross_len;
    cross.fn_set_tensor_d2d = fn_d2d;

    // ensure v_embd is non-empty so graph builders (llama-graph.cpp) use cross.n_enc
    // for sizing instead of falling back to hparams defaults
    if (cross.v_embd.size() != (size_t)(n_target_features * cross_len)) {
        cross.v_embd.resize(n_target_features * cross_len);
    }

    if (seq_id >= 0) {
        auto & entry = cross.v_embd_per_seq[seq_id];
        entry.n_enc      = bucket;
        entry.n_enc_real = cross_len;
        entry.v_embd_gpu = d_staging;
        entry.v_embd_gpu_n_enc_real = cross_len;
        if (entry.v_embd.size() != (size_t)(n_target_features * cross_len)) {
            entry.v_embd.resize(n_target_features * cross_len);
        }
    }
}

void llama_context::set_tree_mask(const uint8_t * visibility, int n_tree_tokens) {
    tree_mask.active = true;
    tree_mask.n_tree_tokens = n_tree_tokens;
    int n2 = n_tree_tokens * n_tree_tokens;
    tree_mask.visibility.assign(visibility, visibility + n2);
}

void llama_context::clear_tree_mask() {
    tree_mask.active = false;
    tree_mask.n_tree_tokens = 0;
    tree_mask.visibility.clear();
}

void llama_context::set_tree_parent_ids(const int32_t * parents, int n_tokens) {
    if (tree_bufs.disabled) {
        return; // multi-GPU: silently use flat chain verify
    }
    if (tree_bufs.max_tree_tokens < n_tokens) {
        // Allocate or reallocate — use exact size + small margin
        int alloc_size = n_tokens + 4;
        allocate_tree_buffers(alloc_size);
    }
    if (tree_bufs.disabled) {
        return; // allocate_tree_buffers detected multi-GPU
    }
    if (n_tokens > tree_bufs.max_tree_tokens) {
        LLAMA_LOG_WARN("%s: tree buffers too small (%d > %d), falling back to flat verify\n",
            __func__, n_tokens, tree_bufs.max_tree_tokens);
        tree_bufs.active = false;
        return;
    }
    tree_bufs.n_tokens = n_tokens;
    tree_bufs.active = true;

    // Copy to CPU buffer
    tree_bufs.parent_ids_cpu.assign(parents, parents + n_tokens);

    // Upload to GPU
    ggml_backend_tensor_set(tree_bufs.parent_ids_gpu, parents, 0, n_tokens * sizeof(int32_t));
}

void llama_context::clear_tree_parent_ids() {
    tree_bufs.active = false;
    tree_bufs.n_tokens = 0;
}

void llama_context::allocate_tree_buffers(int max_tree_tokens) {
    if (tree_bufs.disabled) {
        return;
    }
    if (tree_bufs.max_tree_tokens >= max_tree_tokens) {
        return; // already allocated enough
    }

    // Tree verify buffers live on GPU 0. When the model is split across multiple
    // GPUs, recurrent layers on other devices can't read parent_ids from GPU 0,
    // so the scheduler aborts. Disable tree mode and use the regular SSM_CONV +
    // GATED_DELTA_NET kernels instead. The verify batch is still processed in a
    // single llama_decode call — only the recurrent kernel changes, and for
    // linear chains the sequential kernel produces identical results.
    if (model.n_devices() > 1) {
        LLAMA_LOG_INFO("%s: multi-GPU detected (%zu devices) — disabling tree verify, using flat chain\n",
                       __func__, model.n_devices());
        tree_bufs.disabled = true;
        return;
    }

    if (getenv("GGML_NO_TREE_VERIFY")) {
        LLAMA_LOG_INFO("%s: GGML_NO_TREE_VERIFY set — disabling tree verify, using flat chain\n", __func__);
        tree_bufs.disabled = true;
        return;
    }

    // Free existing
    if (tree_bufs.buffer) {
        ggml_backend_buffer_free(tree_bufs.buffer);
        tree_bufs.buffer = nullptr;
    }
    if (tree_bufs.ggml_ctx) {
        ggml_free(tree_bufs.ggml_ctx);
        tree_bufs.ggml_ctx = nullptr;
    }

    tree_bufs.max_tree_tokens = max_tree_tokens;
    tree_bufs.ssm_intermediates.clear();

    const auto & hparams = model.hparams;
    const int64_t d_inner = hparams.ssm_d_inner;
    const int64_t num_v_heads = hparams.ssm_dt_rank;
    const int64_t head_v_dim = (num_v_heads > 0) ? d_inner / num_v_heads : 0;

    if (head_v_dim == 0 || num_v_heads == 0) {
        return; // not a hybrid model
    }

    // Count recurrent layers
    int n_recurrent = 0;
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_recr(i)) {
            n_recurrent++;
        }
    }
    if (n_recurrent == 0) return;

    // Calculate total buffer size
    // Per layer: [head_v_dim, head_v_dim, num_v_heads, max_tree_tokens] in f16
    const int64_t inter_elems_per_layer = head_v_dim * head_v_dim * num_v_heads * max_tree_tokens;
    const size_t inter_bytes_per_layer = inter_elems_per_layer * sizeof(ggml_fp16_t);
    const size_t parent_ids_bytes = max_tree_tokens * sizeof(int32_t);
    const size_t total_bytes = n_recurrent * inter_bytes_per_layer + parent_ids_bytes;

    // Create ggml context for tensor metadata
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (n_recurrent + 1) + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    tree_bufs.ggml_ctx = ggml_init(params);

    // Create tensors
    tree_bufs.parent_ids_gpu = ggml_new_tensor_1d(tree_bufs.ggml_ctx, GGML_TYPE_I32, max_tree_tokens);
    ggml_set_name(tree_bufs.parent_ids_gpu, "tree_parent_ids");

    tree_bufs.ssm_intermediates.resize(n_recurrent);
    for (int i = 0; i < n_recurrent; i++) {
        // Flat 1D tensor for simplicity, reshape in graph building
        tree_bufs.ssm_intermediates[i] = ggml_new_tensor_1d(tree_bufs.ggml_ctx, GGML_TYPE_F16, inter_elems_per_layer);
        char name[64];
        snprintf(name, sizeof(name), "tree_ssm_inter_%d", i);
        ggml_set_name(tree_bufs.ssm_intermediates[i], name);
    }

    // Allocate GPU buffer
    auto * buft = ggml_backend_get_default_buffer_type(ggml_backend_sched_get_backend(sched.get(), 0));
    tree_bufs.buffer = ggml_backend_alloc_ctx_tensors_from_buft(tree_bufs.ggml_ctx, buft);

    if (!tree_bufs.buffer) {
        LLAMA_LOG_WARN("%s: failed to allocate tree verify buffers (%.1f MB) — using flat chain verify\n", __func__,
                        total_bytes / (1024.0 * 1024.0));
        tree_bufs.max_tree_tokens = 0;
        tree_bufs.disabled = true;
        ggml_free(tree_bufs.ggml_ctx);
        tree_bufs.ggml_ctx = nullptr;
        return;
    }

    LLAMA_LOG_INFO("%s: allocated tree verify buffers: %d layers × %d tokens = %.1f MB\n", __func__,
                   n_recurrent, max_tree_tokens, total_bytes / (1024.0 * 1024.0));

    tree_bufs.parent_ids_cpu.resize(max_tree_tokens);
}

void llama_context::tree_rollback(int commit_n, const int32_t * parents) {
    if (!tree_bufs.active || commit_n < 0) return;

    const auto & hparams = model.hparams;

    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(get_memory());
    llama_memory_recurrent * mem_recr = nullptr;
    if (mem_hybrid) {
        mem_recr = mem_hybrid->get_mem_recr();
    } else {
        mem_recr = dynamic_cast<llama_memory_recurrent *>(get_memory());
    }
    if (!mem_recr) return;

    int32_t cell_idx = -1;
    for (uint32_t i = 0; i < mem_recr->size; ++i) {
        if (mem_recr->cells[i].has_seq_id(0)) {
            cell_idx = (int32_t)i;
            break;
        }
    }
    if (cell_idx < 0) return;

    const uint32_t n_embd_s = hparams.n_embd_s();
    const uint32_t n_embd_r = hparams.n_embd_r();

    (void)parents; // unused for now (linear parents in flat mode)

    // Count recurrent layers
    int n_rec = 0;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (hparams.is_recr(il)) n_rec++;
    }

    // Restore SSM state from f16 intermediates via GPU graph
    if (n_rec > 0) {
        ggml_backend_t gpu_backend = find_gpu_backend();

        size_t ctx_mem = ggml_tensor_overhead() * ((size_t)n_rec * 4 + 2) +
                         ggml_graph_overhead_custom(n_rec * 4, false);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * ctx = ggml_init(ctx_params);

        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_rec * 4, false);

        int recurrent_idx = 0;
        for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
            if (!hparams.is_recr(il)) continue;

            ggml_tensor * inter = tree_bufs.ssm_intermediates[recurrent_idx];
            size_t src_offset = (size_t)commit_n * n_embd_s * sizeof(ggml_fp16_t);

            // Source: f16 view into intermediate buffer at commit_n
            ggml_tensor * src_view = ggml_view_1d(ctx, inter, n_embd_s, src_offset);

            // Destination: f32 view into recurrent state
            ggml_tensor * s_tensor = mem_recr->s_l[il];
            size_t s_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
            ggml_tensor * dst_view = ggml_view_1d(ctx, s_tensor, n_embd_s, s_offset);

            // Copy f16 → f32 (ggml_cpy handles type conversion)
            ggml_tensor * cpy = ggml_cpy(ctx, src_view, dst_view);
            ggml_build_forward_expand(graph, cpy);

            recurrent_idx++;
        }

        // Initialize view buffers (required for direct backend compute)
        struct ggml_tensor * t = ggml_get_first_tensor(ctx);
        while (t) {
            if (t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
            t = ggml_get_next_tensor(ctx, t);
        }

        if (gpu_backend) {
            ggml_backend_graph_compute(gpu_backend, graph);
        } else {
            ggml_backend_sched_graph_compute(sched.get(), graph);
        }
        ggml_free(ctx);
    }

    // Reconstruct conv state: restore backup conv first, then shift by n_accepted
    // (Same approach as tape_replay_conv in dflash_rollback)
    if (dflash_capture && !dflash_capture->tape_layers.empty()) {
        const auto & rec_ids = dflash_capture->recurrent_layer_ids;
        auto & tape_layers = dflash_capture->tape_layers;
        const int n_accepted = commit_n + 1;

        // Find backup cell to restore conv state from
        int32_t backup_cell = -1;
        for (uint32_t i = 0; i < mem_recr->size; ++i) {
            if (mem_recr->cells[i].has_seq_id(1)) { // seq_backup = 1
                backup_cell = (int32_t)i;
                break;
            }
        }

        for (size_t li = 0; li < rec_ids.size(); ++li) {
            int il = rec_ids[li];
            auto & tape = tape_layers[li];

            if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;
            if (tape.qkv_mixed.empty() || !mem_recr->r_l[il]) continue;

            ggml_tensor * r_tensor = mem_recr->r_l[il];
            const size_t r_offset = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);

            const int64_t conv_ch = tape.conv_channels;
            const int64_t conv_window = (int64_t)(n_embd_r / conv_ch);

            // Read pre-verify conv state from backup cell
            std::vector<float> old_window(n_embd_r);
            if (backup_cell >= 0) {
                const size_t backup_offset = (size_t)backup_cell * n_embd_r * ggml_element_size(r_tensor);
                ggml_backend_tensor_get(r_tensor, old_window.data(), backup_offset, n_embd_r * sizeof(float));
            } else {
                // No backup available — read from current (will be slightly wrong for commit_n < 2)
                ggml_backend_tensor_get(r_tensor, old_window.data(), r_offset, n_embd_r * sizeof(float));
            }

            // Shift window forward by n_accepted (same as tape_replay conv rebuild)
            std::vector<float> new_conv(n_embd_r);
            for (int64_t w = 0; w < conv_window; ++w) {
                int src_pos = n_accepted + (int)w;
                for (int64_t ch = 0; ch < conv_ch; ++ch) {
                    float val;
                    if (src_pos < (int)conv_window) {
                        val = old_window[ch * conv_window + src_pos];
                    } else {
                        val = tape.qkv_mixed[(src_pos - conv_window) * conv_ch + ch];
                    }
                    new_conv[ch * conv_window + w] = val;
                }
            }

            ggml_backend_tensor_set(r_tensor, new_conv.data(), r_offset, n_embd_r * sizeof(float));
        }
    }

    // Set cell.pos to the target position (absolute, set by caller via set_tree_seq0_count).
    // In tree mode, prepare() sets cell.pos to last ubatch position which is unpredictable
    // (branches may be last). So we use the absolute target: n_past_before + commit_n.
    const int target_pos = tree_bufs.n_seq0_tokens; // repurposed: caller passes absolute target pos
    if (target_pos >= 0) {
        mem_recr->cells[cell_idx].pos = target_pos;
    }

    clear_tree_parent_ids();
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid < model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        gf = model.build_graph(gparams);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // Staged DFlash decodes answer every eval-callback ask with "no" (hiddens are
    // graph-staged, GPU tape k/v/g/b are graph-copied, qkv is graph-staged). A set
    // sched callback still forces chunked execution with a full backend synchronize
    // per chunk (ggml-backend.cpp compute_splits), so install a null callback for
    // fully-covered decodes and restore it otherwise. Set on both reuse and rebuild
    // paths — the sched retains the previous value across calls.
    if (cparams.cb_eval == dflash_eval_callback && dflash_capture) {
        const bool cb_dormant = dflash_capture->eval_callback_dormant();
        ggml_backend_sched_set_eval_callback(sched.get(),
                cb_dormant ? nullptr : cparams.cb_eval,
                cb_dormant ? nullptr : cparams.cb_eval_user_data);
    }

    // set the input data for the input tensors
    {
        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits argmax/topk (GPU-side, tiny transfer)
    auto * t_argmax_enc = res->t_logits_argmax;
    if (t_argmax_enc && n_tokens > 0) {
        ggml_backend_t backend_argmax = ggml_backend_sched_get_tensor_backend(sched.get(), t_argmax_enc);
        GGML_ASSERT(backend_argmax != nullptr);
        const int64_t total_elems = ggml_nelements(t_argmax_enc);
        const int K = (int)(total_elems / (2 * n_tokens));
        const int n_ids = K * n_tokens;
        logits_argmax_buf.resize(n_ids);
        ggml_backend_tensor_get_async(backend_argmax, t_argmax_enc, logits_argmax_buf.data(), 0, n_ids * sizeof(int32_t));
        logits_argmax_prob_buf.resize(n_ids);
        ggml_backend_tensor_get_async(backend_argmax, t_argmax_enc, logits_argmax_prob_buf.data(), n_ids * sizeof(int32_t), n_ids * sizeof(float));
        logits_argmax_count = n_tokens;
        logits_argmax_k = K;
    }

    // extract logits (skip if GPU argmax available)
    if (logits.data && t_logits && !t_argmax_enc) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;

    // DFlash: reset hidden-state capture so this decode()'s eval callback
    // accumulates across ubatches (prefill with n_tokens > n_ubatch would
    // otherwise leave only the last ubatch's hiddens in layer_hiddens).
    dflash_reset_hidden_capture();

    do {
        const auto & ubatch = mctx->get_ubatch();

        // DFlash: hand the eval callback this ubatch so it can route hidden-state
        // captures per-token (multi-seq) or whole-tensor (single-seq) to the
        // correct layer_hiddens slot. Populate per-seq tape pointers for the
        // graph builder so GPU tape copies target the correct per-slot buffers.
        if (dflash_capture) {
            dflash_capture->ubatch = &ubatch;

            // populate per-seq tape pointers for graph builder
            if (!dflash_capture->tapes.empty()) {
                const int ns = std::min((int) ubatch.n_seqs_unq, (int) LLAMA_DFLASH_MAX_SLOTS);
                bool seqs_changed = (ns != cparams.tape_gpu_n_seqs);
                cparams.tape_gpu_n_seqs = ns;

                for (int s = 0; s < ns; ++s) {
                    const llama_seq_id seq = ubatch.seq_id_unq[s];
                    dflash_tape_gpu * tp = nullptr;
                    if (seq >= 0 && seq < (int) dflash_capture->tapes.size()) {
                        tp = dflash_capture->tapes[seq].get();
                    }
                    if (tp != cparams.tape_gpu_seqs[s]) {
                        seqs_changed = true;
                    }
                    cparams.tape_gpu_seqs[s] = tp;
                }
                for (int s = ns; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
                    cparams.tape_gpu_seqs[s] = nullptr;
                }

                // sentinel for "GPU tape is enabled"
                cparams.tape_gpu = cparams.tape_gpu_seqs[0];

                // graph-staged qkv coverage for this decode (mirrors the builder guard:
                // single-seq and within tape capacity)
                if (cparams.tape_gpu && ubatch.n_seqs_unq == 1 &&
                    (int) ubatch.n_tokens <= cparams.tape_gpu->max_tokens) {
                    dflash_capture->tape_stage_n_tokens = (int) ubatch.n_tokens;
                }

                // graph nodes hold references to tape tensors — invalidate if set changed
                if (seqs_changed && gf_res_prev) {
                    gf_res_prev->reset();
                }
            }

            // track active slot for single-seq (used by active_tape() in eval callback)
            if (ubatch.n_seqs_unq == 1) {
                const llama_seq_id seq = ubatch.seq_id_unq[0];
                if (seq >= 0 && seq < (int) dflash_capture->tapes.size()) {
                    dflash_capture->active_tape_idx = seq;
                }
            }

            // GPU capture staging covers this ubatch iff it is the whole batch (single
            // ubatch), single-slot single-seq, and fits the staging capacity. Toggling
            // changes graph topology (embedded copies), so invalidate the graph cache
            // on a switch.
            {
                const bool stage_ok = dflash_capture->stage_enabled
                    && !dflash_capture->stage_tensors.empty()
                    && ubatch.n_seqs_unq == 1
                    && ubatch.seq_id_unq[0] == 0
                    && dflash_capture->hiddens && dflash_capture->hiddens->size() == 1
                    && (int64_t) ubatch.n_tokens == n_tokens_all
                    && (int) ubatch.n_tokens <= dflash_capture->stage_max_tokens;
                dflash_capture->stage_active = stage_ok;
                ggml_tensor ** stage_want = stage_ok ? dflash_capture->stage_tensors.data() : nullptr;
                if (stage_want != cparams.capture_stage) {
                    cparams.capture_stage = stage_want;
                    if (gf_res_prev) {
                        gf_res_prev->reset();
                    }
                }
                if (stage_ok) {
                    dflash_capture->stage_n_tokens = (int) ubatch.n_tokens;
                }
            }
        }

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;

            if (!cparams.logits_all && !warned_logits_all && n_outputs > (int32_t)cparams.n_seq_max) {
                warned_logits_all = true;
                LLAMA_LOG_WARN("%s: --no-logits-all is set but batch requested %d outputs (> n_seq_max = %d); "
                               "consider removing --no-logits-all for this workload\n",
                               __func__, n_outputs, cparams.n_seq_max);
            }
        }

        ggml_status status;

        const auto * res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings       ? res->get_embd()     : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn()  : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits argmax/topk (GPU-side, tiny transfer)
        auto * t_argmax = res->t_logits_argmax;
        if (t_argmax && n_outputs > 0) {
            ggml_backend_t backend_argmax = ggml_backend_sched_get_tensor_backend(sched.get(), t_argmax);
            GGML_ASSERT(backend_argmax != nullptr);
            // tensor size = 2*K*nrows; derive K
            const int64_t total_elems = ggml_nelements(t_argmax);
            const int K = (int)(total_elems / (2 * n_outputs));
            const int n_ids = K * n_outputs;
            logits_argmax_buf.resize(n_ids);
            ggml_backend_tensor_get_async(backend_argmax, t_argmax, logits_argmax_buf.data(), 0, n_ids * sizeof(int32_t));
            logits_argmax_prob_buf.resize(n_ids);
            ggml_backend_tensor_get_async(backend_argmax, t_argmax, logits_argmax_prob_buf.data(), n_ids * sizeof(int32_t), n_ids * sizeof(float));
            logits_argmax_count = n_outputs;
            logits_argmax_k = K;
        }

        // extract logits (skip if argmax is available and no one needs raw logits)
        if (logits.data && t_logits && n_outputs > 0 && !t_argmax && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool masked    = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs       : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev  : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd  = hparams.n_embd_out();
                float * embd_nextn_out = embd_nextn.data + offset*n_embd;

                GGML_ASSERT((offset + n_rows)*n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows*n_embd*sizeof(float));
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        // DFlash hidden state capture is handled by the eval callback
        // (dflash_eval_callback) — no post-graph readback needed here

        n_outputs_prev += n_outputs;
        n_tokens_prev  += ubatch.n_tokens;
    } while (mctx->next());

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd_nextn.data[i0*n_embd + k], embd_nextn.data[i1*n_embd + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_DEEPSEEK4) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type));

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.tree_mask   =*/ tree_mask.active ? &tree_mask : nullptr,
        /*.tree_parent_ids         =*/ tree_bufs.active ? tree_bufs.parent_ids_gpu : nullptr,
        /*.tree_ssm_intermediates  =*/ tree_bufs.active ? &tree_bufs.ssm_intermediates : nullptr,
        /*.tree_n_recurrent_layers =*/ (int)tree_bufs.ssm_intermediates.size(),
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb() const {
    return [&](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
            }
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
        for (const auto & [buft, size] : memory->memory_breakdown_fixed()) {
            ret[buft].context_fixed += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type));

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.vbr_min_bits                =*/ 0.0,
        /*.vbr_vram_budget_bytes       =*/ 0,
        /*.vbr_growth_headroom_bytes   =*/ 0,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.no_fused_gdn               =*/ false,
        /*.logits_all                  =*/ true,
        /*.vbr_dynamic                 =*/ false,
        /*.vbr_budget_explicit         =*/ false,
        /*.vbr_pin_k                   =*/ false,
        /*.vbr_pin_v                   =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.dflash_n_slots              =*/ 1,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (ggml_is_quantized(params.type_k) || ggml_is_quantized(params.type_v)) {
            LLAMA_LOG_INFO("%s: SPLIT_MODE_TENSOR with quantized KV cache (K=%s, V=%s)\n",
                __func__, ggml_type_name(params.type_k), ggml_type_name(params.type_v));
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    // Auto-enable flash attention for turbo KV cache types
    {
        const bool turbo_k = ggml_is_turbo_kv_type(params.type_k);
        const bool turbo_v = ggml_is_turbo_kv_type(params.type_v);
        const bool vbr_layer_schedule = turbo_vbr_layer_schedule_enabled();
        if ((turbo_k || turbo_v || vbr_layer_schedule) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            LLAMA_LOG_WARN("%s: turbo/VBR KV cache requires flash attention — enabling automatically\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        model->hparams.n_layer_nextn == 0) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const llama_exception & err) {
        // expected during memory fitting (e.g. Gemma4Assistant/EAGLE3 ctx_other not yet set) — warn, don't error
        LLAMA_LOG_WARN("%s: failed to initialize the context: %s\n", __func__, err.what());
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

int32_t * llama_get_logits_argmax(llama_context * ctx) {
    ctx->synchronize();
    return ctx->get_logits_argmax();
}

int32_t llama_get_logits_argmax_n(llama_context * ctx) {
    return ctx->get_logits_argmax_n();
}

int32_t llama_get_logits_argmax_k(llama_context * ctx) {
    return ctx->get_logits_argmax_k();
}

float * llama_get_logits_argmax_probs(llama_context * ctx) {
    ctx->synchronize();
    return ctx->get_logits_argmax_probs();
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

float * llama_get_layer_hidden(llama_context * ctx, int slot) {
    ctx->synchronize();
    return ctx->get_layer_hidden(slot);
}

int64_t llama_get_layer_hidden_n_tokens(llama_context * ctx, int slot) {
    return ctx->get_layer_hidden_n_tokens(slot);
}

int64_t llama_get_layer_hidden_n_embd(llama_context * ctx, int slot) {
    return ctx->get_layer_hidden_n_embd(slot);
}

int32_t llama_get_n_layer_hiddens(llama_context * ctx) {
    return ctx->get_n_layer_hiddens();
}

void llama_set_dflash_capture(llama_context * ctx, const int32_t * layer_ids, int32_t n_layers) {
    ctx->set_dflash_capture(layer_ids, n_layers);
}

void llama_set_dflash_sample_temp(llama_context * ctx, float temp) {
    ctx->set_dflash_sample_temp(temp);
}

void llama_set_dflash_topk(llama_context * ctx, int k) {
    ctx->set_dflash_topk(k);
}

void llama_set_dflash_n_slots(llama_context * ctx, int n) {
    ctx->set_dflash_n_slots(n);
}

void llama_set_tape_recording(llama_context * ctx, bool enable) {
    ctx->set_tape_recording(enable);
}

void llama_set_force_split_seq(llama_context * ctx, bool force) {
    auto * mem = llama_get_memory(ctx);
    if (mem) {
        mem->set_force_split_seq(force);
    }
}

void llama_dflash_allocate_slots(llama_context * ctx, int n_slots) {
    ctx->allocate_tape_gpu(n_slots, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
}

void llama_dflash_set_active_slot(llama_context * ctx, int slot_idx) {
    ctx->set_active_dflash_slot(slot_idx);
}

bool llama_dflash_tape_replay_available(llama_context * ctx) {
    return ctx->tape_replay_available();
}

void llama_tape_replay(llama_context * ctx, llama_seq_id seq_id, int n_accepted) {
    ctx->tape_replay(seq_id, n_accepted);
}

void llama_tape_replay_sync(llama_context * ctx) {
    ctx->tape_replay_sync();
}

void llama_dflash_rollback(llama_context * ctx, llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted) {
    ctx->dflash_rollback(seq_id, seq_backup, n_past_before, n_accepted);
}

void llama_dflash_prepare_branch(llama_context * ctx, llama_seq_id seq_id, llama_seq_id seq_backup, int depth) {
    ctx->dflash_prepare_branch(seq_id, seq_backup, depth);
}

void llama_set_cross_data(llama_context * ctx, const float * data, int64_t n_embd, int64_t n_tokens) {
    ctx->set_cross_data(data, n_embd, n_tokens);
}

void llama_set_cross_data_seq(llama_context * ctx, llama_seq_id seq_id, const float * data, int64_t n_embd, int64_t n_tokens) {
    ctx->set_cross_data_seq(seq_id, data, n_embd, n_tokens);
}

// --- DFlash GPU cross-attention ring ---

struct dflash_cross_ring_handle {
    void * gpu_ring;
    void   (*fn_free)(void *);
    void   (*fn_write)(void *, int, int, const float *, int, int);
    const float * (*fn_interleave)(void *, int, int, int);
    void   (*fn_set_tensor)(void *, const void *, size_t, size_t);
    void   (*fn_write_d2d)(void *, int, int, const void *, int, int);
    void   (*fn_read)(void *, int, int, float *, int, int);
};

void * llama_context::init_cross_ring_gpu(int n_layers, int n_embd, int ring_size) {
    // find CUDA backend registry
    ggml_backend_reg_t cuda_reg = nullptr;
    if (ggml_backend_t gpu_backend = find_gpu_backend()) {
        cuda_reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu_backend));
    }
    if (!cuda_reg) return nullptr;

    // resolve all function pointers
    using alloc_fn_t      = void * (*)(int, int, int);
    using free_fn_t       = void   (*)(void *);
    using write_fn_t      = void   (*)(void *, int, int, const float *, int, int);
    using interleave_fn_t = const float * (*)(void *, int, int, int);
    using set_tensor_fn_t = void   (*)(void *, const void *, size_t, size_t);

    auto fn_alloc      = (alloc_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_alloc");
    auto fn_free       = (free_fn_t)       ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_free");
    auto fn_write      = (write_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_write");
    auto fn_interleave = (interleave_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_interleave");
    auto fn_set_tensor = (set_tensor_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_set_tensor");

    using write_d2d_fn_t = void (*)(void *, int, int, const void *, int, int);
    using read_fn_t      = void (*)(void *, int, int, float *, int, int);
    auto fn_write_d2d = (write_d2d_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_write_d2d");
    auto fn_read      = (read_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_read");

    if (!fn_alloc || !fn_free || !fn_write || !fn_interleave || !fn_set_tensor) {
        return nullptr;
    }

    void * gpu_ring = fn_alloc(n_layers, n_embd, ring_size);
    if (!gpu_ring) return nullptr;

    auto * handle = new dflash_cross_ring_handle();
    handle->gpu_ring      = gpu_ring;
    handle->fn_free       = fn_free;
    handle->fn_write      = fn_write;
    handle->fn_interleave = fn_interleave;
    handle->fn_set_tensor = fn_set_tensor;
    handle->fn_write_d2d  = fn_write_d2d;  // optional: null on backends without the proc
    handle->fn_read       = fn_read;
    return handle;
}

void * llama_dflash_cross_ring_gpu_init(llama_context * ctx, int n_layers, int n_embd, int ring_size) {
    return ctx->init_cross_ring_gpu(n_layers, n_embd, ring_size);
}

void llama_dflash_set_capture_stage_enabled(llama_context * ctx, bool enabled) {
    ctx->set_capture_stage_enabled(enabled);
}

int32_t llama_dflash_capture_stage_get(llama_context * ctx, int32_t layer_idx, const void ** data) {
    return ctx->dflash_capture_stage_get(layer_idx, data);
}

void llama_dflash_cross_ring_gpu_free(void * handle) {
    if (!handle) return;
    auto * h = (dflash_cross_ring_handle *)handle;
    h->fn_free(h->gpu_ring);
    delete h;
}

void llama_dflash_cross_ring_gpu_write(void * handle, int layer, int ring_pos, const float * data, int n_tokens, int n_embd) {
    if (!handle) return;
    auto * h = (dflash_cross_ring_handle *)handle;
    h->fn_write(h->gpu_ring, layer, ring_pos, data, n_tokens, n_embd);
}

bool llama_dflash_cross_ring_gpu_write_d2d(void * handle, int layer, int ring_pos, const void * dev_src, int n_tokens, int n_embd) {
    if (!handle) return false;
    auto * h = (dflash_cross_ring_handle *)handle;
    if (!h->fn_write_d2d) return false;
    h->fn_write_d2d(h->gpu_ring, layer, ring_pos, dev_src, n_tokens, n_embd);
    return true;
}

bool llama_dflash_cross_ring_gpu_read(void * handle, int layer, int ring_pos, float * host_dst, int n_tokens, int n_embd) {
    if (!handle) return false;
    auto * h = (dflash_cross_ring_handle *)handle;
    if (!h->fn_read) return false;
    h->fn_read(h->gpu_ring, layer, ring_pos, host_dst, n_tokens, n_embd);
    return true;
}

void llama_dflash_cross_ring_gpu_set_cross(
        llama_context * ctx, void * handle, llama_seq_id seq_id,
        int ring_write_pos, int ring_filled,
        int n_layers, int n_embd, int ctx_window) {
    if (!handle || !ctx) return;
    auto * h = (dflash_cross_ring_handle *)handle;

    const float * d_staging = h->fn_interleave(h->gpu_ring, ring_write_pos, ring_filled, ctx_window);
    if (!d_staging) return;

    int cross_len = ring_filled < ctx_window ? ring_filled : ctx_window;
    ctx->set_cross_data_gpu(seq_id, d_staging, cross_len, n_layers, n_embd, h->fn_set_tensor);
}

void llama_set_tree_mask(llama_context * ctx, const uint8_t * visibility, int n_tree_tokens) {
    ctx->set_tree_mask(visibility, n_tree_tokens);
}

void llama_clear_tree_mask(llama_context * ctx) {
    ctx->clear_tree_mask();
}

void llama_set_tree_parent_ids(llama_context * ctx, const int32_t * parents, int n_tokens) {
    ctx->set_tree_parent_ids(parents, n_tokens);
}

void llama_clear_tree_parent_ids(llama_context * ctx) {
    ctx->clear_tree_parent_ids();
}

void llama_allocate_tree_buffers(llama_context * ctx, int max_tree_tokens) {
    ctx->allocate_tree_buffers(max_tree_tokens);
}

void llama_tree_rollback(llama_context * ctx, int commit_n, const int32_t * parents, int n_seq0) {
    ctx->set_tree_seq0_count(n_seq0);
    ctx->tree_rollback(commit_n, parents);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

double llama_memory_kv_bpv(llama_memory_t mem) {
    if (!mem) {
        return -1.0;
    }

    return mem->kv_bpv();
}

struct llama_memory_vbr_state_data llama_memory_vbr_state(llama_memory_t mem, llama_seq_id seq_id, uint32_t n_tokens_extra) {
    if (!mem) {
        return {};
    }

    return mem->memory_vbr_state(seq_id, n_tokens_extra);
}

double llama_vbr_floor_bits_per_token(struct llama_context * ctx, enum ggml_type entry_k, enum ggml_type entry_v, double floor_bpv) {
    llama_memory_t mem = ctx ? llama_get_memory(ctx) : nullptr;
    if (!mem) {
        return 0.0;
    }

    return mem->memory_vbr_floor_bits_per_token(entry_k, entry_v, floor_bpv);
}

static llama_memory_recurrent * get_recurrent_mem(llama_memory_t mem) {
    if (auto * h = dynamic_cast<llama_memory_hybrid *>(mem))      return h->get_mem_recr();
    if (auto * h = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) return h->get_mem_recr();
    return dynamic_cast<llama_memory_recurrent *>(mem);
}

bool llama_memory_recurrent_expand(llama_memory_t mem, uint32_t new_n_seq_max) {
    if (!mem) return false;
    auto * recr = get_recurrent_mem(mem);
    return recr ? recr->expand(new_n_seq_max) : true;
}

bool llama_memory_recurrent_shrink(llama_memory_t mem, uint32_t new_n_seq_max) {
    if (!mem) return false;
    auto * recr = get_recurrent_mem(mem);
    return recr ? recr->shrink(new_n_seq_max) : true;
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
