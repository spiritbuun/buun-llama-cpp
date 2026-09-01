#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml-vbr.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

// Internal test helpers.
#include "../src/llama-arch.h"
#include "../src/llama-cparams.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-memory-tree.h"
#include "../src/llama-model.h"
#include "../src/llama-model-loader.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-io.h"
#include "../src/llama-vbr-artifact-adopt.h"
#include "../src/llama-vbr-artifact-capture.h"
#include "../src/llama-vbr-explicit-capture.h"
#include "../src/llama-vbr-qsa-index.h"
#include "../src/models/dflash-selector-family.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

// Narrow production-path instrumentation for the Qwen4 live CUDA contract.
// The same friend name is already owned by the representation-epoch test in a
// separate executable; no hook or counter is compiled into production.
struct llama_kv_cache_vbr_epoch_test {
    static bool active(const llama_kv_cache * kv) {
        return kv != nullptr && kv->vbr_vmm_active() && kv->vbr_budget_bytes_ > 0;
    }

    static bool map_seed_watermark(llama_kv_cache * kv) {
        const uint32_t wm = kv->vbr_watermark_cells(1);
        return wm > 0 && kv->vbr_vmm_try_map(wm);
    }

    static bool force_degrade(llama_kv_cache * kv) {
        std::vector<ggml_type> sim;
        kv->vbr_sim_seed(
            sim, /* pooled_only = */ true,
            GGML_TYPE_COUNT, GGML_TYPE_COUNT,
            nullptr, nullptr, nullptr);
        bool has_mapped_step = false;
        for (size_t i = kv->vbr_degrade_cursor_;
             i < kv->vbr_degrade_order_.size(); ++i) {
            size_t slot = 0;
            const ggml_tensor * tensor = nullptr;
            ggml_type target = GGML_TYPE_COUNT;
            if (!kv->vbr_sim_step(sim, i, slot, tensor, target)) {
                continue;
            }
            const auto & units = kv->vbr_units_of(
                slot / 2, kv->vbr_degrade_order_[i].is_v != 0);
            has_mapped_step = !units.empty();
            for (const auto & [pool, extent] : units) {
                has_mapped_step = has_mapped_step && pool->vmm != nullptr &&
                    pool->wm_cells > 0 && extent->t != nullptr;
            }
            if (has_mapped_step) {
                break;
            }
        }
        if (!has_mapped_step) {
            return false;
        }
        const size_t saved_limit = kv->vbr_degrade_limit_;
        kv->vbr_degrade_limit_ = kv->vbr_degrade_order_.size();
        const bool changed =
            kv->vbr_degrade_next(kv->vbr_watermark_cells(0)) ==
            llama_kv_cache::vbr_degrade_result::applied;
        kv->vbr_degrade_limit_ = saved_limit;
        return changed;
    }

    static std::vector<ggml_type> storage_types(const llama_kv_cache * kv) {
        std::vector<ggml_type> result;
        result.reserve(kv->layers.size() * 2);
        for (const auto & layer : kv->layers) {
            GGML_ASSERT(layer.k != nullptr && layer.v != nullptr);
            result.push_back(layer.k->type);
            result.push_back(layer.v->type);
        }
        return result;
    }

    static uint32_t n_stream(const llama_kv_cache * kv) {
        return kv->n_stream;
    }
};

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    printf("Usage: %s [-a/--arch arch] [-s/--seed seed] [-o/--out dir] [-v/--verbose] [-h/--help]\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_GEMMA4) {
        n_embd = 128;
        n_head = 2;
        n_ff   = 192;
        n_layer = 5; // need at least 5 for swa_pattern (every 5th is full_attention)
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE) {
        n_head_kv = 2; // GQA coverage
    }
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_KIMI_LINEAR ||
            arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_MISTRAL4) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_QWEN4EXP) {
        // Match the shipped Qwen3.8-Flash-Next geometry: partial 64-dim
        // interleaved M-RoPE in 128-dim attention heads.
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(0));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(5));
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,
            arch == LLM_ARCH_QWEN4EXP
                ? std::vector<uint32_t>({11, 11, 10, 0})
                : std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,            std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         arch == LLM_ARCH_DEEPSEEK4 ? uint32_t(4) : uint32_t(2)); // sqrtsoftplus : sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,              true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,       -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   std::vector<float>({0.0f, 4.0f}));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, std::vector<float>({0.0f, 5.0f}));
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,             n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,         uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,            3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,       uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,      4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.cb_eval = cb_eval;
    ctx_params.cb_eval_user_data = cb_eval_user_data;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }

    size_t tmp = seed;
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static void test_qwen4_ple_recurrent_resize(const size_t seed) {
    gguf_context_ptr gguf = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    // These owners precede the model so its borrowed synthetic PLE weights stay
    // alive until after the model/context are destroyed.
    ggml_context_ptr ple_ctx;
    ggml_backend_buffer_ptr ple_buf;
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t cpu_devices[] = { nullptr };
    model_params.devices = cpu_devices;
    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf.get(), set_tensor_data, &tmp, model_params));
    GGML_ASSERT(model != nullptr);

    // Install a small but structurally complete PLE layer on the generated Qwen4
    // model. One 256-wide hash head keeps the embedding input compatible with the
    // ordinary model width without requiring a large real PLE GGUF fixture.
    auto & hp = model->hparams;
    hp.is_ple_impl.set(0);
    hp.ple_ngram_size = 2;
    hp.ple_heads_per_ngram = 1;
    hp.ple_conv_kernel = 2;
    hp.ple_n_heads = 1;
    hp.ple_head_dim = 256;
    hp.n_embd_per_layer = 256;
    hp.ple_eos_token_id = 0;
    hp.ple_layer_multipliers[0] = 1;
    hp.ple_layer_multipliers[1] = 3;
    hp.ple_head_offsets[0] = 0;
    hp.ple_head_vocab_sizes[0] = 128;
    GGML_ASSERT(hp.ple_conv_state() > 0);

    ggml_init_params tensor_params = {
        /* .mem_size   = */ 8*ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ple_ctx.reset(ggml_init(tensor_params));
    GGML_ASSERT(ple_ctx != nullptr);
    auto make_1d = [&](int64_t ne0, const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_1d(ple_ctx.get(), GGML_TYPE_F32, ne0);
        ggml_set_name(tensor, name);
        return tensor;
    };
    auto make_2d = [&](int64_t ne0, int64_t ne1, const char * name) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ple_ctx.get(), GGML_TYPE_F32, ne0, ne1);
        ggml_set_name(tensor, name);
        return tensor;
    };
    model->per_layer_tok_embd      = make_2d(256, 128, "per_layer_token_embd.weight");
    model->layers[0].ple_key        = make_2d(256, 1024, "blk.0.ple_key.weight");
    model->layers[0].ple_value      = make_2d(256, 256, "blk.0.ple_value.weight");
    model->layers[0].ple_norm_key   = make_1d(1024, "blk.0.ple_norm_key.weight");
    model->layers[0].ple_norm_query = make_1d(1024, "blk.0.ple_norm_query.weight");
    model->layers[0].ple_norm_conv  = make_1d(1024, "blk.0.ple_norm_conv.weight");
    model->layers[0].ple_conv1d     = make_2d(2, 1024, "blk.0.ple_conv1d.weight");
    ple_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ple_ctx.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(ple_buf != nullptr);
    for (ggml_tensor * tensor = ggml_get_first_tensor(ple_ctx.get()); tensor != nullptr;
            tensor = ggml_get_next_tensor(ple_ctx.get(), tensor)) {
        set_tensor_data(tensor, &tmp);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 16;
    ctx_params.n_batch = 16;
    ctx_params.n_ubatch = 16;
    ctx_params.n_seq_max = 4;
    ctx_params.n_rs_seq = 3;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;
    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx != nullptr);

    llama_memory_t memory = llama_get_memory(ctx.get());
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(memory);
    GGML_ASSERT(hybrid != nullptr);
    llama_memory_recurrent * recurrent = hybrid->get_mem_recr();
    GGML_ASSERT(recurrent != nullptr && recurrent->size == 4 && recurrent->n_rs_seq == 3);
    GGML_ASSERT(recurrent->p_l[0] != nullptr);

    const size_t row_bytes = ggml_row_size(recurrent->p_l[0]->type, model->hparams.ple_conv_state());
    std::vector<uint8_t> initial(row_bytes);
    for (size_t i = 0; i < initial.size(); ++i) {
        initial[i] = uint8_t((17*i + 29) & 0xff);
    }
    ggml_backend_tensor_set(recurrent->p_l[0], initial.data(), 0, initial.size());

    ggml_tensor * p_before_shrink = recurrent->p_l[0];
    GGML_ASSERT(llama_memory_recurrent_shrink(memory, 1));
    GGML_ASSERT(recurrent->size == 1 && recurrent->p_l[0] != p_before_shrink);
    GGML_ASSERT(recurrent->p_l[0]->ne[1] == int64_t(1 + recurrent->n_rs_seq));
    std::vector<uint8_t> copied(row_bytes);
    ggml_backend_tensor_get(recurrent->p_l[0], copied.data(), 0, copied.size());
    GGML_ASSERT(copied == initial);

    // This is the production failure: graph construction must observe the resized
    // PLE row rather than the stale four-cell allocation left by the old resize.
    llama_token first = 1;
    GGML_ASSERT(llama_decode(ctx.get(), llama_batch_get_one(&first, 1)) == 0);
    llama_synchronize(ctx.get());

    ggml_tensor * p_before_failure = recurrent->p_l[0];
    std::vector<uint8_t> state_before_failure(ggml_nbytes(p_before_failure));
    ggml_backend_tensor_get(p_before_failure, state_before_failure.data(), 0, state_before_failure.size());
#ifdef _WIN32
    _putenv_s("LLAMA_RECURRENT_RESIZE_TEST_FAIL", "before_publish");
#else
    setenv("LLAMA_RECURRENT_RESIZE_TEST_FAIL", "before_publish", 1);
#endif
    const bool failed_expand = llama_memory_recurrent_expand(memory, 2);
#ifdef _WIN32
    _putenv_s("LLAMA_RECURRENT_RESIZE_TEST_FAIL", "");
#else
    unsetenv("LLAMA_RECURRENT_RESIZE_TEST_FAIL");
#endif
    GGML_ASSERT(!failed_expand && recurrent->size == 1 && recurrent->p_l[0] == p_before_failure);
    std::vector<uint8_t> state_after_failure(state_before_failure.size());
    ggml_backend_tensor_get(recurrent->p_l[0], state_after_failure.data(), 0, state_after_failure.size());
    GGML_ASSERT(state_after_failure == state_before_failure);

    GGML_ASSERT(llama_memory_recurrent_expand(memory, 2));
    GGML_ASSERT(recurrent->size == 2 && recurrent->p_l[0] != p_before_failure);
    GGML_ASSERT(recurrent->p_l[0]->ne[1] == int64_t(2*(1 + recurrent->n_rs_seq)));
    llama_token second = 2;
    GGML_ASSERT(llama_decode(ctx.get(), llama_batch_get_one(&second, 1)) == 0);
    llama_synchronize(ctx.get());

    printf("Qwen4 PLE recurrent resize test PASSED\n");
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false);

static void test_dflash_selector_family_contract() {
    using family = llm_dflash_selector_family;

    GGML_ASSERT(llm_dflash_selector_family_from_identity(false, false, false) == family::none);
    GGML_ASSERT(llm_dflash_selector_family_from_identity(true,  false, false) == family::unidentified);
    GGML_ASSERT(llm_dflash_selector_family_from_identity(false, true,  false) == family::fork_dflash2);
    GGML_ASSERT(llm_dflash_selector_family_from_identity(true,  true,  false) == family::fork_dflash2);
    GGML_ASSERT(llm_dflash_selector_family_from_identity(false, false, true)  == family::upstream_compat);
    GGML_ASSERT(llm_dflash_selector_family_from_identity(true,  false, true)  == family::upstream_compat);
    GGML_ASSERT(llm_dflash_selector_family_from_identity(false, true,  true)  == family::mixed);
    GGML_ASSERT(llm_dflash_selector_family_from_identity(true,  true,  true)  == family::mixed);
}

static void test_qwen4_qsa_layout_cpu(llama_model * model, size_t seed) {
    const auto load_with_ratios = [&](const std::array<uint32_t, 2> & ratios) {
        gguf_context_ptr gguf = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        gguf_set_arr_data(gguf.get(), "qwen4exp.attention.compress_ratios", GGUF_TYPE_UINT32,
                          ratios.data(), ratios.size());
        llama_model_params model_params = llama_model_default_params();
        model_params.progress_callback = silent_model_load_progress;
        ggml_backend_dev_t cpu_devices[] = { nullptr };
        model_params.devices = cpu_devices;
        size_t tmp = seed;
        return llama_model_ptr(llama_model_init_from_user(
                gguf.get(), set_tensor_data, &tmp, model_params));
    };
    GGML_ASSERT(load_with_ratios({ 4, 8 }) != nullptr);
    GGML_ASSERT(load_with_ratios({ 4, 0 }) == nullptr);
    GGML_ASSERT(load_with_ratios({ 4, 65 }) == nullptr);
    {
        gguf_context_ptr gguf = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        const std::array<int32_t, 2> ratios = { 4, -1 };
        gguf_set_arr_data(gguf.get(), "qwen4exp.attention.compress_ratios", GGUF_TYPE_INT32,
                          ratios.data(), ratios.size());
        llama_model_params model_params = llama_model_default_params();
        model_params.progress_callback = silent_model_load_progress;
        ggml_backend_dev_t cpu_devices[] = { nullptr };
        model_params.devices = cpu_devices;
        size_t tmp = seed;
        llama_model_ptr invalid(llama_model_init_from_user(
                gguf.get(), set_tensor_data, &tmp, model_params));
        GGML_ASSERT(invalid == nullptr);
    }

    llama_context_params params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.n_seq_max = 2;
    params.kv_unified = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr ctx(llama_init_from_model(model, params));
    GGML_ASSERT(ctx != nullptr);

    auto * memory = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx.get()));
    GGML_ASSERT(memory != nullptr && memory->get_mem_idx() != nullptr);
    auto * idx = memory->get_mem_idx();

    const auto apply_text = [&](llama_seq_id seq, const std::vector<llama_pos> & positions) {
        std::vector<llama_pos> mutable_positions = positions;
        std::vector<llama_token> tokens(positions.size(), 1);
        std::vector<int32_t> n_seq_id(positions.size(), 1);
        std::vector<llama_seq_id> seq_values(positions.size(), seq);
        std::vector<llama_seq_id *> seq_ids(positions.size());
        std::vector<int8_t> output(positions.size(), 0);
        for (size_t i = 0; i < positions.size(); ++i) {
            seq_ids[i] = &seq_values[i];
        }
        llama_ubatch ubatch {
            true, (uint32_t) positions.size(), (uint32_t) positions.size(), 1, 1, 1,
            tokens.data(), nullptr, mutable_positions.data(), n_seq_id.data(), seq_ids.data(),
            seq_values.data(), nullptr, output.data(), {},
        };
        const auto slot = idx->find_slot(ubatch, false);
        GGML_ASSERT(!slot.empty());
        idx->apply_ubatch(slot, ubatch);
    };

    const auto apply_shared_text = [&](const std::vector<llama_pos> & positions) {
        std::vector<llama_pos> mutable_positions = positions;
        std::vector<llama_token> tokens(positions.size(), 1);
        std::vector<int32_t> n_seq_id(positions.size(), 2);
        std::vector<llama_seq_id> seq_values(2*positions.size());
        std::vector<llama_seq_id *> seq_ids(positions.size());
        std::vector<int8_t> output(positions.size(), 0);
        for (size_t i = 0; i < positions.size(); ++i) {
            seq_values[2*i + 0] = 0;
            seq_values[2*i + 1] = 1;
            seq_ids[i] = &seq_values[2*i];
        }
        llama_seq_id seq_id_unq[2] = { 0, 1 };
        llama_ubatch ubatch {
            true, (uint32_t) positions.size(), (uint32_t) positions.size(), 1, 2, 1,
            tokens.data(), nullptr, mutable_positions.data(), n_seq_id.data(), seq_ids.data(),
            seq_id_unq, nullptr, output.data(), {},
        };
        const auto slot = idx->find_slot(ubatch, false);
        GGML_ASSERT(!slot.empty());
        idx->apply_ubatch(slot, ubatch);
    };

    const auto apply_image = [&](llama_seq_id seq, const std::vector<llama_pos> & positions) {
        GGML_ASSERT(positions.size() % 4 == 0);
        std::vector<llama_pos> mutable_positions = positions;
        const size_t n = positions.size()/4;
        std::vector<float> embd(n, 0.0f);
        std::vector<int32_t> n_seq_id(n, 1);
        std::vector<llama_seq_id> seq_values(n, seq);
        std::vector<llama_seq_id *> seq_ids(n);
        std::vector<int8_t> output(n, 0);
        for (size_t i = 0; i < n; ++i) {
            seq_ids[i] = &seq_values[i];
        }
        llama_ubatch ubatch {
            true, (uint32_t) n, (uint32_t) n, 1, 1, 4,
            nullptr, embd.data(), mutable_positions.data(), n_seq_id.data(), seq_ids.data(),
            seq_values.data(), nullptr, output.data(), {},
        };
        const auto slot = idx->find_slot(ubatch, false);
        GGML_ASSERT(!slot.empty());
        idx->apply_ubatch(slot, ubatch);
    };

    struct layout_result {
        int64_t n_kv;
        int64_t n_blocks;
        std::vector<int32_t> cell_blk;
        std::vector<int32_t> blk_cells;
        std::vector<int32_t> blk_pos;
        std::vector<float> bias;
    };

    llama_memory_hybrid_idx_context qsa(memory);
    const auto run = [&](uint32_t ratio, bool blk_bias, llama_seq_id seq,
                         const std::vector<llama_pos> & query_pos) {
        GGML_ASSERT(query_pos.size() == 1 || query_pos.size() == 4);
        const int64_t n_kv = qsa.get_idx()->get_n_kv();
        const int64_t n_blocks = (n_kv + ratio - 1)/ratio;
        ggml_init_params tensor_params = { 128*1024, nullptr, true };
        ggml_context_ptr tensor_ctx(ggml_init(tensor_params));
        GGML_ASSERT(tensor_ctx != nullptr);
        ggml_tensor * cell_blk = ggml_new_tensor_2d(tensor_ctx.get(), GGML_TYPE_I32, n_kv, 1);
        ggml_tensor * blk_cells = ggml_new_tensor_2d(tensor_ctx.get(), GGML_TYPE_I32, ratio*n_blocks, 1);
        ggml_tensor * blk_pos = ggml_new_tensor_1d(tensor_ctx.get(), GGML_TYPE_I32, 4*n_blocks);
        ggml_tensor * bias = ggml_new_tensor_3d(
                tensor_ctx.get(), GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, 1, 1);
        ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        GGML_ASSERT(cpu != nullptr);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(tensor_ctx.get(), cpu.get()));
        GGML_ASSERT(buffer != nullptr);

        llama_token token = 1;
        std::vector<llama_pos> mutable_query_pos = query_pos;
        int32_t n_seq_id = 1;
        llama_seq_id * seq_ids = &seq;
        int8_t output = 1;
        llama_ubatch query {
            true, 1, 1, 1, 1, (uint32_t) query_pos.size(), &token, nullptr, mutable_query_pos.data(),
            &n_seq_id, &seq_ids, &seq, nullptr, &output, {},
        };
        qsa.set_input_qsa(cell_blk, blk_cells, blk_pos, bias, &query, ratio, blk_bias);

        layout_result result {
            n_kv, n_blocks,
            std::vector<int32_t>((int32_t *) cell_blk->data, (int32_t *) cell_blk->data + n_kv),
            std::vector<int32_t>((int32_t *) blk_cells->data, (int32_t *) blk_cells->data + ratio*n_blocks),
            std::vector<int32_t>((int32_t *) blk_pos->data, (int32_t *) blk_pos->data + 4*n_blocks),
            std::vector<float>((float *) bias->data,
                               (float *) bias->data + (blk_bias ? n_blocks : n_kv)),
        };
        return result;
    };

    // An old incomplete block must remain hidden while the actual incomplete causal tail is
    // forced visible. Pin both the compact per-block and general per-cell bias contracts.
    idx->clear(false);
    apply_text(0, { 0, 1, 2, 3, 8, 9, 12, 13 });
    const auto block_layout = run(4, true, 0, { 13 });
    const auto cell_layout  = run(4, false, 0, { 13 });
    const auto & gap_cells = idx->get_cells(0);
    int32_t old_cell = -1;
    int32_t tail_cell = -1;
    for (uint32_t physical = 0; physical < gap_cells.size(); ++physical) {
        if (!gap_cells.is_empty(physical) && gap_cells.seq_has(physical, 0)) {
            old_cell  = gap_cells.pos_get(physical) == 8  ? (int32_t) physical : old_cell;
            tail_cell = gap_cells.pos_get(physical) == 12 ? (int32_t) physical : tail_cell;
        }
    }
    GGML_ASSERT(old_cell >= 0 && tail_cell >= 0);
    GGML_ASSERT(std::isinf(block_layout.bias[block_layout.cell_blk[old_cell]]));
    GGML_ASSERT(block_layout.bias[block_layout.cell_blk[tail_cell]] == 1e9f);
    GGML_ASSERT(std::isinf(cell_layout.bias[old_cell]));
    GGML_ASSERT(cell_layout.bias[tail_cell] == 1e9f);

    // A unified shared-prefix/private-suffix history cannot use one sparse block layout for both
    // queries. It must take the dense-attention fallback instead of dropping the boundary block.
    idx->clear(false);
    apply_shared_text({ 0, 1 });
    apply_text(0, { 2, 3 });
    apply_text(1, { 2, 3 });
    llama_token fork_token[2] = { 1, 1 };
    llama_pos fork_pos[2] = { 7, 7 };
    int32_t fork_n_seq_id[2] = { 1, 1 };
    llama_seq_id fork_seq[2] = { 0, 1 };
    llama_seq_id * fork_seq_ids[2] = { &fork_seq[0], &fork_seq[1] };
    int8_t fork_output[2] = { 1, 1 };
    llama_ubatch fork_query {
        true, 2, 1, 2, 2, 1, fork_token, nullptr, fork_pos,
        fork_n_seq_id, fork_seq_ids, fork_seq, nullptr, fork_output, {},
    };
    GGML_ASSERT(!qsa.qsa_selection_safe(&fork_query));
    llama_ubatch dormant_query {
        true, 1, 1, 1, 1, 1, fork_token, nullptr, fork_pos,
        fork_n_seq_id, fork_seq_ids, fork_seq, nullptr, fork_output, {},
    };
    GGML_ASSERT(qsa.qsa_selection_safe(&dormant_query));

    // A single image sequence ranks repeated temporal positions by spatial coordinates.
    idx->clear(false);
    apply_image(0, {
        10, 10, 10, 10,
         0,  0,  1,  1,
         0,  1,  0,  1,
        10, 10, 10, 10,
    });
    const auto mixed = run(4, false, 0, { 10, 1, 1, 10 });
    const auto mixed_block = run(4, true, 0, { 10, 1, 1, 10 });
    const auto & mixed_cells = idx->get_cells(0);
    int32_t image_bid = -1;
    std::vector<int32_t> image_cells;
    for (uint32_t physical = 0; physical < mixed_cells.size(); ++physical) {
        if (mixed_cells.is_empty(physical)) {
            continue;
        }
        if (mixed_cells.seq_has(physical, 0)) {
            image_cells.push_back((int32_t) physical);
            image_bid = image_bid < 0 ? mixed.cell_blk[physical] : image_bid;
            GGML_ASSERT(mixed.cell_blk[physical] == image_bid);
            GGML_ASSERT(std::isfinite(mixed.bias[physical]));
        }
    }
    GGML_ASSERT(image_bid >= 0 && image_cells.size() == 4);
    std::vector<int32_t> gathered_image(
            mixed.blk_cells.begin() + image_bid*4, mixed.blk_cells.begin() + image_bid*4 + 4);
    std::sort(image_cells.begin(), image_cells.end());
    std::sort(gathered_image.begin(), gathered_image.end());
    GGML_ASSERT(image_cells == gathered_image);
    GGML_ASSERT(mixed.blk_pos[image_bid] == 10);
    GGML_ASSERT(mixed.blk_pos[mixed.n_blocks + image_bid] == 0);
    GGML_ASSERT(mixed.blk_pos[2*mixed.n_blocks + image_bid] == 0);
    GGML_ASSERT(mixed.blk_pos[3*mixed.n_blocks + image_bid] == 10);
    GGML_ASSERT(mixed_block.cell_blk == mixed.cell_blk);
    GGML_ASSERT(mixed_block.blk_cells == mixed.blk_cells);
    GGML_ASSERT(mixed_block.bias[image_bid] == 0.0f);

    // The four-plane position tensor is output only, not rank scratch: ratio 8 M-RoPE must work.
    idx->clear(false);
    apply_image(0, {
        20, 20, 20, 20, 20, 20, 20, 20,
         0,  0,  0,  0,  1,  1,  1,  1,
         0,  1,  2,  3,  0,  1,  2,  3,
        20, 20, 20, 20, 20, 20, 20, 20,
    });
    const auto ratio8 = run(8, false, 0, { 20, 1, 3, 20 });
    const auto & ratio8_cells = idx->get_cells(0);
    int32_t ratio8_bid = -1;
    std::vector<int32_t> expected_ratio8;
    for (uint32_t physical = 0; physical < ratio8_cells.size(); ++physical) {
        if (!ratio8_cells.is_empty(physical)) {
            expected_ratio8.push_back((int32_t) physical);
            ratio8_bid = ratio8_bid < 0 ? ratio8.cell_blk[physical] : ratio8_bid;
            GGML_ASSERT(ratio8.cell_blk[physical] == ratio8_bid);
        }
    }
    GGML_ASSERT(ratio8_bid >= 0 && expected_ratio8.size() == 8);
    std::vector<int32_t> gathered_ratio8(
            ratio8.blk_cells.begin() + ratio8_bid*8, ratio8.blk_cells.begin() + ratio8_bid*8 + 8);
    std::sort(expected_ratio8.begin(), expected_ratio8.end());
    std::sort(gathered_ratio8.begin(), gathered_ratio8.end());
    GGML_ASSERT(expected_ratio8 == gathered_ratio8);
}

static void test_qwen4_indexed_cache_admission(const size_t seed) {
    struct qsa_trace {
        size_t raw_key_nodes = 0;
        size_t score_nodes = 0;
        size_t top_k_nodes = 0;
        bool score_q_reshape_seen = false;
        bool score_q_has_redundant_cont = false;
    };

    const auto trace_qsa = [](ggml_tensor * tensor, bool ask, void * user_data) {
        if (!ask) {
            return true;
        }
        auto & trace = *static_cast<qsa_trace *>(user_data);
        const std::string name = ggml_get_name(tensor);
        trace.raw_key_nodes += name.find("indexer_k_raw") != std::string::npos;
        trace.score_nodes   += name.find("indexer_score") != std::string::npos;
        trace.top_k_nodes   += name.find("indexer_top_k") != std::string::npos;
        if (name.find("indexer_score-") != std::string::npos) {
            std::vector<ggml_tensor *> pending = { tensor };
            while (!pending.empty()) {
                ggml_tensor * node = pending.back();
                pending.pop_back();
                if (node->op == GGML_OP_MUL_MAT && node->src[1] != nullptr) {
                    trace.score_q_reshape_seen |= node->src[1]->op == GGML_OP_RESHAPE;
                    trace.score_q_has_redundant_cont |= node->src[1]->op == GGML_OP_RESHAPE &&
                            node->src[1]->src[0] != nullptr && node->src[1]->src[0]->op == GGML_OP_CONT;
                    continue;
                }
                for (ggml_tensor * src : node->src) {
                    if (src != nullptr) {
                        pending.push_back(src);
                    }
                }
            }
        }
        return false;
    };

    // The index cache must still be populated when every cache cell fits within
    // the QSA budget, but selection-only graph work must disappear.  Pin the exact
    // ratio-4 boundary: top_k 253 selects at most 256 cells, while 252 selects 255.
    {
        gguf_context_ptr dense_gguf = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        gguf_set_val_u32(dense_gguf.get(), "qwen4exp.attention.indexer.top_k", 253);
        qsa_trace dense_trace;
        auto dense = get_model_and_ctx(
                dense_gguf.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                trace_qsa, &dense_trace);
        const auto dense_logits = get_logits(dense.first.get(), dense.second.get(), { 1, 2, 3, 4 });
        GGML_ASSERT(dense_trace.raw_key_nodes > 0);
        GGML_ASSERT(dense_trace.score_nodes == 0);
        GGML_ASSERT(dense_trace.top_k_nodes == 0);

        // Compare against the same loaded weights with QSA disabled.  This keeps
        // the random tensor realization fixed while proving the all-cell shortcut
        // is the ordinary dense-attention path, not merely a smaller graph.
        dense.second.reset();
        const auto compress_ratios = dense.first->hparams.dsv4_compress_ratios;
        std::fill(dense.first->hparams.dsv4_compress_ratios.begin(),
                  dense.first->hparams.dsv4_compress_ratios.end(), 0);
        llama_context_params dense_ref_params = llama_context_default_params();
        dense_ref_params.n_ctx = 0;
        dense_ref_params.n_threads = 4;
        dense_ref_params.n_threads_batch = 4;
        dense_ref_params.n_ubatch = 64;
        llama_context_ptr dense_ref(llama_init_from_model(dense.first.get(), dense_ref_params));
        GGML_ASSERT(dense_ref != nullptr);
        const auto dense_ref_logits = get_logits(dense.first.get(), dense_ref.get(), { 1, 2, 3, 4 });
        GGML_ASSERT(dense_logits.size() == dense_ref_logits.size());
        for (size_t i = 0; i < dense_logits.size(); ++i) {
            GGML_ASSERT(std::abs(dense_logits[i] - dense_ref_logits[i]) <= 1.0e-5f);
        }
        dense_ref.reset();
        dense.first->hparams.dsv4_compress_ratios = compress_ratios;

        gguf_context_ptr sparse_gguf = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        gguf_set_val_u32(sparse_gguf.get(), "qwen4exp.attention.indexer.top_k", 252);
        qsa_trace sparse_trace;
        auto sparse = get_model_and_ctx(
                sparse_gguf.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                trace_qsa, &sparse_trace);
        (void) get_logits(sparse.first.get(), sparse.second.get(), { 1, 2, 3, 4 });
        GGML_ASSERT(sparse_trace.raw_key_nodes > 0);
        GGML_ASSERT(sparse_trace.score_nodes > 0);
        GGML_ASSERT(sparse_trace.top_k_nodes > 0);
        GGML_ASSERT(sparse_trace.score_q_reshape_seen);
        GGML_ASSERT(!sparse_trace.score_q_has_redundant_cont);
    }

    // A graph built for the dense 256-cell watermark must be rejected and rebuilt
    // when the active cache pads to 512.  Also prove that the dense phase writes
    // real raw index keys into the mirrored cells needed by that later sparse graph.
    {
        gguf_context_ptr transition_gguf = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
        gguf_set_val_u32(transition_gguf.get(), "qwen4exp.context_length", 512);
        gguf_set_val_u32(transition_gguf.get(), "qwen4exp.attention.indexer.top_k", 253);
        qsa_trace transition_trace;
        auto transition = get_model_and_ctx(
                transition_gguf.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, false,
                trace_qsa, &transition_trace);

        auto * transition_memory = dynamic_cast<llama_memory_hybrid_idx *>(
                llama_get_memory(transition.second.get()));
        GGML_ASSERT(transition_memory != nullptr);
        auto * transition_idx = transition_memory->get_mem_idx();
        GGML_ASSERT(transition_idx != nullptr);
        const auto idx_layers = transition_idx->get_layer_ids();
        GGML_ASSERT(!idx_layers.empty());
        ggml_tensor * idx_storage = transition_idx->get_k_storage(idx_layers.front());
        GGML_ASSERT(idx_storage != nullptr);
        std::vector<uint8_t> idx_before(ggml_nbytes(idx_storage));
        ggml_backend_tensor_get(idx_storage, idx_before.data(), 0, idx_before.size());

        std::vector<llama_token> first_watermark(256, 1);
        (void) get_logits(transition.first.get(), transition.second.get(), first_watermark);
        GGML_ASSERT(transition_trace.raw_key_nodes > 0);
        GGML_ASSERT(transition_trace.score_nodes == 0);
        GGML_ASSERT(transition_trace.top_k_nodes == 0);
        GGML_ASSERT(transition_idx->seq_pos_max(0) == 255);

        std::vector<uint8_t> idx_after_dense(idx_before.size());
        ggml_backend_tensor_get(idx_storage, idx_after_dense.data(), 0, idx_after_dense.size());
        GGML_ASSERT(idx_after_dense != idx_before);

        const auto & dense_cells = transition_idx->get_cells(0);
        uint32_t late_cell = dense_cells.size();
        for (uint32_t i = 0; i < dense_cells.size(); ++i) {
            if (!dense_cells.is_empty(i) && dense_cells.seq_has(i, 0) && dense_cells.pos_get(i) == 255) {
                late_cell = i;
                break;
            }
        }
        GGML_ASSERT(late_cell < dense_cells.size());
        const size_t idx_row_size = ggml_row_size(idx_storage->type, idx_storage->ne[0]);
        const size_t late_offset = (size_t) late_cell * idx_row_size;
        GGML_ASSERT(late_offset + idx_row_size <= idx_before.size());
        GGML_ASSERT(!std::equal(
                idx_before.begin() + late_offset,
                idx_before.begin() + late_offset + idx_row_size,
                idx_after_dense.begin() + late_offset));

        transition_trace = {};
        // Keep the same 64-token ubatch geometry used at the dense watermark;
        // selection mode is then the only graph-reuse discriminator that changes.
        llama_batch next = llama_batch_init(64, 0, 1);
        for (llama_pos pos = 256; pos < 320; ++pos) {
            common_batch_add(next, 2, pos, { 0 }, true);
        }
        GGML_ASSERT(llama_decode(transition.second.get(), next) == 0);
        GGML_ASSERT(llama_get_logits_ith(transition.second.get(), 63) != nullptr);
        llama_batch_free(next);
        GGML_ASSERT(transition_trace.raw_key_nodes > 0);
        GGML_ASSERT(transition_trace.score_nodes > 0);
        GGML_ASSERT(transition_trace.top_k_nodes > 0);
        GGML_ASSERT(transition_idx->seq_pos_max(0) == 319);
    }

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
    test_qwen4_qsa_layout_cpu(model_and_ctx.first.get(), seed);

    // Phase-0 VBR ownership/accounting contract. Qwen4's attention KV is the only storage whose
    // representation may be selected by Turbo/VBR; recurrent state is fixed and the QSA indexer
    // is a separate context-linear F16 cache. Pin both the production tree and the reference
    // full-model arithmetic before enabling the currently-refused controller.
    {
        constexpr uint64_t n_ctx_train       = 262144;
        constexpr uint64_t n_attn_layers     = 12;
        constexpr uint64_t n_kv_values_side  = 2 * 256;
        constexpr uint64_t n_index_values    = 128;
        constexpr uint64_t bytes_f16         = 2;
        constexpr uint64_t ref_attn_f16      = n_attn_layers * 2 * n_kv_values_side * bytes_f16 * n_ctx_train;
        constexpr uint64_t ref_index_generic = n_attn_layers * 2 * n_index_values * bytes_f16 * n_ctx_train;
        GGML_ASSERT(ref_attn_f16 == 6ULL * 1024 * 1024 * 1024);
        GGML_ASSERT(ref_index_generic == 1536ULL * 1024 * 1024);

        auto * memory = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(model_and_ctx.second.get()));
        GGML_ASSERT(memory != nullptr);
        GGML_ASSERT(memory->get_mem_attn() != nullptr);
        GGML_ASSERT(memory->get_mem_recr() != nullptr);
        GGML_ASSERT(memory->get_mem_idx()  != nullptr);
        GGML_ASSERT(memory->get_mem_idx()->type_k() == GGML_TYPE_F16);
        GGML_ASSERT(memory->get_mem_idx()->type_v() == GGML_TYPE_F16);

        const auto merge = [](auto dst, const auto & src) {
            for (const auto & [buft, size] : src) {
                dst[buft] += size;
            }
            return dst;
        };

        const auto attn    = memory->get_mem_attn()->memory_breakdown();
        const auto recurrent = memory->get_mem_recr()->memory_breakdown();
        const auto index   = memory->get_mem_idx()->memory_breakdown();
        const auto managed = memory->memory_breakdown_vbr_managed();
        const auto fixed   = memory->memory_breakdown_fixed();
        const auto total   = memory->memory_breakdown();

        GGML_ASSERT(managed == memory->get_mem_attn()->memory_breakdown_vbr_managed());
        GGML_ASSERT(managed == attn);
        GGML_ASSERT(fixed == memory->get_mem_recr()->memory_breakdown_fixed());
        GGML_ASSERT(total == merge(merge(attn, recurrent), index));

        const llama_memory_breakdown public_breakdown = llama_get_memory_breakdown(model_and_ctx.second.get());
        for (const auto & [buft, mb] : public_breakdown) {
            const auto it = managed.find(buft);
            GGML_ASSERT(mb.context_vbr_managed == (it == managed.end() ? 0 : it->second));
            GGML_ASSERT(mb.context_vbr_managed <= mb.context);
        }
    }

    const auto make_indexed_memory = [&](ggml_type type_k, ggml_type type_v, const auto & configure) {
        llama_memory_params params = {};
        params.type_k = type_k;
        params.type_v = type_v;
        params.swa_full = true;
        params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;

        llama_cparams cparams = {};
        cparams.n_ctx = 128;
        cparams.n_ctx_seq = 128;
        cparams.n_seq_max = 1;
        configure(cparams);

        std::unique_ptr<llama_memory_i> memory(model_and_ctx.first->create_memory(params, cparams));
        auto * indexed_memory = dynamic_cast<llama_memory_hybrid_idx *>(memory.get());
        GGML_ASSERT(indexed_memory != nullptr);
        GGML_ASSERT(indexed_memory->get_mem_idx() != nullptr);
        GGML_ASSERT(indexed_memory->get_mem_idx()->type_k() == GGML_TYPE_F16);
        GGML_ASSERT(indexed_memory->get_mem_idx()->type_v() == GGML_TYPE_F16);
        return memory;
    };

    // Qwen4-family checkpoints without indexer tensors still use the indexed
    // hybrid wrapper, but they must not advertise an impossible QSA artifact
    // companion. Model the loader's absent-indexer state by clearing the
    // index width before constructing memory and pin the canonical tree.
    {
        const uint32_t saved_indexer_head_size =
            model_and_ctx.first->hparams.indexer_head_size;
        model_and_ctx.first->hparams.indexer_head_size = 0;
        llama_memory_params params = {};
        params.type_k = GGML_TYPE_F16;
        params.type_v = GGML_TYPE_F16;
        params.swa_full = true;
        params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        llama_cparams cparams = {};
        cparams.n_ctx = 128;
        cparams.n_ctx_seq = 128;
        cparams.n_seq_max = 1;
        std::unique_ptr<llama_memory_i> without_qsa(
            model_and_ctx.first->create_memory(params, cparams));
        model_and_ctx.first->hparams.indexer_head_size =
            saved_indexer_head_size;
        auto * wrapper = dynamic_cast<llama_memory_hybrid_idx *>(without_qsa.get());
        GGML_ASSERT(wrapper != nullptr && wrapper->get_mem_idx() == nullptr);
        std::vector<llama_memory_tree_child> tree;
        GGML_ASSERT(llama_memory_tree_collect(wrapper, tree));
        GGML_ASSERT(tree.size() == 2);
        GGML_ASSERT(std::none_of(tree.begin(), tree.end(), [](const auto & child) {
            return child.qsa_index_owner != nullptr;
        }));
    }

    // CPU-bound movable Turbo sides follow the generic partial-offload policy and
    // pin to Q8_0. This is a safe backend fallback, not an architecture refusal;
    // the fixed QSA child remains F16 in both asymmetric cases.
    const auto expect_cpu_turbo_fallback = [&](ggml_type type_k, ggml_type type_v) {
        auto memory = make_indexed_memory(type_k, type_v, [](llama_cparams &) {});
        auto * indexed_memory = dynamic_cast<llama_memory_hybrid_idx *>(memory.get());
        GGML_ASSERT(indexed_memory != nullptr);
        GGML_ASSERT(indexed_memory->get_mem_attn()->type_k() ==
                    (ggml_is_turbo_kv_type(type_k) ? GGML_TYPE_Q8_0 : type_k));
        GGML_ASSERT(indexed_memory->get_mem_attn()->type_v() ==
                    (ggml_is_turbo_kv_type(type_v) ? GGML_TYPE_Q8_0 : type_v));
    };
    expect_cpu_turbo_fallback(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_F16);
    expect_cpu_turbo_fallback(GGML_TYPE_F16, GGML_TYPE_TURBO3_TCQ);

    // Controller requests are admitted and remain scoped to mem_attn. These CPU
    // construction cases deliberately do not claim that a CUDA VMM controller armed;
    // the focused CUDA gate below owns that runtime contract.
    (void) make_indexed_memory(GGML_TYPE_F16, GGML_TYPE_F16, [](llama_cparams & cparams) {
        cparams.vbr_dynamic = true;
    });
    (void) make_indexed_memory(GGML_TYPE_F16, GGML_TYPE_F16, [](llama_cparams & cparams) {
        cparams.vbr_vram_budget_bytes = 1;
    });
    (void) make_indexed_memory(GGML_TYPE_F16, GGML_TYPE_F16, [](llama_cparams & cparams) {
        cparams.vbr_min_bits = 1.0;
    });

    // The internal indexed owner is now representation-ready even though public Turbo/VBR
    // admission remains refused: an ordinary non-F16 attention type must never leak into the
    // fixed QSA index child.
    {
        llama_memory_params params = {};
        params.type_k = GGML_TYPE_Q8_0;
        params.type_v = GGML_TYPE_Q8_0;
        params.swa_full = true;
        params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
        llama_cparams cparams = {};
        cparams.n_ctx = 128;
        cparams.n_ctx_seq = 128;
        cparams.n_seq_max = 1;
        std::unique_ptr<llama_memory_i> memory(model_and_ctx.first->create_memory(params, cparams));
        auto * q8_indexed = dynamic_cast<llama_memory_hybrid_idx *>(memory.get());
        GGML_ASSERT(q8_indexed != nullptr);
        GGML_ASSERT(q8_indexed->get_mem_attn()->type_k() == GGML_TYPE_Q8_0);
        GGML_ASSERT(q8_indexed->get_mem_attn()->type_v() == GGML_TYPE_Q8_0);
        GGML_ASSERT(q8_indexed->get_mem_idx()->type_k() == GGML_TYPE_F16);
        GGML_ASSERT(q8_indexed->get_mem_idx()->type_v() == GGML_TYPE_F16);
        GGML_ASSERT(q8_indexed->memory_breakdown_vbr_managed() ==
                    q8_indexed->get_mem_attn()->memory_breakdown_vbr_managed());
    }

    auto * indexed = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(model_and_ctx.second.get()));
    GGML_ASSERT(indexed != nullptr);
    GGML_ASSERT(llama_memory_can_shift(indexed));

    // All attention-only/transient/copy APIs must treat the QSA indexer as auxiliary attention
    // state. Fill both recurrent slots so the first copy deterministically fails after the base
    // attention mutation; the indexed wrapper must invalidate the destination in all children.
    {
        llama_context_params mutation_params = llama_context_default_params();
        mutation_params.n_ctx = 128;
        mutation_params.n_batch = 64;
        mutation_params.n_ubatch = 64;
        mutation_params.n_seq_max = 2;
        mutation_params.kv_unified = true;
        mutation_params.n_threads = 4;
        mutation_params.n_threads_batch = 4;
        llama_context_ptr mutation_ctx(llama_init_from_model(model_and_ctx.first.get(), mutation_params));
        GGML_ASSERT(mutation_ctx != nullptr);

        auto * mutation_memory = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(mutation_ctx.get()));
        GGML_ASSERT(mutation_memory != nullptr);
        auto * mutation_attn = mutation_memory->get_mem_attn();
        auto * mutation_idx  = mutation_memory->get_mem_idx();
        auto * mutation_recr = mutation_memory->get_mem_recr();
        GGML_ASSERT(mutation_idx != nullptr);

        const auto decode_seq = [&](llama_seq_id seq_id, llama_token token) {
            llama_batch batch = llama_batch_init(4, 0, 1);
            for (llama_pos pos = 0; pos < 4; ++pos) {
                common_batch_add(batch, token, pos, { seq_id }, pos == 3);
            }
            GGML_ASSERT(llama_decode(mutation_ctx.get(), batch) == 0);
            llama_batch_free(batch);
        };

        const auto assert_mirrored = [&](llama_seq_id seq_id) {
            const auto & attn_cells = mutation_attn->get_cells(0);
            const auto & idx_cells  = mutation_idx ->get_cells(0);
            GGML_ASSERT(attn_cells.size() == idx_cells.size());
            for (uint32_t i = 0; i < attn_cells.size(); ++i) {
                const bool in_attn = !attn_cells.is_empty(i) && attn_cells.seq_has(i, seq_id);
                const bool in_idx  = !idx_cells.is_empty(i)  && idx_cells.seq_has(i, seq_id);
                GGML_ASSERT(in_attn == in_idx);
                if (!in_attn) {
                    continue;
                }
                GGML_ASSERT(attn_cells.pos_get(i) == idx_cells.pos_get(i));
                GGML_ASSERT(attn_cells.ext_get(i).tok == idx_cells.ext_get(i).tok);
            }
            GGML_ASSERT(mutation_attn->seq_pos_max(seq_id) == mutation_idx->seq_pos_max(seq_id));
        };

        decode_seq(0, 1);
        decode_seq(1, 2);
        assert_mirrored(0);
        assert_mirrored(1);

        GGML_ASSERT(!mutation_memory->try_seq_cp(0, 1, 0, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_idx ->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == -1);

        decode_seq(1, 2);
        GGML_ASSERT(!mutation_memory->try_seq_cp_transient(0, 1, 0, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_idx ->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == -1);

        GGML_ASSERT(mutation_memory->try_seq_cp(0, 1, 0, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == 3);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == 3);

        GGML_ASSERT(mutation_memory->seq_rm_attn(1, 2, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == 1);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == 3);

        GGML_ASSERT(mutation_memory->seq_rm_transient(1, -1, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_idx ->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == -1);

        GGML_ASSERT(mutation_memory->try_seq_cp_transient(0, 1, 0, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_memory->seq_rm_attn_transient(1, 3, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == 2);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == 3);

        GGML_ASSERT(mutation_memory->seq_rm_transient(1, -1, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_idx ->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == -1);

        mutation_memory->seq_cp(0, 1, 0, -1);
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == 3);
        mutation_memory->seq_keep(0);
        assert_mirrored(0);
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_idx ->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == -1);

        mutation_memory->seq_cp(0, 1, 0, -1);
        GGML_ASSERT(mutation_memory->seq_rm(1, -1, -1));
        assert_mirrored(1);
        GGML_ASSERT(mutation_attn->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_idx ->seq_pos_max(1) == -1);
        GGML_ASSERT(mutation_recr->seq_pos_max(1) == -1);
    }

    // Mutation-resistant structural gate: architecture alone is insufficient,
    // and every component of the admitted layout is mandatory.
    const auto arch = model_and_ctx.first->arch;
    model_and_ctx.first->arch = LLM_ARCH_QWEN35;
    GGML_ASSERT(!llama_memory_can_shift(indexed));
    model_and_ctx.first->arch = arch;

    const auto rope_type = model_and_ctx.first->hparams.rope_type;
    model_and_ctx.first->hparams.rope_type = LLAMA_ROPE_TYPE_MROPE;
    GGML_ASSERT(!llama_memory_can_shift(indexed));
    model_and_ctx.first->hparams.rope_type = rope_type;

    const auto n_rot_full = model_and_ctx.first->hparams.n_rot_full;
    model_and_ctx.first->hparams.n_rot_full = 62;
    GGML_ASSERT(!llama_memory_can_shift(indexed));
    model_and_ctx.first->hparams.n_rot_full = n_rot_full;

    const auto sections = model_and_ctx.first->hparams.rope_sections;
    model_and_ctx.first->hparams.rope_sections[3] = 1;
    GGML_ASSERT(!llama_memory_can_shift(indexed));
    model_and_ctx.first->hparams.rope_sections = sections;
    GGML_ASSERT(llama_memory_can_shift(indexed));

    // Populate both mirrored caches through the real graph, then prove that a
    // text shift rotates only attention K while preserving raw QSA keys.
    (void) get_logits(model_and_ctx.first.get(), model_and_ctx.second.get(), { 1, 2, 3, 4 });

    auto * attn = indexed->get_mem_attn();
    auto * idx  = indexed->get_mem_idx();
    GGML_ASSERT(idx != nullptr);
    GGML_ASSERT(attn->can_shift_qwen4_text_range(0, 2, 4));
    GGML_ASSERT(idx ->can_shift_qwen4_text_range(0, 2, 4));

    const auto snapshot_k = [](const llama_kv_cache * cache) {
        const auto layer_ids = cache->get_layer_ids();
        GGML_ASSERT(!layer_ids.empty());
        ggml_tensor * tensor = cache->get_k_storage(layer_ids.front());
        GGML_ASSERT(tensor != nullptr);
        std::vector<uint8_t> bytes(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
        return bytes;
    };
    const auto attn_k_before = snapshot_k(attn);
    const auto idx_k_before  = snapshot_k(idx);

    indexed->seq_add(0, 2, 4, -1);

    const auto check_shifted_token = [](const llama_kv_cache * cache, llama_token tok, bool rotates) {
        const auto & cells = cache->get_cells(0);
        bool found = false;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i) || !cells.seq_has(i, 0) || cells.ext_get(i).tok != tok) {
                continue;
            }
            found = true;
            GGML_ASSERT(cells.pos_get(i) == 1);
            GGML_ASSERT(cells.ext_get(i).x == 1);
            GGML_ASSERT(cells.ext_get(i).y == 1);
            GGML_ASSERT(cells.get_shift(i) == (rotates ? -1 : 0));
        }
        GGML_ASSERT(found);
    };

    check_shifted_token(attn, 3, true);
    check_shifted_token(idx,  3, false);
    GGML_ASSERT(attn->get_has_shift());
    GGML_ASSERT(!idx->get_has_shift());

    auto update = indexed->init_update(model_and_ctx.second.get(), false);
    GGML_ASSERT(update && update->apply());
    llama_synchronize(model_and_ctx.second.get());
    GGML_ASSERT(!attn->get_has_shift());
    GGML_ASSERT(!idx->get_has_shift());
    GGML_ASSERT(snapshot_k(attn) != attn_k_before);
    GGML_ASSERT(snapshot_k(idx)  == idx_k_before);

    // Exercise the first real QSA decode after the edit.  This traverses the
    // shifted attention cache, raw index cache, block-position construction,
    // and recurrent frontier together; metadata-only assertions cannot catch
    // a graph that consumes the three timelines differently.
    llama_batch continuation = llama_batch_init(1, 0, 1);
    common_batch_add(continuation, 5, 3, { 0 }, true);
    GGML_ASSERT(llama_decode(model_and_ctx.second.get(), continuation) == 0);

    const float * continuation_logits = llama_get_logits_ith(model_and_ctx.second.get(), 0);
    GGML_ASSERT(continuation_logits != nullptr);
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_and_ctx.first.get()));
    for (uint32_t i = 0; i < n_vocab; ++i) {
        GGML_ASSERT(std::isfinite(continuation_logits[i]));
    }
    llama_batch_free(continuation);

    const auto & attn_cells = attn->get_cells(0);
    const auto & idx_cells  = idx ->get_cells(0);
    GGML_ASSERT(attn_cells.size() == idx_cells.size());
    for (uint32_t i = 0; i < attn_cells.size(); ++i) {
        GGML_ASSERT(attn_cells.is_empty(i) == idx_cells.is_empty(i));
        if (attn_cells.is_empty(i)) {
            continue;
        }
        GGML_ASSERT(attn_cells.seq_has(i, 0) == idx_cells.seq_has(i, 0));
        GGML_ASSERT(attn_cells.pos_get(i) == idx_cells.pos_get(i));
        GGML_ASSERT(attn_cells.ext_get(i).x == idx_cells.ext_get(i).x);
        GGML_ASSERT(attn_cells.ext_get(i).y == idx_cells.ext_get(i).y);
        GGML_ASSERT(attn_cells.ext_get(i).tok == idx_cells.ext_get(i).tok);
    }
    GGML_ASSERT(attn->seq_pos_max(0) == 3);
    GGML_ASSERT(idx ->seq_pos_max(0) == 3);
    GGML_ASSERT(indexed->get_mem_recr()->seq_pos_max(0) == 3);
    GGML_ASSERT(!attn->get_has_shift() && !idx->get_has_shift());

    // A real four-plane embedding position is not the broadcast text type.
    // Pin the preflight itself so a future broad "all IMRoPE" gate cannot
    // mutate one child before discovering the unsupported layout in another.
    indexed->clear(false);
    float embd_sentinel = 0.0f;
    llama_pos pos_2d[4] = { 5, 6, 7, 0 };
    int32_t n_seq_id[1] = { 1 };
    llama_seq_id seq_id = 0;
    llama_seq_id * seq_ids[1] = { &seq_id };
    int8_t output[1] = { 0 };
    llama_ubatch ubatch_2d {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ 1,
        /*.n_seq_tokens =*/ 1,
        /*.n_seqs       =*/ 1,
        /*.n_seqs_unq   =*/ 1,
        /*.n_pos        =*/ 4,
        /*.token        =*/ nullptr,
        /*.embd         =*/ &embd_sentinel,
        /*.pos          =*/ pos_2d,
        /*.n_seq_id     =*/ n_seq_id,
        /*.seq_id       =*/ seq_ids,
        /*.seq_id_unq   =*/ &seq_id,
        /*.seq_idx      =*/ nullptr,
        /*.output       =*/ output,
        /*.data         =*/ {},
    };
    const auto attn_slot = attn->find_slot(ubatch_2d, false);
    const auto idx_slot  = idx ->find_slot(ubatch_2d, false);
    GGML_ASSERT(!attn_slot.empty() && !idx_slot.empty());
    attn->apply_ubatch(attn_slot, ubatch_2d);
    idx ->apply_ubatch(idx_slot,  ubatch_2d);
    GGML_ASSERT(!attn->can_shift_qwen4_text_range(0, 0, 8));
    GGML_ASSERT(!idx ->can_shift_qwen4_text_range(0, 0, 8));
}

static void test_qwen4_vbr_cuda(const size_t seed) {
    ggml_backend_load_all();

    ggml_backend_dev_t vbr_device = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        auto * reg = ggml_backend_dev_backend_reg(dev);
        if (!reg) {
            continue;
        }
        const auto get = reinterpret_cast<ggml_backend_vbr_iface_fn_t>(
            ggml_backend_reg_get_proc_address(reg, GGML_VBR_BACKEND_IFACE_PROC));
        if (get != nullptr && get() != nullptr) {
            vbr_device = dev;
            break;
        }
    }
    if (!vbr_device) {
        std::printf("Qwen4 VBR CUDA contract test SKIP: no VBR-capable device\n");
        return;
    }

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::array<ggml_backend_dev_t, 2> devices = { vbr_device, nullptr };
    model_params.devices = devices.data();
    size_t model_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(
        gguf_ctx.get(), set_tensor_data, &model_seed, model_params));
    GGML_ASSERT(model != nullptr);

    // The 256-cell padded dense frontier becomes sparse when the continuation
    // grows to 512. This is the same boundary used by the CPU QSA mutation test.
    model->hparams.indexer_top_k = 253;

    const auto assert_finite = [&](llama_context * ctx, int32_t i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        GGML_ASSERT(logits != nullptr);
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (uint32_t j = 0; j < n_vocab; ++j) {
            GGML_ASSERT(std::isfinite(logits[j]));
        }
    };

    const auto decode_range = [&](llama_context * ctx, llama_pos p0, uint32_t n_tokens) {
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            common_batch_add(batch, 1 + llama_token(i % 7), p0 + llama_pos(i), { 0 }, true);
        }
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
        assert_finite(ctx, int32_t(n_tokens - 1));
        llama_batch_free(batch);
    };

    const auto last_logits = [&](llama_context * ctx) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        GGML_ASSERT(logits != nullptr);
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        return std::vector<float>(logits, logits + n_vocab);
    };

    llama_context_params static_ref_params = llama_context_default_params();
    static_ref_params.n_ctx = 512;
    static_ref_params.n_batch = 64;
    static_ref_params.n_ubatch = 64;
    static_ref_params.n_seq_max = 1;
    static_ref_params.kv_unified = true;
    static_ref_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    static_ref_params.type_k = GGML_TYPE_F16;
    static_ref_params.type_v = GGML_TYPE_F16;
    llama_context_ptr static_ref(llama_init_from_model(model.get(), static_ref_params));
    GGML_ASSERT(static_ref != nullptr);
    decode_range(static_ref.get(), 0, 4);
    const auto static_ref_logits = last_logits(static_ref.get());

    // Every static tier must be owned by mem_attn and execute through the real
    // CUDA attention graph. mem_idx stays F16 for every asymmetric layout.
    for (const ggml_type tier : {
             GGML_TYPE_TURBO8_0,
             GGML_TYPE_TURBO4_0,
             GGML_TYPE_TURBO3_TCQ,
             GGML_TYPE_TURBO2_TCQ,
             GGML_TYPE_TURBO1_TCQ,
         }) {
        llama_context_params params = llama_context_default_params();
        params.n_ctx = 512;
        params.n_batch = 64;
        params.n_ubatch = 64;
        params.n_seq_max = 1;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.type_k = tier;
        params.type_v = tier;
        llama_context_ptr ctx(llama_init_from_model(model.get(), params));
        GGML_ASSERT(ctx != nullptr);
        auto * memory = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx.get()));
        GGML_ASSERT(memory != nullptr);
        GGML_ASSERT(memory->get_mem_attn()->type_k() == tier);
        GGML_ASSERT(memory->get_mem_attn()->type_v() == tier);
        GGML_ASSERT(memory->get_mem_idx()->type_k() == GGML_TYPE_F16);
        GGML_ASSERT(memory->get_mem_idx()->type_v() == GGML_TYPE_F16);
        decode_range(ctx.get(), 0, 4);
        const double tier_nmse = nmse(static_ref_logits, last_logits(ctx.get()));
        GGML_ASSERT(std::isfinite(tier_nmse));
        GGML_ASSERT(tier_nmse <= 1.0e-6);
        printf("Qwen4 static %s vs F16 NMSE %.9g\n", ggml_type_name(tier), tier_nmse);
    }

    // Qwen4 switches from ordinary dense attention to its custom QSA graph once the padded
    // frontier crosses the top-k budget. Keep K and V asymmetric so this specifically pins the
    // value-side representation contract: Turbo K must not add an output inverse WHT, while
    // Turbo V must add one after the sparse attention result.
    struct turbo_wht_trace {
        size_t nodes = 0;
    };
    const auto trace_turbo_wht = [](ggml_tensor * tensor, bool ask, void * user_data) {
        if (ask && tensor->op == GGML_OP_TURBO_WHT) {
            ++static_cast<turbo_wht_trace *>(user_data)->nodes;
        }
        return false;
    };
    const auto sparse_asymmetric_logits = [&](ggml_type type_k, ggml_type type_v, turbo_wht_trace & wht) {
        llama_context_params sparse_params = static_ref_params;
        sparse_params.n_batch = 256;
        sparse_params.type_k = type_k;
        sparse_params.type_v = type_v;
        sparse_params.cb_eval = trace_turbo_wht;
        sparse_params.cb_eval_user_data = &wht;
        llama_context_ptr sparse(llama_init_from_model(model.get(), sparse_params));
        GGML_ASSERT(sparse != nullptr);
        decode_range(sparse.get(), 0, 256);
        wht.nodes = 0;
        decode_range(sparse.get(), 256, 64);
        return last_logits(sparse.get());
    };

    turbo_wht_trace sparse_ref_wht;
    turbo_wht_trace sparse_k_wht;
    turbo_wht_trace sparse_v_wht;
    const auto sparse_ref_logits = sparse_asymmetric_logits(
        GGML_TYPE_F16, GGML_TYPE_F16, sparse_ref_wht);
    const auto sparse_k_logits = sparse_asymmetric_logits(
        GGML_TYPE_TURBO8_0, GGML_TYPE_F16, sparse_k_wht);
    const auto sparse_v_logits = sparse_asymmetric_logits(
        GGML_TYPE_F16, GGML_TYPE_TURBO8_0, sparse_v_wht);
    const double sparse_k_nmse = nmse(sparse_ref_logits, sparse_k_logits);
    const double sparse_v_nmse = nmse(sparse_ref_logits, sparse_v_logits);
    GGML_ASSERT(sparse_ref_wht.nodes == 0);
    GGML_ASSERT(sparse_k_wht.nodes == 0);
    GGML_ASSERT(sparse_v_wht.nodes > 0);
    GGML_ASSERT(std::isfinite(sparse_k_nmse) && sparse_k_nmse <= 1.0e-5);
    GGML_ASSERT(std::isfinite(sparse_v_nmse) && sparse_v_nmse <= 1.0e-5);
    printf("Qwen4 sparse asymmetric Turbo8 K/F16 V NMSE %.9g, F16 K/Turbo8 V NMSE %.9g\n",
        sparse_k_nmse, sparse_v_nmse);

    // Force the long-context gather decision at a small synthetic budget. The ordinary sparse
    // tests above intentionally use the shipped-width boundary and therefore exercise the scan
    // path at this tiny fixture size. This contract pins the selected-cell production branch and
    // its V-cache restoration rather than merely testing graph construction helpers.
    struct qsa_gather_trace {
        size_t selected_k = 0;
        size_t selected_v = 0;
        size_t top_k = 0;
        size_t inverse_wht = 0;
    };
    const auto trace_qsa_gather = [](ggml_tensor * tensor, bool ask, void * user_data) {
        if (!ask) {
            return true;
        }
        auto & value = *static_cast<qsa_gather_trace *>(user_data);
        const std::string name = ggml_get_name(tensor);
        value.selected_k += name.find("qsa_k_sel") != std::string::npos;
        value.selected_v += name.find("qsa_v_sel") != std::string::npos;
        value.top_k += name.find("indexer_top_k") != std::string::npos;
        value.inverse_wht += tensor->op == GGML_OP_TURBO_WHT;
        return false;
    };
    const uint32_t saved_top_k = model->hparams.indexer_top_k;
    model->hparams.indexer_top_k = 8;
    const auto gathered_logits = [&](ggml_type type_k, ggml_type type_v, qsa_gather_trace & gather_trace) {
        llama_context_params gather_params = static_ref_params;
        gather_params.n_ctx = 128;
        gather_params.n_batch = 64;
        gather_params.n_ubatch = 64;
        gather_params.type_k = type_k;
        gather_params.type_v = type_v;
        gather_params.cb_eval = trace_qsa_gather;
        gather_params.cb_eval_user_data = &gather_trace;
        llama_context_ptr gather_ctx(llama_init_from_model(model.get(), gather_params));
        GGML_ASSERT(gather_ctx != nullptr);
        decode_range(gather_ctx.get(), 0, 64);
        gather_trace = {};
        decode_range(gather_ctx.get(), 64, 1);
        GGML_ASSERT(gather_trace.selected_k > 0 && gather_trace.selected_v > 0);
        return last_logits(gather_ctx.get());
    };
    qsa_gather_trace gather_f16_trace;
    qsa_gather_trace gather_turbo_k_trace;
    qsa_gather_trace gather_turbo_v_trace;
    const auto gather_f16_logits = gathered_logits(
        GGML_TYPE_F16, GGML_TYPE_F16, gather_f16_trace);
    const auto gather_turbo_k_logits = gathered_logits(
        GGML_TYPE_TURBO8_0, GGML_TYPE_F16, gather_turbo_k_trace);
    const auto gather_turbo_v_logits = gathered_logits(
        GGML_TYPE_F16, GGML_TYPE_TURBO8_0, gather_turbo_v_trace);
    const double gather_k_nmse = nmse(gather_f16_logits, gather_turbo_k_logits);
    const double gather_v_nmse = nmse(gather_f16_logits, gather_turbo_v_logits);
    GGML_ASSERT(gather_f16_trace.inverse_wht == 0);
    GGML_ASSERT(gather_turbo_k_trace.inverse_wht > 0);
    GGML_ASSERT(gather_turbo_v_trace.inverse_wht > 0);
    GGML_ASSERT(std::isfinite(gather_k_nmse) && gather_k_nmse <= 1.0e-5);
    GGML_ASSERT(std::isfinite(gather_v_nmse) && gather_v_nmse <= 1.0e-5);

    // The same eight-token continuation runs through gather when decoded token by token and scan
    // when decoded as one ubatch. Compare the established F16 results and pin the negative branch.
    qsa_gather_trace sequential_trace;
    llama_context_params sequential_params = static_ref_params;
    sequential_params.n_ctx = 128;
    sequential_params.n_batch = 64;
    sequential_params.n_ubatch = 64;
    sequential_params.cb_eval = trace_qsa_gather;
    sequential_params.cb_eval_user_data = &sequential_trace;
    llama_context_ptr sequential_ctx(llama_init_from_model(model.get(), sequential_params));
    GGML_ASSERT(sequential_ctx != nullptr);
    decode_range(sequential_ctx.get(), 0, 64);
    sequential_trace = {};
    for (llama_pos pos = 64; pos < 72; ++pos) {
        llama_batch one = llama_batch_init(1, 0, 1);
        common_batch_add(one, 1 + llama_token((pos - 64) % 7), pos, { 0 }, true);
        GGML_ASSERT(llama_decode(sequential_ctx.get(), one) == 0);
        llama_batch_free(one);
    }
    const auto sequential_logits = last_logits(sequential_ctx.get());
    GGML_ASSERT(sequential_trace.selected_k > 0 && sequential_trace.selected_v > 0);

    qsa_gather_trace scan_trace;
    llama_context_params scan_params = sequential_params;
    scan_params.cb_eval_user_data = &scan_trace;
    llama_context_ptr scan_ctx(llama_init_from_model(model.get(), scan_params));
    GGML_ASSERT(scan_ctx != nullptr);
    decode_range(scan_ctx.get(), 0, 64);
    scan_trace = {};
    decode_range(scan_ctx.get(), 64, 8);
    const auto scan_logits = last_logits(scan_ctx.get());
    GGML_ASSERT(scan_trace.selected_k == 0 && scan_trace.selected_v == 0);
    const double gather_scan_nmse = nmse(scan_logits, sequential_logits);
    GGML_ASSERT(std::isfinite(gather_scan_nmse) && gather_scan_nmse <= 1.0e-5);

    qsa_gather_trace non_fa_trace;
    llama_context_params non_fa_params = sequential_params;
    non_fa_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    non_fa_params.cb_eval_user_data = &non_fa_trace;
    llama_context_ptr non_fa_qsa(llama_init_from_model(model.get(), non_fa_params));
    GGML_ASSERT(non_fa_qsa != nullptr);
    decode_range(non_fa_qsa.get(), 0, 64);
    non_fa_trace = {};
    decode_range(non_fa_qsa.get(), 64, 1);
    GGML_ASSERT(non_fa_trace.selected_k == 0 && non_fa_trace.selected_v == 0);

    // Two non-unified streams exercise the gather's stream dimension. Compare each output against
    // an isolated context so stream/index association cannot be swapped without failing.
    const auto decode_prefix = [&](llama_context * ctx, llama_seq_id seq, llama_token base) {
        llama_batch batch = llama_batch_init(64, 0, 1);
        for (llama_pos pos = 0; pos < 64; ++pos) {
            common_batch_add(batch, base + llama_token(pos % 7), pos, { seq }, true);
        }
        GGML_ASSERT(llama_decode(ctx, batch) == 0);
        llama_batch_free(batch);
    };
    const auto logits_ith = [&](llama_context * ctx, int32_t i) {
        const float * logits = llama_get_logits_ith(ctx, i);
        GGML_ASSERT(logits != nullptr);
        const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        return std::vector<float>(logits, logits + n_vocab);
    };
    const auto isolated_stream = [&](llama_token base, llama_token query_token) {
        llama_context_ptr isolated(llama_init_from_model(model.get(), sequential_params));
        GGML_ASSERT(isolated != nullptr);
        decode_prefix(isolated.get(), 0, base);
        llama_batch query = llama_batch_init(1, 0, 1);
        common_batch_add(query, query_token, 64, { 0 }, true);
        GGML_ASSERT(llama_decode(isolated.get(), query) == 0);
        llama_batch_free(query);
        return last_logits(isolated.get());
    };
    const auto isolated0 = isolated_stream(1, 17);
    const auto isolated1 = isolated_stream(9, 23);

    qsa_gather_trace streams_trace;
    llama_context_params streams_params = sequential_params;
    streams_params.n_ctx = 256;
    streams_params.n_batch = 128;
    streams_params.n_seq_max = 2;
    streams_params.kv_unified = false;
    streams_params.cb_eval_user_data = &streams_trace;
    llama_context_ptr streams_ctx(llama_init_from_model(model.get(), streams_params));
    GGML_ASSERT(streams_ctx != nullptr);
    decode_prefix(streams_ctx.get(), 0, 1);
    decode_prefix(streams_ctx.get(), 1, 9);
    streams_trace = {};
    llama_batch stream_query = llama_batch_init(2, 0, 1);
    common_batch_add(stream_query, 17, 64, { 0 }, true);
    common_batch_add(stream_query, 23, 64, { 1 }, true);
    GGML_ASSERT(llama_decode(streams_ctx.get(), stream_query) == 0);
    llama_batch_free(stream_query);
    GGML_ASSERT(streams_trace.selected_k > 0 && streams_trace.selected_v > 0);
    const double stream0_nmse = nmse(isolated0, logits_ith(streams_ctx.get(), 0));
    const double stream1_nmse = nmse(isolated1, logits_ith(streams_ctx.get(), 1));
    GGML_ASSERT(std::isfinite(stream0_nmse) && stream0_nmse <= 1.0e-10);
    GGML_ASSERT(std::isfinite(stream1_nmse) && stream1_nmse <= 1.0e-10);

    // Unified KV has one physical QSA layout. A shared prefix followed by private continuations
    // therefore falls back to dense attention; compare each query with its isolated history.
    const auto isolated_fork = [&](llama_token private_base, llama_token query_token) {
        llama_context_ptr isolated(llama_init_from_model(model.get(), sequential_params));
        GGML_ASSERT(isolated != nullptr);
        llama_batch prefix = llama_batch_init(64, 0, 1);
        for (llama_pos pos = 0; pos < 2; ++pos) {
            common_batch_add(prefix, 1 + llama_token(pos % 7), pos, { 0 }, true);
        }
        for (llama_pos pos = 2; pos < 64; ++pos) {
            common_batch_add(prefix, private_base + llama_token(pos % 7), pos, { 0 }, true);
        }
        GGML_ASSERT(llama_decode(isolated.get(), prefix) == 0);
        llama_batch_free(prefix);
        llama_batch query = llama_batch_init(1, 0, 1);
        common_batch_add(query, query_token, 64, { 0 }, true);
        GGML_ASSERT(llama_decode(isolated.get(), query) == 0);
        llama_batch_free(query);
        return last_logits(isolated.get());
    };
    const auto saved_ratios = model->hparams.dsv4_compress_ratios;
    std::fill(model->hparams.dsv4_compress_ratios.begin(),
              model->hparams.dsv4_compress_ratios.end(), 0);
    const auto fork_ref0 = isolated_fork(1, 17);
    const auto fork_ref1 = isolated_fork(9, 23);
    model->hparams.dsv4_compress_ratios = saved_ratios;

    qsa_gather_trace fork_trace;
    llama_context_params fork_params = streams_params;
    fork_params.kv_unified = true;
    fork_params.cb_eval_user_data = &fork_trace;
    const auto populate_fork = [&](llama_context * ctx) {
        llama_batch shared = llama_batch_init(2, 0, 2);
        for (llama_pos pos = 0; pos < 2; ++pos) {
            common_batch_add(shared, 1 + llama_token(pos % 7), pos, { 0, 1 }, true);
        }
        GGML_ASSERT(llama_decode(ctx, shared) == 0);
        llama_batch_free(shared);
        llama_batch private_suffix = llama_batch_init(124, 0, 1);
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            const llama_token base = seq == 0 ? 1 : 9;
            for (llama_pos pos = 2; pos < 64; ++pos) {
                common_batch_add(private_suffix, base + llama_token(pos % 7), pos, { seq }, true);
            }
        }
        GGML_ASSERT(llama_decode(ctx, private_suffix) == 0);
        llama_batch_free(private_suffix);
    };

    // A dormant second sequence is harmless: a one-sequence ubatch gets its own filtered layout.
    llama_context_ptr dormant_ctx(llama_init_from_model(model.get(), fork_params));
    GGML_ASSERT(dormant_ctx != nullptr);
    populate_fork(dormant_ctx.get());
    fork_trace = {};
    llama_batch dormant_query = llama_batch_init(1, 0, 1);
    common_batch_add(dormant_query, 17, 64, { 0 }, true);
    GGML_ASSERT(llama_decode(dormant_ctx.get(), dormant_query) == 0);
    llama_batch_free(dormant_query);
    GGML_ASSERT(fork_trace.top_k > 0 && fork_trace.selected_k > 0 && fork_trace.selected_v > 0);
    const double dormant_nmse = nmse(isolated0, last_logits(dormant_ctx.get()));
    GGML_ASSERT(std::isfinite(dormant_nmse) && dormant_nmse <= 1.0e-10);

    llama_context_ptr fork_ctx(llama_init_from_model(model.get(), fork_params));
    GGML_ASSERT(fork_ctx != nullptr);
    populate_fork(fork_ctx.get());
    fork_trace = {};
    llama_batch fork_queries = llama_batch_init(2, 0, 1);
    common_batch_add(fork_queries, 17, 64, { 0 }, true);
    common_batch_add(fork_queries, 23, 64, { 1 }, true);
    GGML_ASSERT(llama_decode(fork_ctx.get(), fork_queries) == 0);
    llama_batch_free(fork_queries);
    GGML_ASSERT(fork_trace.top_k == 0 && fork_trace.selected_k == 0 && fork_trace.selected_v == 0);
    const double fork0_nmse = nmse(fork_ref0, logits_ith(fork_ctx.get(), 0));
    const double fork1_nmse = nmse(fork_ref1, logits_ith(fork_ctx.get(), 1));
    GGML_ASSERT(std::isfinite(fork0_nmse) && fork0_nmse <= 1.0e-10);
    GGML_ASSERT(std::isfinite(fork1_nmse) && fork1_nmse <= 1.0e-10);

    model->hparams.indexer_top_k = saved_top_k;
    printf("Qwen4 selected-cell gather Turbo8 K/V NMSE %.9g/%.9g, scan parity %.9g, "
           "streams %.9g/%.9g, dormant %.9g, unified fork %.9g/%.9g\n",
           gather_k_nmse, gather_v_nmse, gather_scan_nmse, stream0_nmse, stream1_nmse,
           dormant_nmse, fork0_nmse, fork1_nmse);

    // Dynamic device VBR needs a non-transposed FA cache, so an explicit FA-off
    // request is promoted before memory construction and must really arm. With
    // KV offload disabled, the host fallback instead stays static Q8 and usable;
    // it must never advertise a controller that cannot retier.
    {
        llama_context_params fallback = static_ref_params;
        fallback.vbr_dynamic = true;
        fallback.vbr_budget_explicit = true;
        fallback.vbr_vram_budget_bytes = 64ull * 1024 * 1024;
        fallback.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        llama_context_ptr non_fa(llama_init_from_model(model.get(), fallback));
        GGML_ASSERT(non_fa != nullptr);
        auto * memory = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(non_fa.get()));
        GGML_ASSERT(memory != nullptr);
        GGML_ASSERT(llama_kv_cache_vbr_epoch_test::active(memory->get_mem_attn()));
        GGML_ASSERT(memory->get_mem_idx()->type_k() == GGML_TYPE_F16);
        decode_range(non_fa.get(), 0, 4);

        fallback.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        fallback.offload_kqv = false;
        llama_context_ptr cpu_kv(llama_init_from_model(model.get(), fallback));
        GGML_ASSERT(cpu_kv != nullptr);
        memory = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(cpu_kv.get()));
        GGML_ASSERT(memory != nullptr);
        GGML_ASSERT(!llama_kv_cache_vbr_epoch_test::active(memory->get_mem_attn()));
        GGML_ASSERT(memory->get_mem_attn()->type_k() == GGML_TYPE_Q8_0);
        GGML_ASSERT(memory->get_mem_attn()->type_v() == GGML_TYPE_Q8_0);
        GGML_ASSERT(memory->get_mem_idx()->type_k() == GGML_TYPE_F16);
        decode_range(cpu_kv.get(), 0, 4);
    }

    struct qsa_trace {
        size_t raw_key_nodes = 0;
        size_t score_nodes = 0;
        size_t top_k_nodes = 0;
    } trace;
    const auto trace_qsa = [](ggml_tensor * tensor, bool ask, void * user_data) {
        if (!ask) {
            return true;
        }
        auto & value = *static_cast<qsa_trace *>(user_data);
        const std::string name = ggml_get_name(tensor);
        value.raw_key_nodes += name.find("indexer_k_raw") != std::string::npos;
        value.score_nodes   += name.find("indexer_score") != std::string::npos;
        value.top_k_nodes   += name.find("indexer_top_k") != std::string::npos;
        return false;
    };

    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 256;
    params.n_ubatch = 64;
    params.n_seq_max = 2;
    params.kv_unified = false; // dynamic VBR must force the three children to unified storage
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.vbr_dynamic = true;
    params.vbr_budget_explicit = true;
    params.vbr_vram_budget_bytes = 64ull * 1024 * 1024;
    params.cb_eval = trace_qsa;
    params.cb_eval_user_data = &trace;
    llama_context_ptr ctx(llama_init_from_model(model.get(), params));
    GGML_ASSERT(ctx != nullptr);
    auto * memory = dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx.get()));
    GGML_ASSERT(memory != nullptr);
    auto * attn = memory->get_mem_attn();
    auto * idx = memory->get_mem_idx();
    GGML_ASSERT(llama_kv_cache_vbr_epoch_test::active(attn));
    GGML_ASSERT(llama_kv_cache_vbr_epoch_test::n_stream(attn) == 1);
    GGML_ASSERT(llama_kv_cache_vbr_epoch_test::n_stream(idx) == 1);
    GGML_ASSERT(llama_kv_cache_vbr_epoch_test::map_seed_watermark(attn));
    GGML_ASSERT(idx->type_k() == GGML_TYPE_F16 && idx->type_v() == GGML_TYPE_F16);

    decode_range(ctx.get(), 0, 256);
    GGML_ASSERT(trace.raw_key_nodes > 0);
    GGML_ASSERT(trace.score_nodes == 0 && trace.top_k_nodes == 0);

    const auto snapshot_index = [&]() {
        std::vector<std::vector<uint8_t>> result;
        for (const uint32_t il : idx->get_layer_ids()) {
            ggml_tensor * tensor = idx->get_k_storage(il);
            result.emplace_back(ggml_nbytes(tensor));
            ggml_backend_tensor_get(
                tensor, result.back().data(), 0, result.back().size());
        }
        return result;
    };
    const auto assert_index_snapshot = [&](const std::vector<std::vector<uint8_t>> & before) {
        size_t index_layer = 0;
        for (const uint32_t il : idx->get_layer_ids()) {
            ggml_tensor * tensor = idx->get_k_storage(il);
            std::vector<uint8_t> after(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, after.data(), 0, after.size());
            GGML_ASSERT(after == before.at(index_layer++));
        }
        GGML_ASSERT(index_layer == before.size());
    };
    const auto tier_ordinal = [](ggml_type type) -> int {
        switch (type) {
            case GGML_TYPE_TURBO8_0:   return 0;
            case GGML_TYPE_TURBO4_0:   return 1;
            case GGML_TYPE_TURBO3_TCQ: return 2;
            case GGML_TYPE_TURBO2_TCQ: return 3;
            case GGML_TYPE_TURBO1_TCQ: return 4;
            default:                   return -1;
        }
    };

    // Drive the real production order through every rung. Each applied step
    // must advance only the attention epoch, leave the fixed QSA index bytes
    // untouched, and remain executable before the next step is attempted.
    std::array<bool, 5> seen_tier = {};
    llama_pos next_pos = 256;
    trace = {};
    for (size_t step = 0; step < 64 && !std::all_of(
             seen_tier.begin(), seen_tier.end(), [](bool seen) { return seen; }); ++step) {
        const auto index_before = snapshot_index();
        const auto types_before = llama_kv_cache_vbr_epoch_test::storage_types(attn);
        const auto epoch_before = memory->vbr_representation_identity();
        GGML_ASSERT(llama_kv_cache_vbr_epoch_test::force_degrade(attn));
        const auto epoch_after = memory->vbr_representation_identity();
        const auto types_after = llama_kv_cache_vbr_epoch_test::storage_types(attn);
        GGML_ASSERT(types_after != types_before);
        GGML_ASSERT(epoch_after.tier_epoch > epoch_before.tier_epoch);
        GGML_ASSERT(epoch_after.tier_epoch_swa == epoch_before.tier_epoch_swa);
        assert_index_snapshot(index_before);
        for (const ggml_type type : types_after) {
            const int ordinal = tier_ordinal(type);
            if (ordinal >= 0) {
                seen_tier.at(size_t(ordinal)) = true;
            }
        }
        decode_range(ctx.get(), next_pos++, 1);
        GGML_ASSERT(idx->type_k() == GGML_TYPE_F16 && idx->type_v() == GGML_TYPE_F16);
    }
    GGML_ASSERT(std::all_of(
        seen_tier.begin(), seen_tier.end(), [](bool seen) { return seen; }));

    GGML_ASSERT(next_pos <= 320);
    if (next_pos < 320) {
        decode_range(ctx.get(), next_pos, uint32_t(320 - next_pos));
    }
    GGML_ASSERT(trace.raw_key_nodes > 0);
    GGML_ASSERT(trace.score_nodes > 0 && trace.top_k_nodes > 0);
    GGML_ASSERT(attn->seq_pos_max(0) == 319);
    GGML_ASSERT(idx ->seq_pos_max(0) == 319);
    GGML_ASSERT(idx->type_k() == GGML_TYPE_F16 && idx->type_v() == GGML_TYPE_F16);

    llama_batch two_seq = llama_batch_init(2, 0, 2);
    common_batch_add(two_seq, 3, 320, { 0 }, true);
    common_batch_add(two_seq, 4,   0, { 1 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), two_seq) == 0);
    assert_finite(ctx.get(), 0);
    assert_finite(ctx.get(), 1);
    llama_batch_free(two_seq);
    GGML_ASSERT(attn->seq_pos_max(0) == 320 && idx->seq_pos_max(0) == 320);
    GGML_ASSERT(attn->seq_pos_max(1) ==   0 && idx->seq_pos_max(1) ==   0);

    // The indexed hybrid tree is now artifact-capable only as a complete
    // attention+QSA/recurrent unit. Prove that discovery binds the QSA owner
    // to the attention child and that its typed bytes seal this exact frontier.
    std::vector<llama_memory_tree_child> artifact_tree;
    GGML_ASSERT(llama_memory_tree_collect(memory, artifact_tree));
    GGML_ASSERT(artifact_tree.size() == 2);
    const auto qsa_child = std::find_if(
        artifact_tree.begin(), artifact_tree.end(), [](const auto & child) {
            return child.attention != nullptr && child.qsa_index_owner != nullptr;
        });
    const auto recurrent_child = std::find_if(
        artifact_tree.begin(), artifact_tree.end(), [](const auto & child) {
            return child.recurrent != nullptr;
        });
    GGML_ASSERT(qsa_child != artifact_tree.end());
    GGML_ASSERT(recurrent_child != artifact_tree.end());
    GGML_ASSERT(qsa_child->attention == attn);
    GGML_ASSERT(qsa_child->qsa_index_owner == memory);
    GGML_ASSERT(recurrent_child->qsa_index_owner == nullptr);

    const auto qsa_provider = vbr_qsa_index_capture_provider(*memory);
    uint64_t qsa_size = 0;
    std::vector<uint8_t> qsa_bytes;
    llama_pos qsa_terminal = -1;
    llama_pos serialized_terminal = -1;
    GGML_ASSERT(qsa_provider.kind ==
        vbr_artifact_companion_kind::qsa_index);
    GGML_ASSERT(qsa_provider.size && qsa_provider.capture &&
        qsa_provider.terminal_position);
    GGML_ASSERT(qsa_provider.size(qsa_provider.context, 0, qsa_size));
    GGML_ASSERT(qsa_provider.capture(
        qsa_provider.context, 0, qsa_bytes));
    GGML_ASSERT(qsa_size == qsa_bytes.size() && qsa_size != 0);
    GGML_ASSERT(qsa_provider.terminal_position(
        qsa_provider.context, 0, qsa_terminal));
    GGML_ASSERT(vbr_qsa_index_companion_terminal(
        qsa_bytes.data(), qsa_bytes.size(), serialized_terminal));
    GGML_ASSERT(qsa_terminal == 320 && serialized_terminal == qsa_terminal);

    // The attention artifact layout intentionally carries no token, whereas
    // the mirrored QSA index does.  Exercise the production relocation key:
    // position/x/y select the target cell and the authenticated native QSA
    // payload restores the token itself.
    artifact_segment_chain qsa_chain(qsa_bytes.size());
    GGML_ASSERT(qsa_chain.append(qsa_bytes.data(), qsa_bytes.size()));
    vbr_artifact_companion_payload qsa_descriptor;
    qsa_descriptor.kind = vbr_artifact_companion_kind::qsa_index;
    qsa_descriptor.format_version =
        vbr_qsa_index_companion_format_version();
    qsa_descriptor.build_identity_digest =
        vbr_qsa_index_companion_build_identity();
    qsa_descriptor.payload_bytes = qsa_chain.size();
    vbr_target_companion_snapshot qsa_target;
    qsa_target.kind = qsa_descriptor.kind;
    qsa_target.format_version = qsa_descriptor.format_version;
    qsa_target.build_identity_digest = qsa_descriptor.build_identity_digest;
    qsa_target.available = true;
    qsa_target.target_cookie = memory;
    const auto parse_qsa = [&](const artifact_segment_chain & source) {
        std::unique_ptr<vbr_parsed_companion_image> parsed;
        GGML_ASSERT(vbr_parse_qsa_index_companion(
            nullptr, qsa_descriptor, source, qsa_target, parsed));
        return parsed;
    };

    // The descriptor and the native header independently pin the codec. A
    // version drift must be refused before it can mutate the destination.
    {
        auto wrong_version = qsa_descriptor;
        ++wrong_version.format_version;
        std::unique_ptr<vbr_parsed_companion_image> refused;
        GGML_ASSERT(!vbr_parse_qsa_index_companion(
            nullptr, wrong_version, qsa_chain, qsa_target, refused));
        GGML_ASSERT(!refused);
    }

    using qsa_test_key = std::tuple<llama_pos, int32_t, int32_t>;
    std::map<qsa_test_key, llama_token> qsa_tokens;
    std::map<qsa_test_key, std::vector<uint8_t>> qsa_rows;
    const auto & qsa_cells_before = idx->get_cells(0);
    for (uint32_t physical = 0; physical < qsa_cells_before.size(); ++physical) {
        if (!qsa_cells_before.seq_has(physical, 0)) {
            continue;
        }
        const auto ext = qsa_cells_before.ext_get(physical);
        const qsa_test_key key {
            qsa_cells_before.pos_get(physical), ext.x, ext.y,
        };
        GGML_ASSERT(qsa_tokens.emplace(key, ext.tok).second);
        auto & rows = qsa_rows[key];
        for (const uint32_t il : idx->get_layer_ids()) {
            ggml_tensor * tensor = idx->get_k_storage(il);
            const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            const size_t start = rows.size();
            rows.resize(start + row_size);
            ggml_backend_tensor_get(
                tensor, rows.data() + start, physical*tensor->nb[1], row_size);
        }
    }
    vbr_companion_attention_layout qsa_layout;
    qsa_layout.child_id = qsa_child->child_id;
    const auto & attention_cells = attn->get_cells(0);
    const uint32_t attention_stream = attn->get_stream_for_seq(0);
    for (uint32_t physical = 0; physical < attention_cells.size(); ++physical) {
        if (!attention_cells.seq_has(physical, 0)) {
            continue;
        }
        const auto ext = attention_cells.ext_get(physical);
        qsa_layout.cells.push_back({
            attention_stream, physical, attention_cells.pos_get(physical),
            ext.x, ext.y, LLAMA_TOKEN_NULL,
        });
    }
    GGML_ASSERT(!qsa_tokens.empty() && !qsa_layout.cells.empty());

    // Record the exact continuation before removing the QSA state. The final
    // relocated restore must reproduce it, proving the restored index bytes
    // are not merely plausible metadata.
    GGML_ASSERT(memory->seq_rm(1, -1, -1));
    GGML_ASSERT(memory->try_seq_cp(0, 1, 0, 321));
    llama_batch qsa_reference_batch = llama_batch_init(1, 0, 1);
    common_batch_add(qsa_reference_batch, 1, 321, { 1 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), qsa_reference_batch) == 0);
    const auto qsa_continuation_reference = last_logits(ctx.get());
    llama_batch_free(qsa_reference_batch);
    GGML_ASSERT(memory->seq_rm(1, -1, -1));
    GGML_ASSERT(attn->seq_pos_max(0) == 320 && idx->seq_pos_max(0) == 320);

    idx->seq_rm(0, -1, -1);
    idx->seq_rm(1, -1, -1);
    GGML_ASSERT(idx->state_empty());
    const auto qsa_adoption =
        vbr_qsa_index_adoption_provider(*memory, qsa_child->child_id);

    const auto read_qsa_rows = [&](uint32_t physical) {
        std::vector<uint8_t> restored;
        for (const uint32_t il : idx->get_layer_ids()) {
            ggml_tensor * tensor = idx->get_k_storage(il);
            const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            const size_t start = restored.size();
            restored.resize(start + row_size);
            ggml_backend_tensor_get(
                tensor, restored.data() + start,
                physical*tensor->nb[1], row_size);
        }
        return restored;
    };

    const auto assert_qsa_image = [&](const vbr_companion_attention_layout & layout) {
        const auto & cells = idx->get_cells(0);
        size_t occupied = 0;
        for (const auto & expected : layout.cells) {
            GGML_ASSERT(expected.physical_cell < cells.size());
            GGML_ASSERT(cells.seq_has(expected.physical_cell, 0));
            const auto ext = cells.ext_get(expected.physical_cell);
            const qsa_test_key key {
                cells.pos_get(expected.physical_cell), ext.x, ext.y,
            };
            const auto token = qsa_tokens.find(key);
            const auto rows = qsa_rows.find(key);
            GGML_ASSERT(token != qsa_tokens.end() && token->second == ext.tok);
            GGML_ASSERT(rows != qsa_rows.end());
            GGML_ASSERT(read_qsa_rows(expected.physical_cell) == rows->second);
            ++occupied;
        }
        GGML_ASSERT(occupied == qsa_tokens.size());
    };

    // Restore into deliberately different physical cells. This catches both
    // ignored relocation maps and tensor readers that write the wrong rows.
    auto relocated_layout = qsa_layout;
    GGML_ASSERT(relocated_layout.cells.size() > 1);
    const uint32_t first_physical = relocated_layout.cells.front().physical_cell;
    for (size_t i = 0; i + 1 < relocated_layout.cells.size(); ++i) {
        relocated_layout.cells[i].physical_cell =
            relocated_layout.cells[i + 1].physical_cell;
    }
    relocated_layout.cells.back().physical_cell = first_physical;
    std::unique_ptr<vbr_prepared_companion_image> qsa_prepared;
    GGML_ASSERT(qsa_adoption.prepare_with_layout &&
        qsa_adoption.prepare_with_layout(
            qsa_adoption.context, parse_qsa(qsa_chain), 0,
            relocated_layout, qsa_prepared));
    GGML_ASSERT(qsa_prepared && qsa_adoption.recheck(
        qsa_adoption.context, *qsa_prepared));
    assert_qsa_image(relocated_layout);
    GGML_ASSERT(qsa_adoption.rollback(
        qsa_adoption.context, *qsa_prepared));
    GGML_ASSERT(idx->state_empty());

    // Install the production-aligned image and prove both exact tensor bytes
    // and the next-token computation match the pre-capture frontier.
    qsa_prepared.reset();
    GGML_ASSERT(qsa_adoption.prepare_with_layout(
        qsa_adoption.context, parse_qsa(qsa_chain), 0,
        qsa_layout, qsa_prepared));
    GGML_ASSERT(qsa_prepared && qsa_adoption.recheck(
        qsa_adoption.context, *qsa_prepared));
    assert_qsa_image(qsa_layout);
    qsa_adoption.publish_swap(qsa_adoption.context, *qsa_prepared);

    // Occupied adoption has a distinct destructive rollback contract. Make a
    // valid incoming image observably different from the incumbent by changing
    // one known native K-row byte while preserving its layout and metadata.
    // This proves both the incoming install and the recovery reinstall execute.
    const auto & changed_cell = qsa_layout.cells.front();
    const qsa_test_key changed_key {
        changed_cell.logical_position, changed_cell.ext_x, changed_cell.ext_y,
    };
    const auto changed_row = qsa_rows.find(changed_key);
    GGML_ASSERT(changed_row != qsa_rows.end());
    const auto index_layers = idx->get_layer_ids();
    GGML_ASSERT(!index_layers.empty());
    ggml_tensor * first_index_tensor = idx->get_k_storage(index_layers.front());
    const size_t first_row_size =
        ggml_row_size(first_index_tensor->type, first_index_tensor->ne[0]);
    GGML_ASSERT(first_row_size > 0 && changed_row->second.size() >= first_row_size);

    // Walk the exact v1 prefix and native KV metadata to the first layer's K
    // rows. The native rows are serialized in ascending physical-cell order.
    size_t native_offset = 0;
    const auto read_native_scalar = [&](auto & value) {
        GGML_ASSERT(native_offset + sizeof(value) <= qsa_bytes.size());
        std::memcpy(&value, qsa_bytes.data() + native_offset, sizeof(value));
        native_offset += sizeof(value);
    };
    uint32_t qsa_magic = 0, qsa_version = 0, qsa_cache_size = 0,
             qsa_streams = 0, qsa_layer_count = 0;
    ggml_type qsa_type_k = GGML_TYPE_COUNT, qsa_type_v = GGML_TYPE_COUNT;
    read_native_scalar(qsa_magic);
    read_native_scalar(qsa_version);
    read_native_scalar(qsa_cache_size);
    read_native_scalar(qsa_streams);
    read_native_scalar(qsa_type_k);
    read_native_scalar(qsa_type_v);
    read_native_scalar(qsa_layer_count);
    GGML_ASSERT(qsa_magic == 0x49534151 &&
        qsa_version == vbr_qsa_index_companion_format_version() &&
        qsa_cache_size == idx->get_size() && qsa_streams == idx->get_n_stream() &&
        qsa_type_k == idx->type_k() && qsa_type_v == idx->type_v() &&
        qsa_layer_count == index_layers.size());
    for (uint32_t i = 0; i < qsa_layer_count; ++i) {
        uint32_t layer = 0;
        uint64_t width = 0;
        read_native_scalar(layer);
        read_native_scalar(width);
        GGML_ASSERT(layer == index_layers[i] && width ==
            uint64_t(idx->get_k_storage(layer)->ne[0]));
    }
    llama_pos encoded_terminal = -1;
    uint32_t encoded_stream = 0, encoded_cells = 0;
    read_native_scalar(encoded_terminal);
    read_native_scalar(encoded_stream);
    read_native_scalar(encoded_cells);
    GGML_ASSERT(encoded_terminal == qsa_terminal && encoded_stream == 0 &&
        encoded_cells == qsa_tokens.size());
    for (uint32_t i = 0; i < encoded_cells; ++i) {
        llama_pos position = -1;
        llama_kv_cell_ext ext;
        read_native_scalar(position);
        read_native_scalar(ext);
        GGML_ASSERT(position >= 0);
    }
    uint32_t native_streams = 0, native_cells = 0;
    read_native_scalar(native_streams);
    read_native_scalar(native_cells);
    GGML_ASSERT(native_streams == 1 && native_cells == encoded_cells);
    for (uint32_t i = 0; i < native_cells; ++i) {
        llama_pos position = -1;
        uint32_t sequence_count = 0;
        llama_kv_cell_ext ext;
        read_native_scalar(position);
        read_native_scalar(sequence_count);
        read_native_scalar(ext);
        GGML_ASSERT(sequence_count == 1);
        llama_seq_id sequence = -1;
        read_native_scalar(sequence);
        GGML_ASSERT(position >= 0 && sequence == 0);
    }
    uint32_t v_trans = 0, native_layers = 0;
    int32_t native_k_type = -1;
    uint64_t native_row_size = 0;
    read_native_scalar(v_trans);
    read_native_scalar(native_layers);
    read_native_scalar(native_k_type);
    read_native_scalar(native_row_size);
    GGML_ASSERT(v_trans <= 1 && native_layers == index_layers.size() &&
        native_k_type == int32_t(first_index_tensor->type) &&
        native_row_size == first_row_size);
    size_t physical_ordinal = 0;
    for (uint32_t physical = 0; physical < changed_cell.physical_cell; ++physical) {
        physical_ordinal += qsa_cells_before.seq_has(physical, 0);
    }
    GGML_ASSERT(physical_ordinal < native_cells);

    std::vector<uint8_t> incoming_bytes = qsa_bytes;
    const size_t changed_offset =
        native_offset + physical_ordinal*size_t(native_row_size);
    GGML_ASSERT(changed_offset < incoming_bytes.size());
    incoming_bytes[changed_offset] ^= 1;
    std::vector<uint8_t> expected_incoming_row = changed_row->second;
    expected_incoming_row[0] ^= 1;
    artifact_segment_chain incoming_chain(incoming_bytes.size());
    GGML_ASSERT(incoming_chain.append(
        incoming_bytes.data(), incoming_bytes.size()));

    std::unique_ptr<vbr_prepared_companion_image> qsa_replacement;
    GGML_ASSERT(qsa_adoption.prepare_replacement_with_layout &&
        qsa_adoption.prepare_replacement_with_layout(
            qsa_adoption.context, parse_qsa(incoming_chain),
            parse_qsa(qsa_chain), 0,
            qsa_layout, qsa_replacement));
    GGML_ASSERT(qsa_replacement && qsa_adoption.recheck(
        qsa_adoption.context, *qsa_replacement));
    GGML_ASSERT(read_qsa_rows(changed_cell.physical_cell) ==
        expected_incoming_row);
    GGML_ASSERT(qsa_adoption.rollback(
        qsa_adoption.context, *qsa_replacement));
    assert_qsa_image(qsa_layout);

    decode_range(ctx.get(), 321, 1);
    GGML_ASSERT(nmse(qsa_continuation_reference, last_logits(ctx.get())) <= 1e-12);

    // Qwen4 recurrent sequence images interleave the PLE convolution-history
    // row immediately after each R row.  Prove the atomic recurrent companion
    // parser consumes that native layout instead of accepting only R/S-only
    // model state.
    class recurrent_bytes_writer final : public llama_io_write_i {
    public:
        void write(const void * data, size_t size) override {
            const auto * bytes = static_cast<const uint8_t *>(data);
            output.insert(output.end(), bytes, bytes + size);
        }
        void write_tensor(
                ggml_tensor * tensor, size_t offset, size_t size) override {
            const size_t start = output.size();
            output.resize(start + size);
            ggml_backend_tensor_get(
                tensor, output.data() + start, offset, size);
        }
        size_t n_bytes() override { return output.size(); }
        std::vector<uint8_t> output;
    } recurrent_writer;
    static constexpr uint32_t recurrent_magic = 0xaf143cd8;
    recurrent_writer.write(&recurrent_magic, sizeof(recurrent_magic));
    const llama_seq_id recurrent_source_sequence = 0;
    recurrent_writer.write(
        &recurrent_source_sequence, sizeof(recurrent_source_sequence));
    recurrent_child->recurrent->state_write(recurrent_writer, 0, 0);
    artifact_segment_chain recurrent_chain(recurrent_writer.output.size());
    GGML_ASSERT(recurrent_chain.append(
        recurrent_writer.output.data(), recurrent_writer.output.size()));
    vbr_artifact_companion_payload recurrent_descriptor;
    recurrent_descriptor.kind = vbr_artifact_companion_kind::recurrent;
    recurrent_descriptor.format_version = 1;
    recurrent_descriptor.build_identity_digest =
        vbr_explicit_recurrent_companion_build_identity();
    recurrent_descriptor.payload_bytes = recurrent_chain.size();
    vbr_target_companion_snapshot recurrent_target;
    recurrent_target.kind = vbr_artifact_companion_kind::recurrent;
    recurrent_target.format_version = 1;
    recurrent_target.build_identity_digest =
        recurrent_descriptor.build_identity_digest;
    recurrent_target.available = true;
    recurrent_target.target_cookie = recurrent_child->recurrent;
    std::unique_ptr<vbr_parsed_companion_image> recurrent_parsed;
    GGML_ASSERT(vbr_parse_recurrent_companion(
        nullptr, recurrent_descriptor, recurrent_chain,
        recurrent_target, recurrent_parsed));
    GGML_ASSERT(recurrent_parsed &&
        recurrent_parsed->kind() == vbr_artifact_companion_kind::recurrent);

    printf("Qwen4 static Turbo and dynamic dense/sparse QSA VBR CUDA test PASSED\n");
}

struct file_deleter {
    void operator()(FILE * file) const {
        if (file) {
            fclose(file);
        }
    }
};

using file_ptr = std::unique_ptr<FILE, file_deleter>;

static file_ptr make_test_tmpfile() {
#if defined(_WIN32)
    // tmpfile() can still require administrator privileges on Windows; callers
    // already treat an unavailable round-trip file as a skipped check.
    return file_ptr(tmpfile());
#else
    const char * tmpdir = std::getenv("TMPDIR");
    std::string path = std::string(tmpdir && tmpdir[0] ? tmpdir : "/tmp") +
        "/test-llama-archs-XXXXXX";
    const int fd = mkstemp(path.data());
    if (fd < 0) {
        return {};
    }
    FILE * file = fdopen(fd, "w+b");
    if (file == nullptr) {
        close(fd);
        unlink(path.c_str());
        return {};
    }
    // Retain tmpfile()'s anonymous lifetime while honoring the caller's
    // build-local TMPDIR instead of filling the system tmpfs.
    unlink(path.c_str());
    return file_ptr(file);
#endif
}

static file_ptr make_dflash_selector_identity_file(std::initializer_list<const char *> tensor_names) {
    file_ptr file = make_test_tmpfile();
    if (!file) {
        return {};
    }

    ggml_init_params tensor_params = {
        /* .mem_size   = */ 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * tensor_ctx = ggml_init(tensor_params);
    GGML_ASSERT(tensor_ctx != nullptr);

    gguf_context_ptr gguf_ctx(gguf_init_empty());
    gguf_set_val_str(gguf_ctx.get(), "general.architecture", "dflash");
    gguf_set_val_f32(gguf_ctx.get(), "dflash.attention.layer_norm_rms_epsilon", 1.0e-5f);
    gguf_set_val_u32(gguf_ctx.get(), "dflash.conv_kernel_size", 2);
    gguf_set_val_u32(gguf_ctx.get(), "dflash.conv_group_size", 1);
    gguf_set_val_u32(gguf_ctx.get(), "dflash.selector_rank", 1);
    gguf_set_val_u32(gguf_ctx.get(), "dflash.selector_top_k", 1);
    const int32_t target_layer = 0;
    gguf_set_arr_data(gguf_ctx.get(), "dflash.target_layers", GGUF_TYPE_INT32,
            &target_layer, 1);
    for (const char * tensor_name : tensor_names) {
        ggml_tensor * tensor = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 1);
        ggml_set_name(tensor, tensor_name);
        gguf_add_tensor(gguf_ctx.get(), tensor);
    }
    GGML_ASSERT(gguf_write_to_file_ptr(gguf_ctx.get(), file.get(), false));

    ggml_free(tensor_ctx);
    rewind(file.get());
    return file;
}

static void test_dflash_loader_exact_identity() {
    // Some Windows test environments cannot create anonymous temporary files.
    // The pure classifier still runs above; skip only the serialized-loader
    // identity proof when the platform refuses the file primitive.
    file_ptr tmpfile_probe = make_test_tmpfile();
    if (!tmpfile_probe) {
        std::fprintf(stderr, "DFlash exact-name loader test skipped: temporary file unavailable\n");
        return;
    }

    const auto check = [](const char * stored_name, bool expect_fork, llm_dflash_selector_family expected_family) {
        file_ptr file = make_dflash_selector_identity_file({ stored_name });
        GGML_ASSERT(file != nullptr);
        std::vector<std::string> splits;
        llama_model_loader loader(
                /* metadata        */ nullptr,
                /* set_tensor_data */ nullptr,
                /* user_data       */ nullptr,
                /* fname           */ "",
                splits,
                file.get(),
                LLAMA_LOAD_MODE_NONE,
                /* check_tensors   */ false,
                /* no_alloc        */ true,
                /* load_mtp        */ false,
                /* kv_overrides    */ nullptr,
                /* tensor_overrides */ nullptr);

        // Generic tensor lookup must not cross the two wire families. Admission
        // below owns the compatibility decision before any tensor is loaded.
        GGML_ASSERT((loader.get_tensor_meta("selector.hidden_proj.weight") != nullptr) == expect_fork);
        GGML_ASSERT((loader.get_tensor_meta("selector_hidden.weight") != nullptr) != expect_fork);
        GGML_ASSERT((loader.get_tensor_meta_exact("selector.hidden_proj.weight") != nullptr) == expect_fork);
        GGML_ASSERT((loader.get_tensor_meta_exact("selector_hidden.weight") != nullptr) != expect_fork);
        GGML_ASSERT(llm_dflash_selector_family_from_loader(true, 1, loader) == expected_family);

        llama_model_params params = llama_model_default_params();
        llama_model_ptr model(llama_model_create(loader, params));
        GGML_ASSERT(model != nullptr);
        model->hparams.n_layer_all = 1;
        model->hparams.n_embd = 1;
        bool admitted = true;
        std::string refusal;
        try {
            model->load_arch_hparams(loader);
        } catch (const std::runtime_error & error) {
            admitted = false;
            refusal = error.what();
        }
        if (admitted != expect_fork) {
            std::fprintf(stderr, "DFlash admission mismatch stored=%s admitted=%d refusal=%s\n",
                    stored_name, int(admitted), refusal.c_str());
        }
        GGML_ASSERT(admitted == expect_fork);
        if (expect_fork) {
            GGML_ASSERT(model->hparams.dflash2_selector_rank == 1);
        } else {
            GGML_ASSERT(refusal.find("upstream DFlash convolution/selector tensor schema is unsupported") !=
                    std::string::npos);
        }
    };

    check("selector.hidden_proj.weight", true,  llm_dflash_selector_family::fork_dflash2);
    check("selector_hidden.weight",      false, llm_dflash_selector_family::upstream_compat);

    file_ptr mixed_file = make_dflash_selector_identity_file({
        "selector.hidden_proj.weight",
        "selector_hidden.weight",
    });
    GGML_ASSERT(mixed_file != nullptr);
    std::vector<std::string> splits;
    llama_model_loader mixed_loader(
            nullptr, nullptr, nullptr, "", splits, mixed_file.get(), LLAMA_LOAD_MODE_NONE,
            false, true, false, nullptr, nullptr);
    GGML_ASSERT(llm_dflash_selector_family_from_loader(true, 1, mixed_loader) ==
            llm_dflash_selector_family::mixed);
    {
        llama_model_params params = llama_model_default_params();
        llama_model_ptr model(llama_model_create(mixed_loader, params));
        GGML_ASSERT(model != nullptr);
        model->hparams.n_layer_all = 1;
        model->hparams.n_embd = 1;
        bool refused = false;
        try {
            model->load_arch_hparams(mixed_loader);
        } catch (const std::runtime_error & error) {
            refused = std::string(error.what()).find("mixes mutually exclusive") != std::string::npos;
        }
        GGML_ASSERT(refused);
    }

    file_ptr partial_mixed_file = make_dflash_selector_identity_file({
        "selector.hidden_proj.weight",
        "blk.0.attn_conv_base",
    });
    GGML_ASSERT(partial_mixed_file != nullptr);
    splits.clear();
    llama_model_loader partial_mixed_loader(
            nullptr, nullptr, nullptr, "", splits, partial_mixed_file.get(), LLAMA_LOAD_MODE_NONE,
            false, true, false, nullptr, nullptr);
    GGML_ASSERT(llm_dflash_selector_family_from_loader(true, 1, partial_mixed_loader) ==
            llm_dflash_selector_family::mixed);
    {
        llama_model_params params = llama_model_default_params();
        llama_model_ptr model(llama_model_create(partial_mixed_loader, params));
        GGML_ASSERT(model != nullptr);
        model->hparams.n_layer_all = 1;
        model->hparams.n_embd = 1;
        bool refused = false;
        try {
            model->load_arch_hparams(partial_mixed_loader);
        } catch (const std::runtime_error & error) {
            refused = std::string(error.what()).find("mixes mutually exclusive") != std::string::npos;
        }
        GGML_ASSERT(refused);
    }

    file_ptr unidentified_file = make_dflash_selector_identity_file({ "unrelated.weight" });
    GGML_ASSERT(unidentified_file != nullptr);
    splits.clear();
    llama_model_loader unidentified_loader(
            nullptr, nullptr, nullptr, "", splits, unidentified_file.get(), LLAMA_LOAD_MODE_NONE,
            false, true, false, nullptr, nullptr);
    {
        llama_model_params params = llama_model_default_params();
        llama_model_ptr model(llama_model_create(unidentified_loader, params));
        GGML_ASSERT(model != nullptr);
        model->hparams.n_layer_all = 1;
        model->hparams.n_embd = 1;
        bool refused = false;
        try {
            model->load_arch_hparams(unidentified_loader);
        } catch (const std::runtime_error & error) {
            refused = std::string(error.what()).find("has no recognized selector tensor schema") !=
                std::string::npos;
        }
        GGML_ASSERT(refused);
    }
}

static file_ptr make_qwen35_mtp_sidecar(const ggml_type d2t_type, const size_t seed) {
    GGML_ASSERT(d2t_type == GGML_TYPE_I32 || d2t_type == GGML_TYPE_I64);
    file_ptr file = make_test_tmpfile();
    if (!file) {
        return file;
    }

    gguf_context_ptr source_gguf = get_gguf_ctx(LLM_ARCH_QWEN35, false);
    llama_model_saver source_meta(LLM_ARCH_QWEN35, source_gguf.get());
    source_meta.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));

    llama_model_params source_params = llama_model_default_params();
    source_params.progress_callback = silent_model_load_progress;
    source_params.load_mtp = true;
    ggml_backend_dev_t cpu_devices[] = { nullptr };
    source_params.devices = cpu_devices;

    size_t tmp = seed;
    llama_model_ptr source(llama_model_init_from_user(
            source_gguf.get(), set_tensor_data, &tmp, source_params));
    if (!source) {
        throw std::runtime_error("failed to create synthetic Qwen3.5 MTP model");
    }

    // Write a genuine MTP-only sidecar: omit the trunk and the optional full-vocab
    // NextN embedding/head so the normal loader must select tok_embd/output+d2t.
    gguf_context_ptr sidecar_gguf(gguf_init_empty());
    gguf_set_kv(sidecar_gguf.get(), source_gguf.get());
    llama_model_saver saver(LLM_ARCH_QWEN35, sidecar_gguf.get());
    for (const auto & entry : llama_internal_get_tensor_map(source.get())) {
        const std::string & name = entry.first;
        if (name.rfind("blk.0.", 0) == 0 ||
                name.find(".nextn.embed_tokens.") != std::string::npos ||
                name.find(".nextn.shared_head_head.") != std::string::npos ||
                name == "output.weight") {
            continue;
        }
        saver.add_tensor(entry.second);
    }

    ggml_init_params tensor_params = {
        /*.mem_size   =*/ 64*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context_ptr tensor_ctx(ggml_init(tensor_params));
    ggml_tensor * output = ggml_new_tensor_2d(tensor_ctx.get(), GGML_TYPE_F16, 256, 32);
    ggml_set_name(output, "output.weight");
    memset(output->data, 0, ggml_nbytes(output));
    saver.add_tensor(output);

    ggml_tensor * d2t = ggml_new_tensor_1d(tensor_ctx.get(), d2t_type, 32);
    ggml_set_name(d2t, "d2t");
    if (d2t_type == GGML_TYPE_I32) {
        std::vector<int32_t> values(32);
        for (int32_t i = 0; i < 32; ++i) {
            values[i] = i;
        }
        memcpy(d2t->data, values.data(), ggml_nbytes(d2t));
    } else {
        std::vector<int64_t> values(32);
        for (int64_t i = 0; i < 32; ++i) {
            values[i] = i;
        }
        memcpy(d2t->data, values.data(), ggml_nbytes(d2t));
    }
    saver.add_tensor(d2t);
    saver.save(file.get());
    fflush(file.get());
    rewind(file.get());
    return file;
}

static std::pair<llama_model_ptr, llama_context_ptr> load_qwen35_mtp_sidecar(FILE * file) {
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t cpu_devices[] = { nullptr };
    model_params.devices = cpu_devices;

    llama_model_ptr model(llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to load synthetic Qwen3.5 MTP sidecar");
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.n_ctx = 8;
    ctx_params.n_batch = 8;
    ctx_params.n_ubatch = 8;
    ctx_params.n_seq_max = 1;
    ctx_params.n_outputs_max = 1;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    if (!ctx) {
        throw std::runtime_error("failed to create Qwen3.5 MTP context");
    }
    return std::make_pair(std::move(model), std::move(ctx));
}

static void test_qwen35_mtp_d2t_contract(const size_t seed) {
    auto check_dense_fallback = [seed](const ggml_type d2t_type) {
        file_ptr file = make_qwen35_mtp_sidecar(d2t_type, seed);
        if (!file) {
            return false;
        }
        auto model_and_ctx = load_qwen35_mtp_sidecar(file.get());
        ggml_cgraph * gf = llama_graph_reserve(model_and_ctx.second.get(), 1, 1, 1);
        GGML_ASSERT(gf != nullptr);

        ggml_tensor * output = ggml_graph_get_tensor(gf, "result_output_d2t");
        GGML_ASSERT(output != nullptr);
        GGML_ASSERT(output->type == GGML_TYPE_F32 && output->ne[0] == 128 && output->ne[1] == 1);
        GGML_ASSERT(output->op == GGML_OP_RESHAPE && output->src[0] != nullptr);

        ggml_tensor * set_rows = output->src[0];
        GGML_ASSERT(set_rows->op == GGML_OP_SET_ROWS);
        GGML_ASSERT(set_rows->src[0] != nullptr && set_rows->src[0]->ne[1] == 32);
        GGML_ASSERT(set_rows->src[1] != nullptr && set_rows->src[1]->type == d2t_type);
        GGML_ASSERT(set_rows->src[1]->ne[0] == 32);
        GGML_ASSERT(set_rows->src[2] != nullptr && set_rows->src[2]->ne[1] == 128);
        return true;
    };

    if (!check_dense_fallback(GGML_TYPE_I64) || !check_dense_fallback(GGML_TYPE_I32)) {
        printf("Qwen3.5 MTP d2t loader/graph contract test SKIPPED (tmpfile unavailable)\n");
        return;
    }

    // A backend sampler may be valid without understanding a compact token-id
    // domain. Such chains must keep the dense scatter fallback. Greedy would
    // otherwise return the compact row index, while logit bias would use a full
    // token id as a compact-row index.
    for (int unsafe_kind = 0; unsafe_kind < 2; ++unsafe_kind) {
        file_ptr file = make_qwen35_mtp_sidecar(GGML_TYPE_I32, seed);
        if (!file) {
            printf("Qwen3.5 MTP d2t loader/graph contract test SKIPPED (tmpfile unavailable)\n");
            return;
        }
        auto model_and_ctx = load_qwen35_mtp_sidecar(file.get());

        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        if (unsafe_kind == 0) {
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        } else {
            const llama_logit_bias bias = { 101, 5.0f };
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_logit_bias(128, 1, &bias));
        }
        GGML_ASSERT(llama_set_sampler(model_and_ctx.second.get(), 0, sampler.get()));

        ggml_cgraph * gf = llama_graph_reserve(model_and_ctx.second.get(), 1, 1, 1);
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(ggml_graph_get_tensor(gf, "result_output_d2t") != nullptr);
    }

    {
        file_ptr file = make_qwen35_mtp_sidecar(GGML_TYPE_I32, seed);
        if (!file) {
            printf("Qwen3.5 MTP d2t loader/graph contract test SKIPPED (tmpfile unavailable)\n");
            return;
        }
        auto model_and_ctx = load_qwen35_mtp_sidecar(file.get());

        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        // Disabled default stages lower to a backend-capable no-op. They must
        // preserve compact candidate-domain eligibility.
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_penalties(128, 0, 1.0f, 0.0f, 0.0f));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(4));
        GGML_ASSERT(llama_set_sampler(model_and_ctx.second.get(), 0, sampler.get()));

        ggml_cgraph * gf = llama_graph_reserve(model_and_ctx.second.get(), 1, 1, 1);
        GGML_ASSERT(gf != nullptr);
        GGML_ASSERT(ggml_graph_get_tensor(gf, "result_output_d2t") == nullptr);

        ggml_tensor * output = ggml_graph_get_tensor(gf, "result_output");
        GGML_ASSERT(output != nullptr);
        GGML_ASSERT(output->type == GGML_TYPE_F32 && output->ne[0] == 32 && output->ne[1] == 1);

        ggml_tensor * candidates = ggml_graph_get_tensor(gf, "top_k_candidates");
        GGML_ASSERT(candidates != nullptr && candidates->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(candidates) == 4 && candidates->op == GGML_OP_GET_ROWS);
        GGML_ASSERT(candidates->src[0] != nullptr && candidates->src[0]->op == GGML_OP_RESHAPE);
        GGML_ASSERT(candidates->src[0]->src[0] != nullptr);
        GGML_ASSERT(candidates->src[0]->src[0]->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_nelements(candidates->src[0]->src[0]) == 32);
    }

    printf("Qwen3.5 MTP d2t loader/graph contract test PASSED\n");
}

static gguf_context_ptr get_qwen4_mtp_gguf_ctx(uint32_t n_nextn = 1) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    llama_model_saver metadata(LLM_ARCH_QWEN4EXP, gguf_ctx.get());

    // The ordinary fixture has two trunk blocks. Reinterpret the second as the
    // single dense MTP block and pin its lack of QSA compression in metadata.
    metadata.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, n_nextn);
    metadata.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(1));
    metadata.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>({ 4, 0 }));
    return gguf_ctx;
}

static file_ptr make_qwen4_mtp_sidecar(
        const size_t seed, std::vector<float> & target_h, const char * omit_tensor = nullptr) {
    file_ptr file = make_test_tmpfile();
    if (!file) {
        return file;
    }

    gguf_context_ptr source_gguf = get_qwen4_mtp_gguf_ctx();
    llama_model_params source_params = llama_model_default_params();
    source_params.progress_callback = silent_model_load_progress;
    source_params.load_mtp = true;
    ggml_backend_dev_t cpu_devices[] = { nullptr };
    source_params.devices = cpu_devices;

    size_t tmp = seed;
    llama_model_ptr source(llama_model_init_from_user(
            source_gguf.get(), set_tensor_data, &tmp, source_params));
    if (!source) {
        throw std::runtime_error("failed to create synthetic Qwen4 MTP model");
    }
    GGML_ASSERT(source->hparams.n_layer() == 1);
    GGML_ASSERT(source->hparams.n_layer_nextn == 1);
    GGML_ASSERT(source->hparams.n_layer_all == 2);
    GGML_ASSERT(!source->hparams.is_recr(1));

    // Exercise the target half of the contract: it must publish the wide HC
    // residual that the draft head consumes, rather than the collapsed output.
    llama_context_params target_params = llama_context_default_params();
    target_params.n_ctx = 8;
    target_params.n_batch = 8;
    target_params.n_ubatch = 8;
    target_params.n_seq_max = 1;
    target_params.n_outputs_max = 1;
    target_params.n_threads = 1;
    target_params.n_threads_batch = 1;
    target_params.n_rs_seq = 3;
    llama_context_ptr target(llama_init_from_model(source.get(), target_params));
    if (!target) {
        throw std::runtime_error("failed to create synthetic Qwen4 target context");
    }
    GGML_ASSERT(llama_n_rs_seq(target.get()) == 3);
    llama_set_embeddings_nextn(target.get(), true, false);

    ggml_cgraph * target_gf = llama_graph_reserve(target.get(), 3, 1, 1);
    GGML_ASSERT(target_gf != nullptr);
    ggml_tensor * target_h_graph = ggml_graph_get_tensor(target_gf, "h_nextn");
    GGML_ASSERT(target_h_graph != nullptr);
    GGML_ASSERT(target_h_graph->ne[0] == source->hparams.n_embd);
    GGML_ASSERT(target_h_graph->ne[1] == source->hparams.dsv4_hc_mult);
    GGML_ASSERT(target_h_graph->ne[2] == 3);

    llama_token tokens[3] = { 5, 6, 7 };
    llama_batch target_batch = llama_batch_get_one(tokens, 3);
    GGML_ASSERT(llama_decode(target.get(), target_batch) == 0);
    llama_synchronize(target.get());

    const int64_t n_embd_out = source->hparams.n_embd_out();
    for (int32_t row = 0; row < 3; ++row) {
        const float * h = llama_get_embeddings_nextn_ith(target.get(), row);
        GGML_ASSERT(h != nullptr);
        GGML_ASSERT(std::all_of(h, h + n_embd_out, [](float value) {
            return std::isfinite(value);
        }));
    }
    const float * h_last = llama_get_embeddings_nextn_ith(target.get(), 2);
    target_h.assign(h_last, h_last + n_embd_out);
    GGML_ASSERT(std::all_of(target_h.begin(), target_h.end(), [](float value) {
        return std::isfinite(value);
    }));

    // Serialize a genuine standalone sidecar: retain global embedding/head
    // tensors and block 1, but omit every block-0 trunk tensor. The loader must
    // recognize this shape and still build the draft graph.
    gguf_context_ptr sidecar_gguf(gguf_init_empty());
    gguf_set_kv(sidecar_gguf.get(), source_gguf.get());
    llama_model_saver saver(LLM_ARCH_QWEN4EXP, sidecar_gguf.get());
    for (const auto & entry : llama_internal_get_tensor_map(source.get())) {
        if (entry.first.rfind("blk.0.", 0) == 0 ||
                entry.first.rfind("output_hc_", 0) == 0 ||
                (omit_tensor != nullptr && entry.first == omit_tensor)) {
            continue;
        }
        saver.add_tensor(entry.second);
    }
    saver.save(file.get());
    fflush(file.get());
    rewind(file.get());
    return file;
}

static file_ptr make_qwen4_mtp_combined(
        const size_t seed, bool add_bogus_tensor = false, bool add_fused_qkv = false) {
    file_ptr file = make_test_tmpfile();
    if (!file) {
        return file;
    }

    gguf_context_ptr source_gguf = get_qwen4_mtp_gguf_ctx();
    llama_model_params source_params = llama_model_default_params();
    source_params.progress_callback = silent_model_load_progress;
    source_params.load_mtp = true;
    ggml_backend_dev_t cpu_devices[] = { nullptr };
    source_params.devices = cpu_devices;

    size_t tmp = seed;
    llama_model_ptr source(llama_model_init_from_user(
            source_gguf.get(), set_tensor_data, &tmp, source_params));
    GGML_ASSERT(source != nullptr);

    gguf_context_ptr combined_gguf(gguf_init_empty());
    gguf_set_kv(combined_gguf.get(), source_gguf.get());
    llama_model_saver saver(LLM_ARCH_QWEN4EXP, combined_gguf.get());
    for (const auto & entry : llama_internal_get_tensor_map(source.get())) {
        saver.add_tensor(entry.second);
    }

    ggml_context_ptr extra_ctx;
    if (add_bogus_tensor || add_fused_qkv) {
        ggml_init_params params = {
            /* .mem_size   = */ 4*1024*1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        extra_ctx.reset(ggml_init(params));
        GGML_ASSERT(extra_ctx != nullptr);
        if (add_bogus_tensor) {
            ggml_tensor * bogus = ggml_new_tensor_1d(extra_ctx.get(), GGML_TYPE_F32, 1);
            ggml_set_name(bogus, "qwen4exp.unrecognized_sibling.weight");
            saver.add_tensor(bogus);
        }
        if (add_fused_qkv) {
            const auto & mtp = source->layers[1];
            const int64_t n_qkv = mtp.wq->ne[1] + mtp.wk->ne[1] + mtp.wv->ne[1];
            ggml_tensor * qkv = ggml_new_tensor_2d(
                    extra_ctx.get(), GGML_TYPE_F32, source->hparams.n_embd, n_qkv);
            ggml_set_name(qkv, "blk.1.attn_qkv.weight");
            saver.add_tensor(qkv);
            for (const char * suffix : { "scale", "input_scale" }) {
                ggml_tensor * scale = ggml_new_tensor_1d(extra_ctx.get(), GGML_TYPE_F32, 1);
                ggml_set_name(scale, format("blk.1.attn_qkv.%s", suffix).c_str());
                saver.add_tensor(scale);
            }
        }
    }
    saver.save(file.get());
    fflush(file.get());
    rewind(file.get());
    return file;
}

static void test_qwen4_mtp_sidecar_contract(const size_t seed) {
    {
        llama_hparams predicate_hparams = {};
        predicate_hparams.n_layer_nextn = 1;
        predicate_hparams.router_layer = 0;
        GGML_ASSERT(!predicate_hparams.has_mtp());
        predicate_hparams.router_layer = -1;
        GGML_ASSERT(predicate_hparams.has_mtp());
    }

    // The target load must ignore the attached draft block entirely. This pins
    // TENSOR_SKIP for both split and optional fused-expert representations.
    {
        file_ptr integrated_file = make_qwen4_mtp_combined(seed);
        GGML_ASSERT(integrated_file != nullptr);
        llama_model_params target_model_params = llama_model_default_params();
        target_model_params.progress_callback = silent_model_load_progress;
        target_model_params.load_mtp = false;
        ggml_backend_dev_t cpu_devices[] = { nullptr };
        target_model_params.devices = cpu_devices;
        llama_model_ptr target_model(llama_model_load_from_file_ptr(
                integrated_file.get(), target_model_params));
        GGML_ASSERT(target_model != nullptr);
        GGML_ASSERT(llama_model_has_mtp(target_model.get()));
        GGML_ASSERT(target_model->layers[1].nextn.eh_proj == nullptr);
        GGML_ASSERT(target_model->layers[1].nextn.hc_head_norm == nullptr);
        GGML_ASSERT(target_model->layers[1].ffn_gate_exps == nullptr);
        GGML_ASSERT(target_model->layers[1].ffn_up_exps == nullptr);
        GGML_ASSERT(target_model->layers[1].ffn_gate_up_exps == nullptr);

        // Ordinary embedding extraction uses the model's advertised wide HC
        // width. It must publish the wide residual rather than overread the
        // collapsed output-head input.
        llama_context_params embd_params = llama_context_default_params();
        embd_params.n_ctx = 4;
        embd_params.n_batch = 4;
        embd_params.n_ubatch = 4;
        embd_params.n_seq_max = 1;
        embd_params.embeddings = true;
        embd_params.n_threads = 1;
        embd_params.n_threads_batch = 1;
        llama_context_ptr embd_ctx(llama_init_from_model(target_model.get(), embd_params));
        GGML_ASSERT(embd_ctx != nullptr);
        llama_token embd_token = 5;
        llama_batch embd_batch = llama_batch_get_one(&embd_token, 1);
        GGML_ASSERT(llama_decode(embd_ctx.get(), embd_batch) == 0);
        llama_synchronize(embd_ctx.get());
        const float * embd = llama_get_embeddings_ith(embd_ctx.get(), 0);
        GGML_ASSERT(embd != nullptr);
        GGML_ASSERT(std::all_of(embd, embd + target_model->hparams.n_embd_out(), [](float value) {
            return std::isfinite(value);
        }));

        file_ptr bogus_file = make_qwen4_mtp_combined(seed, true);
        GGML_ASSERT(bogus_file != nullptr);
        llama_model_ptr bogus_model(llama_model_load_from_file_ptr(
                bogus_file.get(), target_model_params));
        GGML_ASSERT(bogus_model == nullptr);

        file_ptr fused_qkv_file = make_qwen4_mtp_combined(seed, false, true);
        GGML_ASSERT(fused_qkv_file != nullptr);
        llama_model_ptr fused_qkv_model(llama_model_load_from_file_ptr(
                fused_qkv_file.get(), target_model_params));
        GGML_ASSERT(fused_qkv_model != nullptr);
        GGML_ASSERT(fused_qkv_model->layers[1].wqkv == nullptr);
    }

    // Reject unsupported multi-block artifacts during model load, before graph
    // construction can hit an assertion after allocating the draft weights.
    {
        gguf_context_ptr multi_gguf = get_qwen4_mtp_gguf_ctx(2);
        llama_model_params multi_params = llama_model_default_params();
        multi_params.progress_callback = silent_model_load_progress;
        multi_params.load_mtp = true;
        ggml_backend_dev_t cpu_devices[] = { nullptr };
        multi_params.devices = cpu_devices;
        size_t multi_seed = seed;
        llama_model_ptr multi_model(llama_model_init_from_user(
                multi_gguf.get(), set_tensor_data, &multi_seed, multi_params));
        GGML_ASSERT(multi_model == nullptr);
    }

    std::vector<float> target_h;
    file_ptr file = make_qwen4_mtp_sidecar(seed, target_h);
    if (!file) {
        printf("Qwen4 MTP sidecar contract test SKIPPED (tmpfile unavailable)\n");
        return;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t cpu_devices[] = { nullptr };
    model_params.devices = cpu_devices;

    // A standalone sidecar is not a valid target model. Without load_mtp the
    // absent trunk must fail model load rather than yielding null graph inputs.
    llama_model_params wrong_role_params = model_params;
    wrong_role_params.load_mtp = false;
    llama_model_ptr wrong_role(llama_model_load_from_file_ptr(file.get(), wrong_role_params));
    GGML_ASSERT(wrong_role == nullptr);
    rewind(file.get());

    llama_model_ptr model(llama_model_load_from_file_ptr(file.get(), model_params));
    if (!model) {
        throw std::runtime_error("failed to load synthetic Qwen4 MTP sidecar");
    }
    GGML_ASSERT(model->hparams.n_layer() == 1);
    GGML_ASSERT(model->hparams.n_layer_nextn == 1);
    GGML_ASSERT(llama_model_has_mtp(model.get()));
    GGML_ASSERT(model->hc_head_norm == nullptr);
    GGML_ASSERT(model->layers[0].hc_attn_norm == nullptr);
    GGML_ASSERT(model->layers[1].nextn.eh_proj != nullptr);
    GGML_ASSERT(model->layers[1].nextn.enorm != nullptr);
    GGML_ASSERT(model->layers[1].nextn.hnorm != nullptr);
    GGML_ASSERT(model->layers[1].nextn.hc_head_norm != nullptr);
    GGML_ASSERT(model->layers[1].nextn.hc_head_down != nullptr);
    GGML_ASSERT(model->layers[1].nextn.hc_head_up != nullptr);
    GGML_ASSERT(model->layers[1].index_q_proj == nullptr);
    GGML_ASSERT(model->layers[1].index_k_proj == nullptr);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    ctx_params.n_ctx = 8;
    ctx_params.n_batch = 8;
    ctx_params.n_ubatch = 8;
    ctx_params.n_seq_max = 2;
    ctx_params.kv_unified = true;
    ctx_params.n_outputs_max = 2;
    ctx_params.n_threads = 1;
    ctx_params.n_threads_batch = 1;

    // A sidecar MTP context has the same linkage contract as native MTP: it
    // points at a target context through ctx_other. Qwen4 nevertheless owns a
    // separate filtered draft cache, so context linkage must not imply shared
    // physical KV cells.
    file_ptr target_file = make_qwen4_mtp_combined(seed);
    GGML_ASSERT(target_file != nullptr);
    llama_model_params target_model_params = llama_model_default_params();
    target_model_params.progress_callback = silent_model_load_progress;
    target_model_params.load_mtp = false;
    target_model_params.devices = cpu_devices;
    llama_model_ptr target_model(llama_model_load_from_file_ptr(
            target_file.get(), target_model_params));
    GGML_ASSERT(target_model != nullptr);
    uint8_t semantic_digest[32] = {};
    // This synthetic fixture intentionally has no production vocabulary, so
    // semantic identity is unavailable. It must nevertheless traverse the
    // appended MTP layer without calling a target-only layer accessor out of
    // bounds (the real integrated model used to abort here).
    (void) llama_model_semantic_family_digest(
            target_model.get(), semantic_digest);
    llama_context_params target_ctx_params = ctx_params;
    target_ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    llama_context_ptr target_ctx(llama_init_from_model(target_model.get(), target_ctx_params));
    GGML_ASSERT(target_ctx != nullptr);
    ctx_params.ctx_other = target_ctx.get();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    if (!ctx) {
        throw std::runtime_error("failed to create synthetic Qwen4 MTP context");
    }
    GGML_ASSERT(dynamic_cast<llama_memory_hybrid_idx *>(llama_get_memory(ctx.get())) == nullptr);
    GGML_ASSERT(llama_get_ctx_other(ctx.get()) == target_ctx.get());
    GGML_ASSERT(!llama_memory_has_shared_cells(llama_get_memory(ctx.get())));
    GGML_ASSERT(llama_n_ctx(ctx.get()) >= 8);
    GGML_ASSERT(llama_n_ctx_seq(ctx.get()) == llama_n_ctx(ctx.get()));

    llama_set_embeddings_nextn(ctx.get(), true, true);
    ggml_cgraph * gf = llama_graph_reserve(ctx.get(), 2, 1, 1);
    GGML_ASSERT(gf != nullptr);
    GGML_ASSERT(ggml_graph_get_tensor(gf, "mtp_h_input") != nullptr);
    ggml_tensor * eh_proj = ggml_graph_get_tensor(gf, "mtp_eh_proj-1");
    GGML_ASSERT(eh_proj != nullptr);
    ggml_tensor * eh_concat = nullptr;
    std::vector<ggml_tensor *> pending = { eh_proj };
    while (!pending.empty() && eh_concat == nullptr) {
        ggml_tensor * node = pending.back();
        pending.pop_back();
        if (node->op == GGML_OP_CONCAT) {
            eh_concat = node;
            break;
        }
        for (ggml_tensor * src : node->src) {
            if (src != nullptr) {
                pending.push_back(src);
            }
        }
    }
    GGML_ASSERT(eh_concat != nullptr);
    GGML_ASSERT(strcmp(ggml_get_name(eh_concat->src[0]), "mtp_enorm-1") == 0);
    GGML_ASSERT(strcmp(ggml_get_name(eh_concat->src[1]), "mtp_hnorm-1") == 0);
    GGML_ASSERT(ggml_graph_get_tensor(gf, "mtp_hc_attn_pre-1") != nullptr);
    GGML_ASSERT(ggml_graph_get_tensor(gf, "mtp_hc_head") != nullptr);
    GGML_ASSERT(ggml_graph_get_tensor(gf, "indexer_k_raw-1") == nullptr);
    GGML_ASSERT(ggml_graph_get_tensor(gf, "result_output") != nullptr);
    ggml_tensor * draft_h_graph = ggml_graph_get_tensor(gf, "h_nextn");
    GGML_ASSERT(draft_h_graph != nullptr);
    GGML_ASSERT(draft_h_graph->ne[0] == model->hparams.n_embd_out());
    GGML_ASSERT(draft_h_graph->ne[1] == 1);

    // Feed the exact target-published HC state together with one proposed token
    // for each logical sequence. This exercises both token and hidden inputs,
    // multi-slot unified draft KV, the one-layer graph, and chained t_h_nextn
    // publication.
    llama_batch batch = llama_batch_init(2, (int32_t) target_h.size(), 1);
    batch.token = (llama_token *) malloc(2 * sizeof(llama_token));
    GGML_ASSERT(batch.token != nullptr);
    batch.n_tokens = 2;
    batch.token[0] = 8;
    batch.token[1] = 9;
    memcpy(batch.embd, target_h.data(), target_h.size() * sizeof(float));
    memcpy(batch.embd + target_h.size(), target_h.data(), target_h.size() * sizeof(float));
    batch.pos[0] = 0;
    batch.pos[1] = 0;
    batch.n_seq_id[0] = 1;
    batch.n_seq_id[1] = 1;
    batch.seq_id[0][0] = 0;
    batch.seq_id[1][0] = 1;
    batch.logits[0] = 1;
    batch.logits[1] = 1;
    GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
    llama_synchronize(ctx.get());

    for (int32_t out = 0; out < 2; ++out) {
        const float * logits = llama_get_logits_ith(ctx.get(), out);
        GGML_ASSERT(logits != nullptr);
        for (int32_t i = 0; i < llama_vocab_n_tokens(llama_model_get_vocab(model.get())); ++i) {
            GGML_ASSERT(std::isfinite(logits[i]));
        }
        const float * next_h = llama_get_embeddings_nextn_ith(ctx.get(), out);
        GGML_ASSERT(next_h != nullptr);
        for (size_t i = 0; i < target_h.size(); ++i) {
            GGML_ASSERT(std::isfinite(next_h[i]));
        }
    }
    llama_batch_free(batch);

    // The three NextN input tensors are mandatory. A missing hidden-state norm
    // must fail load rather than silently build a graph with an absent operand.
    std::vector<float> ignored_h;
    file_ptr incomplete = make_qwen4_mtp_sidecar(seed, ignored_h, "blk.1.nextn.hnorm.weight");
    GGML_ASSERT(incomplete != nullptr);
    llama_model_ptr incomplete_model(llama_model_load_from_file_ptr(incomplete.get(), model_params));
    GGML_ASSERT(incomplete_model == nullptr);

    file_ptr incomplete_mixer = make_qwen4_mtp_sidecar(
            seed, ignored_h, "blk.1.nextn.hc_head_norm.weight");
    GGML_ASSERT(incomplete_mixer != nullptr);
    llama_model_ptr incomplete_mixer_model(
            llama_model_load_from_file_ptr(incomplete_mixer.get(), model_params));
    GGML_ASSERT(incomplete_mixer_model == nullptr);

    printf("Qwen4 MTP standalone sidecar/target handoff contract test PASSED\n");
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (encode) {
        if (llama_encode(lctx, batch)) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        return false; // FIXME @ngxson
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

// Archs whose graphs the meta (tensor-parallel) split planner cannot split yet.
// These pass on single devices and are skipped ONLY for SPLIT_MODE_TENSOR.
static bool arch_tensor_split_supported(const llm_arch arch) {
    if (llm_arch_is_diffusion(arch)) {
        // diffusion graphs reshape a permuted tensor (ggml-backend-meta.cpp
        // handle_reshape asserts); verified dream + llada, family-wide skip
        return false;
    }
    if (arch == LLM_ARCH_DFLASH_DRAFT || arch == LLM_ARCH_GEMMA4_DFLASH_DRAFT) {
        // drafters are never tensor-split in production — TP spec-decode pins the
        // drafter to a single device (--spec-draft-device); same permuted-reshape
        // limitation as the diffusion family when forced onto the meta device
        return false;
    }
    return true;
}

static int save_models(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_model_saver_supports_arch(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {});
            const std::string path = dir + "/" + llm_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: Saving %s model (%s) to %s...\n", __func__, llm_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model_and_ctx.first.get(), path.c_str());
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return 0;
}

static int test_backends(const llm_arch target_arch, const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode) {}
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, ggml_backend_dev_description(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llm_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llm_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|\n";

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip");
    printf("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        printf("-");
    }
    printf("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        printf("-");
    }
    printf("|------|---------------|---------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (target_arch != LLM_ARCH_UNKNOWN && arch != target_arch) {
            continue;
        }
        if (arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue; // FIXME: ISWA KV cache initialization needs more fixture params
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                printf(template_row_cfg.c_str(),
                    llm_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR &&
                        (dc.devs.empty() || !arch_tensor_split_supported(arch)));
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llm_arch_supports_sm_tensor(arch)) {
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, dc.devs, dc.split_mode, encode);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            all_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }
                    }

                    file_ptr file = make_test_tmpfile();
                    // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                    //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                    if (file && llama_model_saver_supports_arch(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        llama_model_saver ms = llama_model_saver(model_and_ctx_dev.first.get());
                        ms.add_kv_from_model();
                        ms.add_tensors_from_model();
                        ms.save(file.get());
                        rewind(file.get());

                        auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file.get(), seed, dc.devs, dc.split_mode, encode);
                        const std::vector<float> logits_roundtrip = get_logits(
                            model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                        status_roundtrip = "\033[1;32mOK\033[0m";
                        GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                        for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                            if (logits_roundtrip[i] != logits_dev[i]) {
                                all_ok = false;
                                status_roundtrip = "\033[1;31mFAIL\033[0m";
                                break;
                            }
                        }
                    }
                }

                // log the results for this test case
                printf(template_row_res.c_str(),
                    status_nmse.c_str(), nmse_str, status_roundtrip.c_str());
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // FIXME these tests are disabled in the CI for macOS-latest-cmake-arm64 because they are segfaulting
    common_init();
    std::random_device rd;

    llm_arch arch = LLM_ARCH_UNKNOWN;
    size_t seed = rd();
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;
    std::string out;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        }
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                arch = llm_arch_from_string(arch_name);
                if (arch == LLM_ARCH_UNKNOWN) {
                    LOG_ERR("%s: unkown LLM architecture: %s\n", __func__, arch_name.c_str());
                    return 1;
                }
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        }
    }
    printf("%s: using seed %zu\n", __func__, seed);

    try {
        test_dflash_selector_family_contract();
        test_dflash_loader_exact_identity();
        if (!out.empty()) {
            return save_models(arch, seed, log_level, out);
        }
        if (arch == LLM_ARCH_UNKNOWN || arch == LLM_ARCH_QWEN35) {
            test_qwen35_mtp_d2t_contract(seed);
        }
        if (arch == LLM_ARCH_UNKNOWN || arch == LLM_ARCH_QWEN4EXP) {
            test_qwen4_ple_recurrent_resize(seed);
            test_qwen4_indexed_cache_admission(seed);
            test_qwen4_vbr_cuda(seed);
            test_qwen4_mtp_sidecar_contract(seed);
        }
        return test_backends(arch, seed, log_level);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
