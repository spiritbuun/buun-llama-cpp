// Various helper functions and utilities

#pragma once

#include "llama-cpp.h"
#include "common-cache-family.h"

#include "ggml-opt.h"
#include "ggml.h"
#include "llama.h"

#include <list>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <map>
#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif

#ifdef _WIN32
#define DIRECTORY_SEPARATOR '\\'
#else
#define DIRECTORY_SEPARATOR '/'
#endif // _WIN32

#define COM_DBG(fmt, ...) LOG_DBG("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_TRC(fmt, ...) LOG_TRC("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_INF(fmt, ...) LOG_INF("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_WRN(fmt, ...) LOG_WRN("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_ERR(fmt, ...) LOG_ERR("cmn  %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define COM_CNT(fmt, ...) LOG_CNT(""              fmt,               __VA_ARGS__)

#define die(msg)          do { fputs("error: " msg "\n", stderr);                exit(1); } while (0)
#define die_fmt(fmt, ...) do { fprintf(stderr, "error: " fmt "\n", __VA_ARGS__); exit(1); } while (0)

struct common_time_meas {
    common_time_meas(int64_t & t_acc, bool disable = false);
    ~common_time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

// Defined by common-cache-plan.h. Fixed underlying type permits common_params
// to carry the closed value without introducing the common.h <-> checkpoint-
// shadow include cycle.
enum class common_cache_plan_authority_level : uint8_t;

struct common_adapter_lora_info {
    std::string path;
    float scale;

    std::string task_name;
    std::string prompt_prefix;

    struct llama_adapter_lora * ptr;
};

using llama_tokens = std::vector<llama_token>;

struct common_control_vector_load_info;

//
// CPU utils
//

struct common_cpu_params {
    int      n_threads                   = -1;
    bool     cpumask[GGML_MAX_N_THREADS] = {false}; // CPU affinity mask.
    bool     mask_valid                  = false;   // Default: any CPU
    enum ggml_sched_priority  priority   = GGML_SCHED_PRIO_NORMAL;  // Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
    bool     strict_cpu                  = false;   // Use strict CPU placement
    uint32_t poll                        = 50;      // Polling (busywait) level (0 - no polling, 100 - mostly polling)
};

int32_t common_cpu_get_num_physical_cores();
int32_t common_cpu_get_num_math();

//
// Common params
//

enum llama_example {
    LLAMA_EXAMPLE_BATCHED,
    LLAMA_EXAMPLE_DEBUG,
    LLAMA_EXAMPLE_COMMON,
    LLAMA_EXAMPLE_SPECULATIVE,
    LLAMA_EXAMPLE_COMPLETION,
    LLAMA_EXAMPLE_CLI,
    LLAMA_EXAMPLE_EMBEDDING,
    LLAMA_EXAMPLE_PERPLEXITY,
    LLAMA_EXAMPLE_RETRIEVAL,
    LLAMA_EXAMPLE_PASSKEY,
    LLAMA_EXAMPLE_IMATRIX,
    LLAMA_EXAMPLE_BENCH,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CVECTOR_GENERATOR,
    LLAMA_EXAMPLE_EXPORT_LORA,
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_LOOKUP,
    LLAMA_EXAMPLE_PARALLEL,
    LLAMA_EXAMPLE_TTS,
    LLAMA_EXAMPLE_DIFFUSION,
    LLAMA_EXAMPLE_FINETUNE,
    LLAMA_EXAMPLE_FIT_PARAMS,
    LLAMA_EXAMPLE_RESULTS,
    LLAMA_EXAMPLE_EXPORT_GRAPH_OPS,
    LLAMA_EXAMPLE_DOWNLOAD,
    LLAMA_EXAMPLE_TOKENIZE,

    LLAMA_EXAMPLE_COUNT,
};

enum common_sampler_type {
    COMMON_SAMPLER_TYPE_NONE        = 0,
    COMMON_SAMPLER_TYPE_DRY         = 1,
    COMMON_SAMPLER_TYPE_TOP_K       = 2,
    COMMON_SAMPLER_TYPE_TOP_P       = 3,
    COMMON_SAMPLER_TYPE_MIN_P       = 4,
  //COMMON_SAMPLER_TYPE_TFS_Z       = 5,
    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6,
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7,
    COMMON_SAMPLER_TYPE_XTC         = 8,
    COMMON_SAMPLER_TYPE_INFILL      = 9,
    COMMON_SAMPLER_TYPE_PENALTIES   = 10,
    COMMON_SAMPLER_TYPE_TOP_N_SIGMA = 11,
    COMMON_SAMPLER_TYPE_ADAPTIVE_P  = 12,
};

// dimensionality reduction methods, used by cvector-generator
enum dimre_method {
    DIMRE_METHOD_PCA,
    DIMRE_METHOD_MEAN,
};

enum common_conversation_mode {
    COMMON_CONVERSATION_MODE_DISABLED = 0,
    COMMON_CONVERSATION_MODE_ENABLED  = 1,
    COMMON_CONVERSATION_MODE_AUTO     = 2,
};

enum common_grammar_trigger_type {
    COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN,
    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
};

struct common_grammar_trigger {
    common_grammar_trigger_type type;
    std::string value;
    llama_token token = LLAMA_TOKEN_NULL;
};

enum common_params_sampling_config : uint64_t {
    COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS        = 1 << 0,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_K           = 1 << 1,
    COMMON_PARAMS_SAMPLING_CONFIG_TOP_P           = 1 << 2,
    COMMON_PARAMS_SAMPLING_CONFIG_MIN_P           = 1 << 3,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY = 1 << 4,
    COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD   = 1 << 5,
    COMMON_PARAMS_SAMPLING_CONFIG_TEMP            = 1 << 6,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N  = 1 << 7,
    COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT  = 1 << 8,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT        = 1 << 9,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU    = 1 << 10,
    COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA    = 1 << 11,
};

enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,          // no speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE,  // standalone draft model speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3,  // Eagle3 speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,     // Multi-token prediction
    COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,  // DFlash speculative decoding
    COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK,  // DSpark speculative decoding (DFlash + Markov head)
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,  // simple self-speculative decoding based on n-grams
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,   // self-speculative decoding with n-gram keys only
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, // self-speculative decoding with n-gram keys and 4 m-gram values
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,   // self-speculative decoding with 3-level n-gram cache
    COMMON_SPECULATIVE_TYPE_SUFFIX,        // model-free suffix tree speculative decoding
    COMMON_SPECULATIVE_TYPE_COPYSPEC,      // model-free copy-from-context speculative decoding
    COMMON_SPECULATIVE_TYPE_RECYCLE,       // model-free token recycling (adjacency matrix)
    COMMON_SPECULATIVE_TYPE_DFLASH,        // DFlash block-diffusion speculative decoding
    COMMON_SPECULATIVE_TYPE_COUNT          // number of types, unknown type
};

// Grammar type enumeration
enum common_grammar_type {
    COMMON_GRAMMAR_TYPE_NONE,           // no grammar set
    COMMON_GRAMMAR_TYPE_USER,           // user-provided GBNF (--grammar / "grammar" API field)
    COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT,  // auto-generated from JSON schema (--json-schema / "json_schema" API field)
    COMMON_GRAMMAR_TYPE_TOOL_CALLS,     // auto-generated by chat template parser for function calling
};

// Grammar variant struct with type and grammar string
struct common_grammar {
    common_grammar_type type = COMMON_GRAMMAR_TYPE_NONE;
    std::string grammar;

    // Default constructor - no grammar
    common_grammar() = default;

    // Constructor with type and grammar string
    common_grammar(common_grammar_type t, std::string g) : type(t), grammar(std::move(g)) {
        GGML_ASSERT(type != COMMON_GRAMMAR_TYPE_NONE || !grammar.empty());
    }

    // Check if a grammar is set
    bool empty() const { return type == COMMON_GRAMMAR_TYPE_NONE || grammar.empty(); }
};

// Returns the raw grammar string, or empty string if no grammar is set.
inline const std::string & common_grammar_value(const common_grammar & g) {
    return g.grammar;
}

// Returns true when the generation_prompt should be prefilled into the grammar sampler.
// Only output-format and tool-call grammars need prefill; user-supplied grammars must not be prefilled.
inline bool common_grammar_needs_prefill(const common_grammar & g) {
    return g.type == COMMON_GRAMMAR_TYPE_OUTPUT_FORMAT
        || g.type == COMMON_GRAMMAR_TYPE_TOOL_CALLS;
}

// sampling parameters
struct common_params_sampling {
    uint32_t seed = LLAMA_DEFAULT_SEED; // the seed used to initialize llama_sampler

    int32_t n_prev             = 64;     // number of previous tokens to remember
    int32_t n_probs            = 0;      // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t min_keep           = 0;      // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t top_k              = 40;     // <= 0 to use vocab size
    float   top_p              = 0.95f;  // 1.0 = disabled
    float   min_p              = 0.05f;  // 0.0 = disabled
    float   xtc_probability    = 0.00f;  // 0.0 = disabled
    float   xtc_threshold      = 0.10f;  // > 0.5 disables XTC
    float   typ_p              = 1.00f;  // typical_p, 1.0 = disabled
    float   temp               = 0.80f;  // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float   dynatemp_range     = 0.00f;  // 0.0 = disabled
    float   dynatemp_exponent  = 1.00f;  // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t penalty_last_n     = 64;     // last n tokens to penalize (0 = disable penalty)
    float   penalty_repeat     = 1.00f;  // 1.0 = disabled
    float   penalty_freq       = 0.00f;  // 0.0 = disabled
    float   penalty_present    = 0.00f;  // 0.0 = disabled
    float   dry_multiplier     = 0.0f;   // 0.0 = disabled;      DRY repetition penalty for tokens extending repetition:
    float   dry_base           = 1.75f;  // 0.0 = disabled;      multiplier * base ^ (length of sequence before token - allowed length)
    int32_t dry_allowed_length = 2;      // tokens extending repetitions beyond this receive penalty
    int32_t dry_penalty_last_n = 64;     // how many tokens to scan for repetitions (0 = disable penalty)
    float   adaptive_target    = -1.0f;  // select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    float   adaptive_decay     = 0.90f;  // EMA decay for adaptation; history ≈ 1/(1-decay) tokens (0.0 - 0.99)
    int32_t mirostat           = 0;      // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   top_n_sigma        = -1.00f; // -1.0 = disabled
    float   mirostat_tau       = 5.00f;  // target entropy
    float   mirostat_eta       = 0.10f;  // learning rate
    bool    ignore_eos         = false;
    bool    no_perf            = false;  // disable performance metrics
    bool    timing_per_token   = false;

    uint64_t user_sampling_config = 0; // bitfield to track user-specified samplers

    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};     // default sequence breakers for DRY

    std::vector<enum common_sampler_type> samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_DRY,
        COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TYPICAL_P,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_XTC,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };

    common_grammar              grammar;      // optional grammar constraint (user / output-format / tool-calls)
    bool                                grammar_lazy = false;
    std::vector<common_grammar_trigger> grammar_triggers; // optional triggers (for lazy grammars)
    std::set<llama_token>               preserved_tokens;

    std::vector<llama_logit_bias> logit_bias;     // logit biases to apply
    std::vector<llama_logit_bias> logit_bias_eog; // pre-calculated logit biases for EOG tokens

    // The assistant generation prompt already prefilled into the prompt.
    // Fed to the grammar sampler (to advance past pre-existing tokens) and used
    // to determine the reasoning budget sampler's initial state.
    // Only applied when the grammar is of output-format or tool-calls type.
    std::string generation_prompt;

    // reasoning budget sampler parameters
    // these are populated by the server/CLI based on chat template params
    int32_t                   reasoning_budget_tokens   = -1;  // -1 = disabled, >= 0 = token budget
    std::vector<llama_token>  reasoning_budget_start;          // start tag token sequence
    std::vector<llama_tokens> reasoning_budget_end;            // end tag token sequences; the first tag is used as the forcing sequence
    std::vector<llama_token>  reasoning_budget_forced;         // forced sequence (message + first end tag)
    std::string               reasoning_budget_message;        // message injected before end tag when budget exhausted
    bool                      reasoning_control = false;       // create the budget sampler on demand so reasoning can be ended at runtime

    bool backend_sampling = false;

    // print the parameters into a string
    std::string print() const;
};

struct common_params_model {
    std::string path        = ""; // model local path
    std::string url         = ""; // model url to download
    std::string hf_repo     = ""; // HF repo
    std::string hf_file     = ""; // HF file
    std::string docker_repo = ""; // Docker repo

    std::string get_name() const {
        if (!hf_repo.empty()) {
            return hf_repo;
        }
        if (!docker_repo.empty()) {
            return docker_repo;
        }
        return path;
    }

    bool empty() const {
        return get_name().empty();
    }
};

// draft-model-based speculative decoding parameters
struct common_params_speculative_draft {
    int32_t n_max = 3; // maximum number of tokens to draft during speculative decoding
    int32_t n_min = 0; // minimum number of draft tokens to use for speculative decoding
    bool n_max_set = false; // true when the user explicitly overrides the draft depth

    // Qwen-27B MTP-only sidecars: 32768 enables the experimental public
    // balanced FR-Spec map; 0 keeps the full vocabulary (default).
    uint32_t mtp_vocab_size = 0;

    float p_split = 0.1f; // speculative decoding split probability
    float p_min   = 0.0f; // minimum speculative decoding probability (greedy)

    bool backend_sampling = true; // offload draft sampling to the backend (default: on)
    bool dspark_gpu_assist = true; // keep lightweight DSpark layers/tail on a GPU when its backbone is CPU-resident

    common_params_model mparams;

    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    int32_t n_gpu_layers = -1; // number of layers to store in VRAM for the draft model (-1 - use default)

    ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
    ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    // fork: drafter context size override (0 = use model default)
    int32_t n_ctx = 0;

    // fork: string replacements for cross-model compat
    std::vector<std::pair<std::string, std::string>> replacements;
};

struct common_params_speculative_ngram_mod {
    int32_t n_match = 24;

    int32_t n_max = 64;
    int32_t n_min = 48;
};

struct common_params_speculative_ngram_map {
    uint16_t size_n   = 12; // ngram size for lookup
    uint16_t size_m   = 48; // mgram size for speculative tokens
    uint16_t min_hits = 1;  // minimum hits at ngram/mgram lookup for mgram to be proposed
};

struct common_params_speculative_ngram_cache {
    std::string lookup_cache_static;  // path of static ngram cache file for lookup decoding
    std::string lookup_cache_dynamic; // path of dynamic ngram cache file for lookup decoding
};

struct common_params_speculative {
    std::vector<enum common_speculative_type> types = { COMMON_SPECULATIVE_TYPE_NONE };

    double synth_len = -1.0;
    std::vector<double> synth_rates;

    // used by Simple, MTP, Eagle3, etc. - all methods that require some kind of draft model
    common_params_speculative_draft draft;

    common_params_speculative_ngram_mod ngram_mod;
    common_params_speculative_ngram_map ngram_simple;
    common_params_speculative_ngram_map ngram_map_k;
    common_params_speculative_ngram_map ngram_map_k4v;

    common_params_speculative_ngram_cache ngram_cache;

    // DFlash-specific params (top-level for fork compat)
    int32_t n_max        = 16; // maximum number of tokens to draft during speculative decoding
    int32_t n_min        = 0;  // minimum number of draft tokens to use for speculative decoding
    int32_t tree_budget  = 0;  // DDTree node budget (0 = flat DFlash, >0 = tree verification)
    int32_t dflash_max_slots = 1; // max concurrent server slots that keep DFlash state
    float   p_split = 0.1f;   // speculative decoding split probability
    float   p_min   = 0.0f;   // minimum speculative decoding probability (0 = disabled)
    float   sample_temp = 0.0f; // drafter sampling temperature (0 = greedy, >0 = Gumbel sampling)
    bool    sample_temp_set = false; // true when --spec-draft-temp explicitly overrides DFlash2 auto-match
    int32_t draft_topk  = 1;   // top-K candidates per drafter position (1 = argmax only)

    // DFlash draft model (separate from upstream's draft.model)
    struct common_params_model mparams_dft;
    llama_model * model_dft = nullptr;
    llama_context_params cparams_dft;

    // copyspec: copy from context
    int32_t copyspec_gamma      = 6;    // window size for rolling hash match

    // token recycling: adjacency matrix
    int32_t recycle_k            = 8;    // top-k successors per token

    // suffix tree speculative decoding
    int32_t suffix_max_depth    = 64;   // maximum depth of suffix tree
    float   suffix_spec_factor  = 2.0f; // max_spec = factor * match_len + offset
    float   suffix_spec_offset  = 0.0f; // additive offset for max speculative tokens
    float   suffix_min_prob     = 0.1f; // prune branches below this probability

    bool has_dft() const {
        return !draft.mparams.empty();
    }

    bool has_type(common_speculative_type t) const {
        return std::find(types.begin(), types.end(), t) != types.end();
    }

    bool has_synth() const {
        return synth_len != -1.0 || !synth_rates.empty();
    }

    uint32_t need_n_rs_seq() const {
        bool needs_rs_seq = std::any_of(types.begin(), types.end(), [&](auto t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 || t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
        });

        return needs_rs_seq ? draft.n_max : 0u;
    }

    // model-free self-speculation (no -md): drafts come from the target context alone but are
    // still VERIFIED through the target, so hybrid/recurrent targets need the same rollback
    // backup sequence as draft-model speculation (see server load_model n_parallel doubling)
    bool has_model_free_type() const {
        for (const auto t : types) {
            switch (t) {
                case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                case COMMON_SPECULATIVE_TYPE_SUFFIX:
                case COMMON_SPECULATIVE_TYPE_COPYSPEC:
                case COMMON_SPECULATIVE_TYPE_RECYCLE:
                    return true;
                default:
                    break;
            }
        }
        return false;
    }

    // fork: single-type compat helper (most fork code checks one type at a time)
    common_speculative_type type() const {
        if (types.empty() || (types.size() == 1 && types[0] == COMMON_SPECULATIVE_TYPE_NONE)) {
            return COMMON_SPECULATIVE_TYPE_NONE;
        }
        return types.back();
    }

    void set_type(common_speculative_type t) {
        types = { t };
    }
};

struct common_params_diffusion {
    int32_t steps         = 128;
    bool    visual_mode   = false;

    float   eps           = 0;        // epsilon for timesteps
    int32_t block_length  = 0;        // block length for generation

    int32_t algorithm     = 4;        // default algorithm: low-confidence
    float   alg_temp      = 0.0f;     // algorithm temperature

    float   cfg_scale     = 0;        // classifier-free guidance scale
    float   threshold     = 0;        // confidence threshold for token commitment (0 = disabled)
    bool    add_gumbel_noise = false; // add gumbel noise to the logits if temp > 0.0

    bool    self_spec     = false;    // use linear self-speculation (bidirectional draft + causal verify)
    int32_t draft_length  = 16;       // number of tokens to draft per self-spec cycle
};

// reasoning API response format (not to be confused as chat template's reasoning format)
// only used by server
enum common_reasoning_format {
    COMMON_REASONING_FORMAT_NONE,
    COMMON_REASONING_FORMAT_AUTO,            // Same as deepseek, using `message.reasoning_content`
    COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY, // Extract thinking tag contents and return as `message.reasoning_content`, or leave inline in <think> tags in stream mode
    COMMON_REASONING_FORMAT_DEEPSEEK,        // Extract thinking tag contents and return as `message.reasoning_content`, including in streaming deltas.
    // do not extend this enum unless you absolutely have to
    // in most cases, use COMMON_REASONING_FORMAT_AUTO
    // see: https://github.com/ggml-org/llama.cpp/pull/15408
};


struct lr_opt {
    float    lr0          = 1e-5; // learning rate at first epoch
    float    lr_min       = -1;
    float    decay_epochs = -1;   // if >0, the learning rate starts at lr0 and decays to lr_min after this many epochs
    float    scale_epoch  = 0;
    float    wd           = 0;
    unsigned epochs       = 2;

    unsigned epoch; // set by optimizer outer (epochs) loop
    // learning rate decay - constant LR per epoch only for now
    float get_lr(float e) const;
    float get_lr() const { return get_lr(epoch); }
    // must call after arg parse, before get_lr
    void init();
};

struct ggml_opt_optimizer_params common_opt_lr_pars(void * userdata);

enum common_moe_cache_mode {
    COMMON_MOE_CACHE_MODE_OFF,
    COMMON_MOE_CACHE_MODE_AUTO,
    COMMON_MOE_CACHE_MODE_ON,
    COMMON_MOE_CACHE_MODE_SOFT,
};

struct common_moe_cache_params {
    common_moe_cache_mode mode = COMMON_MOE_CACHE_MODE_AUTO;
    size_t budget_mib          = 0;
    int expert_parallel        = 0;
    bool mode_explicit         = false;
    bool fit_selected          = false;
};

struct common_params {
    int32_t n_predict             =    -1; // max. number of new tokens to predict, -1 == no limit
    int32_t n_ctx                 =     0; // context size, 0 == context the model was trained with
    int32_t n_batch               =  2048; // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_ubatch              =   512; // physical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep                =     0; // number of tokens to keep from initial prompt
    int32_t n_chunks              =    -1; // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel            =     1; // number of parallel sequences to decode
    int32_t n_sequences           =     1; // number of sequences to decode
    int32_t n_outputs_max         =     0; // max outputs in a batch (0 = n_batch)
    int32_t n_outputs_max_per_seq =     1; // max outputs per sequence
    int32_t grp_attn_n            =     1; // group-attention factor
    int32_t grp_attn_w            =   512; // group-attention width
    int32_t n_print               =    -1; // print token count every n tokens (-1 = disabled)
    float   rope_freq_base        =  0.0f; // RoPE base frequency
    float   rope_freq_scale       =  0.0f; // RoPE frequency scaling factor
    float   yarn_ext_factor       = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor      = -1.0f; // YaRN magnitude scaling factor
    float   yarn_beta_fast        = -1.0f; // YaRN low correction dim
    float   yarn_beta_slow        = -1.0f; // YaRN high correction dim
    int32_t yarn_orig_ctx         =     0; // YaRN original context length

    // offload params
    std::vector<ggml_backend_dev_t> devices; // devices to use for offloading

    int32_t n_gpu_layers       = -1;    // number of layers to store in VRAM, -1 is auto, <= -2 is all
    int32_t main_gpu           = 0;     // the GPU that is used for scratch and small tensors
    float   tensor_split[128]  = {0};   // how split tensors should be distributed across GPUs
    bool    fit_params         = true;  // whether to fit unset model/context parameters to free device memory
    bool    fit_params_print   = false; // print the estimated required memory to run the model
    int32_t fit_params_min_ctx = 4096;  // minimum context size to set when trying to reduce memory use

    // margin per device in bytes for fitting parameters to free memory:
    std::vector<size_t> fit_params_target = std::vector<size_t>(llama_max_devices(), 1024 * 1024*1024);
    // Server-side auxiliary reservations may raise fit_params_target. Dynamic VBR should retain
    // the caller's ordinary per-device safety headroom and account auxiliaries through live free
    // memory instead of treating their bytes as permanently unavailable a second time.
    uint64_t fit_params_vbr_growth_headroom_bytes = 0; // 0 = derive from fit_params_target

    enum llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER; // how to split the model across GPUs
    enum llama_load_mode  load_mode  = LLAMA_LOAD_MODE_AUTO; // how to load the model

    enum llama_lazy_mode lazy_mode = LLAMA_LAZY_MODE_AUTO; // on-demand reading of tensors marked by the arch

    common_cpu_params cpuparams;
    common_cpu_params cpuparams_batch;

    ggml_backend_sched_eval_callback cb_eval = nullptr;
    void * cb_eval_user_data                 = nullptr;

    ggml_numa_strategy numa = GGML_NUMA_STRATEGY_DISABLED;

    enum llama_rope_scaling_type rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    enum llama_pooling_type      pooling_type      = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling type for embeddings
    enum llama_attention_type    attention_type    = LLAMA_ATTENTION_TYPE_UNSPECIFIED; // attention type for embeddings
    enum llama_flash_attn_type   flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO; // whether to use Flash Attention
    bool no_fused_gdn = false;

    struct common_params_sampling    sampling;
    struct common_params_speculative speculative;
    struct common_params_diffusion   diffusion;

    struct common_params_model model;

    std::set<std::string> model_alias;     // model aliases                                                 // NOLINT
    std::set<std::string> model_tags;      // model tags (informational, not used for routing)              // NOLINT
    std::string hf_token             = ""; // HF token (aka bearer token)                                   // NOLINT
    std::string prompt               = "";                                                                  // NOLINT
    std::string system_prompt        = "";                                                                  // NOLINT
    std::string prompt_file          = ""; // store the external prompt file name                           // NOLINT
    std::string path_prompt_cache    = ""; // path to file for saving/loading prompt eval state             // NOLINT
    std::string input_prefix         = ""; // string to prefix user inputs with                             // NOLINT
    std::string input_suffix         = ""; // string to suffix user inputs with                             // NOLINT
    std::string logits_file          = ""; // file for saving *all* logits                                  // NOLINT
    std::string path_prompts_log_dir = ""; // directory with logged prompts                                 // NOLINT

    // llama-debug specific options
    std::string logits_output_dir = "data"; // directory for saving logits output files                     // NOLINT
    bool        save_logits       = false;  // whether to save logits to files                              // NOLINT
    std::vector<std::string> tensor_filter; // filter tensor names for debug output (regex)                 // NOLINT

    std::vector<std::string> in_files;   // all input files
    std::vector<std::string> antiprompt; // strings upon which more user input is prompted (a.k.a. reverse prompts)
    std::vector<llama_model_kv_override> kv_overrides;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    bool lora_init_without_apply = false; // only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_adapter_lora_apply)
    std::vector<common_adapter_lora_info> lora_adapters; // lora adapter path with user defined scale

    std::vector<common_control_vector_load_info> control_vectors; // control vector with user defined scale

    int32_t verbosity                  = 3;  // LOG_LEVEL_INFO
    int32_t control_vector_layer_start = -1; // layer range for control vector
    int32_t control_vector_layer_end   = -1; // layer range for control vector
    bool    offline                    = false;

    int32_t ppl_stride      = 0;     // stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
    int32_t ppl_output_type = 0;     // = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line
                                     //                                       (which is more convenient to use for plotting)
                                     //
    bool   hellaswag        = false; // compute HellaSwag score over random tasks from datafile supplied in prompt
    size_t hellaswag_tasks  = 400;   // number of tasks to use when computing the HellaSwag score

    bool   winogrande       = false; // compute Winogrande score over random tasks from datafile supplied in prompt
    size_t winogrande_tasks = 0;     // number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

    bool   multiple_choice  = false;  // compute TruthfulQA score over random tasks from datafile supplied in prompt
    size_t multiple_choice_tasks = 0; // number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

    bool   kl_divergence    = false; // compute KL divergence

    bool check             = false; // check rather than generate results for llama-results

    bool usage             = false; // print usage
    bool completion        = false; // print source-able completion script
    bool use_color         = false; // use color to distinguish generations and inputs
    bool special           = false; // enable special token output
    bool interactive       = false; // interactive mode
    bool interactive_first = false; // wait for user input immediately
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool escape            = true;  // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = true;  // insert new sequences for decoding on-the-fly
    bool no_perf           = false; // disable performance metrics
    bool show_timings      = true;  // show timing information on CLI
    bool ctx_shift         = false; // context shift on infinite text generation
    bool swa_full          = false; // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
    bool kv_unified        = false; // enable unified KV cache
    bool logits_all        = true;  // see llama_context_params.logits_all

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool display_prompt    = true;  // print prompt before generation
    bool no_kv_offload     = false; // disable KV offloading
    bool warmup            = true;  // warmup run
    bool check_tensors     = false; // validate tensor data
    bool no_op_offload     = false; // globally disable offload host tensor operations to device
    bool no_extra_bufts    = false; // disable extra buffer types (used for weight repacking)
    common_moe_cache_params moe_cache;
    bool no_host           = false; // bypass host buffer allowing extra buffers to be used

    bool single_turn       = false; // single turn chat conversation

    ggml_type cache_type_k = GGML_TYPE_F16; // KV cache data type for the K
    ggml_type cache_type_v = GGML_TYPE_F16; // KV cache data type for the V
    bool cache_type_k_explicit = false;      // whether -ct/-ctk explicitly selected the K type
    bool cache_type_v_explicit = false;      // whether -ct/-ctv explicitly selected the V type
    std::string vbr_budget = "dynamic"; // VBR target budget: dynamic or a fixed tier/bit width
    std::string vbr_min_bits = "auto";  // VBR aggregate effective bits/value floor for dynamic capacity planning
    std::string vbr_vram_budget = "auto"; // VBR KV VRAM budget: auto or explicit byte/suffixed size
    std::string vbr_policy = "auto";    // VBR policy ladder JSON/path; auto checks VBR_POLICY_LADDER env
    std::string vbr_selected_family;     // selected VBR ladder family, static or dynamic
    std::string vbr_selected_policy;     // selected VBR measured policy/rung name
    std::string vbr_selected_schedule;   // selected VBR schedule path
    double vbr_min_bits_value = 0.0;     // requested aggregate floor in effective bits/value, 0 == auto/none; not a per-codec ban
    double vbr_capacity_bits = 0.0;      // selected supported capacity floor in effective bits/value, 0 == auto/none
    double vbr_selected_bpv = 0.0;       // measured BPV of the selected policy/rung
    double vbr_selected_kld = 0.0;       // measured KLD of the selected policy/rung
    uint64_t vbr_vram_budget_bytes = 0;  // explicit VBR KV VRAM budget in bytes, 0 == auto
    bool vbr_budget_explicit = false;   // whether --vbr-budget/--vbr-bits was provided
    bool vbr_min_bits_explicit = false; // whether --vbr-min-bits/--vbr-floor was provided
    bool vbr_vram_budget_explicit = false; // whether --vbr-vram/--vbr-vram-budget was provided
    bool vbr_policy_explicit = false;   // whether --vbr-policy was provided
    // Common CLI default: dynamic VBR on both sides. The underlying entry tensors remain F16;
    // postprocessing supplies the friendly implicit t4 floor. Explicit `-ct vbr` is tracked
    // separately and deliberately retains the full t1 ladder when no floor was typed.
    bool vbr_cache_type_k = true;
    bool vbr_cache_type_v = true;
    bool vbr_cache_type_k_explicit = false;
    bool vbr_cache_type_v_explicit = false;
    // dynamic VBR server policy: clear idle slots' KV before a degrade wave would land the
    // aggregate BELOW this bits/value. 8.125 = protect the f16/t8 near-lossless band (above it,
    // degrading beats destroying another client's re-prefillable cache); 0 = never reclaim;
    // >= 16 = reclaim before any degrade (single-user maximal-quality profile)
    float vbr_reclaim_floor_bpv = 8.125f;
    // dynamic VBR server policy: when a DEGRADED conversation would keep less than this
    // fraction of its prompt as reusable prefix anyway, drop the prefix entirely so the
    // empty-cache lossless reset restores the entry tier (turn-N cache quality = turn-1).
    // 0 disables the trade.
    float vbr_reset_keep_frac = 0.25f;
    // canonical predicates — use these instead of re-deriving the flag combinations
    bool vbr_enabled() const {
        return vbr_cache_type_k || vbr_cache_type_v || vbr_budget_explicit ||
               vbr_min_bits_explicit || vbr_vram_budget_explicit || vbr_policy_explicit;
    }
    bool vbr_dynamic() const {
        return vbr_enabled() && (vbr_budget == "dynamic" || vbr_budget == "auto" || vbr_budget.empty());
    }
    // mixed config: a side that did NOT select the vbr alias while the other did is PINNED at
    // its explicit type (arg.cpp warns at parse time; the runtime ladder skips it). Whole-cache
    // turbo configs driven by --vbr-* knobs alone pin nothing — both sides degrade.
    bool vbr_pin_k() const { return vbr_dynamic() && !vbr_cache_type_k && vbr_cache_type_v; }
    bool vbr_pin_v() const { return vbr_dynamic() && !vbr_cache_type_v && vbr_cache_type_k; }

    common_conversation_mode conversation_mode = COMMON_CONVERSATION_MODE_AUTO;

    // multimodal models (see tools/mtmd)
    struct common_params_model mmproj;
    bool mmproj_use_gpu = true;                 // use GPU for multimodal model
    ggml_backend_dev_t mmproj_device = nullptr; // GPU device to use for multimodal model
    bool mmproj_gpu_swap = false;               // swap MTP↔mmproj VRAM on vision requests
    bool no_mmproj = false;                     // explicitly disable multimodal model
    std::vector<std::string> image;             // path to image file(s) ; TODO: change the name to "media"
    int image_min_tokens = -1;
    int image_max_tokens = -1;
    int mtmd_batch_max_tokens = 1024;

    // for video input
    float       video_fps                   = 4.0f;
    int64_t     video_timestamp_interval_ms = 5000;
    std::string video_ffmpeg_bin_dir        = "";

    // finetune
    struct lr_opt lr;
    enum ggml_opt_optimizer_type optimizer = GGML_OPT_OPTIMIZER_TYPE_ADAMW;
    float val_split = 0.05f; // fraction of the data used for the validation set

    // embedding
    bool embedding         = false; // get only sentence embedding
    int32_t embd_normalize = 2;     // normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
    std::string embd_out   = "";    // empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
    std::string embd_sep   = "\n";  // separator of embeddings
    std::string cls_sep    = "\t";  // separator of classification sequences

    // server params
    int32_t port                = 8080;          // server listens on this network port
    bool    reuse_port          = false;         // allow multiple sockets to bind to the same port
    int32_t timeout_read        = 3600;          // http read timeout in seconds
    int32_t timeout_write       = timeout_read;  // http write timeout in seconds
    int32_t sse_ping_interval   = 30;            // SSE ping interval in seconds
    int32_t n_threads_http      = -1;    // number of threads to process HTTP requests (TODO: support threadpool)
    int32_t n_cache_reuse       = 0;     // min chunk size to reuse from the cache via KV shifting
    bool    cache_prompt        = true;  // whether to enable prompt caching
    bool    cache_idle_slots    = true;  // save and clear idle slots upon starting a new task
    bool    vbr_prompt_cache    = false; // resolved projected VBR host-cache publication request
    bool    vbr_prompt_cache_explicit = false; // whether --[no-]vbr-prompt-cache was provided
    int32_t vbr_anchor_cache_mib = 0;    // optional extra quality-anchor pool for VBR artifacts
    int32_t n_ctx_checkpoints   = 32;    // max number of context checkpoints per slot
    int32_t kv_unified_per_slot = 0;     // max context per parallel slot; 0 = unset
    int32_t checkpoint_min_step = 8192;  // minimum spacing between context checkpoints
    int32_t cache_ram_mib       = 8192;  // -1 = no limit, 0 - disable, 1 = 1 MiB, etc.

    std::string hostname      = "127.0.0.1";
    std::string public_path   = "";                                                                         // NOLINT
    std::string api_prefix    = "";                                                                         // NOLINT
    std::string chat_template = "";                                                                         // NOLINT
    bool use_jinja = true;                                                                                  // NOLINT

    // server CORS params
    std::string cors_origins = "*";
    std::string cors_methods = "GET, POST, DELETE, OPTIONS";
    std::string cors_headers = "*";
    bool cors_credentials = true;
    bool cors_origins_explicit = false; // for --agent option

    bool enable_chat_template = true;
    bool force_pure_content_parser = false;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    int enable_reasoning = -1; // -1 = auto, 0 = disable, 1 = enable
    bool prefill_assistant = true; // if true, any trailing assistant message will be prefilled into the response
    int sleep_idle_seconds = -1;   // if >0, server will sleep after this many seconds of idle time

    std::vector<std::string> api_keys;

    std::string ssl_file_key  = "";                                                                         // NOLINT
    std::string ssl_file_cert = "";                                                                         // NOLINT

    std::map<std::string, std::string> default_template_kwargs;

    // CLI params
    std::string server_base; // if set, connect to this server instead of starting a new one

    // UI configs
    bool ui = true;
    bool ui_mcp_proxy = false;
    std::string ui_config_json;

    // "advanced" endpoints are disabled by default for better security
    bool endpoint_slots   = true;
    bool endpoint_props   = false; // only control POST requests, not GET
    bool endpoint_metrics = false;

    // Cache-plan observer: strictly zero observer work when disabled.
    bool cache_debug = false;

    // Trusted-local, single-principal cache-plan preview surface. This flag
    // only exposes the route; ordinary requests allocate no observer/planner
    // state merely because it is enabled.
    bool cache_plan_preflight = false;

    // Trusted-local, single-principal cache-control HTTP surface. Server
    // startup enables its required cache-lifecycle authority; this flag also
    // registers the reviewed routes.
    bool cache_control_api = false;

    // Graduated cache-plan authority request. Non-off levels remain
    // observation-only until the corresponding authority ratchet closes.
    common_cache_plan_authority_level cache_plan_authority{}; // zero = off

    // Cache-lifecycle authority substrate (accounting-gated admission). The
    // parser value records an explicit request; server initialization also
    // enables it automatically when the prompt cache is present. It remains
    // independent of --cache-debug.
    bool cache_lifecycle = false;

    // enable built-in tools
    std::vector<std::string> server_tools;
    std::string server_tools_runtime;

    // MCP server configs (Cursor-compatible JSON)
    std::string mcp_servers_config;   // path to JSON file with MCP server definitions
    std::string mcp_servers_json;     // inline JSON with MCP server definitions

    // router server configs
    std::string models_dir    = "";     // directory containing models for the router server
    std::string models_preset = "";     // directory containing model presets for the router server
    int models_max = 4;                 // maximum number of models to load simultaneously
    bool models_autoload = true;        // automatically load models when requested via the router server
    std::string models_preset_hf = "";  // show a warning about remote presets on router loaded (if not empty)

    bool log_json = false;

    std::string slot_save_path;
    std::string media_path; // path to directory for loading media files

    // Cache receipt: untrusted divergence-location hint
    // on responses. Keyed chain by default; unkeyed only behind the debug flag.
    bool        cache_receipt = false;
    std::string cache_receipt_key;          // per-session/tenant comparison key
    bool        cache_receipt_unkeyed_debug = false;

    float slot_prompt_similarity = 0.1f;

    // batched-bench params
    bool is_pp_shared   = false;
    bool is_tg_separate = false;

    std::vector<int32_t> n_pp;
    std::vector<int32_t> n_tg;
    std::vector<int32_t> n_pl;

    // retrieval params
    std::vector<std::string> context_files; // context files to embed

    int32_t chunk_size = 64; // chunk size for context embedding

    std::string chunk_separator = "\n"; // chunk separator for context embedding

    // passkey params
    int32_t n_junk = 250; // number of times to repeat the junk text
    int32_t i_pos  = -1;  // position of the passkey in the junk text

    // imatrix params
    int32_t n_out_freq  = 10; // output the imatrix every n_out_freq iterations
    int32_t n_save_freq =  0; // save the imatrix every n_save_freq iterations
    int32_t i_chunk     =  0; // start processing from this chunk
    int8_t  imat_dat    =  0; // whether the legacy imatrix.dat format should be output (gguf <= 0 < dat)

    bool process_output  = false; // collect data for the output tensor
    bool compute_ppl     = true;  // whether to compute perplexity
    bool show_statistics = false; // show imatrix statistics per tensor
    bool parse_special   = false; // whether to parse special tokens during imatrix tokenization

    // cvector-generator params
    int n_pca_batch = 100;
    int n_pca_iterations = 1000;
    dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    std::string cvector_positive_file = "tools/cvector-generator/positive.txt";
    std::string cvector_negative_file = "tools/cvector-generator/negative.txt";

    bool spm_infill = false; // suffix/prefix/middle pattern for infill

    // batched-bench params
    bool batched_bench_output_jsonl = false;

    // tokenize params
    bool tokenize_ids        = false; // if true, only print the token IDs
    bool tokenize_stdin      = false; // if true, read the prompt from stdin
    bool tokenize_no_bos     = false; // if true, do not add the BOS token
    bool tokenize_show_count = false; // if true, print the total token count

    // common params
    std::string out_file; // output filename for all example programs
    // optional callback for model loading progress and cancellation:
    // called with a progress value between 0.0 and 1.0.
    // return false from callback to abort model loading or true to continue
    llama_progress_callback load_progress_callback = NULL;
    void *                  load_progress_callback_user_data = NULL;
    bool no_alloc = false; // Don't allocate model buffers

    // TTS params
    std::string tts_lang = "";
    std::string tts_speaker_file = "";

    bool is_gen_docs = false; // whether we are running inside llama-gen-docs
};

enum class common_vbr_cpu_fallback_result {
    not_needed,
    applied,
    explicit_vbr,
};

enum class common_vbr_prompt_cache_mode {
    disabled_cache_ram,
    disabled_static,
    disabled_explicit,
    enabled_explicit,
    enabled_automatic,
};

// Default prompt-cache policy. Dynamic VBR follows the ordinary nonzero --cache-ram
// default unless the legacy VBR-specific switch was explicitly set. A zero
// cache budget is authoritative and prevents all host-cache activation.
common_vbr_prompt_cache_mode common_vbr_prompt_cache_mode_for(
    const common_params & params);

// Common-layer policy seam for the implicit dynamic-VBR default. `has_gpu`
// describes the resolved placement inventory; explicit VBR is never rewritten.
common_vbr_cpu_fallback_result common_params_apply_vbr_cpu_fallback(
    common_params & params, bool has_gpu);

// call once at the start of a program if it uses libcommon
// initializes the logging system and prints info about the build
void common_init();

void common_params_print_info(const common_params & params, bool print_devices = true);
std::string common_params_get_system_info(const common_params & params);

// Resolve a VBR floor spec ("t8"/"t4"/"t3tcq"/"t2tcq"/"t1tcq", "auto"/"none", or a bits value) to an
// aggregate floor in effective bits/value (0 == bottom-tier floor). Throws std::invalid_argument on
// bad input. Single source of truth for the floor→bits mapping, shared by the main CLI and llama-bench.
double common_vbr_floor_bits(const std::string & floor);

// Resolve a VBR VRAM budget spec ("auto"/"none" or a size with optional K/M/G[i]B suffix) to bytes
// (0 == auto). Throws std::invalid_argument on bad input. Shared with the main CLI's parser.
uint64_t common_vbr_vram_bytes(const std::string & vram);

bool parse_cpu_range(const std::string & range, bool(&boolmask)[GGML_MAX_N_THREADS]);
bool parse_cpu_mask(const std::string & mask, bool(&boolmask)[GGML_MAX_N_THREADS]);
void postprocess_cpu_params(common_cpu_params & cpuparams, const common_cpu_params * role_model = nullptr);
bool set_process_priority(enum ggml_sched_priority prio);

//
// String utils
//

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define LLAMA_COMMON_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define LLAMA_COMMON_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_COMMON_ATTRIBUTE_FORMAT(1, 2)
std::string string_format(const char * fmt, ...);

std::string string_strip(const std::string & str);
std::string string_get_sortable_timestamp();
std::string string_lcs(std::string_view a, std::string_view b);

std::string string_join(const std::vector<std::string> & values, const std::string & separator);
std::vector<std::string> string_split(const std::string & str, const std::string & delimiter);
std::string string_repeat(const std::string & str, size_t n);

void string_replace_all(std::string & s, const std::string & search, const std::string & replace);

std::string regex_escape(const std::string & s);

template<class T>
static std::vector<T> string_split(const std::string & str, char delim) {
    static_assert(!std::is_same<T, std::string>::value, "Please use the specialized version for std::string");
    std::vector<T> values;
    std::istringstream str_stream(str);
    std::string token;
    while (std::getline(str_stream, token, delim)) {
        T value;
        std::istringstream token_stream(token);
        token_stream >> value;
        values.push_back(value);
    }
    return values;
}

template<>
inline std::vector<std::string> string_split<std::string>(const std::string & str, char delim)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t delim_pos = str.find(delim);
    while (delim_pos != std::string::npos) {
        std::string part = str.substr(begin_pos, delim_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = delim_pos + 1;
        delim_pos = str.find(delim, begin_pos);
    }
    parts.emplace_back(str.substr(begin_pos));
    return parts;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, std::string_view prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

// remove when moving to c++20
inline bool string_starts_with(std::string_view str, char prefix) {
    return !str.empty() && str.front() == prefix;
}

// remove when moving to c++20
inline bool string_ends_with(std::string_view str, std::string_view suffix) {
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline bool string_remove_suffix(std::string & str, std::string_view suffix) {
    if (string_ends_with(str, suffix)) {
        str.resize(str.size() - suffix.size());
        return true;
    }
    return false;
}

inline size_t string_find_partial_stop(std::string_view str, std::string_view stop) {
    if (!str.empty() && !stop.empty()) {
        const size_t max_len = std::min(str.size(), stop.size());
        const char last_char = str.back();
        for (size_t len = max_len; len > 0; --len) {
            if (stop[len - 1] == last_char) {
                if (string_ends_with(str, stop.substr(0, len))) {
                    return str.size() - len;
                }
            }
        }
    }
    return std::string::npos;
}

bool string_parse_kv_override(const char * data, std::vector<llama_model_kv_override> & overrides);
void string_process_escapes(std::string & input);

std::string string_from(bool value);
std::string string_from(const std::vector<int> & values);
std::string string_from(const struct llama_context * ctx, const std::vector<llama_token> & tokens);
std::string string_from(const struct llama_context * ctx, const struct llama_batch & batch);

bool glob_match(const std::string & pattern, const std::string & str);

//
// Environment utils
//

// portable environment access, an unset variable reads as an empty string
// and setting an empty value unsets the variable
std::string common_get_env(const std::string & name);
void        common_set_env(const std::string & name, const std::string & value);

//
// Filesystem utils
//

bool fs_validate_filename(const std::string & filename, bool allow_subdirs = false);
bool fs_create_directory_with_parents(const std::string & path);
bool fs_is_directory(const std::string & path);

std::string fs_get_cache_directory();
std::string fs_get_cache_file(const std::string & filename);
std::string fs_get_config_directory();

struct common_file_info {
    std::string path;
    std::string name;
    size_t      size = 0; // in bytes
    bool        is_dir = false;
};
std::vector<common_file_info> fs_list(const std::string & path, bool include_directories);

// fs open, also handle UTF8 on Windows
std::ifstream fs_open_ifstream(const std::string & fname, std::ios_base::openmode mode);

//
// TTY utils
//

// Auto-detect if colors can be enabled based on terminal and environment
bool tty_can_use_colors();

//
// Model utils
//

struct common_sampler;

// note: defines the model, context, samplers, ets. lifetimes
struct common_init_result {
    common_init_result(common_params & params, bool model_only = false);
    ~common_init_result();

    llama_model * model();
    llama_context * context();

    common_sampler * sampler(llama_seq_id seq_id);
    void reset_samplers();

    std::vector<llama_adapter_lora_ptr> & lora();

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using common_init_result_ptr = std::unique_ptr<common_init_result>;

common_init_result_ptr common_init_from_params(common_params & params, bool model_only = false);

struct llama_model_params   common_model_params_to_llama  (      common_params & params);
struct llama_context_params common_context_params_to_llama(const common_params & params);

// clear LoRA adapters from context, then apply new list of adapters
void common_set_adapter_lora(struct llama_context * ctx, std::vector<common_adapter_lora_info> & lora);

// model endpoint from env
std::string common_get_model_endpoint();

// for testing purposes
char * common_get_model_or_exit(int, char*[]);

//
// Threadpool utils
//

struct ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const common_cpu_params & params);

struct common_threadpools {
    common_threadpools() = default;
    ~common_threadpools();

    common_threadpools(const common_threadpools &) = delete;
    common_threadpools & operator=(const common_threadpools &) = delete;

    void init(llama_context * ctx, const common_params & params);

private:
    ggml_threadpool * threadpool       = nullptr;
    ggml_threadpool * threadpool_batch = nullptr;

    decltype(ggml_threadpool_free) * free_fn = nullptr;
};

//
// Context utils
//

enum common_context_seq_rm_type {
    COMMON_CONTEXT_SEQ_RM_TYPE_NO           = 0, // seq_rm not supported (e.g. no memory module)
    COMMON_CONTEXT_SEQ_RM_TYPE_PART         = 1, // can seq_rm partial sequences
    COMMON_CONTEXT_SEQ_RM_TYPE_FULL         = 2, // can seq_rm full sequences only
    COMMON_CONTEXT_SEQ_RM_TYPE_RS = 3, // can seq_rm partial sequences, bounded by n_rs_seq
};

// check if the llama_context can remove sequences
// note: clears the memory of the context
common_context_seq_rm_type common_context_can_seq_rm(llama_context * ctx);

// fork: kept external — the server's censused destruction doors call these directly
void common_context_seq_rm (llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1);
void common_context_seq_cp (llama_context * ctx, llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
void common_context_seq_add(llama_context * ctx, llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta);

struct common_memory {
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    void init(llama_context * ctx_tgt, llama_context * ctx_dft = nullptr);

    // aborts execution on failure
    void seq_rm (llama_seq_id seq_id, llama_pos p0, llama_pos p1) const;
    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos delta) const;
    void seq_cp (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) const;
};

//
// Batch utils
//

void common_batch_clear(struct llama_batch & batch);

void common_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits);

// decodes a single batch of tokens for a prompt and manages session tokens
//
// Note: We save state before the last token so that we can replay it to ensure
// compatibility with all memory types. Recurrent/hybrid models cannot remove
// tokens from memory, so this approach works across all model architectures.
bool common_prompt_batch_decode(
              struct llama_context * ctx,
    const std::vector<llama_token> & all_tokens,
                               int   n_new,
                               int & n_past,
                               int   n_batch,
                  std::string_view   state_path,
                              bool   save_state);

// replays the last token after loading state to regenerate logits
// used after loading session state to ensure the sampling context has valid logits
bool common_replay_last_token(struct llama_context * ctx, llama_token last_token, int32_t pos);

//
// Vocab utils
//

// tokenizes a string into a vector of tokens
// should work similar to Python's `tokenizer.encode`
std::vector<llama_token> common_tokenize(
  const struct llama_context * ctx,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

std::vector<llama_token> common_tokenize(
    const struct llama_vocab * vocab,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false);

// tokenizes a token into a piece, optionally renders special/control tokens
// should work similar to Python's `tokenizer.id_to_piece`
std::string common_token_to_piece(
        const struct llama_context * ctx,
                       llama_token   token,
                       bool          special = true);

std::string common_token_to_piece(
          const struct llama_vocab * vocab,
                       llama_token   token,
                       bool          special = true);

// detokenizes a vector of tokens into a string
// should work similar to Python's `tokenizer.decode`
// optionally renders special/control tokens
std::string common_detokenize(
            const struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

std::string common_detokenize(
              const struct llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
                                  bool   special = true);

//
// Embedding utils
//

// TODO: replace embd_norm with an enum
void common_embd_normalize(const float * inp, float * out, int n, int embd_norm);

float common_embd_similarity_cos(const float * embd1, const float * embd2, int n);

//
// Control vector utils
//

struct common_control_vector_data {
    int n_embd;

    // stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
    std::vector<float> data;
};

struct common_control_vector_load_info {
    float strength;

    std::string fname;
};

// Load control vectors, scale each by strength, and add them together.
// On error, returns {-1, empty}
common_control_vector_data common_control_vector_load(const std::vector<common_control_vector_load_info> & load_infos);

//
// Split utils
//

namespace {

const char * const LLM_KV_SPLIT_NO            = "split.no";
const char * const LLM_KV_SPLIT_COUNT         = "split.count";
const char * const LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count";

}

//
// FFN offload utils
//

const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

const char * const LLM_FFN_DENSE_REGEX = "\\.ffn_(up|down|gate)\\.";

inline std::string llm_ffn_block_regex(int idx, const char * ffn_regex) {
    return string_format("blk\\.%d%s", idx, ffn_regex);
}

inline llama_model_tensor_buft_override llm_ffn_exps_cpu_override() {
    return { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() };
}

inline void llm_add_n_cpu_ffn_overrides(int n, const char * ffn_regex, std::vector<llama_model_tensor_buft_override> & overrides) {
    // keep strings alive and avoid leaking memory by storing them in a static list
    static std::list<std::string> buft_override_strings;
    for (int i = 0; i < n; ++i) {
        buft_override_strings.push_back(llm_ffn_block_regex(i, ffn_regex));
        overrides.push_back({buft_override_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
}

//
// training utils
//

ggml_opt_dataset_t common_opt_dataset_init(struct llama_context * ctx, const std::vector<llama_token> & tokens, int64_t stride);

// "adamw" or "sgd" (case insensitive)
enum ggml_opt_optimizer_type common_opt_get_optimizer(const char *);

//
// prompt utils
//

// Server/common-layer logical computation frontier. This is the state after
// processing the half-open logical prefix [0, token_count); next_position is the next
// effective model position for that prefix. The three identity keys are opaque,
// comparison-only server keys:
//   - execution_identity: this loaded model/runtime instance
//   - adapter_config_identity: active adapter weights + exact scales
//   - media_content_identity: media content/shape in the logical prefix
//
// This deliberately lives beside common_prompt_checkpoint rather than in libllama:
// sequence lineage, per-request adapters and mtmd media are server-layer concepts.
// version == 0 means a legacy checkpoint with no dual-written frontier.
struct common_computation_frontier {
    static constexpr uint32_t VERSION = 1;

    uint32_t version = 0;

    uint64_t sequence_epoch = 0;
    int64_t  token_count    = 0;
    llama_pos next_position = 0;

    std::string execution_identity;
    std::string adapter_config_identity;
    std::string media_content_identity;

    bool valid() const {
        return version == VERSION &&
               sequence_epoch != 0 &&
               token_count >= 0 &&
               next_position >= 0 &&
               !execution_identity.empty() &&
               !adapter_config_identity.empty() &&
               !media_content_identity.empty();
    }

    void clear() {
        version = 0;
        sequence_epoch = 0;
        token_count = 0;
        next_position = 0;
        execution_identity.clear();
        adapter_config_identity.clear();
        media_content_identity.clear();
    }
};

// Copy-on-write byte owner for immutable checkpoint planes. Copying a
// checkpoint into the host cache or a non-consuming restore delivery shares
// the exact allocation; the first live mutation detaches only that plane.
// Empty buffers allocate nothing. Read access is immutable; replacement uses
// one scoped writer so fixed-cache fan-out stays zero-copy without letting a
// pointer or reference escape the copy-on-write boundary.
class common_shared_byte_buffer {
    struct storage {
        storage() = default;
        explicit storage(size_t size) : bytes(size) {}
        explicit storage(const std::vector<uint8_t> & source)
            : bytes(source) {}

        std::vector<uint8_t> bytes;
        const void * accounting_owner = nullptr;
        uint64_t accounting_allocation = 0;
    };

public:
    common_shared_byte_buffer() = default;
    common_shared_byte_buffer(const common_shared_byte_buffer & other);
    common_shared_byte_buffer & operator=(
        const common_shared_byte_buffer & other);
    common_shared_byte_buffer(common_shared_byte_buffer && other) noexcept;
    common_shared_byte_buffer & operator=(
        common_shared_byte_buffer && other) noexcept;

    size_t size() const noexcept;
    bool empty() const noexcept;
    const uint8_t * data() const noexcept;
    void clear() noexcept;

    // Publishes newly filled storage only after the synchronous writer
    // returns. The writer must not retain the supplied pointer.
    template<class Writer>
    void overwrite(size_t size, Writer && writer) {
        if (accounting_owned_) {
            throw std::logic_error(
                "cannot overwrite an accounted checkpoint allocation");
        }
        if (size == 0) {
            std::forward<Writer>(writer)(nullptr, 0);
            clear();
            return;
        }
        auto replacement =
            std::make_shared<storage>(size);
        std::forward<Writer>(writer)(replacement->bytes.data(), size);
        bytes_ = std::move(replacement);
    }

    const uint8_t & operator[](size_t index) const noexcept;

    const std::vector<uint8_t> & view() const noexcept;
    bool shares_storage_with(
        const common_shared_byte_buffer & other) const noexcept;
    const void * storage_identity() const noexcept;
    long storage_use_count() const noexcept;
    bool accounting_binding(
        const void *& owner, uint64_t & allocation) const noexcept;
    bool owns_accounting_binding(
        const void * owner, uint64_t allocation) const noexcept;
    bool bind_accounting(
        const void * owner, uint64_t allocation) const noexcept;
    bool unbind_accounting(
        const void * owner, uint64_t allocation,
        bool clear_storage_binding = true) const noexcept;

    friend bool operator==(
        const common_shared_byte_buffer & a,
        const common_shared_byte_buffer & b) noexcept;
    friend bool operator!=(
        const common_shared_byte_buffer & a,
        const common_shared_byte_buffer & b) noexcept {
        return !(a == b);
    }

private:
    std::shared_ptr<storage> bytes_;
    mutable bool accounting_owned_ = false;
};

inline bool operator==(
        const common_computation_frontier & a,
        const common_computation_frontier & b) noexcept {
    return a.version == b.version &&
           a.sequence_epoch == b.sequence_epoch &&
           a.token_count == b.token_count &&
           a.next_position == b.next_position &&
           a.execution_identity == b.execution_identity &&
           a.adapter_config_identity == b.adapter_config_identity &&
           a.media_content_identity == b.media_content_identity;
}

struct common_prompt_checkpoint {
    int64_t n_tokens;

    // (optional) id of the task that created the checkpoint
    int id_task = -1;

    llama_pos pos_min;
    llama_pos pos_max;

    // Attention-content lineage epochs at capture time. A recurrent-only checkpoint restores
    // exact recurrent state while retaining the live attention KV. Lossless in-place retiering
    // preserves that lineage; occupied-cell reuse, clear/reset and import do not. Both are 0 when
    // VBR is inactive, making the restore-time check a no-op.
    uint64_t checkpoint_epoch     = 0;
    uint64_t checkpoint_epoch_swa = 0;

    // Logical validity record, dual-written beside the legacy physical fields
    // during the typed-companion migration. Legacy checkpoints have version == 0.
    common_computation_frontier computation_frontier;

    // Declared-family provenance. This is policy metadata only: it follows
    // checkpoint copies/restores but never enters checkpoint payload bytes.
    common_cache_family_binding cache_family;

    common_shared_byte_buffer data_tgt;
    common_shared_byte_buffer data_dft;
    // Draft checkpoints used as VBR companions contain the complete sequence
    // image so they can populate an empty draft context. Legacy speculative
    // checkpoints may contain only PARTIAL_ONLY state. Retain the wire mode
    // with the bytes rather than making restore infer it from current flags.
    bool data_dft_full_sequence = false;

    // Typed accelerator state stashed with the checkpoint (typed
    // accelerators). Exact restore readiness is conjunctive (PROPOSAL §6
    // invariant 3): a component that is mandatory-on-presence and fails to
    // apply fails the WHOLE checkpoint restore fail-closed; purely optional
    // components may degrade drafting quality only, never correctness.
    struct accel_state {
        // DFlash drafter ring buffer bytes. Mandatory-on-presence: a non-empty
        // ring that fails to load fails the restore (shipped behavior at the
        // ring-restore site).
        common_shared_byte_buffer ring;

        // Speculative-impl state (e.g. eagle3's deferred-boundary g_embd row).
        // Applied unconditionally; absence resets the impl state. Optional.
        common_shared_byte_buffer spec;

        size_t size()  const { return ring.size() + spec.size(); }
        bool   empty() const { return ring.empty() && spec.empty(); }
        void   clear()       { ring.clear(); spec.clear(); }
    };
    accel_state accel;

    size_t size() const;

    bool empty() const;
    void clear();

    void update_pos(
            int64_t n_tokens,
            llama_pos pos_min,
            llama_pos pos_max);

    void update_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    void update_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags);

    void load_tgt(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    void load_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    // Server restore paths must be able to reject an incompatible draft image
    // without terminating the process. The aborting load_dft() wrapper remains
    // for callers whose checkpoint mismatch is an invariant violation.
    bool try_load_dft(
            llama_context * ctx,
            llama_seq_id seq_id,
            llama_state_seq_flags flags) const;

    void clear_tgt();
    void clear_dft();
};

inline bool common_prompt_checkpoint_lineage_matches(
        const common_prompt_checkpoint & checkpoint,
        const llama_memory_vbr_state_data & state) noexcept {
    return checkpoint.checkpoint_epoch == state.checkpoint_epoch &&
           checkpoint.checkpoint_epoch_swa == state.checkpoint_epoch_swa;
}
