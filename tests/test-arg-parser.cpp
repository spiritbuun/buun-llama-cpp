#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"
#include "speculative.h"

#include <cstdlib>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_set>

#undef NDEBUG
#include <cassert>

#ifndef _WIN32
#include <unistd.h>
#endif

static void test(void) {
    common_params params;

    auto assert_output_limits = [](int32_t n_batch, int32_t n_parallel, int32_t n_draft,
                                   int32_t total, int32_t per_seq) {
        const auto limits = common_speculative_get_output_limits(n_batch, n_parallel, n_draft);
        assert(limits.total == total);
        assert(limits.per_seq == per_seq);
    };

    assert_output_limits(16, 2,  3, 8, 4);
    assert_output_limits(16, 2, -1, 2, 1);
    assert_output_limits( 6, 2,  3, 6, 4);
    assert_output_limits( 2, 1,  3, 2, 2);
    assert_output_limits(
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max());

    {
        common_params_speculative spec;
        spec.synth_len = 3.4;

        auto assert_invalid = [](const common_params_speculative & value, int32_t n_max) {
            try {
                common_speculative_synth_rates_resolve(&value, n_max);
                assert(false);
            } catch (const std::invalid_argument &) {
            }
        };

        const auto rates = common_speculative_synth_rates_resolve(&spec, 4);
        assert(rates.size() == 4);
        assert(std::abs(rates[0] - 0.80581) < 1e-5);
        assert(std::abs(rates[1] - 0.64933) < 1e-5);
        assert(std::abs(rates[2] - 0.52323) < 1e-5);
        assert(std::abs(rates[3] - 0.42163) < 1e-5);
        assert(std::abs(1.0 + rates[0] + rates[1] + rates[2] + rates[3] - 3.4) < 1e-8);

        spec.synth_len = 1.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({0.0, 0.0, 0.0, 0.0}));

        spec.synth_len = 5.0;
        assert(common_speculative_synth_rates_resolve(&spec, 4) == std::vector<double>({1.0, 1.0, 1.0, 1.0}));

        spec.synth_len = 5.1;
        assert_invalid(spec, 4);

        spec.synth_len = std::numeric_limits<double>::quiet_NaN();
        assert_invalid(spec, 4);

        spec.synth_len = 0.0;
        assert_invalid(spec, 4);

        spec.synth_len = -1.0;
        spec.synth_rates = {0.8, 0.6, 0.4};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        assert(common_speculative_synth_rates_resolve(&spec, 4) == spec.synth_rates);

        spec.synth_rates = {0.8, 0.9, 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, std::numeric_limits<double>::quiet_NaN(), 0.4, 0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, -0.2};
        assert_invalid(spec, 4);

        spec.synth_rates = {0.8, 0.6, 0.4, 0.2};
        spec.synth_len = 3.0;
        assert_invalid(spec, 4);
    }

    {
        common_params base;
        base.n_parallel = 4;
        base.n_outputs_max_per_seq = 8;

        const auto draft = common_base_params_to_speculative(base);
        assert(draft.n_outputs_max == 4);
        assert(draft.n_outputs_max_per_seq == 1);
    }

    printf("test-arg-parser: make sure there is no duplicated arguments in any examples\n\n");
    for (int ex = 0; ex < LLAMA_EXAMPLE_COUNT; ex++) {
        try {
            auto ctx_arg = common_params_parser_init(params, (enum llama_example)ex);
            common_params_add_preset_options(ctx_arg.options);
            std::unordered_set<std::string> seen_args;
            std::unordered_set<std::string> seen_env_vars;
            for (const auto & opt : ctx_arg.options) {
                // check for args duplications
                for (const auto & arg : opt.get_args()) {
                    if (seen_args.find(arg) == seen_args.end()) {
                        seen_args.insert(arg);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same argument: %s", arg.c_str());
                        exit(1);
                    }
                }
                // check for env var duplications
                for (const auto & env : opt.get_env()) {
                    if (seen_env_vars.find(env) == seen_env_vars.end()) {
                        seen_env_vars.insert(env);
                    } else {
                        fprintf(stderr, "test-arg-parser: found different handlers for the same env var: %s", env.c_str());
                        exit(1);
                    }
                }

                // exclude spec args from this check
                // ref: https://github.com/ggml-org/llama.cpp/pull/22397
                const bool skip = opt.is_spec;

                // ensure shorter argument precedes longer argument
                if (!skip && opt.args.size() > 1) {
                    const std::string first(opt.args.front());
                    const std::string last(opt.args.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }

                // same check for negated arguments
                if (opt.args_neg.size() > 1) {
                    const std::string first(opt.args_neg.front());
                    const std::string last(opt.args_neg.back());

                    if (first.length() > last.length()) {
                        fprintf(stderr, "test-arg-parser: shorter negated argument should come before longer one: %s, %s\n",
                                first.c_str(), last.c_str());
                        assert(false);
                    }
                }
            }
        } catch (std::exception & e) {
            printf("%s\n", e.what());
            assert(false);
        }
    }

    auto list_str_to_char = [](std::vector<std::string> & argv) -> std::vector<char *> {
        std::vector<char *> res;
        for (auto & arg : argv) {
            res.push_back(const_cast<char *>(arg.data()));
        }
        return res;
    };

#ifndef _WIN32
    auto clear_vbr_runtime_env = []() {
        // the arg layer's own exports + the developer-override envs llama-kv-cache honors —
        // scrub between blocks so one test's env cannot leak into the next
        for (const char * name : {
                "VBR_VMM",
                "VBR_MODE",
                "VBR_BUDGET_MIB",
                "VBR_MIN_BITS",
                "VBR_LAYER_SCHEDULE",
                "VBR_LAYER_SCHEDULE_FROM_POLICY",
                "VBR_LAYER_STRICT",
                "VBR_SCHEDULE_CTX",
                "VBR_POLICY_LADDER",
                "VBR_BUDGET",
                "VBR_CAPACITY_BITS",
                "VBR_VRAM_BUDGET",
                "VBR_SELECTED_FAMILY",
                "VBR_SELECTED_POLICY",
                "VBR_SELECTED_BPV",
                "VBR_SELECTED_KLD",
                "VBR_SELECTED_SCHEDULE"}) {
            unsetenv(name);
        }
    };
#endif

    std::vector<std::string> argv;

    printf("test-arg-parser: test invalid usage\n\n");

    // missing value
    argv = {"binary_name", "-m"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (int)
    argv = {"binary_name", "-ngl", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    // wrong value (enum)
    argv = {"binary_name", "-sm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    {
        common_params penalty_params;
        assert(penalty_params.sampling.penalty_last_n == 64);
        assert(penalty_params.sampling.dry_penalty_last_n == 64);

        argv = {"binary_name", "--repeat-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--dry-penalty-last-n", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "0"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-1"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "nan"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        argv = {"binary_name", "--repeat-penalty", "-inf"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));

        const char * penalty_options[] = {"--frequency-penalty", "--presence-penalty"};
        const char * nonfinite_values[] = {"nan", "inf", "-inf"};
        for (const char * option : penalty_options) {
            for (const char * value : nonfinite_values) {
                argv = {"binary_name", option, value};
                assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), penalty_params, LLAMA_EXAMPLE_COMMON));
            }
        }
    }

    // non-existence arg in specific example (--draft cannot be used outside llama-speculative)
    argv = {"binary_name", "--draft", "123"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_EMBEDDING));

    argv = {"binary_name", "-lm", "hello"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    printf("test-arg-parser: test valid usage\n\n");

    argv = {"binary_name", "-m", "model_file.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "model_file.gguf");

    argv = {"binary_name", "-t", "1234"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.cpuparams.n_threads == 1234);

    argv = {"binary_name", "--verbose"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.verbosity > 1);

    argv = {"binary_name", "-m", "abc.gguf", "--predict", "6789", "--batch-size", "9090"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "abc.gguf");
    assert(params.n_predict == 6789);
    assert(params.n_batch == 9090);

    // --draft cannot be used outside llama-speculative
    argv = {"binary_name", "--spec-draft-n-max", "123"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SPECULATIVE));
    assert(params.speculative.draft.n_max == 123);
    assert(params.speculative.draft.n_max_set);

    {
        common_params draft_temp_params;
        assert(draft_temp_params.speculative.sample_temp == 0.0f);
        assert(!draft_temp_params.speculative.sample_temp_set);

        argv = {"binary_name", "--spec-draft-temp", "0"};
        assert(true == common_params_parse(
                argv.size(), list_str_to_char(argv).data(), draft_temp_params, LLAMA_EXAMPLE_SERVER));
        assert(draft_temp_params.speculative.sample_temp == 0.0f);
        assert(draft_temp_params.speculative.sample_temp_set);
    }

    {
        common_params default_params;
        assert(!default_params.speculative.draft.n_max_set);
        assert(default_params.speculative.draft.mtp_vocab_size == 0);
    }

    argv = {"binary_name", "--spec-mtp-vocab-size", "32768"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.draft.mtp_vocab_size == 32768);

    argv = {"binary_name", "--spec-mtp-vocab-size", "0"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));
    assert(params.speculative.draft.mtp_vocab_size == 0);

    argv = {"binary_name", "--spec-mtp-vocab-size", "16384"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_SERVER));

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_len == 3.4);
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-rates", "0.8,0.6,0.2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
        assert(synth_params.speculative.synth_rates == std::vector<double>({0.8, 0.6, 0.2}));
    }

    {
        common_params synth_params;
        argv = {"binary_name", "--spec-synth-len", "3.4x"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), synth_params, LLAMA_EXAMPLE_SERVER));
    }

    argv = {"binary_name", "-lm", "none"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);

    argv = {"binary_name", "-lm", "mmap"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    argv = {"binary_name", "-lm", "mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    argv = {"binary_name", "-lm", "mmap+mlock"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    argv = {"binary_name", "-lm", "dio"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    // multi-value args (CSV)
    argv = {"binary_name", "--lora", "file1.gguf,\"file2,2.gguf\",\"file3\"\"3\"\".gguf\",file4\".gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.lora_adapters.size() == 4);
    assert(params.lora_adapters[0].path == "file1.gguf");
    assert(params.lora_adapters[1].path == "file2,2.gguf");
    assert(params.lora_adapters[2].path == "file3\"3\".gguf");
    assert(params.lora_adapters[3].path == "file4\".gguf");

    {
        printf("test-arg-parser: test VBR cache type and budget flags\n\n");

        // The common CLI defaults to dynamic VBR with a t4 quality floor. Entry tensors still
        // start at F16; the floor only limits how far pressure may degrade them.
        common_params vbr_default;
        argv = {"binary_name", "-m", "model.gguf"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_default, LLAMA_EXAMPLE_COMMON));
        assert(vbr_default.cache_type_k == GGML_TYPE_F16);
        assert(vbr_default.cache_type_v == GGML_TYPE_F16);
        assert(vbr_default.vbr_cache_type_k);
        assert(vbr_default.vbr_cache_type_v);
        assert(!vbr_default.vbr_cache_type_k_explicit);
        assert(!vbr_default.vbr_cache_type_v_explicit);
        assert(vbr_default.vbr_dynamic());
        assert(vbr_default.vbr_min_bits_value == 4.125);
        assert(vbr_default.vbr_capacity_bits == 4.125);
        assert(!vbr_default.vbr_prompt_cache_explicit);
        assert(common_vbr_prompt_cache_mode_for(vbr_default) ==
               common_vbr_prompt_cache_mode::enabled_automatic);

        common_params vbr_cache_disabled;
        argv = {"binary_name", "-m", "model.gguf", "--cache-ram", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_cache_disabled, LLAMA_EXAMPLE_SERVER));
        assert(common_vbr_prompt_cache_mode_for(vbr_cache_disabled) ==
               common_vbr_prompt_cache_mode::disabled_cache_ram);

        common_params vbr_cache_explicit_off;
        argv = {"binary_name", "-m", "model.gguf", "--no-vbr-prompt-cache"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_cache_explicit_off, LLAMA_EXAMPLE_SERVER));
        assert(vbr_cache_explicit_off.vbr_prompt_cache_explicit);
        assert(!vbr_cache_explicit_off.vbr_prompt_cache);
        assert(common_vbr_prompt_cache_mode_for(vbr_cache_explicit_off) ==
               common_vbr_prompt_cache_mode::disabled_explicit);

        common_params vbr_cache_explicit_on;
        argv = {"binary_name", "-m", "model.gguf", "--vbr-prompt-cache"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_cache_explicit_on, LLAMA_EXAMPLE_SERVER));
        assert(vbr_cache_explicit_on.vbr_prompt_cache_explicit);
        assert(vbr_cache_explicit_on.vbr_prompt_cache);
        assert(common_vbr_prompt_cache_mode_for(vbr_cache_explicit_on) ==
               common_vbr_prompt_cache_mode::enabled_explicit);

        common_params implicit_cpu = vbr_default;
        assert(common_params_apply_vbr_cpu_fallback(implicit_cpu, false) ==
               common_vbr_cpu_fallback_result::applied);
        assert(!implicit_cpu.vbr_enabled());
        assert(implicit_cpu.cache_type_k == GGML_TYPE_F16);
        assert(implicit_cpu.cache_type_v == GGML_TYPE_F16);
        assert(implicit_cpu.vbr_min_bits_value == 0.0);
        assert(implicit_cpu.vbr_capacity_bits == 0.0);
        const auto implicit_cpu_context =
            common_context_params_to_llama(implicit_cpu);
        assert(!implicit_cpu_context.vbr_dynamic);
        assert(implicit_cpu_context.vbr_min_bits == 0.0);
        common_params implicit_zero_layers = vbr_default;
        implicit_zero_layers.n_gpu_layers = 0;
        assert(common_params_apply_vbr_cpu_fallback(
                   implicit_zero_layers, true) ==
               common_vbr_cpu_fallback_result::applied);
        common_params implicit_host_kv = vbr_default;
        implicit_host_kv.no_kv_offload = true;
        assert(common_params_apply_vbr_cpu_fallback(
                   implicit_host_kv, true) ==
               common_vbr_cpu_fallback_result::applied);

        // A side the user set to an explicit non-VBR type keeps that type when
        // the OTHER (implicit-vbr) side falls back — intent preservation.
        common_params mixed_explicit_k;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "q8_0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mixed_explicit_k, LLAMA_EXAMPLE_COMMON));
        assert(!mixed_explicit_k.vbr_cache_type_k);
        assert(mixed_explicit_k.vbr_cache_type_v); // V still implicit vbr
        assert(common_params_apply_vbr_cpu_fallback(mixed_explicit_k, false) ==
               common_vbr_cpu_fallback_result::applied);
        assert(mixed_explicit_k.cache_type_k == GGML_TYPE_Q8_0);
        assert(mixed_explicit_k.cache_type_v == GGML_TYPE_F16);
        assert(!mixed_explicit_k.vbr_enabled());

        // A concrete cache type opts out of the implicit VBR default.
        common_params vbr_opt_out;
        argv = {"binary_name", "-m", "model.gguf", "-ct", "f16"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_opt_out, LLAMA_EXAMPLE_COMMON));
        assert(!vbr_opt_out.vbr_cache_type_k);
        assert(!vbr_opt_out.vbr_cache_type_v);
        assert(vbr_opt_out.cache_type_k_explicit);
        assert(vbr_opt_out.cache_type_v_explicit);
        assert(!vbr_opt_out.vbr_enabled());
        assert(common_vbr_prompt_cache_mode_for(vbr_opt_out) ==
               common_vbr_prompt_cache_mode::disabled_static);

        // Explicit VBR preserves the historical full ladder: the cache STARTS at F16 (full quality
        // until budget pressure; the measured fp16->t8 band degrades first) and the runtime
        // controller walks it toward the floor; the runtime channel is cparams
        // (llama_context_params), postprocess exports NO runtime env.
        common_params vbr_params;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "VBR", "-ctv", "vbr"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_params, LLAMA_EXAMPLE_COMMON));
        assert(vbr_params.cache_type_k == GGML_TYPE_F16);
        assert(vbr_params.cache_type_v == GGML_TYPE_F16);
        assert(vbr_params.vbr_cache_type_k);
        assert(vbr_params.vbr_cache_type_v);
        assert(vbr_params.vbr_cache_type_k_explicit);
        assert(vbr_params.vbr_cache_type_v_explicit);
        assert(vbr_params.vbr_budget == "dynamic");
        assert(vbr_params.vbr_dynamic());
        assert(vbr_params.vbr_min_bits_value == 0.0);
        assert(vbr_params.vbr_capacity_bits == 1.25); // capacity advertised at the default t1 floor
        common_params explicit_cpu = vbr_params;
        assert(common_params_apply_vbr_cpu_fallback(explicit_cpu, false) ==
               common_vbr_cpu_fallback_result::explicit_vbr);
        assert(explicit_cpu.vbr_dynamic());
        assert(explicit_cpu.vbr_cache_type_k_explicit);
        assert(explicit_cpu.vbr_cache_type_v_explicit);
#ifndef _WIN32
        assert(getenv("VBR_VMM") == nullptr);
        assert(getenv("VBR_MODE") == nullptr);
        assert(getenv("VBR_BUDGET_MIB") == nullptr);
        assert(getenv("VBR_MIN_BITS") == nullptr);
#endif

        // --vbr-floor is a LITERAL aggregate floor (no snap-up to the next tier); the entry type
        // stays turbo8 regardless
        common_params vbr_t2_floor;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "-ctv", "vbr", "--vbr-min-bits", "2.25"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_t2_floor, LLAMA_EXAMPLE_COMMON));
        assert(vbr_t2_floor.cache_type_k == GGML_TYPE_F16);
        assert(vbr_t2_floor.vbr_min_bits == "2.25");
        assert(vbr_t2_floor.vbr_min_bits_value == 2.25);
        assert(vbr_t2_floor.vbr_capacity_bits == 2.25);

        common_params vbr_literal_floor;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-floor", "2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_literal_floor, LLAMA_EXAMPLE_COMMON));
        assert(vbr_literal_floor.vbr_min_bits_value == 2.0);
        assert(vbr_literal_floor.vbr_capacity_bits == 2.0); // literal, NOT snapped to 2.25

        common_params vbr_tier_alias_floor;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-floor", "t2"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_tier_alias_floor, LLAMA_EXAMPLE_COMMON));
        assert(vbr_tier_alias_floor.vbr_min_bits == "2.25");
        assert(vbr_tier_alias_floor.vbr_min_bits_value == 2.25);

        common_params vbr_fractional_floor;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-floor", "2.75"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_fractional_floor, LLAMA_EXAMPLE_COMMON));
        assert(vbr_fractional_floor.vbr_min_bits_value == 2.75);
        assert(vbr_fractional_floor.vbr_capacity_bits == 2.75);

        // dynamic floors outside the degrade ladder [t1, t8] clamp with a warning
        common_params vbr_low_floor;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-floor", "0.5"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_low_floor, LLAMA_EXAMPLE_COMMON));
        assert(vbr_low_floor.vbr_min_bits_value == 1.25);

        common_params vbr_high_floor;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-floor", "f16"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_high_floor, LLAMA_EXAMPLE_COMMON));
        assert(vbr_high_floor.vbr_min_bits_value == 16.0); // f16 tops the ladder now (= never degrade)

        // one-sided explicit vbr in dynamic mode keeps the default-VBR opposite side active;
        // an explicitly non-default side stays pinned at its type
        common_params vbr_imply_v;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_imply_v, LLAMA_EXAMPLE_COMMON));
        assert(vbr_imply_v.vbr_cache_type_k);
        assert(vbr_imply_v.vbr_cache_type_v);
        assert(vbr_imply_v.cache_type_v == GGML_TYPE_F16);

        common_params vbr_pin_v;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "-ctv", "q8_0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_pin_v, LLAMA_EXAMPLE_COMMON));
        assert(vbr_pin_v.vbr_cache_type_k);
        assert(!vbr_pin_v.vbr_cache_type_v);
        assert(vbr_pin_v.cache_type_k == GGML_TYPE_F16);
        assert(vbr_pin_v.cache_type_v == GGML_TYPE_Q8_0);

        // An explicit F16 side is a pin, not an untouched default. This must work both with
        // the side-specific spelling and when -ct establishes F16 before one side selects VBR.
        common_params vbr_pin_k_f16;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "f16", "-ctv", "vbr"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_pin_k_f16, LLAMA_EXAMPLE_COMMON));
        assert(!vbr_pin_k_f16.vbr_cache_type_k);
        assert(vbr_pin_k_f16.vbr_cache_type_v);
        assert(vbr_pin_k_f16.cache_type_k == GGML_TYPE_F16);
        assert(vbr_pin_k_f16.cache_type_k_explicit);
        assert(vbr_pin_k_f16.vbr_pin_k());

        common_params vbr_pin_k_f16_shorthand;
        argv = {"binary_name", "-m", "model.gguf", "-ct", "f16", "-ctv", "vbr"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_pin_k_f16_shorthand, LLAMA_EXAMPLE_COMMON));
        assert(!vbr_pin_k_f16_shorthand.vbr_cache_type_k);
        assert(vbr_pin_k_f16_shorthand.vbr_cache_type_v);
        assert(vbr_pin_k_f16_shorthand.cache_type_k == GGML_TYPE_F16);
        assert(vbr_pin_k_f16_shorthand.cache_type_k_explicit);
        assert(vbr_pin_k_f16_shorthand.vbr_pin_k());

        common_params vbr_pin_v_f16;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "-ctv", "f16"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_pin_v_f16, LLAMA_EXAMPLE_COMMON));
        assert(vbr_pin_v_f16.vbr_cache_type_k);
        assert(!vbr_pin_v_f16.vbr_cache_type_v);
        assert(vbr_pin_v_f16.cache_type_v == GGML_TYPE_F16);
        assert(vbr_pin_v_f16.cache_type_v_explicit);
        assert(vbr_pin_v_f16.vbr_pin_v());

        // --vbr-vram alone implies -ctk/-ctv vbr (dynamic)
        common_params vbr_vram_budget;
        argv = {"binary_name", "-m", "model.gguf", "--vbr-vram", "24G"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_vram_budget, LLAMA_EXAMPLE_COMMON));
        assert(vbr_vram_budget.vbr_cache_type_k);
        assert(vbr_vram_budget.vbr_cache_type_v);
        assert(vbr_vram_budget.vbr_vram_budget == "25769803776");
        assert(vbr_vram_budget.vbr_vram_budget_bytes == 25769803776ull);
        assert(vbr_vram_budget.vbr_dynamic());
        assert(vbr_vram_budget.vbr_min_bits_value == 4.125);

        // Explicitly opting both sides out cannot be silently undone by a VBR sizing flag.
        common_params vbr_f16_conflict;
        argv = {"binary_name", "-m", "model.gguf", "-ct", "f16", "--vbr-vram", "24G"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_f16_conflict, LLAMA_EXAMPLE_COMMON));

        // A fixed --vbr-budget is an explicit static codec choice and does not inherit the
        // implicit dynamic t4 floor.
        common_params vbr_fixed_default;
        argv = {"binary_name", "-m", "model.gguf", "--vbr-budget", "t3"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_fixed_default, LLAMA_EXAMPLE_COMMON));
        assert(vbr_fixed_default.cache_type_k == GGML_TYPE_TURBO3_TCQ);
        assert(vbr_fixed_default.cache_type_v == GGML_TYPE_TURBO3_TCQ);
        assert(vbr_fixed_default.vbr_min_bits_value == 0.0);
        assert(!vbr_fixed_default.vbr_dynamic());

        // fixed mode: a tier budget selects the static cache type
        common_params vbr_k_only;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "-ctv", "f16", "--vbr-budget", "t4"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_k_only, LLAMA_EXAMPLE_COMMON));
        assert(vbr_k_only.cache_type_k == GGML_TYPE_TURBO4_0);
        assert(vbr_k_only.cache_type_v == GGML_TYPE_F16);
        assert(vbr_k_only.vbr_cache_type_k);
        assert(!vbr_k_only.vbr_cache_type_v);
        assert(vbr_k_only.vbr_capacity_bits == 4.125);
        assert(!vbr_k_only.vbr_dynamic());

        common_params vbr_bad_budget;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-budget", "nonsense"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_bad_budget, LLAMA_EXAMPLE_COMMON));

        common_params vbr_bad_floor;
        argv = {"binary_name", "-m", "model.gguf", "--vbr-min-bits", "17"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_bad_floor, LLAMA_EXAMPLE_COMMON));

        common_params vbr_bad_vram;
        argv = {"binary_name", "-m", "model.gguf", "--vbr-vram", "abc"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_bad_vram, LLAMA_EXAMPLE_COMMON));

        // --vbr-* never clobbers an explicitly non-vbr cache side; with BOTH sides explicit
        // there is nothing to apply it to
        common_params vbr_ct_conflict;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "q8_0", "-ctv", "q8_0", "--vbr-budget", "t3"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_ct_conflict, LLAMA_EXAMPLE_COMMON));

        // one free side: applies to it (V here), leaves the explicit side alone
        common_params vbr_ct_partial;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "q8_0", "--vbr-budget", "t3"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_ct_partial, LLAMA_EXAMPLE_COMMON));
        assert(vbr_ct_partial.cache_type_k == GGML_TYPE_Q8_0);
        assert(vbr_ct_partial.cache_type_v == GGML_TYPE_TURBO3_TCQ);
        assert(!vbr_ct_partial.vbr_cache_type_k);
        assert(vbr_ct_partial.vbr_cache_type_v);

        // a floor above the fixed budget is contradictory
        common_params vbr_floor_over_budget;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-budget", "t2", "--vbr-floor", "t3"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_floor_over_budget, LLAMA_EXAMPLE_COMMON));

        // --vbr-policy needs a fixed budget (dynamic mode uses the baked degrade order)
        common_params vbr_solo_policy;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "--vbr-policy", "does-not-matter.json"};
        assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_solo_policy, LLAMA_EXAMPLE_COMMON));

#ifndef _WIN32
        clear_vbr_runtime_env();

        // fixed mode + policy ladder: rung selection, schedule export
        const std::string policy_prefix = "test-vbr-policy-" + std::to_string((long long) getpid());
        const std::string policy_file = policy_prefix + ".json";
        const std::string schedule_file = policy_prefix + ".sched";

        {
            std::ofstream schedule(schedule_file);
            schedule << "0-0:k:t3tcq\n";
        }
        {
            std::ofstream policy(policy_file);
            policy
                << "{\n"
                << "  \"static_ladder\": [\n"
                << "    {\n"
                << "      \"name\": \"compact-test\",\n"
                << "      \"bpv\": 3.25,\n"
                << "      \"full_kld\": 0.05,\n"
                << "      \"schedule_file\": \"" << schedule_file << "\"\n"
                << "    }\n"
                << "  ]\n"
                << "}\n";
        }

        common_params vbr_policy_fixed;
        argv = {"binary_name", "-m", "model.gguf", "-ctk", "vbr", "-ctv", "vbr", "--vbr-budget", "3.5", "--vbr-policy", policy_file, "--vbr-floor", "3"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_policy_fixed, LLAMA_EXAMPLE_SERVER));
        assert(vbr_policy_fixed.vbr_capacity_bits == 3.25);
        assert(vbr_policy_fixed.vbr_selected_policy == "compact-test");
        assert(getenv("VBR_LAYER_SCHEDULE") && std::string(getenv("VBR_LAYER_SCHEDULE")) == "@" + schedule_file);
        assert(getenv("VBR_SELECTED_POLICY") && std::string(getenv("VBR_SELECTED_POLICY")) == "compact-test");

        clear_vbr_runtime_env();
        std::remove(policy_file.c_str());
        std::remove(schedule_file.c_str());
#endif
    }

    printf("test-arg-parser: test MoE cache and repack modes\n\n");

    {
        common_params mode_params;
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO);
        assert(mode_params.moe_cache.mode_explicit == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_UNSPECIFIED);
    }

    const std::vector<std::vector<std::string>> invalid_moe_cache_args = {
        {"binary_name", "--moe-cache"},
        {"binary_name", "--moe-cache", "invalid"},
        {"binary_name", "--moe-cache", "-1"},
        {"binary_name", "--moe-cache", "1048577"},
        {"binary_name", "--moe-cache", "256x"},
        {"binary_name", "--moe-cache", "999999999999999999999999"},
        {"binary_name", "--moe-cache-expert-parallel", "-1"},
        {"binary_name", "--moe-cache-expert-parallel", "9"},
        {"binary_name", "--moe-cache-expert-parallel", "invalid"},
    };
    for (auto invalid_argv : invalid_moe_cache_args) {
        common_params mode_params;
        assert(false == common_params_parse(
                invalid_argv.size(), list_str_to_char(invalid_argv).data(),
                mode_params, LLAMA_EXAMPLE_COMMON));
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf",
                "--moe-cache-expert-parallel", "auto"};
        assert(true == common_params_parse(
                argv.size(), list_str_to_char(argv).data(),
                mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.expert_parallel == -1);
        assert(common_context_params_to_llama(mode_params).moe_cache_expert_parallel == -1);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf",
                "--moe-cache-expert-parallel", "3"};
        assert(true == common_params_parse(
                argv.size(), list_str_to_char(argv).data(),
                mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.expert_parallel == 3);
        assert(common_context_params_to_llama(mode_params).moe_cache_expert_parallel == 3);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.moe_cache.mode_explicit == true);
        assert(mode_params.no_extra_bufts == false);
        const llama_context_params cparams = common_context_params_to_llama(mode_params);
        assert(cparams.moe_cache_mode == LLAMA_MOE_CACHE_MODE_AUTO);
        assert(cparams.moe_cache_budget_mib == 0);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "on", "--repack"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == true);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_ON);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--repack", "--moe-cache", "256"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.budget_mib == 256);
        assert(mode_params.no_extra_bufts == true);
        const llama_context_params cparams = common_context_params_to_llama(mode_params);
        assert(cparams.moe_cache_mode == LLAMA_MOE_CACHE_MODE_ON);
        assert(cparams.moe_cache_budget_mib == 256);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "1048576"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.budget_mib == 1048576);
        assert(mode_params.no_extra_bufts == true);
        assert(common_context_params_to_llama(mode_params).moe_cache_budget_mib == 1048576);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "256", "--moe-cache", "auto"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_AUTO);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "off", "--repack"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_OFF);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_OFF);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "-m", "model.gguf", "--moe-cache", "0"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_OFF);
        assert(mode_params.moe_cache.budget_mib == 0);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_OFF);
    }

// skip this part on windows, because setenv is not supported
#ifdef _WIN32
    printf("test-arg-parser: skip on windows build\n");
#else
    printf("test-arg-parser: test environment variables (valid + invalid usages)\n\n");

    setenv("LLAMA_ARG_THREADS", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "blah.gguf");
    assert(params.cpuparams.n_threads == 1010);

    setenv("LLAMA_ARG_LOAD_MODE", "blah", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_LOAD_MODE", "mmap", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP);

    setenv("LLAMA_ARG_LOAD_MODE", "mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MLOCK);

    unsetenv("LLAMA_ARG_LOAD_MODE");
    setenv("LLAMA_ARG_MOE_CACHE", "invalid", true);
    argv = {"binary_name"};
    assert(false == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));

    setenv("LLAMA_ARG_MOE_CACHE", "on", true);
    {
        common_params mode_params;
        argv = {"binary_name"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_ON);
        assert(mode_params.moe_cache.mode_explicit == true);
        assert(mode_params.no_extra_bufts == true);
    }

    {
        common_params mode_params;
        argv = {"binary_name", "--moe-cache", "off"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), mode_params, LLAMA_EXAMPLE_COMMON));
        assert(mode_params.moe_cache.mode == COMMON_MOE_CACHE_MODE_OFF);
        assert(mode_params.moe_cache.mode_explicit == true);
        assert(mode_params.no_extra_bufts == false);
        assert(common_context_params_to_llama(mode_params).moe_cache_mode == LLAMA_MOE_CACHE_MODE_OFF);
    }
    unsetenv("LLAMA_ARG_MOE_CACHE");

    setenv("LLAMA_ARG_LOAD_MODE", "mmap+mlock", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK);

    setenv("LLAMA_ARG_LOAD_MODE", "dio", true);
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO);

    printf("test-arg-parser: test negated environment variables\n\n");

    setenv("LLAMA_ARG_LOAD_MODE", "none", true);
    setenv("LLAMA_ARG_NO_PERF", "1", true); // legacy format
    argv = {"binary_name"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.load_mode == LLAMA_LOAD_MODE_NONE);
    assert(params.no_perf == true);

    printf("test-arg-parser: test environment variables being overwritten\n\n");

    setenv("LLAMA_ARG_MODEL", "blah.gguf", true);
    setenv("LLAMA_ARG_THREADS", "1010", true);
    argv = {"binary_name", "-m", "overwritten.gguf"};
    assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), params, LLAMA_EXAMPLE_COMMON));
    assert(params.model.path == "overwritten.gguf");
    assert(params.cpuparams.n_threads == 1010);
#endif // _WIN32

    printf("test-arg-parser: test download functions\n\n");
    const char * GOOD_URL = "http://ggml.ai/";
    const char * BAD_URL  = "http://ggml.ai/404";

    {
        printf("test-arg-parser: test good URL\n\n");
        auto res = common_remote_get_content(GOOD_URL, {});
        assert(res.first == 200);
        assert(res.second.size() > 0);
        std::string str(res.second.data(), res.second.size());
        assert(str.find("llama.cpp") != std::string::npos);
    }

    {
        printf("test-arg-parser: test bad URL\n\n");
        auto res = common_remote_get_content(BAD_URL, {});
        assert(res.first == 404);
    }

    {
        printf("test-arg-parser: test max size error\n");
        common_remote_params params;
        params.max_size = 1;
        try {
            common_remote_get_content(GOOD_URL, params);
            assert(false && "it should throw an error");
        } catch (std::exception & e) {
            printf("  expected error: %s\n\n", e.what());
        }
    }

    printf("test-arg-parser: all tests OK\n\n");
}

int main(void) {
    try {
        test();
    } catch (std::exception & e) {
        fprintf(stderr, "test-arg-parser: exception: %s\n", e.what());
        return 1;
    }
    return 0;
}
