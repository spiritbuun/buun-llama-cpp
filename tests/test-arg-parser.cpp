#include "arg.h"
#include "common.h"
#include "download.h"
#include "llama.h"

#include <cstdlib>
#include <fstream>
#include <cstdio>
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

        // A concrete cache type opts out of the implicit VBR default.
        common_params vbr_opt_out;
        argv = {"binary_name", "-m", "model.gguf", "-ct", "f16"};
        assert(true == common_params_parse(argv.size(), list_str_to_char(argv).data(), vbr_opt_out, LLAMA_EXAMPLE_COMMON));
        assert(!vbr_opt_out.vbr_cache_type_k);
        assert(!vbr_opt_out.vbr_cache_type_v);
        assert(!vbr_opt_out.vbr_enabled());

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
