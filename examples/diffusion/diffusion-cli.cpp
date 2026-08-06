#include "arg.h"
#include "chat.h"
#include "common.h"
#include "diffusion.h"
#include "llama.h"
#include "log.h"

#include <clocale>
#include <cstring>
#include <string>
#include <vector>

static std::string format_input_text(const std::string & prompt, const std::string & system_prompt, bool use_chat_template, llama_model * model) {
    if (!use_chat_template) {
        return prompt;
    }

    auto chat_templates = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    common_chat_msg system_msg;

    if (!system_prompt.empty()) {
        system_msg.role = "system";
        system_msg.content = system_prompt;
        inputs.messages.push_back(system_msg);
    }

    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = prompt;

    inputs.messages.push_back(user_msg);
    inputs.add_generation_prompt = true;

    auto result = common_chat_templates_apply(chat_templates.get(), inputs);

    return result.prompt;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = params.n_gpu_layers;
    model_params.devices            = params.devices.data();
    model_params.load_mode          = params.load_mode;
    model_params.check_tensors      = params.check_tensors;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: unsupported model for diffusion");
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = params.n_ctx;
    ctx_params.n_batch              = params.n_batch;
    ctx_params.n_ubatch             = params.n_ubatch;
    ctx_params.flash_attn_type      = params.flash_attn_type;
    ctx_params.no_perf              = params.no_perf;
    ctx_params.type_k               = params.cache_type_k;
    ctx_params.type_v               = params.cache_type_v;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const llama_vocab * vocab            = llama_model_get_vocab(model);

    bool use_chat_template = params.enable_chat_template && params.prompt.find("<|im_start|>") == std::string::npos;
    std::string formatted_prompt = format_input_text(params.prompt, params.system_prompt, use_chat_template, model);

    // Close open <think> tag from jinja template. Self-spec needs newlines for AR mode;
    // block diffusion uses compact form since masks fill content directly.
    {
        auto pos = formatted_prompt.rfind("<think>");
        if (pos != std::string::npos && formatted_prompt.find("</think>", pos) == std::string::npos) {
            if (params.diffusion.self_spec) {
                formatted_prompt = formatted_prompt.substr(0, pos) + "<think>\n</think>\n";
            } else {
                formatted_prompt = formatted_prompt.substr(0, pos) + "<think></think>";
            }
        }
    }

    std::vector<llama_token> input_tokens = common_tokenize(vocab,
                                                            formatted_prompt,
                                                            /*add special tokens*/ true,
                                                            /*parse special*/ true);

    int n_input = input_tokens.size();

    if (static_cast<uint32_t>(n_input) >= llama_n_ctx(ctx)) {
        LOG_ERR("error: input too long (%d tokens), max context is %d\n", n_input, llama_n_ctx(ctx));
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_token mask_token_id = llama_vocab_mask(vocab);

    GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL);

    bool shift_logits = false;
    char shift_logits_str[8];
    if (llama_model_meta_val_str(model, "diffusion.shift_logits", shift_logits_str, sizeof(shift_logits_str)) >= 0) {
        shift_logits = (strcmp(shift_logits_str, "true") == 0);
    }

    int32_t max_new_tokens  = params.n_predict > 0 ? params.n_predict : 256;

    std::vector<llama_token> output_tokens(max_new_tokens);
    int32_t n_generated = 0;

    int64_t t0 = ggml_time_us();

    if (params.diffusion.self_spec) {
        diffusion_self_spec_params ss_params;
        ss_params.mask_token_id  = mask_token_id;
        ss_params.seed           = params.sampling.seed;
        ss_params.draft_length   = params.diffusion.draft_length > 0 ? params.diffusion.draft_length : 16;
        ss_params.max_new_tokens = max_new_tokens;

        LOG_INF("diffusion: self-spec mode, draft_length = %d, max_new_tokens = %d\n",
                ss_params.draft_length, max_new_tokens);
        LOG_INF("\n");

        n_generated = diffusion_self_spec_generate(ctx,
                                                    input_tokens.data(),
                                                    n_input,
                                                    output_tokens.data(),
                                                    max_new_tokens,
                                                    ss_params);
    } else {
        int32_t block_length    = params.diffusion.block_length > 0 ? params.diffusion.block_length : 32;
        int32_t steps_per_block = params.diffusion.steps > 0 ? params.diffusion.steps : block_length;

        diffusion_block_params block_params;
        block_params.mask_token_id   = mask_token_id;
        block_params.seed            = params.sampling.seed;
        block_params.temperature     = params.sampling.temp;
        block_params.block_length    = block_length;
        block_params.steps_per_block = steps_per_block;
        block_params.max_new_tokens  = max_new_tokens;
        block_params.shift_logits    = shift_logits;
        block_params.algorithm       = static_cast<diffusion_algorithm>(params.diffusion.algorithm);
        block_params.top_p           = params.sampling.top_p;
        block_params.top_k           = params.sampling.top_k;
        block_params.threshold       = params.diffusion.threshold;

        LOG_INF("diffusion: block mode, shift_logits = %s\n", shift_logits ? "true" : "false");
        LOG_INF("diffusion: block_length = %d, steps_per_block = %d, max_new_tokens = %d\n",
                block_length, steps_per_block, max_new_tokens);
        LOG_INF("diffusion: algorithm = %d, temperature = %.3f, threshold = %.3f\n",
                (int)block_params.algorithm, block_params.temperature, block_params.threshold);
        LOG_INF("\n");

        n_generated = diffusion_generate_blocks(ctx,
                                                 input_tokens.data(),
                                                 n_input,
                                                 output_tokens.data(),
                                                 max_new_tokens,
                                                 block_params);
    }

    int64_t t1 = ggml_time_us();
    double elapsed_ms = (t1 - t0) / 1000.0;

    std::string output_text = common_detokenize(vocab,
        std::vector<llama_token>(output_tokens.begin(), output_tokens.begin() + n_generated), false);
    LOG_INF("%s", output_text.c_str());

    LOG_INF("\n\n[%d tokens in %.1fms, %.1f t/s]\n",
            n_generated, elapsed_ms, n_generated / (elapsed_ms / 1000.0));

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
