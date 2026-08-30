#include "server-task.h"
#include "server-cache-plan-authority.h"
#include "server-cache-destruction-quote.h"

#include "../../common/common-cache-plan-estimate.h"

#include "build-info.h"
#include "server-cache-authority.h"
#include "server-cache-retention-proof.h"
#include "server-vbr-artifact-store.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <limits>
#include <new>
#include <sstream>
#include <thread>
#include <tuple>
#include <type_traits>

namespace {

json server_json_from_ordered(nlohmann::ordered_json value) {
    return json::parse(value.dump());
}

} // namespace

json server_task_result_cache_control::to_json() {
    return server_json_from_ordered(server_cache_control_json(operation, result));
}

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    common_chat_msg new_msg;
    try {
        new_msg = common_chat_parse(
            generated_text,
            is_partial,
            chat_parser_params);
    } catch (const std::exception & e) {
        // A parse failure of a malformed generation must never take down the caller: the PEG
        // parser throws on a FINAL parse it cannot match (and on a partial parse that fails at
        // position 0), which aborted llama-cli on an uncaught exception and would fail a fully
        // generated server request. Degrade to the raw text as plain content — and skip the
        // incremental diff machinery entirely: the raw fallback is not prefix-consistent with
        // the earlier partial parses, and string_diff throws (by design) on non-prefix updates.
        SRV_WRN("chat parse failed (%s) — falling back to raw content\n", e.what());
        chat_msg         = {};
        chat_msg.role    = "assistant";
        chat_msg.content = generated_text;
        chat_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        return chat_msg;
    }
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// server_result_stats
//

json server_result_stats::to_json() const {
    json base = server_slot_stats::to_json();
    if (kv_bpv >= 0.0) {
        base["kv_bpv"] = kv_bpv;
    }

    return base;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    if (!cache_receipt.is_null()) {
        res["cache_receipt"] = cache_receipt;
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

json server_task_result_cache_capture::to_json() {
    const char * consistency_name = "_count";
    switch (consistency) {
        case server_cache_capture_consistency::unavailable:
            consistency_name = "unavailable";
            break;
        case server_cache_capture_consistency::capture_exact:
            consistency_name = "capture_exact";
            break;
        case server_cache_capture_consistency::_count:
            break;
    }
    return json {
        { "status", server_vbr_artifact_capture_status_name(status) },
        { "consistency", consistency_name },
        { "reference", reference },
        { "controllers", controllers },
        { "units", units },
        { "companions", companions },
        { "payload_bytes", payload_bytes },
        { "stash_bytes", stash_bytes },
        { "companion_bytes", companion_bytes },
        { "chunks", chunks },
        { "backpressure_waits", backpressure_waits },
        { "event_completions", event_completions },
        { "synchronous_fallbacks", synchronous_fallbacks },
        { "dedup", dedup },
    };
}

const char * server_cache_import_consistency_name(
        server_cache_import_consistency consistency) noexcept {
    switch (consistency) {
        case server_cache_import_consistency::unavailable: return "unavailable";
        case server_cache_import_consistency::capture_exact: return "capture_exact";
        case server_cache_import_consistency::live_rebased: return "live_rebased";
        case server_cache_import_consistency::_count: break;
    }
    return "_count";
}

json server_task_result_cache_import::to_json() {
    const char * consistency_name =
        server_cache_import_consistency_name(consistency);
    return json {
        { "status", server_vbr_artifact_import_status_name(status) },
        { "validation_status",
          vbr_manifest_validation_status_name(validation_status) },
        { "stage_status", vbr_adopt_stage_status_name(stage_status) },
        { "downward_reserve_status",
          vbr_downward_reserve_status_name(downward_reserve_status) },
        { "adopt_status", vbr_adopt_status_name(adopt_status) },
        { "recovery", vbr_adopt_recovery_outcome_name(recovery) },
        { "phase", adopt_attempted
              ? json(vbr_adopt_phase_name(phase)) : json(nullptr) },
        { "downward_subphase",
          adopt_attempted
              ? json(vbr_downward_adopt_subphase_name(downward_subphase))
              : json(nullptr) },
        { "downward_edge",
          downward_edge == UINT32_MAX ? json(nullptr) : json(downward_edge) },
        { "schedule", vbr_import_schedule_status_name(schedule_status) },
        { "destination_schedule",
          vbr_import_destination_status_name(destination_status) },
        { "destination_policy_steps", destination_policy_steps },
        { "destination_logical_bytes", destination_logical_bytes },
        { "destination_physical_growth_bytes",
          destination_physical_growth_bytes },
        { "destination_max_deficit", destination_max_deficit },
        { "decision", vbr_import_decision_name(decision) },
        { "consistency", consistency_name },
        { "units", units },
        { "companions", companions },
        { "payload_bytes", payload_bytes },
        { "companion_bytes", companion_bytes },
    };
}

json server_task_result_cache_plan_preflight::to_json() {
    return server_json_from_ordered(server_cache_plan_preflight_json(view));
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
server_prompt_cache::server_prompt_cache(
        int32_t limit_size_mib,
        size_t limit_tokens) {
    limit_size = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
    this->limit_tokens = limit_tokens;
}

bool server_prompt_cache::enable_retention_shadow() noexcept {
    if (!retention_shadow_rows) {
        retention_shadow_rows.reset(new (std::nothrow)
            server_prompt_cache_shadow_row[
                SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES]);
    }
    if (!retention_shadow_rows || !retention_capacity_authority) {
        return bool(retention_shadow_rows);
    }
    if (retention_shadow_artifacts && retention_shadow_lineages) {
        return true;
    }
    std::unique_ptr<server_prompt_cache_shadow_artifact_slot[]> artifacts(
        new (std::nothrow) server_prompt_cache_shadow_artifact_slot[
            SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY]);
    std::unique_ptr<server_prompt_cache_shadow_lineage_slot[]> lineages(
        new (std::nothrow) server_prompt_cache_shadow_lineage_slot[
            SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY]);
    if (!artifacts || !lineages) {
        return false;
    }
    retention_shadow_artifacts = std::move(artifacts);
    retention_shadow_lineages = std::move(lineages);
    return true;
}

namespace {

struct server_prompt_cache_budget_bytes {
    size_t compact = 0;
    size_t anchor = 0;
    bool exact = false;
};

bool server_prompt_cache_measure_fixed_states(
        const std::vector<const server_prompt_cache_state *> & fixed,
        llama_cache_acct_ledger * ledger,
        size_t & bytes) noexcept {
    bytes = 0;
    try {
        bool shared_plane = false;
        for (const auto * state : fixed) {
            if (!state || state->payload.kind() ==
                    server_prompt_cache_payload_kind::vbr_artifact ||
                state->size() > SIZE_MAX - bytes) {
                return false;
            }
            bytes += state->size();
            for (const auto & checkpoint : state->prompt.checkpoints) {
                shared_plane |=
                    checkpoint.data_tgt.storage_use_count() > 1 ||
                    checkpoint.data_dft.storage_use_count() > 1 ||
                    checkpoint.accel.ring.storage_use_count() > 1 ||
                    checkpoint.accel.spec.storage_use_count() > 1;
            }
        }
        if (!shared_plane) {
            return true;
        }
        if (!ledger) {
            return false;
        }
        std::vector<llama_cache_acct_op_id> ops;
        for (const auto * state : fixed) {
            if (!state->accounting_complete ||
                state->release_ops().empty()) {
                return false;
            }
            ops.insert(ops.end(), state->release_ops().begin(),
                       state->release_ops().end());
        }
        std::sort(ops.begin(), ops.end());
        if (ops.empty() ||
            std::adjacent_find(ops.begin(), ops.end()) != ops.end()) {
            return false;
        }
        llama_cache_acct_release_set_preview preview;
        if (!ledger->preview_release_set(ops, ledger->serial(), preview)) {
            return false;
        }
        uint64_t exact = 0;
        for (const auto & row : preview.rows) {
            if (row.resident_allocated > UINT64_MAX - exact) {
                return false;
            }
            exact += row.resident_allocated;
        }
        if (exact > SIZE_MAX) {
            return false;
        }
        bytes = size_t(exact);
        return true;
    } catch (...) {
        bytes = 0;
        return false;
    }
}

server_prompt_cache_budget_bytes server_prompt_cache_measure_budgets(
        const std::list<server_prompt_cache_state> & states,
        llama_cache_acct_ledger * ledger) noexcept {
    server_prompt_cache_budget_bytes out;
    try {
        std::vector<const server_prompt_cache_payload *> payloads;
        std::vector<const server_prompt_cache_state *> fixed;
        bool payloads_reserved = false;
        for (const auto & state : states) {
            if (state.payload.kind() ==
                    server_prompt_cache_payload_kind::vbr_artifact) {
                if (!payloads_reserved) {
                    payloads.reserve(states.size());
                    payloads_reserved = true;
                }
                payloads.push_back(&state.payload);
            } else {
                fixed.push_back(&state);
            }
        }
        size_t fixed_bytes = 0;
        if (!server_prompt_cache_measure_fixed_states(
                fixed, ledger, fixed_bytes)) {
            return out;
        }
        server_prompt_cache_vbr_budget_summary summary;
        if (!server_prompt_cache_payload::summarize_vbr_budgets(
                payloads, summary) ||
            summary.compact_resident_bytes > SIZE_MAX - fixed_bytes ||
            summary.anchor_resident_bytes > SIZE_MAX -
                fixed_bytes - size_t(summary.compact_resident_bytes)) {
            return out;
        }
        out.compact = fixed_bytes +
            size_t(summary.compact_resident_bytes);
        out.anchor = size_t(summary.anchor_resident_bytes);
        out.exact = true;
        return out;
    } catch (...) {
        return {};
    }
}

} // namespace

static size_t server_prompt_cache_effective_token_limit(
    size_t limit_size,
    size_t limit_tokens,
    size_t cache_bytes,
    size_t cache_tokens) noexcept;

struct server_prompt_cache_vbr_pressure_plan {
    std::array<server_prompt_cache_state *, 2> victims {};
    std::array<llama_cache_acct_artifact_id, 2> artifacts {};
    std::array<bool, 2> soft_leased {};
    size_t count = 0;
};

static bool server_prompt_cache_plan_vbr_pressure(
    server_prompt_cache & cache,
    size_t projected_bytes,
    size_t projected_tokens,
    size_t max_victims,
    bool begin_competition_wave,
    server_prompt_cache_vbr_pressure_plan & plan,
    server_prompt_cache_shadow_row * shadow_rows = nullptr,
    server_prompt_cache_shadow_artifact_slot * shadow_artifacts = nullptr,
    server_prompt_cache_shadow_lineage_slot * shadow_lineages = nullptr,
    llama_cache_acct_artifact_id ignored_artifact = {}) noexcept;

static bool server_prompt_cache_revalidate_vbr_victim(
    server_prompt_cache & cache,
    server_prompt_cache_state * expected_incumbent,
    llama_cache_acct_artifact_id expected_artifact) noexcept;

server_prompt_cache_vbr_capacity_claim::
server_prompt_cache_vbr_capacity_claim(
        server_prompt_cache_vbr_capacity_claim && other) noexcept
    : cache_(other.cache_),
      victims_(other.victims_),
      victim_artifacts_(other.victim_artifacts_),
      victim_count_(other.victim_count_),
      destination_artifact_(other.destination_artifact_),
      incoming_compact_bytes_(other.incoming_compact_bytes_),
      incoming_tokens_(other.incoming_tokens_),
      scheduler_owner_(other.scheduler_owner_) {
    other.clear();
}

server_prompt_cache_vbr_capacity_claim &
server_prompt_cache_vbr_capacity_claim::operator=(
        server_prompt_cache_vbr_capacity_claim && other) noexcept {
    if (this != &other) {
        cache_ = other.cache_;
        victims_ = other.victims_;
        victim_artifacts_ = other.victim_artifacts_;
        victim_count_ = other.victim_count_;
        destination_artifact_ = other.destination_artifact_;
        incoming_compact_bytes_ = other.incoming_compact_bytes_;
        incoming_tokens_ = other.incoming_tokens_;
        scheduler_owner_ = other.scheduler_owner_;
        other.clear();
    }
    return *this;
}

void server_prompt_cache_vbr_capacity_claim::clear() noexcept {
    cache_ = nullptr;
    victims_ = {};
    victim_artifacts_ = {};
    victim_count_ = 0;
    destination_artifact_ = {};
    incoming_compact_bytes_ = 0;
    incoming_tokens_ = 0;
    scheduler_owner_ = {};
}

bool server_prompt_cache::prepare_vbr_publication_capacity(
        server_prompt_cache_vbr_publication_metadata * const * prepared,
        size_t prepared_count,
        uint64_t incoming_compact_bytes,
        server_prompt_cache_vbr_capacity_claim & claim,
        server_prompt_cache_vbr_capacity_status * status) noexcept {
    if (status) {
        *status = server_prompt_cache_vbr_capacity_status::invalid;
    }
    if (claim.ready() || !prepared || prepared_count == 0 ||
        prepared_count > VBR_PROJECTED_CAPTURE_MAX_MANIFESTS ||
        incoming_compact_bytes == 0 ||
        incoming_compact_bytes > SIZE_MAX) {
        return false;
    }
    size_t incoming_tokens = 0;
    for (size_t i = 0; i < prepared_count; ++i) {
        const auto * item = prepared[i];
        if (!item || !item->ready() || item->cache_ != this ||
            item->entry_.size() != 1) {
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (prepared[j] == item) {
                return false;
            }
        }
        const int tokens = item->entry_.front().prompt.n_tokens();
        if (tokens <= 0 || size_t(tokens) > SIZE_MAX - incoming_tokens) {
            return false;
        }
        incoming_tokens += size_t(tokens);
    }

    const auto fits = [&](size_t bytes, size_t tokens) {
        if (limit_size > 0 && bytes > limit_size) {
            return false;
        }
        const size_t effective = server_prompt_cache_effective_token_limit(
            limit_size, limit_tokens, bytes, tokens);
        return limit_tokens == 0 || tokens <= effective;
    };
    // No victim can make a singleton fit if the incoming artifact itself
    // exceeds the empty-cache byte/token envelope. Keep this typed refusal
    // ahead of both the incumbent scan and pressure planning so the idle
    // coordinator can retry a smaller authenticated stem.
    if (prepared_count == 1 &&
        !fits(size_t(incoming_compact_bytes), incoming_tokens)) {
        if (status) {
            *status = server_prompt_cache_vbr_capacity_status::
                incoming_exceeds_hard_limit;
        }
        return false;
    }

    size_t naive_bytes = 0;
    size_t current_tokens = 0;
    for (const auto & state : states) {
        const int tokens = state.prompt.n_tokens();
        const size_t total = state.size();
        const uint64_t anchor = quality_anchor_budget_enabled &&
                state.payload.kind() ==
                    server_prompt_cache_payload_kind::vbr_artifact
            ? state.payload.vbr_anchor_resident_bytes() : 0;
        if (tokens < 0 || size_t(tokens) > SIZE_MAX - current_tokens ||
            anchor > total || total - size_t(anchor) >
                SIZE_MAX - naive_bytes) {
            return false;
        }
        naive_bytes += total - size_t(anchor);
        current_tokens += size_t(tokens);
    }
    if (incoming_tokens > SIZE_MAX - current_tokens ||
        size_t(incoming_compact_bytes) > SIZE_MAX - naive_bytes) {
        return false;
    }
    const size_t projected_tokens = current_tokens + incoming_tokens;
    const size_t projected_naive = naive_bytes +
        size_t(incoming_compact_bytes);
    size_t projected_bytes = projected_naive;
    if (!fits(projected_bytes, projected_tokens)) {
        // The allocation-free per-entry sum is conservative. Only a byte
        // bound can make exact shared-allocation accounting admit a shape the
        // upper bound rejects; token-only pressure is already exact above.
        if (limit_size != 0) {
            const auto measured =
                server_prompt_cache_measure_budgets(states, acct);
            if (!measured.exact ||
                measured.compact >
                    SIZE_MAX - size_t(incoming_compact_bytes) ||
                (!quality_anchor_budget_enabled &&
                 measured.anchor > SIZE_MAX - measured.compact -
                     size_t(incoming_compact_bytes))) {
                return false;
            }
            projected_bytes = measured.compact +
                (quality_anchor_budget_enabled ? 0 : measured.anchor) +
                size_t(incoming_compact_bytes);
        }
    }
    if (!fits(projected_bytes, projected_tokens)) {
        if (prepared_count != 1) {
            if (status) {
                *status = server_prompt_cache_vbr_capacity_status::
                    pressure_batch_unsupported;
            }
            return false;
        }
        server_prompt_cache_vbr_pressure_plan pressure;
        if (!publish_authority || !acct || !retention_obs || !lease_obs ||
            !lease_execution_identity ||
            !server_prompt_cache_plan_vbr_pressure(
                *this, projected_bytes, projected_tokens, 2, true,
                pressure,
                retention_shadow_rows.get(),
                retention_shadow_artifacts.get(),
                retention_shadow_lineages.get(),
                prepared[0]->destination_artifact_)) {
            claim.clear();
            return false;
        }
        claim.victims_ = pressure.victims;
        claim.victim_artifacts_ = pressure.artifacts;
        claim.victim_count_ = pressure.count;
    }
    if (claim.victim_count_ != 0) {
        claim.destination_artifact_ = prepared[0]->destination_artifact_;
        claim.incoming_compact_bytes_ = incoming_compact_bytes;
        claim.incoming_tokens_ = incoming_tokens;
    }
    claim.cache_ = this;
    claim.scheduler_owner_ = std::this_thread::get_id();
    if (status) {
        *status = claim.victim_count_ != 0
            ? server_prompt_cache_vbr_capacity_status::pressure_cited
            : server_prompt_cache_vbr_capacity_status::fit;
    }
    return true;
}

bool server_prompt_cache::consume_vbr_publication_capacity(
        server_prompt_cache_vbr_capacity_claim & claim) noexcept {
    if (!claim.ready() || claim.cache_ != this ||
        claim.scheduler_owner_ != std::this_thread::get_id() ||
        claim.victim_count_ != 0) {
        return false;
    }
    claim.clear();
    return true;
}

size_t server_prompt_cache::size() const {
    size_t naive = 0;
    bool has_vbr = false;
    bool has_anchor = false;
    bool has_shared_fixed_plane = false;
    for (const auto & state : states) {
        naive += state.size();
        if (state.payload.kind() ==
                server_prompt_cache_payload_kind::vbr_artifact) {
            has_vbr = true;
            has_anchor |= state.payload.vbr_has_quality_anchor();
        } else {
            for (const auto & checkpoint : state.prompt.checkpoints) {
                has_shared_fixed_plane |=
                    checkpoint.data_tgt.storage_use_count() > 1 ||
                    checkpoint.data_dft.storage_use_count() > 1 ||
                    checkpoint.accel.ring.storage_use_count() > 1 ||
                    checkpoint.accel.spec.storage_use_count() > 1;
            }
        }
    }
    if (!has_vbr) {
        if (!has_shared_fixed_plane || !acct) {
            return naive;
        }
        const auto measured = server_prompt_cache_measure_budgets(
            states, acct);
        return measured.exact ? measured.compact : naive;
    }
    if (!has_anchor) {
        if (!acct) {
            return naive;
        }
        if (has_shared_fixed_plane) {
            const auto measured = server_prompt_cache_measure_budgets(
                states, acct);
            return measured.exact
                ? measured.compact + measured.anchor : naive;
        }
        try {
            std::vector<const server_prompt_cache_payload *> payloads;
            payloads.reserve(states.size());
            size_t fixed_bytes = 0;
            for (const auto & state : states) {
                if (state.payload.kind() ==
                        server_prompt_cache_payload_kind::vbr_artifact) {
                    payloads.push_back(&state.payload);
                } else {
                    fixed_bytes += state.size();
                }
            }
            llama_cache_acct_release_set_preview preview;
            if (!server_prompt_cache_payload::preview_vbr_retire_union(
                    payloads, acct->serial(), preview)) {
                return naive;
            }
            uint64_t vbr_bytes = 0;
            for (const auto & row : preview.rows) {
                if (row.resident_allocated > UINT64_MAX - vbr_bytes) {
                    return naive;
                }
                vbr_bytes += row.resident_allocated;
            }
            return vbr_bytes <= SIZE_MAX - fixed_bytes
                ? fixed_bytes + size_t(vbr_bytes) : naive;
        } catch (...) {
            return naive;
        }
    }
    const auto measured = server_prompt_cache_measure_budgets(states, acct);
    return measured.exact ? measured.compact + measured.anchor : naive;
}

size_t server_prompt_cache::anchor_size() const {
    size_t naive = 0;
    bool has_anchor = false;
    for (const auto & state : states) {
        const auto bytes = state.payload.vbr_anchor_resident_bytes();
        has_anchor |= bytes != 0 || state.payload.vbr_has_quality_anchor();
        if (bytes > SIZE_MAX - naive) {
            return SIZE_MAX;
        }
        naive += size_t(bytes);
    }
    if (!has_anchor) {
        return 0;
    }
    const auto measured = server_prompt_cache_measure_budgets(states, acct);
    return measured.exact ? measured.anchor : naive;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache::iterator server_prompt_cache::find_state_exact(
        const server_tokens & tokens,
        const std::string & adapter_config_key) noexcept {
    return std::find_if(states.begin(), states.end(), [&](const auto & state) {
        // Identity-scoped: token equality under another adapter is not a
        // durable copy. Equal length closes the recurrent/hybrid prefix hole.
        // contains() is the existing durable-recovery predicate used before
        // a live slot may be cleared. VBR artifacts use their separate typed
        // restore/recovery authority and are intentionally excluded here.
        return state.payload.fixed_state_restorable() &&
               state.adapter_config_key == adapter_config_key &&
               state.prompt.tokens.size() == tokens.size() &&
               state.prompt.tokens.get_common_prefix(tokens) == tokens.size();
    });
}

server_prompt_cache::const_iterator server_prompt_cache::find_state_exact(
        const server_tokens & tokens,
        const std::string & adapter_config_key) const noexcept {
    return const_cast<server_prompt_cache *>(this)->find_state_exact(
        tokens, adapter_config_key);
}

bool server_prompt_cache::contains(
        const server_tokens & tokens,
        const std::string & adapter_config_key) const {
    return find_state_exact(tokens, adapter_config_key) != states.end();
}

void server_prompt_cache::cache_plan_begin_inventory() noexcept {
    cache_plan_next_source_id = 0;
    for (auto & state : states) {
        state.cache_plan_source_id = -1;
    }
}

bool server_prompt_cache::cache_plan_get_source_id(
        server_prompt_cache_state & state,
        int32_t & source_id) noexcept {
    return server_cache_plan_assign_source_id(
        state.cache_plan_source_id, cache_plan_next_source_id, source_id);
}

std::list<server_prompt_cache_state> server_prompt_cache::stage(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft, std::string adapter_config_key) {
    // Calculate the entry's marginal physical bytes before allocating it.
    // Lifecycle-off retains the historical logical sum. Under authority, a
    // live checkpoint plane already bound to this ledger is resident and the
    // host save adds only another reference; unbound aliases within the same
    // prompt are counted once.
    size_t checkpoints_size = 0;
    if (publish_authority && acct) {
        try {
            std::vector<const void *> unbound;
            unbound.reserve(prompt.checkpoints.size() * 4);
            const auto add_plane = [&](const common_shared_byte_buffer & plane) {
                if (plane.empty()) {
                    return true;
                }
                const void * owner = nullptr;
                uint64_t allocation = 0;
                if (plane.accounting_binding(owner, allocation)) {
                    return owner == acct;
                }
                const auto identity = plane.storage_identity();
                if (!identity) {
                    return false;
                }
                if (std::find(unbound.begin(), unbound.end(), identity) !=
                        unbound.end()) {
                    return true;
                }
                if (plane.size() > SIZE_MAX - checkpoints_size) {
                    return false;
                }
                unbound.push_back(identity);
                checkpoints_size += plane.size();
                return true;
            };
            for (const auto & ckpt : prompt.checkpoints) {
                if (!add_plane(ckpt.data_tgt) ||
                    !add_plane(ckpt.data_dft) ||
                    !add_plane(ckpt.accel.ring) ||
                    !add_plane(ckpt.accel.spec)) {
                    return {};
                }
            }
        } catch (...) {
            return {};
        }
    } else {
        for (const auto & ckpt : prompt.checkpoints) {
            if (ckpt.size() > SIZE_MAX - checkpoints_size) {
                return {};
            }
            checkpoints_size += ckpt.size();
        }
    }

    if (state_size_dft > SIZE_MAX - state_size_tgt ||
        checkpoints_size > SIZE_MAX - state_size_tgt - state_size_dft) {
        return {};
    }
    const size_t state_size_new =
        state_size_tgt + state_size_dft + checkpoints_size;

    // this state can't be cached at all; report failure (the caller keeps the live slot)
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return {};
    }

    // Allocate the entry as a DETACHED single-node list, entirely outside `states`. Every allocation
    // that can throw (the list node, the state vectors, the token clone, the checkpoint copy) is
    // performed here; on any failure we return an empty list and leave the cache completely
    // untouched: no eviction or limit reduction for a save that did not happen. publish()
    // then splices this node in without allocating.
    std::list<server_prompt_cache_state> staged;
    try {
        staged.emplace_back();
        auto & entry = staged.back();
        auto * fixed = entry.payload.fixed_state();
        GGML_ASSERT(fixed != nullptr);

        fixed->main.resize(state_size_tgt);
        fixed->drft.resize(state_size_dft);
        entry.prompt.tokens      = prompt.tokens.clone();
        entry.prompt.checkpoints = prompt.checkpoints;
        entry.prompt.sequence_epoch = prompt.sequence_epoch;
        entry.adapter_config_key = std::move(adapter_config_key);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());
        return {};
    }

    return staged;
}

static size_t server_prompt_cache_effective_token_limit(
        size_t limit_size,
        size_t limit_tokens,
        size_t cache_bytes,
        size_t cache_tokens) noexcept {
    if (limit_size == 0) {
        return limit_tokens;
    }
    const float size_per_token = std::max<float>(
        1.0f, float(cache_bytes) / std::max<size_t>(1, cache_tokens));
    return std::max<size_t>(limit_tokens, limit_size / size_per_token);
}

static bool server_prompt_cache_vbr_frontier_matches(
        const server_prompt & prompt,
        const server_prompt_cache_payload & payload,
        const std::string & execution_identity,
        const std::string & adapter_config_key,
        bool * raw_token_comparison = nullptr) noexcept {
    if (raw_token_comparison) {
        *raw_token_comparison = false;
    }
    try {
        const auto * artifact = payload.vbr_artifact();
        if (!artifact || execution_identity.empty()) {
            return false;
        }
        const auto & package = artifact->package();
        if (!package) {
            return false;
        }
        const auto & manifest = package.manifest();
        const auto & identity = manifest.identity;
        if (identity.execution_identity != execution_identity ||
            identity.adapter_config_identity != adapter_config_key ||
            identity.sequence_epoch != prompt.sequence_epoch ||
            identity.token_count != prompt.n_tokens() ||
            identity.next_position != prompt.tokens.pos_next()) {
            return false;
        }
        if (raw_token_comparison) {
            *raw_token_comparison = true;
        }
        if (manifest.token_block.tokens !=
                prompt.tokens.retention_token_ids()) {
            return false;
        }
        std::string media_identity;
        return prompt.tokens.media_content_identity(
                   prompt.n_tokens(), media_identity) &&
               media_identity == identity.media_content_identity;
    } catch (...) {
        return false;
    }
}

static bool server_prompt_retention_exact_scope(
        const server_tokens & tokens,
        const std::string & adapter_config_key,
        int64_t coverage_tokens,
        std::string & out) noexcept;

bool server_prompt_cache::contains_vbr_frontier(
        const server_prompt & prompt,
        const std::string & execution_identity,
        const std::string & adapter_config_key) const noexcept {
    for (const auto & state : states) {
        if (state.payload.kind() ==
                server_prompt_cache_payload_kind::vbr_artifact &&
            state.adapter_config_key == adapter_config_key &&
            server_prompt_cache_vbr_frontier_matches(
                prompt, state.payload, execution_identity,
                adapter_config_key)) {
            return true;
        }
    }
    return false;
}

bool server_prompt_cache::mark_vbr_frontiers(
        server_prompt_cache_vbr_frontier_query * queries,
        size_t query_count,
        server_prompt_cache_vbr_frontier_batch_diagnostics * diagnostics)
        const noexcept {
    if (diagnostics) {
        *diagnostics = {};
    }
    const auto identity_less = [](const auto & lhs, const auto & rhs) {
        return std::tie(
                   lhs.sequence_epoch, lhs.token_count, lhs.next_position,
                   lhs.execution_identity, lhs.adapter_config_identity,
                   lhs.media_content_identity) <
               std::tie(
                   rhs.sequence_epoch, rhs.token_count, rhs.next_position,
                   rhs.execution_identity, rhs.adapter_config_identity,
                   rhs.media_content_identity);
    };
    try {
        if ((query_count != 0 && !queries) ||
            query_count > SERVER_RETENTION_MAX_CANDIDATES) {
            return false;
        }
        if (query_count == 0) {
            return true;
        }
        vbr_stem_witness_arena.clear();
        bool has_stem_witness = false;
        for (size_t i = 0; i < query_count; ++i) {
            auto & query = queries[i];
            query.durable = false;
            query.token_identity_digest = {};
            query.token_identity_ready = false;
            query.stem_durable = false;
            const auto & identity = query.identity;
            if (query.slot_id < 0 ||
                !query.prompt ||
                identity.execution_identity.empty() ||
                identity.adapter_config_identity.empty() ||
                identity.media_content_identity.empty() ||
                identity.sequence_epoch == 0 ||
                identity.token_count <= 0 ||
                identity.next_position <= 0) {
                return false;
            }
            if (query.expected_stem_artifact.v != 0) {
                if (!retention_obs) {
                    return false;
                }
                has_stem_witness = true;
            }
        }
        std::sort(queries, queries + query_count, [&](const auto & lhs,
                                                      const auto & rhs) {
            if (identity_less(lhs.identity, rhs.identity)) {
                return true;
            }
            if (identity_less(rhs.identity, lhs.identity)) {
                return false;
            }
            return lhs.slot_id < rhs.slot_id;
        });
        // Sorting moves query objects in place. Capture their final addresses
        // only after identity ordering so each pointer retains its witness.
        if (has_stem_witness &&
            vbr_stem_witness_arena.capacity() < query_count) {
            vbr_stem_witness_arena.reserve(query_count);
        }
        for (size_t i = 0; i < query_count; ++i) {
            if (queries[i].expected_stem_artifact.v != 0) {
                vbr_stem_witness_arena.push_back(&queries[i]);
            }
        }
        std::sort(
            vbr_stem_witness_arena.begin(),
            vbr_stem_witness_arena.end(),
            [](const auto * lhs, const auto * rhs) {
                return std::tie(
                           lhs->expected_stem_artifact.v, lhs->slot_id) <
                       std::tie(
                           rhs->expected_stem_artifact.v, rhs->slot_id);
            });
        for (const auto & state : states) {
            if (diagnostics) {
                ++diagnostics->states_visited;
            }
            if (state.payload.kind() !=
                    server_prompt_cache_payload_kind::vbr_artifact) {
                continue;
            }
            if (diagnostics) {
                ++diagnostics->vbr_states_visited;
            }
            const auto * artifact = state.payload.vbr_artifact();
            if (!artifact || !artifact->package()) {
                return false;
            }
            const auto & identity = artifact->package().manifest().identity;
            std::array<uint8_t, 32> host_token_identity = {};
            auto current = std::lower_bound(
                queries, queries + query_count, identity,
                [&](const auto & query, const auto & key) {
                    return identity_less(query.identity, key);
                });
            for (; current != queries + query_count &&
                    !identity_less(identity, current->identity) &&
                    !identity_less(current->identity, identity);
                 ++current) {
                if (!current->token_identity_ready) {
                    if (!current->prompt->tokens.retention_token_digest(
                            current->token_identity_digest)) {
                        return false;
                    }
                    current->token_identity_ready = true;
                }
                if (!state.prompt.tokens.retention_token_digest(
                        host_token_identity)) {
                    return false;
                }
                if (state.adapter_config_key ==
                        current->identity.adapter_config_identity &&
                    state.vbr_execution_identity ==
                        current->identity.execution_identity &&
                    host_token_identity ==
                        current->token_identity_digest) {
                    current->durable = true;
                }
            }
            if (!vbr_stem_witness_arena.empty()) {
                if (diagnostics) {
                    ++diagnostics->stem_artifact_lookups;
                }
                const auto host_artifact = retention_obs->artifact_id(
                    server_retention_instance_key::for_host_entry(&state));
                const auto found = std::lower_bound(
                    vbr_stem_witness_arena.begin(),
                    vbr_stem_witness_arena.end(),
                    host_artifact.v,
                    [](const auto * witness, uint64_t value) {
                        return witness->expected_stem_artifact.v < value;
                    });
                for (auto current_stem = found;
                     current_stem != vbr_stem_witness_arena.end() &&
                     (*current_stem)->expected_stem_artifact == host_artifact;
                     ++current_stem) {
                    (*current_stem)->stem_durable = true;
                    if (diagnostics) {
                        ++diagnostics->stem_matches;
                    }
                }
            }
        }
        return true;
    } catch (...) {
        for (size_t i = 0; i < query_count; ++i) {
            queries[i].durable = false;
            queries[i].stem_durable = false;
        }
        return false;
    }
}

llama_cache_acct_artifact_id server_prompt_cache::vbr_host_artifact_id(
        const_iterator host) const noexcept {
    if (!retention_obs || host == states.cend() ||
        host->payload.kind() !=
            server_prompt_cache_payload_kind::vbr_artifact) {
        return {};
    }
    return retention_obs->artifact_id(
        server_retention_instance_key::for_host_entry(&*host));
}

server_prompt_cache_vbr_publication_metadata::
~server_prompt_cache_vbr_publication_metadata() {
    if (cache_) {
        cache_->abandon_vbr_publication_metadata(*this);
    }
}

server_prompt_cache_vbr_publication_metadata::
server_prompt_cache_vbr_publication_metadata(
        server_prompt_cache_vbr_publication_metadata && other) noexcept
    : cache_(other.cache_),
      source_(other.source_),
      source_slot_(other.source_slot_),
      source_artifact_(other.source_artifact_),
      destination_artifact_(other.destination_artifact_),
      source_sequence_epoch_(other.source_sequence_epoch_),
      coverage_tokens_(other.coverage_tokens_),
      source_prefix_digest_(other.source_prefix_digest_),
      stem_(other.stem_) {
    entry_.splice(entry_.end(), other.entry_);
    other.clear();
}

server_prompt_cache_vbr_publication_metadata &
server_prompt_cache_vbr_publication_metadata::operator=(
        server_prompt_cache_vbr_publication_metadata && other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (cache_) {
        cache_->abandon_vbr_publication_metadata(*this);
    }
    cache_ = other.cache_;
    source_ = other.source_;
    source_slot_ = other.source_slot_;
    source_artifact_ = other.source_artifact_;
    destination_artifact_ = other.destination_artifact_;
    source_sequence_epoch_ = other.source_sequence_epoch_;
    coverage_tokens_ = other.coverage_tokens_;
    source_prefix_digest_ = other.source_prefix_digest_;
    stem_ = other.stem_;
    entry_.splice(entry_.end(), other.entry_);
    other.clear();
    return *this;
}

bool server_prompt_cache_vbr_publication_metadata::ready() const noexcept {
    return cache_ && source_ && source_slot_ >= 0 &&
        source_artifact_.v != 0 && destination_artifact_.v != 0 &&
        coverage_tokens_ != 0 && entry_.size() == 1 &&
        entry_.front().prompt.n_tokens() >= 0 &&
        uint64_t(entry_.front().prompt.n_tokens()) == coverage_tokens_;
}

void server_prompt_cache_vbr_publication_metadata::clear() noexcept {
    cache_ = nullptr;
    source_ = nullptr;
    source_slot_ = -1;
    source_artifact_ = {};
    destination_artifact_ = {};
    source_sequence_epoch_ = 0;
    coverage_tokens_ = 0;
    source_prefix_digest_ = {};
    stem_ = false;
    entry_.clear();
}

server_prompt_cache_vbr_refresh_status
server_prompt_cache::refresh_vbr_compact(
        const server_prompt & source_prompt,
        server_prompt_cache_vbr_owner incoming,
        const std::string & execution_identity,
        const std::string & adapter_config_key,
        int32_t source_slot) noexcept {
    using status = server_prompt_cache_vbr_refresh_status;
    if (!incoming || !acct || !retention_obs || source_slot < 0 ||
        execution_identity.empty() || adapter_config_key.empty() ||
        !incoming->accounted_by(acct) || !incoming->retirement_owned() ||
        !vbr_retention_source_available(source_slot)) {
        return status::invalid;
    }
    try {
        const auto probe = server_prompt_cache_payload::from_vbr(incoming);
        if (!probe.valid() || !server_prompt_cache_vbr_frontier_matches(
                source_prompt, probe, execution_identity,
                adapter_config_key)) {
            return status::invalid;
        }

        iterator target = states.end();
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (it->payload.kind() !=
                    server_prompt_cache_payload_kind::vbr_artifact ||
                it->adapter_config_key != adapter_config_key ||
                it->vbr_execution_identity != execution_identity ||
                !server_prompt_cache_vbr_frontier_matches(
                    source_prompt, it->payload, execution_identity,
                    adapter_config_key)) {
                continue;
            }
            if (target != states.end()) {
                return status::ambiguous;
            }
            target = it;
        }
        if (target == states.end()) {
            return status::not_found;
        }
        if (target->recovery_pins != 0) {
            return status::busy;
        }

        const auto * old_variants = target->payload.vbr_variants();
        if (!old_variants || !old_variants->compact_current()) {
            return status::invalid;
        }
        if (incoming.get() == old_variants->compact_current().get() ||
            incoming.get() == old_variants->quality_anchor().get()) {
            return status::unchanged;
        }
        const bool had_anchor = bool(old_variants->quality_anchor());
        const auto * old_compact = old_variants->compact_current().get();
        server_prompt_cache_payload replacement;
        bool unchanged = false;
        if (!target->payload.prepare_vbr_refresh(
                std::move(incoming), replacement,
                quality_anchor_budget_enabled, unchanged)) {
            return unchanged ? status::unchanged : status::internal_error;
        }

        struct budget_result {
            bool measured = false;
            bool compact_fits = false;
            bool anchor_fits = false;
        };
        const auto measure = [&](const server_prompt_cache_payload & value) {
            budget_result result;
            std::vector<const server_prompt_cache_payload *> payloads;
            std::vector<const server_prompt_cache_state *> fixed_states;
            payloads.reserve(states.size());
            fixed_states.reserve(states.size());
            for (const auto & state : states) {
                if (&state == &*target) {
                    payloads.push_back(&value);
                } else if (state.payload.kind() ==
                        server_prompt_cache_payload_kind::vbr_artifact) {
                    payloads.push_back(&state.payload);
                } else {
                    fixed_states.push_back(&state);
                }
            }
            server_prompt_cache_vbr_budget_summary vbr;
            size_t fixed_bytes = 0;
            if (!server_prompt_cache_payload::summarize_vbr_budgets(
                    payloads, vbr) ||
                !server_prompt_cache_measure_fixed_states(
                    fixed_states, acct, fixed_bytes) ||
                vbr.compact_resident_bytes > SIZE_MAX - fixed_bytes) {
                return result;
            }
            const size_t compact_bytes = fixed_bytes +
                size_t(vbr.compact_resident_bytes);
            result.measured = true;
            result.compact_fits =
                limit_size == 0 || compact_bytes <= limit_size;
            result.anchor_fits = !value.vbr_has_quality_anchor() ||
                (quality_anchor_budget_enabled &&
                 vbr.anchor_resident_bytes <= limit_anchor_size);
            return result;
        };

        const server_prompt_cache_vbr_payload * retired_owner = nullptr;
        if (had_anchor) {
            retired_owner = old_compact;
        } else if (!replacement.vbr_has_quality_anchor()) {
            retired_owner = old_compact;
        }
        const auto budget = measure(replacement);
        if (!budget.measured || !budget.compact_fits) {
            return status::budget_refused;
        }
        if (!budget.anchor_fits) {
            if (had_anchor) {
                return status::budget_refused;
            }
            auto compact = replacement.vbr_compact_owner();
            replacement = server_prompt_cache_payload::from_vbr(
                std::move(compact));
            retired_owner = old_compact;
            if (!replacement.valid()) {
                return status::budget_refused;
            }
        }

        vbr_artifact_prepared_retire prepared;
        bool physical_retire = false;
        const uint64_t serial = acct->serial();
        if (retired_owner &&
            !target->payload.vbr_logical_erase_only()) {
            std::vector<const vbr_artifact_package_view *> retiring_packages {
                &retired_owner->package(),
            };
            if (!retiring_packages.front()->prepare_owned_retire(
                    retiring_packages, serial, prepared)) {
                return status::accounting_unavailable;
            }
            physical_retire = true;
        }
        if (acct->serial() != serial) {
            return status::accounting_unavailable;
        }

        target->payload.swap_vbr_storage(replacement);
        replacement.reset_vbr_storage();
        if (physical_retire) {
            const auto retired = prepared.commit();
            GGML_ASSERT(retired !=
                        vbr_artifact_prepared_retire_status::unavailable);
        }
        return target->payload.vbr_has_quality_anchor()
            ? status::updated_with_anchor
            : status::updated_compact_only;
    } catch (...) {
        return status::internal_error;
    }
}

bool server_prompt_cache::preview_vbr_compact_refresh_capacity(
        const server_prompt & source_prompt,
        const std::string & execution_identity,
        const std::string & adapter_config_key,
        uint64_t incoming_compact_bytes) const noexcept {
    if (incoming_compact_bytes == 0 || execution_identity.empty() ||
        adapter_config_key.empty()) {
        return false;
    }
    try {
        const_iterator target = states.end();
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (it->payload.kind() !=
                    server_prompt_cache_payload_kind::vbr_artifact ||
                it->adapter_config_key != adapter_config_key ||
                it->vbr_execution_identity != execution_identity ||
                !server_prompt_cache_vbr_frontier_matches(
                    source_prompt, it->payload, execution_identity,
                    adapter_config_key)) {
                continue;
            }
            if (target != states.end()) {
                return false;
            }
            target = it;
        }
        if (target == states.end() || target->recovery_pins != 0) {
            return false;
        }
        if (limit_size == 0) {
            return true;
        }
        const auto * variants = target->payload.vbr_variants();
        const auto compact = variants ? variants->compact_current() : nullptr;
        if (!compact) {
            return false;
        }
        const uint64_t committed = size();
        // Only claim release credit when this logical row is the sole compact
        // owner. Shared owners remain charged until the exact refresh terminal
        // proves their union after publication.
        const uint64_t replaceable =
            target->payload.vbr_retirement_exclusive()
                ? compact->resident_bytes() : 0;
        const uint64_t base = committed >= replaceable
            ? committed-replaceable : committed;
        return incoming_compact_bytes <= UINT64_MAX-base &&
            base+incoming_compact_bytes <= limit_size;
    } catch (...) {
        return false;
    }
}

bool server_prompt_cache::prepare_vbr_restore(
        const server_tokens & request_tokens,
        const std::string & execution_identity,
        const std::string & adapter_config_key,
        server_prompt_cache_vbr_restore_candidate & candidate,
        bool allow_prefix_projection) noexcept {
    candidate = {};
    // The first automatic-import slice is text-only. A later-media suffix has
    // a different exact DF scope from its cached media stem; fail closed until
    // a dedicated frontier-media lookup authority is wired.
    if (request_tokens.empty() || request_tokens.has_media() ||
        execution_identity.empty() ||
        adapter_config_key.empty() || !retention_obs ||
        !retention_obs->prefix_tracking_enabled() ||
        !retention_obs->prefix_tracking_available()) {
        return false;
    }
    try {
        std::string scope;
        if (!server_prompt_retention_exact_scope(
                request_tokens, adapter_config_key,
                int64_t(request_tokens.size()), scope)) {
            return false;
        }
        struct selection {
            server_prompt_cache_state * best;
            const server_tokens * request;
            const std::string * execution;
            const std::string * adapter;
            uint64_t prefix;
            uint64_t source_tokens;
            llama_pos selected_next_position;
            bool quality;
            uint64_t artifact;
            bool projected;
        } exact {
            nullptr, &request_tokens, &execution_identity,
            &adapter_config_key, 0, 0, -1, false, 0, false,
        };
        const auto select =
            [](void * opaque, const server_retention_instance_key & key,
               uint64_t prefix) noexcept {
                auto & current = *static_cast<selection *>(opaque);
                if (key.kind != common_retention_artifact_kind::host_entry ||
                    key.owner_slot != -1 || key.instance == 0 ||
                    prefix < current.prefix) {
                    return true;
                }
                auto * state = reinterpret_cast<server_prompt_cache_state *>(
                    key.instance);
                const uint64_t source_tokens = state && state->prompt.n_tokens() > 0
                    ? uint64_t(state->prompt.n_tokens()) : 0;
                if (!state || state->payload.kind() !=
                        server_prompt_cache_payload_kind::vbr_artifact ||
                    state->adapter_config_key != *current.adapter ||
                    state->vbr_execution_identity != *current.execution ||
                    source_tokens == 0 ||
                    (current.projected
                        ? prefix >= source_tokens ||
                          state->prompt.tokens.has_media() ||
                          !state->prompt.checkpoints.empty()
                        : source_tokens != prefix) ||
                    state->recovery_pins == UINT32_MAX) {
                    return true;
                }
                const auto * artifact = state->payload.vbr_artifact();
                if (!artifact || !artifact->package()) {
                    return true;
                }
                const auto & manifest = artifact->package().manifest();
                const llama_pos selected_next =
                    current.request->pos_next(int64_t(prefix));
                // The retention callback is reached through this state's
                // immutable indexed token block, and stage_vbr authenticated
                // that block against the sealed package before publication.
                // Re-scanning the complete parent here for every alias would
                // turn an 8192-owner terminal into O(owners*tokens).
                if (selected_next <= 0 ||
                    manifest.identity.token_count <= 0 ||
                    uint64_t(manifest.identity.token_count) != source_tokens ||
                    state->prompt.tokens.pos_next() !=
                        manifest.identity.next_position ||
                    (!current.projected && selected_next !=
                        manifest.identity.next_position)) {
                    return true;
                }
                const auto * variants = state->payload.vbr_variants();
                const bool quality = variants && variants->quality_anchor();
                const auto & preferred = quality
                    ? variants->quality_anchor()
                    : variants->compact_current();
                const uint64_t artifact_id = preferred
                    ? preferred->reference_artifact().v : 0;
                if (artifact_id == 0) {
                    return true;
                }
                if (prefix == current.prefix && current.best &&
                    (quality < current.quality ||
                     (quality == current.quality &&
                      artifact_id >= current.artifact))) {
                    return true;
                }
                current.best = state;
                current.prefix = prefix;
                current.source_tokens = source_tokens;
                current.selected_next_position = selected_next;
                current.quality = quality;
                current.artifact = artifact_id;
                return true;
            };
        const bool indexed_attention = retention_obs->visit_prefix_instances(
            common_retention_pool::attention, scope,
            request_tokens.retention_token_ids(), &exact, select);
        const bool indexed_recurrent = retention_obs->visit_prefix_instances(
            common_retention_pool::recurrent, scope,
            request_tokens.retention_token_ids(), &exact, select);
        if (!indexed_attention || !indexed_recurrent) {
            return false;
        }
        selection projected {
            nullptr, &request_tokens, &execution_identity,
            &adapter_config_key, 0, 0, -1, false, 0, true,
        };
        if (allow_prefix_projection) {
            if (!retention_obs->visit_common_prefix_instances(
                    common_retention_pool::attention, scope,
                    request_tokens.retention_token_ids(), &projected,
                    select)) {
                return false;
            }
        }
        const auto better = [](const selection & left,
                               const selection & right) noexcept {
            return left.best &&
                (!right.best || left.prefix > right.prefix ||
                 (left.prefix == right.prefix &&
                  (left.quality > right.quality ||
                   (left.quality == right.quality &&
                    left.artifact < right.artifact))));
        };
        const selection & selected = allow_prefix_projection &&
                better(projected, exact)
            ? projected : exact;
        if (!selected.best) {
            return false;
        }
        const auto * variants = selected.best->payload.vbr_variants();
        if (!variants || !variants->compact_current()) {
            return false;
        }
        auto compact = variants->compact_current();
        auto preferred = variants->quality_anchor()
            ? variants->quality_anchor() : compact;
        ++selected.best->recovery_pins;
        candidate.cache_ = this;
        candidate.source_ = selected.best;
        candidate.payload_ = std::move(preferred);
        if (variants->quality_anchor()) {
            candidate.fallback_payload_ = std::move(compact);
        }
        candidate.cache_family_ = selected.best->cache_family;
        candidate.prefix_tokens_ = selected.prefix;
        candidate.source_tokens_ = selected.source_tokens;
        candidate.selected_next_position_ = selected.selected_next_position;
        candidate.requires_prefix_projection_ = selected.projected;
        int32_t selected_source_id = -1;
        (void) cache_plan_get_source_id(*selected.best, selected_source_id);
        candidate.source_id_ = selected_source_id;
        return true;
    } catch (...) {
        candidate = {};
        return false;
    }
}

void server_prompt_cache::release_vbr_restore(
        server_prompt_cache_vbr_restore_candidate & candidate) noexcept {
    if (candidate.cache_ != this) {
        return;
    }
    if (!candidate.source_) {
        candidate.clear();
        return;
    }
    if (candidate.prepared_slot_ >= 0 &&
        !candidate.adopted_destination_ && retention_obs) {
        const auto destination =
            server_retention_instance_key::for_slot(candidate.prepared_slot_);
        retention_obs->abandon_prepared_launch(destination);
        retention_obs->retire(destination);
    }
    // The pin excludes every cache eraser, so the prepared node's stable
    // std::list address remains authoritative until this capability releases
    // it. No list scan is needed on the restore/rollback path.
    GGML_ASSERT(candidate.source_->recovery_pins > 0);
    --candidate.source_->recovery_pins;
    candidate.clear();
}

server_prompt_cache_vbr_restore_candidate::
~server_prompt_cache_vbr_restore_candidate() {
    if (cache_) {
        cache_->release_vbr_restore(*this);
    }
}

server_prompt_cache_vbr_restore_candidate::
server_prompt_cache_vbr_restore_candidate(
        server_prompt_cache_vbr_restore_candidate && other) noexcept {
    *this = std::move(other);
}

server_prompt_cache_vbr_restore_candidate &
server_prompt_cache_vbr_restore_candidate::operator=(
        server_prompt_cache_vbr_restore_candidate && other) noexcept {
    if (this != &other) {
        if (cache_) {
            cache_->release_vbr_restore(*this);
        }
        cache_ = other.cache_;
        source_ = other.source_;
        payload_ = std::move(other.payload_);
        fallback_payload_ = std::move(other.fallback_payload_);
        cache_family_ = other.cache_family_;
        prefix_tokens_ = other.prefix_tokens_;
        source_tokens_ = other.source_tokens_;
        selected_next_position_ = other.selected_next_position_;
        requires_prefix_projection_ = other.requires_prefix_projection_;
        source_id_ = other.source_id_;
        prepared_slot_ = other.prepared_slot_;
        prepared_destination_ = other.prepared_destination_;
        adopted_destination_ = other.adopted_destination_;
        prepared_prompt_ = std::move(other.prepared_prompt_);
        other.clear();
    }
    return *this;
}

bool server_prompt_cache_vbr_restore_candidate::ready() const noexcept {
    return cache_ && source_ && payload_ && prefix_tokens_ > 0 &&
        source_tokens_ >= prefix_tokens_ && selected_next_position_ > 0 &&
        requires_prefix_projection_ == (prefix_tokens_ < source_tokens_);
}

const server_prompt_cache_vbr_owner &
server_prompt_cache_vbr_restore_candidate::payload() const noexcept {
    return payload_;
}

const server_prompt_cache_vbr_owner &
server_prompt_cache_vbr_restore_candidate::fallback_payload()
        const noexcept {
    return fallback_payload_;
}

bool server_prompt_cache_vbr_restore_candidate::use_fallback_payload()
        noexcept {
    if (!ready() || !fallback_payload_ || adopted_destination_) {
        return false;
    }
    payload_ = std::move(fallback_payload_);
    return true;
}

const common_cache_family_binding &
server_prompt_cache_vbr_restore_candidate::cache_family() const noexcept {
    return cache_family_;
}

uint64_t server_prompt_cache_vbr_restore_candidate::prefix_tokens()
        const noexcept {
    return prefix_tokens_;
}

uint64_t server_prompt_cache_vbr_restore_candidate::source_tokens()
        const noexcept {
    return source_tokens_;
}

llama_pos server_prompt_cache_vbr_restore_candidate::selected_next_position()
        const noexcept {
    return selected_next_position_;
}

bool server_prompt_cache_vbr_restore_candidate::requires_prefix_projection()
        const noexcept {
    return requires_prefix_projection_;
}

int32_t server_prompt_cache_vbr_restore_candidate::source_id()
        const noexcept {
    return source_id_;
}

void server_prompt_cache_vbr_restore_candidate::clear() noexcept {
    cache_ = nullptr;
    source_ = nullptr;
    payload_.reset();
    fallback_payload_.reset();
    cache_family_ = {};
    prefix_tokens_ = 0;
    source_tokens_ = 0;
    selected_next_position_ = -1;
    requires_prefix_projection_ = false;
    source_id_ = -1;
    prepared_slot_ = -1;
    prepared_destination_ = nullptr;
    adopted_destination_ = nullptr;
    prepared_prompt_.reset();
}

static bool server_prompt_cache_vbr_owner_is_current(
        const server_prompt_cache_state * state,
        const server_prompt_cache_vbr_owner & owner) noexcept {
    const auto * variants = state ? state->payload.vbr_variants() : nullptr;
    return variants && owner &&
        (variants->compact_current() == owner ||
         variants->quality_anchor() == owner);
}

server_prompt_cache_vbr_replacement_ticket::
~server_prompt_cache_vbr_replacement_ticket() {
    if (cache_) {
        cache_->release_vbr_occupied_replacement(*this);
    }
}

server_prompt_cache_vbr_replacement_ticket::
server_prompt_cache_vbr_replacement_ticket(
        server_prompt_cache_vbr_replacement_ticket && other) noexcept {
    *this = std::move(other);
}

server_prompt_cache_vbr_replacement_ticket &
server_prompt_cache_vbr_replacement_ticket::operator=(
        server_prompt_cache_vbr_replacement_ticket && other) noexcept {
    if (this != &other) {
        if (cache_) {
            cache_->release_vbr_occupied_replacement(*this);
        }
        cache_ = other.cache_;
        incoming_ = std::move(other.incoming_);
        recovery_source_ = other.recovery_source_;
        recovery_owner_ = std::move(other.recovery_owner_);
        recovery_pin_ = std::move(other.recovery_pin_);
        recovery_ops_ = std::move(other.recovery_ops_);
        incumbent_ = other.incumbent_;
        incumbent_family_current_ = other.incumbent_family_current_;
        incumbent_family_ = other.incumbent_family_;
        incoming_family_ = other.incoming_family_;
        execution_identity_ = std::move(other.execution_identity_);
        adapter_config_key_ = std::move(other.adapter_config_key_);
        replacement_prompt_ = std::move(other.replacement_prompt_);
        provisional_key_ = other.provisional_key_;
        destination_slot_ = other.destination_slot_;
        incoming_prefix_tokens_ = other.incoming_prefix_tokens_;
        incumbent_tokens_ = other.incumbent_tokens_;
        incumbent_live_lcp_ = other.incumbent_live_lcp_;
        incumbent_sequence_epoch_ = other.incumbent_sequence_epoch_;
        incoming_token_digest_ = other.incoming_token_digest_;
        incumbent_token_digest_ = other.incumbent_token_digest_;
        incumbent_lease_identity_ = std::move(other.incumbent_lease_identity_);
        incumbent_artifact_ = other.incumbent_artifact_;
        incumbent_lineage_ = other.incumbent_lineage_;
        incoming_owner_artifact_ = other.incoming_owner_artifact_;
        recovery_owner_artifact_ = other.recovery_owner_artifact_;
        recovery_host_artifact_ = other.recovery_host_artifact_;
        provisional_artifact_ = other.provisional_artifact_;
        publish_prepared_ = other.publish_prepared_;
        published_ = other.published_;
        other.clear();
    }
    return *this;
}

bool server_prompt_cache_vbr_replacement_ticket::ready() const noexcept {
    if (published_ || !cache_ || !incoming_.ready() || incoming_.cache_ != cache_ ||
        !recovery_source_ ||
        !recovery_owner_ || !recovery_pin_ || !recovery_pin_->valid() ||
        !incumbent_ || !incumbent_family_current_ ||
        *incumbent_family_current_ != incumbent_family_ ||
        !replacement_prompt_ || destination_slot_ < 0 ||
        incoming_prefix_tokens_ <= incumbent_live_lcp_ ||
        incumbent_tokens_ == 0 || incumbent_sequence_epoch_ == 0 ||
        incumbent_artifact_.v == 0 || incumbent_lineage_ == 0 ||
        incoming_owner_artifact_.v == 0 ||
        recovery_owner_artifact_.v == 0 ||
        recovery_host_artifact_.v == 0 || provisional_artifact_.v == 0 ||
        !incumbent_lease_identity_.valid() ||
        !recovery_pin_->binds_exact(
            recovery_host_artifact_, recovery_ops_) ||
        incumbent_->sequence_epoch != incumbent_sequence_epoch_ ||
        incumbent_->n_tokens() <= 0 ||
        uint64_t(incumbent_->n_tokens()) != incumbent_tokens_ ||
        incumbent_->tokens.has_media() ||
        replacement_prompt_->tokens.has_media() ||
        !replacement_prompt_->checkpoints.empty() ||
        replacement_prompt_->sequence_epoch !=
            incoming_.source_->prompt.sequence_epoch ||
        uint64_t(replacement_prompt_->n_tokens()) !=
            incoming_prefix_tokens_ ||
        !server_prompt_cache_vbr_owner_is_current(
            incoming_.source_, incoming_.payload_) ||
        recovery_source_->payload.vbr_compact_owner() != recovery_owner_ ||
        recovery_source_->adapter_config_key != adapter_config_key_ ||
        recovery_source_->vbr_execution_identity != execution_identity_ ||
        recovery_source_->cache_family != incumbent_family_ ||
        recovery_source_->prompt.sequence_epoch !=
            incumbent_sequence_epoch_ ||
        recovery_source_->prompt.n_tokens() != incumbent_->n_tokens() ||
        recovery_source_->recovery_pins == 0 ||
        recovery_owner_->reference_artifact() !=
            recovery_owner_artifact_ ||
        incoming_.payload_->reference_artifact() !=
            incoming_owner_artifact_ ||
        !cache_->retention_obs ||
        cache_->retention_obs->artifact_id(
            server_retention_instance_key::for_slot(destination_slot_)) !=
                incumbent_artifact_ ||
        cache_->retention_obs->artifact_id(provisional_key_) !=
            provisional_artifact_ ||
        !cache_->retention_obs->prepared_for_launch(provisional_key_)) {
        return false;
    }
    std::array<uint8_t, 32> digest = {};
    std::array<uint8_t, 32> incoming_digest = {};
    std::array<uint8_t, 32> recovery_digest = {};
    common_retention_lineage_record lineage;
    const auto recovery_key =
        server_retention_instance_key::for_host_entry(recovery_source_);
    if (!incumbent_->tokens.retention_token_digest(digest) ||
        !replacement_prompt_->tokens.retention_token_digest(incoming_digest) ||
        !recovery_source_->prompt.tokens.retention_token_digest(
            recovery_digest)) {
        return false;
    }
    const auto lease = cache_->lease_obs
        ? cache_->lease_obs->inspect_range(
            incumbent_artifact_, incumbent_lease_identity_,
            incumbent_sequence_epoch_, 0, incumbent_tokens_)
        : server_cache_lease_evaluation {};
    return digest == incumbent_token_digest_ &&
        incoming_digest == incoming_token_digest_ &&
        recovery_digest == incumbent_token_digest_ &&
        cache_->retention_obs->lineage_for_instance(
            server_retention_instance_key::for_slot(destination_slot_),
            lineage) &&
        lineage.lineage_id == incumbent_lineage_ &&
        cache_->retention_obs->artifact_id(recovery_key) ==
            recovery_host_artifact_ &&
        recovery_ops_.size() == 1 && recovery_ops_.front() &&
        lease.state == server_cache_lease_eval_state::known &&
        !server_cache_lease_is_hard(lease);
}

const server_prompt &
server_prompt_cache_vbr_replacement_ticket::replacement_prompt()
        const noexcept {
    static const server_prompt empty;
    return replacement_prompt_ ? *replacement_prompt_ : empty;
}

const server_prompt_cache_vbr_owner &
server_prompt_cache_vbr_replacement_ticket::incoming_payload() const noexcept {
    return incoming_.payload_;
}

const server_prompt_cache_vbr_owner &
server_prompt_cache_vbr_replacement_ticket::recovery_payload() const noexcept {
    return recovery_owner_;
}

int32_t server_prompt_cache_vbr_replacement_ticket::destination_slot()
        const noexcept {
    return destination_slot_;
}

uint64_t
server_prompt_cache_vbr_replacement_ticket::incoming_prefix_tokens()
        const noexcept {
    return incoming_prefix_tokens_;
}

uint64_t server_prompt_cache_vbr_replacement_ticket::incumbent_tokens()
        const noexcept {
    return incumbent_tokens_;
}

uint64_t server_prompt_cache_vbr_replacement_ticket::incumbent_live_lcp()
        const noexcept {
    return incumbent_live_lcp_;
}

llama_cache_acct_artifact_id
server_prompt_cache_vbr_replacement_ticket::incumbent_artifact()
        const noexcept {
    return incumbent_artifact_;
}

llama_cache_acct_artifact_id
server_prompt_cache_vbr_replacement_ticket::incoming_owner_artifact()
        const noexcept {
    return incoming_owner_artifact_;
}

llama_cache_acct_artifact_id
server_prompt_cache_vbr_replacement_ticket::recovery_owner_artifact()
        const noexcept {
    return recovery_owner_artifact_;
}

llama_cache_acct_artifact_id
server_prompt_cache_vbr_replacement_ticket::recovery_host_artifact()
        const noexcept {
    return recovery_host_artifact_;
}

llama_cache_acct_artifact_id
server_prompt_cache_vbr_replacement_ticket::provisional_artifact()
        const noexcept {
    return provisional_artifact_;
}

void server_prompt_cache_vbr_replacement_ticket::clear() noexcept {
    incoming_ = {};
    cache_ = nullptr;
    recovery_source_ = nullptr;
    recovery_owner_.reset();
    recovery_pin_.reset();
    recovery_ops_.clear();
    incumbent_ = nullptr;
    incumbent_family_current_ = nullptr;
    incumbent_family_ = {};
    incoming_family_ = {};
    execution_identity_.clear();
    adapter_config_key_.clear();
    replacement_prompt_.reset();
    provisional_key_ = {};
    destination_slot_ = -1;
    incoming_prefix_tokens_ = 0;
    incumbent_tokens_ = 0;
    incumbent_live_lcp_ = 0;
    incumbent_sequence_epoch_ = 0;
    incoming_token_digest_ = {};
    incumbent_token_digest_ = {};
    incumbent_lease_identity_ = {};
    incumbent_artifact_ = {};
    incumbent_lineage_ = 0;
    incoming_owner_artifact_ = {};
    recovery_owner_artifact_ = {};
    recovery_host_artifact_ = {};
    provisional_artifact_ = {};
    publish_prepared_ = false;
    published_ = false;
}

std::list<server_prompt_cache_state> server_prompt_cache::stage_vbr(
        const server_prompt & prompt,
        server_prompt_cache_payload payload,
        const std::string & execution_identity,
        std::string adapter_config_key) {
    if (payload.kind() != server_prompt_cache_payload_kind::vbr_artifact ||
        !payload.publishable() ||
        !server_prompt_cache_vbr_frontier_matches(
            prompt, payload, execution_identity, adapter_config_key)) {
        return {};
    }
    const size_t payload_size = payload.size();
    if (payload_size == 0 ||
        (limit_size > 0 && payload_size > limit_size) ||
        (limit_size == 0 && limit_tokens > 0 &&
         size_t(prompt.n_tokens()) > limit_tokens)) {
        return {};
    }

    std::array<uint8_t, 32> token_identity_digest = {};
    if (!prompt.tokens.retention_token_digest(token_identity_digest)) {
        return {};
    }
    std::list<server_prompt_cache_state> staged;
    try {
        staged.emplace_back();
        auto & entry = staged.back();
        entry.prompt.tokens = prompt.tokens.clone();
        entry.prompt.sequence_epoch = prompt.sequence_epoch;
        entry.payload = std::move(payload);
        entry.adapter_config_key = std::move(adapter_config_key);
        entry.vbr_execution_identity = execution_identity;
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate VBR prompt cache metadata: %s\n", e.what());
        return {};
    }
    return staged;
}

bool server_prompt_cache::payload_bytes(
        const server_prompt_cache_state & st,
        uint64_t & snapshot_bytes,
        uint64_t & checkpoint_bytes,
        uint64_t & accelerator_bytes) noexcept {
    snapshot_bytes = checkpoint_bytes = accelerator_bytes = 0;
    const auto * fixed = st.payload.fixed_state();
    if (!fixed) {
        return false;
    }
    snapshot_bytes    = uint64_t(fixed->size());
    const auto add_checked = [](uint64_t & acc, size_t value) {
        if (uint64_t(value) > std::numeric_limits<uint64_t>::max() - acc) {
            return false;
        }
        acc += uint64_t(value);
        return true;
    };
    for (const auto & ckpt : st.prompt.checkpoints) {
        if (!add_checked(checkpoint_bytes, ckpt.data_tgt.size()) ||
            !add_checked(checkpoint_bytes, ckpt.data_dft.size()) ||
            !add_checked(accelerator_bytes, ckpt.accel.size())) {
            snapshot_bytes = checkpoint_bytes = accelerator_bytes = 0;
            return false;
        }
    }
    return true;
}

bool server_prompt_cache::payload_leaves(
        server_prompt_cache_state & st,
        std::array<server_prompt_cache_payload_leaf, 3> & leaves) noexcept {
    leaves = {{
        {
            llama_cache_acct_category::full_snapshot_payload,
            0,
        },
        {
            llama_cache_acct_category::checkpoint_state_payload,
            0,
        },
        {
            llama_cache_acct_category::typed_accelerator_payload,
            0,
        },
    }};
    uint64_t snapshot_bytes = 0;
    uint64_t checkpoint_bytes = 0;
    uint64_t accelerator_bytes = 0;
    if (!payload_bytes(
            st, snapshot_bytes, checkpoint_bytes, accelerator_bytes)) {
        return false;
    }
    leaves[0].bytes = snapshot_bytes;
    leaves[1].bytes = checkpoint_bytes;
    leaves[2].bytes = accelerator_bytes;
    return true;
}

// Shadow-accounting producer: one transaction per charged leaf category of a published
// entry, at the publication boundary (the splice into `states`), released when the entry leaves
// `states` on any path. Aggregate entry size is a provider grouping and is NEVER charged — the
// leaves below are mutually exclusive so their sum cannot double-count. The fill-failure abort
// mapping (stage() → abort) lands with F's real artifact transaction. The ledger is
// non-throwing by contract, so no accounting failure can escape into the shipped cache path;
// the `acct_unavailable` fault seam proves that invariance in the gate.
void server_prompt_cache::acct_charge_entry(server_prompt_cache_state & st) {
    if (!acct || st.payload.kind() ==
            server_prompt_cache_payload_kind::vbr_artifact) {
        // Sealed VBR allocations were already admitted and charged by their
        // catalog transaction. The logical cache node owns a borrow, not a
        // second physical allocation.
        return;
    }
    const auto domain = llama_cache_acct_resource_domain::non_device(
        llama_cache_acct_residency::pageable_host);

    // checked sums: an overflowing observation latches the leaf unavailable instead of
    // charging a fabricated value (the shipped path is untouched either way)
    std::array<server_prompt_cache_payload_leaf, 3> leaves;
    const bool sums_ok = payload_leaves(st, leaves);

    if (!sums_ok || server_fault("acct_unavailable")) { // Shipped-path invariance seam.
        for (const auto & leaf : leaves) {
            server_cache_acct_mark_shadow_unavailable(
                *acct, leaf.category, domain,
                llama_cache_acct_producer::host_cache);
        }
        return;
    }

    try {
        st.acct_ops.clear();
        st.acct_ops.reserve(leaves.size());
        for (const auto & leaf : leaves) {
            const auto op = server_cache_acct_charge_shadow(
                *acct, leaf.category, domain,
                llama_cache_acct_producer::host_cache, {},
                leaf.bytes, leaf.bytes);
            if (!op) {
                for (const auto committed : st.acct_ops) {
                    (void) acct->release(committed);
                }
                st.acct_ops.clear();
                st.accounting_complete = false;
                return;
            }
            st.acct_ops.push_back(op);
        }
        st.accounting_complete = true;
    } catch (...) {
        for (const auto committed : st.acct_ops) {
            (void) acct->release(committed);
        }
        st.acct_ops.clear();
        st.accounting_complete = false;
    }
}

void server_prompt_cache::acct_release_entry(server_prompt_cache_state & st) {
    if (!acct) {
        return;
    }
    for (const auto op : st.release_ops()) {
        if (op) {
            acct->release(op);
        }
    }
    st.acct_ops.clear();
    st.accounting_complete = false;
}

bool server_cache_lease_build_identity(
        const std::string & execution_identity,
        const std::string & adapter_identity,
        const server_tokens & tokens,
        int64_t coverage_tokens,
        server_cache_lease_identity & out) {
    if (execution_identity.empty() ||
        adapter_identity.empty() ||
        coverage_tokens < 0) {
        return false;
    }
    out.execution_identity = execution_identity;
    out.adapter_config_identity = adapter_identity;
    return tokens.media_content_identity(
               coverage_tokens, out.media_content_identity) &&
           out.valid();
}

static bool server_prompt_retention_exact_scope(
        const server_tokens & tokens,
        const std::string & adapter_config_key,
        int64_t coverage_tokens,
        std::string & out) noexcept {
    out.clear();
    if (coverage_tokens < 0 ||
        uint64_t(coverage_tokens) > tokens.size()) {
        return false;
    }
    try {
        std::string media_identity;
        if (!tokens.media_content_identity(
                coverage_tokens, media_identity)) {
            return false;
        }
        const uint64_t adapter_size = adapter_config_key.size();
        const uint64_t media_size = media_identity.size();
        size_t total = sizeof(adapter_size);
        if (adapter_size > SIZE_MAX - total) {
            return false;
        }
        total += size_t(adapter_size);
        if (sizeof(media_size) > SIZE_MAX - total) {
            return false;
        }
        total += sizeof(media_size);
        if (media_size > SIZE_MAX - total) {
            return false;
        }
        total += size_t(media_size);
        out.reserve(total);
        out.append(
            reinterpret_cast<const char *>(&adapter_size),
            sizeof(adapter_size));
        out.append(adapter_config_key);
        out.append(
            reinterpret_cast<const char *>(&media_size),
            sizeof(media_size));
        out.append(media_identity);
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

static bool server_prompt_retention_exact_scope(
        const server_prompt & prompt,
        const std::string & adapter_config_key,
        int64_t coverage_tokens,
        std::string & out) noexcept {
    return server_prompt_retention_exact_scope(
        prompt.tokens, adapter_config_key, coverage_tokens, out);
}

bool server_prompt_retention_publish_exact_prefix(
        server_retention_sidecar_store & retention,
        const server_retention_instance_key & key,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens) noexcept {
    if (!retention.prefix_tracking_enabled()) {
        return true;
    }
    std::string scope;
    if (!server_prompt_retention_exact_scope(
            prompt, adapter_identity, coverage_tokens, scope)) {
        return retention.publish_prefix(
            key, {}, prompt.tokens.retention_token_ids());
    }
    return retention.publish_prefix(
        key, scope, prompt.tokens.retention_token_ids());
}

static void server_prompt_cache_mirror_prefix(
        server_prompt_cache & cache,
        const server_retention_instance_key & key,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens) noexcept {
    if (cache.retention_obs) {
        (void) server_prompt_retention_publish_exact_prefix(
            *cache.retention_obs, key, prompt, adapter_identity,
            coverage_tokens);
    }
}

static void server_prompt_cache_mirror_lease(
        server_prompt_cache & cache,
        bool mirrored,
        const server_cache_lease_subject * source,
        const server_cache_lease_subject & destination,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens) noexcept {
    if (!mirrored || !cache.lease_obs) {
        return;
    }
    try {
        server_cache_lease_identity identity;
        if (destination.valid() &&
            cache.lease_execution_identity &&
            server_cache_lease_build_identity(
                *cache.lease_execution_identity, adapter_identity,
                prompt.tokens, coverage_tokens, identity)) {
            if (source) {
                (void) cache.lease_obs->artifact_cloned(
                    *source, destination, identity);
            } else {
                (void) cache.lease_obs->artifact_rebound(
                    destination.artifact, identity);
            }
        } else {
            cache.lease_obs->artifact_identity_unavailable(destination);
        }
    } catch (...) {
        cache.lease_obs->artifact_identity_unavailable(destination);
    }
}

enum class server_prompt_cache_prefix_clone_mode : uint8_t {
    publish_from_prompt = 0,
    share_source,
    publish_stem,
    branch_prefix,
};

static bool server_prompt_cache_mirror_artifact_clone(
        server_prompt_cache & cache,
        const server_retention_instance_key & source_key,
        common_retention_artifact_kind source_kind,
        int32_t source_slot,
        const server_retention_instance_key & destination_key,
        common_retention_artifact_kind destination_kind,
        int32_t destination_slot,
        const server_prompt & prompt,
        const std::string & adapter_identity,
        int64_t coverage_tokens,
        server_prompt_cache_prefix_clone_mode prefix_mode =
            server_prompt_cache_prefix_clone_mode::publish_from_prompt,
        int64_t source_turn_tokens = -1)
        noexcept {
    if (!cache.retention_obs) {
        return true;
    }

    common_chat_msg_spans no_spans;
    bool cloned = false;
    bool prefix_prepared = false;
    switch (prefix_mode) {
        case server_prompt_cache_prefix_clone_mode::publish_stem:
            cloned = source_turn_tokens > 0 && coverage_tokens > 0 &&
                coverage_tokens <= source_turn_tokens &&
                cache.retention_obs->publish(
                    destination_key, common_retention_pool::attention,
                    no_spans, true, uint64_t(source_turn_tokens),
                    uint64_t(coverage_tokens), true,
                    nullptr, nullptr, &source_key);
            break;
        case server_prompt_cache_prefix_clone_mode::branch_prefix:
            cloned = coverage_tokens > 0 &&
                cache.retention_obs->branch_prefix(
                    source_key, destination_key,
                    uint64_t(coverage_tokens), true);
            break;
        case server_prompt_cache_prefix_clone_mode::publish_from_prompt:
            if (coverage_tokens > 0) {
                std::string scope;
                cloned = server_prompt_retention_exact_scope(
                        prompt, adapter_identity, coverage_tokens, scope) &&
                    cache.retention_obs->clone_exact_prefix(
                        source_key, destination_key, scope,
                        prompt.tokens.retention_token_ids());
                prefix_prepared = cloned;
            }
            break;
        case server_prompt_cache_prefix_clone_mode::share_source:
            cloned = cache.retention_obs->clone(
                source_key, destination_key);
            break;
    }
    const bool indexed = cloned && (prefix_prepared ||
        (destination_kind == common_retention_artifact_kind::checkpoint ||
         (prefix_mode == server_prompt_cache_prefix_clone_mode::share_source
              ? cache.retention_obs->clone_prefix(
                    source_key, destination_key)
              : server_prompt_retention_publish_exact_prefix(
                    *cache.retention_obs, destination_key, prompt,
                    adapter_identity, coverage_tokens))));
    const server_cache_lease_subject source {
        cache.retention_obs->artifact_id(source_key),
        source_kind,
        source_slot,
    };
    const server_cache_lease_subject destination {
        cache.retention_obs->artifact_id(destination_key),
        destination_kind,
        destination_slot,
    };
    // A divergent prefix is a new lineage, not a continuation of the parent.
    // In particular it must not inherit the parent's hard/soft lease. The
    // ordinary exact restore and shared-source publication modes remain true
    // clones and keep their existing lease transfer semantics.
    if (prefix_mode != server_prompt_cache_prefix_clone_mode::branch_prefix) {
        server_prompt_cache_mirror_lease(
            cache, indexed, &source, destination, prompt,
            adapter_identity, coverage_tokens);
    }
    return indexed;
}

bool server_prompt_cache::prepare_vbr_publication_metadata(
        const server_prompt & source_prompt,
        const std::string & execution_identity,
        std::string adapter_config_key,
        int32_t source_slot,
        server_prompt_cache_vbr_publication_metadata & prepared) noexcept {
    return prepare_vbr_publication_metadata_impl(
        source_prompt, source_prompt.n_tokens(), false,
        execution_identity, std::move(adapter_config_key), source_slot,
        prepared);
}

bool server_prompt_cache::prepare_vbr_stem_publication_metadata(
        const server_prompt & source_prompt,
        int64_t coverage_tokens,
        const std::string & execution_identity,
        std::string adapter_config_key,
        int32_t source_slot,
        server_prompt_cache_vbr_publication_metadata & prepared) noexcept {
    return prepare_vbr_publication_metadata_impl(
        source_prompt, coverage_tokens, true, execution_identity,
        std::move(adapter_config_key), source_slot, prepared);
}

bool server_prompt_cache::prepare_vbr_publication_metadata_impl(
        const server_prompt & source_prompt,
        int64_t coverage_tokens,
        bool stem,
        const std::string & execution_identity,
        std::string adapter_config_key,
        int32_t source_slot,
        server_prompt_cache_vbr_publication_metadata & prepared) noexcept {
    if (prepared.ready() || !acct || !retention_obs || source_slot < 0 ||
        execution_identity.empty() || adapter_config_key.empty() ||
        source_prompt.tokens.empty() ||
        coverage_tokens <= 0 ||
        coverage_tokens > source_prompt.n_tokens() ||
        (stem && source_prompt.sequence_epoch == 0) ||
        !retention_obs->prefix_tracking_enabled() ||
        !retention_obs->prefix_tracking_available() ||
        !vbr_retention_source_available(source_slot)) {
        return false;
    }
    std::array<uint8_t, 32> source_prefix_digest = {};
    if (stem &&
        !source_prompt.tokens.retention_token_prefix_digest(
            size_t(coverage_tokens), source_prefix_digest)) {
        return false;
    }
    std::list<server_prompt_cache_state> entry;
    try {
        entry.emplace_back();
        auto & state = entry.front();
        state.prompt.tokens = source_prompt.tokens.clone();
        if (stem) {
            state.prompt.tokens.keep_first(size_t(coverage_tokens));
        }
        state.prompt.sequence_epoch = source_prompt.sequence_epoch;
        state.adapter_config_key = std::move(adapter_config_key);
        state.vbr_execution_identity = execution_identity;
    } catch (...) {
        return false;
    }
    auto & state = entry.front();
    const auto source_key =
        server_retention_instance_key::for_slot(source_slot);
    const auto destination_key =
        server_retention_instance_key::for_host_entry(&state);
    const bool mirrored =
        !server_fault("vbr_prompt_cache_prefix_fail") &&
        server_prompt_cache_mirror_artifact_clone(
            *this,
            source_key, common_retention_artifact_kind::live_slot,
            source_slot,
            destination_key, common_retention_artifact_kind::host_entry,
            -1,
            state.prompt, state.adapter_config_key,
            state.prompt.n_tokens(),
            stem ? server_prompt_cache_prefix_clone_mode::publish_stem
                 : server_prompt_cache_prefix_clone_mode::publish_from_prompt,
            source_prompt.n_tokens());
    if (!mirrored) {
        retention_obs->retire(destination_key);
        return false;
    }
    const auto source_artifact = retention_obs->artifact_id(source_key);
    const auto destination_artifact =
        retention_obs->artifact_id(destination_key);
    if (source_artifact.v == 0 || destination_artifact.v == 0) {
        retention_obs->retire(destination_key);
        return false;
    }
    prepared.cache_ = this;
    prepared.source_ = &source_prompt;
    prepared.source_slot_ = source_slot;
    prepared.source_artifact_ = source_artifact;
    prepared.destination_artifact_ = destination_artifact;
    prepared.source_sequence_epoch_ = source_prompt.sequence_epoch;
    prepared.coverage_tokens_ = uint64_t(coverage_tokens);
    prepared.source_prefix_digest_ = source_prefix_digest;
    prepared.stem_ = stem;
    prepared.entry_.splice(prepared.entry_.end(), entry);
    return true;
}

void server_prompt_cache::abandon_vbr_publication_metadata(
        server_prompt_cache_vbr_publication_metadata & prepared) noexcept {
    if (prepared.cache_ != this) {
        return;
    }
    if (retention_obs && prepared.entry_.size() == 1) {
        retention_obs->retire(
            server_retention_instance_key::for_host_entry(
                &prepared.entry_.front()));
    }
    prepared.clear();
}

bool server_prompt_cache::publish_vbr(
        server_prompt_cache_vbr_publication_metadata & prepared,
        server_prompt_cache_payload payload,
        common_cache_family_binding family,
        bool automatic_main_family,
        iterator * published,
        server_prompt_cache_vbr_capacity_claim * capacity) noexcept {
    if (published) {
        *published = states.end();
    }
    if (!prepared.ready() || prepared.cache_ != this ||
        payload.kind() != server_prompt_cache_payload_kind::vbr_artifact ||
        !payload.publishable()) {
        return false;
    }
    auto & staged = prepared.entry_.front();
    const auto source_prefix_matches = [&]() noexcept {
        if (!prepared.stem_) {
            return true;
        }
        try {
            if (!prepared.source_ ||
                prepared.source_->sequence_epoch !=
                    prepared.source_sequence_epoch_ ||
                staged.prompt.sequence_epoch !=
                    prepared.source_sequence_epoch_ ||
                prepared.coverage_tokens_ > uint64_t(INT64_MAX) ||
                prepared.source_->n_tokens() < 0 ||
                uint64_t(prepared.source_->n_tokens()) <
                    prepared.coverage_tokens_ ||
                staged.prompt.n_tokens() < 0 ||
                uint64_t(staged.prompt.n_tokens()) !=
                    prepared.coverage_tokens_ ||
                !server_prompt_cache_vbr_frontier_matches(
                    staged.prompt, payload,
                    staged.vbr_execution_identity,
                    staged.adapter_config_key)) {
                return false;
            }
            std::array<uint8_t, 32> live_digest = {};
            std::array<uint8_t, 32> staged_digest = {};
            return prepared.source_->tokens.retention_token_prefix_digest(
                       size_t(prepared.coverage_tokens_), live_digest) &&
                   staged.prompt.tokens.retention_token_prefix_digest(
                       size_t(prepared.coverage_tokens_), staged_digest) &&
                   live_digest == prepared.source_prefix_digest_ &&
                   staged_digest == prepared.source_prefix_digest_ &&
                   prepared.source_->tokens.get_common_prefix(
                       staged.prompt.tokens) == prepared.coverage_tokens_;
        } catch (...) {
            return false;
        }
    };
    if (!source_prefix_matches()) {
        return false;
    }
    server_prompt_cache_vbr_pressure_citation required_victims;
    if (capacity) {
        if (!capacity->ready() || capacity->cache_ != this ||
            capacity->scheduler_owner_ != std::this_thread::get_id()) {
            return false;
        }
        if (capacity->victim_count_ != 0) {
            const auto compact = payload.vbr_compact_owner();
            if (prepared.destination_artifact_ !=
                    capacity->destination_artifact_ ||
                prepared.entry_.front().prompt.n_tokens() < 0 ||
                size_t(prepared.entry_.front().prompt.n_tokens()) !=
                    capacity->incoming_tokens_ ||
                !compact || compact->resident_bytes() >
                    capacity->incoming_compact_bytes_) {
                capacity->clear();
                return false;
            }
            for (size_t i = 0; i < capacity->victim_count_; ++i) {
                if (!server_prompt_cache_revalidate_vbr_victim(
                        *this, capacity->victims_[i],
                        capacity->victim_artifacts_[i])) {
                    capacity->clear();
                    return false;
                }
            }
            required_victims.artifacts = capacity->victim_artifacts_;
            required_victims.count = capacity->victim_count_;
        }
        capacity->clear();
    }
    const auto source_key =
        server_retention_instance_key::for_slot(prepared.source_slot_);
    const auto destination_key =
        server_retention_instance_key::for_host_entry(&staged);
    if (!retention_obs ||
        retention_obs->artifact_id(source_key) !=
            prepared.source_artifact_ ||
        retention_obs->artifact_id(destination_key) !=
            prepared.destination_artifact_) {
        return false;
    }
    staged.payload = std::move(payload);
    server_prompt_cache_apply_family(
        staged, family, automatic_main_family);
    const auto * source = prepared.source_;
    const int32_t source_slot = prepared.source_slot_;
    const int64_t source_vbr_coverage_tokens = prepared.stem_
        ? int64_t(prepared.coverage_tokens_) : -1;
    std::list<server_prompt_cache_state> entry;
    entry.splice(entry.end(), prepared.entry_);
    prepared.clear();
    try {
        if (publish_impl(
                std::move(entry), source, source_slot, published, true,
                required_victims, source_vbr_coverage_tokens)) {
            return true;
        }
    } catch (...) {
    }
    if (retention_obs) {
        retention_obs->retire(destination_key);
    }
    if (published) {
        *published = states.end();
    }
    return false;
}

static server_cache_destruction_admission server_prompt_cache_observe_drop(
        server_prompt_cache & cache,
        const server_prompt_cache_state & state,
        server_cache_destruction_reason reason) noexcept {
    if (!cache.destruction_obs) {
        return {};
    }

    server_cache_destruction_request request;
    request.cls    = server_cache_destruction_class::host_artifact_drop;
    request.reason = reason;
    request.add_target(
        server_cache_destruction_target_kind::host_artifact,
        -1,
        cache.retention_obs ? cache.retention_obs->artifact_id(
            server_retention_instance_key::for_host_entry(&state))
            : llama_cache_acct_artifact_id{});

    const auto & ops = state.release_ops();
    llama_cache_acct_release_set_preview preview;
    const bool known = cache.acct && !ops.empty() &&
        cache.acct->preview_release_set(
            ops, cache.acct->serial(), preview, true);
    if (known) {
        // The ledger has already aggregated the complete logical owner by
        // (category, domain). At most two measures per bounded accounting
        // category enter the fixed destruction request, independent of the
        // checkpoint-ring cardinality and shared-allocation fanout.
        for (const auto & row : preview.yield_rows) {
            for (const auto measure : {
                    llama_cache_acct_measure::logical_payload,
                    llama_cache_acct_measure::resident_allocated }) {
                server_cache_destruction_yield value;
                value.category = row.category;
                value.measure  = measure;
                value.domain_known = true;
                value.domain = row.domain;
                value.value = llama_cache_acct_value::measured(
                    measure == llama_cache_acct_measure::logical_payload
                        ? row.logical_payload : row.resident_allocated);
                request.add_yield(value);
            }
        }
    } else {
        // One bounded unavailable pair preserves fail-closed lease behavior
        // without turning a large logical checkpoint ring into request
        // overflow.
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated }) {
            server_cache_destruction_yield value;
            value.measure = measure;
            request.add_yield(value);
        }
    }
    return server_cache_retention_admit(cache.destruction_obs, request);
}

static server_cache_destruction_admission
server_prompt_cache_observe_drop_pair(
        server_prompt_cache & cache,
        const server_prompt_cache_state & first,
        const server_prompt_cache_state & second,
        server_cache_destruction_reason reason,
        const llama_cache_acct_release_set_preview & preview) noexcept {
    if (!cache.destruction_obs) {
        return {};
    }
    server_cache_destruction_request request;
    request.cls = server_cache_destruction_class::host_artifact_drop;
    request.reason = reason;
    for (const auto * state : { &first, &second }) {
        request.add_target(
            server_cache_destruction_target_kind::host_artifact, -1,
            cache.retention_obs ? cache.retention_obs->artifact_id(
                server_retention_instance_key::for_host_entry(state))
                : llama_cache_acct_artifact_id {});
    }
    for (const auto & row : preview.yield_rows) {
        for (const auto measure : {
                llama_cache_acct_measure::logical_payload,
                llama_cache_acct_measure::resident_allocated }) {
            server_cache_destruction_yield value;
            value.category = row.category;
            value.measure = measure;
            value.domain_known = true;
            value.domain = row.domain;
            value.value = llama_cache_acct_value::measured(
                measure == llama_cache_acct_measure::logical_payload
                    ? row.logical_payload : row.resident_allocated);
            request.add_yield(value);
        }
    }
    return server_cache_retention_admit(cache.destruction_obs, request);
}

struct server_prompt_cache_retirement_manifest {
    server_retention_instance_key host;
    std::vector<server_retention_instance_key> checkpoints;
};

static bool server_prompt_cache_capture_retirement(
        server_prompt_cache & cache,
        server_prompt_cache::iterator it,
        server_prompt_cache_retirement_manifest & manifest) noexcept {
    try {
        manifest = {};
        if (!cache.retention_obs) {
            return true;
        }
        manifest.host =
            server_retention_instance_key::for_host_entry(&*it);
        manifest.checkpoints.reserve(it->prompt.checkpoints.size());
        for (auto & checkpoint : it->prompt.checkpoints) {
            manifest.checkpoints.push_back(
                server_retention_instance_key::for_checkpoint(
                    -1, &checkpoint));
        }
        return true;
    } catch (...) {
        manifest = {};
        return false;
    }
}

static void server_prompt_cache_retire_manifest(
        server_prompt_cache & cache,
        const server_prompt_cache_retirement_manifest & manifest) noexcept {
    if (!cache.retention_obs) {
        return;
    }
    cache.retention_obs->retire(manifest.host);
    for (const auto & checkpoint : manifest.checkpoints) {
        cache.retention_obs->retire(checkpoint);
    }
}

static void server_prompt_cache_retire_entry(
        server_prompt_cache & cache,
        server_prompt_cache::iterator it) noexcept {
    // Retention retirement is a post-capability finalizer on authoritative
    // paths because it releases the sidecar provenance op. The legacy wrapper
    // invokes it at its historical pre-erase position.
    if (!cache.retention_obs) {
        return;
    }
    cache.retention_obs->retire(
        server_retention_instance_key::for_host_entry(&*it));
    for (auto & checkpoint : it->prompt.checkpoints) {
        cache.retention_obs->retire(
            server_retention_instance_key::for_checkpoint(
                -1, &checkpoint));
    }
}

static server_prompt_cache::iterator server_prompt_cache_destroy_entry_impl(
        server_prompt_cache & cache,
        server_prompt_cache::iterator it) {
    GGML_ASSERT(it->recovery_pins == 0);
    return cache.states.erase(it);
}

server_prompt_cache::iterator server_prompt_cache::destroy_entry(
        iterator it,
        server_cache_destruction_reason reason) {
    return destroy_entry_impl(it, reason, states.end());
}

using server_cache_checkpoint_iterator =
    server_cache_checkpoint_authority_context::checkpoint_iterator;

void server_cache_checkpoint_ring_changed(
        server_cache_checkpoint_authority_context & context) noexcept {
    context.attempts.ring_changed();
    context.seam_heuristic = nullptr;
    context.thinning_refusal =
        common_cache_plan_destruction_reason::none;
    context.floor_refusal =
        common_cache_plan_destruction_reason::mandatory_anchor;
}

bool server_cache_checkpoint_thinning_attempt_begin(
        server_cache_checkpoint_authority_context & context,
        bool capacity_mode) noexcept {
    return context.attempts.begin(
        capacity_mode
            ? server_cache_checkpoint_attempt_lane::capacity_thinning
            : server_cache_checkpoint_attempt_lane::optional_thinning);
}

bool server_cache_checkpoint_refusal_state_changed(
        server_cache_checkpoint_authority_context & context,
        common_cache_plan_destruction_reason reason,
        bool publication_skip) noexcept {
    return context.attempts.refusal_changed(reason, publication_skip);
}

server_cache_destruction_admission server_cache_checkpoint_observe_drop(
        const server_cache_checkpoint_authority_context & context,
        server_cache_destruction_reason reason,
        llama_cache_acct_artifact_id artifact) noexcept {
    server_cache_destruction_request request;
    request.cls = server_cache_destruction_class::checkpoint_drop;
    request.reason = reason;
    request.add_target(
        server_cache_destruction_target_kind::checkpoint_ring,
        context.slot_id, artifact);
    request.add_yield(
        llama_cache_acct_category::checkpoint_state_payload);
    return server_cache_retention_admit(context.destruction, request);
}

namespace {

bool build_checkpoint_destruction_artifact(
        const server_cache_checkpoint_authority_context & context,
        server_cache_checkpoint_iterator checkpoint,
        server_cache_destruction_artifact & out) noexcept {
    out = {};
    try {
        if (!context.retention || !context.leases ||
            checkpoint == context.checkpoints.end()) {
            return false;
        }
        const auto key = server_retention_instance_key::for_checkpoint(
            context.slot_id, &*checkpoint);
        server_retention_checkpoint_inventory inventory;
        server_retention_candidate catalog;
        if (!context.retention->checkpoint_inventory(key, inventory) ||
            !inventory.identity_known || !inventory.release_owned ||
            !context.retention->candidate_for_instance(key, catalog) ||
            catalog.artifact_id.v == 0 ||
            catalog.record.kind !=
                common_retention_artifact_kind::checkpoint ||
            catalog.release_ops.empty()) {
            return false;
        }
        out.candidate.artifact_id = catalog.artifact_id;
        out.candidate.record = catalog.record;
        out.candidate.lineage = catalog.lineage;
        out.candidate.availability = catalog.avail;
        out.candidate.release_ops = catalog.release_ops;
        out.candidate.identity_known = true;
        out.candidate.lease = inventory.lease;
        out.kind = common_retention_artifact_kind::checkpoint;
        out.owner_slot = context.slot_id;
        out.pool = catalog.record.stamp.pool;
        out.mandatory_anchor =
            catalog.record.stamp.mandatory_anchor;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

void emit_checkpoint_destruction(
        const server_cache_checkpoint_authority_context & context,
        const common_cache_plan_destruction_receipt & receipt,
        uint64_t projected_bytes,
        uint64_t price_us,
        uint32_t weight_milli,
        uint32_t ordinal) noexcept {
    if (!context.debug_observability) {
        return;
    }
    try {
        json payload = server_json_from_ordered(server_cache_destruction_receipt_json(
            receipt, projected_bytes, "checkpoint_drop"));
        payload["price_us"] = price_us;
        payload["retention_weight_milli"] = weight_milli;
        payload["rank_ordinal"] = ordinal;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n",
                payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb checkpoint ownership.
    }
}

bool checkpoint_drop_certified(
        server_cache_checkpoint_authority_context & context,
        server_cache_checkpoint_iterator victim,
        server_cache_checkpoint_iterator recovery,
        server_cache_destruction_reason reason,
        uint64_t price_us,
        uint32_t weight_milli,
        uint32_t ordinal,
        server_cache_checkpoint_iterator & next) noexcept {
    if (!context.authority || !context.retention || !context.destruction ||
        victim == context.checkpoints.end() ||
        recovery == context.checkpoints.end() || victim == recovery) {
        return false;
    }
    auto & authority = *context.authority;
    const uint64_t sequence = ++authority.destruction_quote_sequence;
    const auto refuse = [&](common_cache_plan_destruction_receipt * existing,
                            common_cache_plan_destruction_reason why) {
        context.thinning_refusal = why;
        if (!server_cache_checkpoint_refusal_state_changed(context, why)) {
            return;
        }
        common_cache_plan_destruction_receipt receipt = existing
            ? std::move(*existing)
            : common_cache_plan_destruction_receipt{};
        receipt.state = common_cache_plan_destruction_state::refused;
        receipt.reason = why;
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::checkpoint_member_drop);
        receipt.admission_sequence = sequence;
        authority.observe_host_destruction(receipt, true);
        context.destruction->note_checkpoint_thin_refused();
        emit_checkpoint_destruction(context,
            receipt, 0, price_us, weight_milli, ordinal);
    };

    server_cache_destruction_artifact victim_artifact;
    server_cache_destruction_artifact recovery_artifact;
    if (!build_checkpoint_destruction_artifact(context,
            victim, victim_artifact) ||
        !build_checkpoint_destruction_artifact(context,
            recovery, recovery_artifact)) {
        refuse(nullptr,
               common_cache_plan_destruction_reason::manifest_incomplete);
        return false;
    }
    const auto recovery_key =
        server_retention_instance_key::for_checkpoint(context.slot_id, &*recovery);
    auto pin = context.retention->acquire_recovery_pin(recovery_key);
    if (!pin.valid() || !pin.binds_exact(
            recovery_artifact.candidate.artifact_id,
            recovery_artifact.candidate.release_ops)) {
        refuse(nullptr,
               common_cache_plan_destruction_reason::recovery_unavailable);
        return false;
    }

    const auto preview = [&](const auto & ops, uint64_t serial,
                             auto & released) {
        return authority.ledger.preview_release_set(
            ops, serial, released);
    };
    const auto project = [&](const auto & released, auto & domains) {
        return authority.project_release(released, domains);
    };
    const uint64_t accounting_serial = authority.ledger.serial();
    auto quote = server_cache_destruction_quote_single_artifact(
        victim_artifact,
        common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::checkpoint_member_drop),
        accounting_serial, sequence,
        preview, project);
    if (quote.receipt.state !=
            common_cache_plan_destruction_state::quoted) {
        const auto why = quote.receipt.reason;
        refuse(&quote.receipt, why);
        return false;
    }
    authority.observe_host_destruction(quote.receipt, false);
    std::vector<server_cache_destruction_artifact> current;
    try {
        current.push_back(std::move(victim_artifact));
    } catch (...) {
        refuse(&quote.receipt,
               common_cache_plan_destruction_reason::internal_fault);
        return false;
    }
    auto prepared = server_cache_prepare_release_set(
        quote, current, authority.ledger, authority.ledger.serial(),
        project, std::move(pin));
    if (prepared.status !=
            server_cache_prepare_release_status::prepared) {
        refuse(&quote.receipt, prepared.reason);
        return false;
    }
    uint64_t projected_bytes = 0;
    for (const auto & row : quote.projected_domains) {
        if (row.projected_release_bytes.state !=
                llama_cache_acct_known::known ||
            row.projected_release_bytes.value >
                std::numeric_limits<uint64_t>::max() - projected_bytes) {
            refuse(&quote.receipt,
                   common_cache_plan_destruction_reason::
                       accounting_unavailable);
            return false;
        }
        projected_bytes += row.projected_release_bytes.value;
    }
    quote.receipt.displaced_fate =
        common_cache_plan_displaced_fate::exact_replay_recipe;
    quote.receipt.recovery_citation =
        common_cache_plan_recovery_citation::resolved;
    quote.receipt.recovery_source_artifact_id =
        recovery_artifact.candidate.artifact_id;
    quote.receipt.recovery_source_manifest_digest =
        server_cache_destruction_recovery_source_digest(
            recovery_artifact.candidate.artifact_id,
            recovery_artifact.candidate.release_ops);
    quote.receipt.state =
        common_cache_plan_destruction_state::certified;
    authority.observe_host_destruction(quote.receipt, true);
    emit_checkpoint_destruction(context,
        quote.receipt, projected_bytes,
        price_us, weight_milli, ordinal);

    const auto victim_key =
        server_retention_instance_key::for_checkpoint(context.slot_id, &*victim);
    const auto admission = server_cache_checkpoint_observe_drop(context,
        reason, current.front().candidate.artifact_id);
    const std::thread::id scheduler_owner = std::this_thread::get_id();
    GGML_ASSERT(context.raw_owner && context.raw_drop);
    next = context.raw_drop(
        context.raw_owner, victim, std::next(victim));
    // The typed raw_drop adapter is pinned to the slot's X-macro _impl door;
    // that door only advances the ring latch and erases this list node. The
    // node destructor frees checkpoint-owned vectors and shadow metadata and
    // cannot write C, so no ledger producer can interleave before commit.
    GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
    server_cache_recovery_pin retained_pin;
    const auto committed = prepared.capability.commit(retained_pin);
    GGML_ASSERT(committed ==
                common_cache_plan_destruction_reason::none);
    context.retention->retire_after_committed_release(victim_key);
    quote.receipt.state =
        common_cache_plan_destruction_state::executed;
    quote.receipt.actual_accounting_serial =
        authority.ledger.serial();
    authority.observe_host_destruction(quote.receipt, false);
    emit_checkpoint_destruction(context,
        quote.receipt, projected_bytes,
        price_us, weight_milli, ordinal);
    context.destruction->note_checkpoint_thin_executed(
        admission.sequence, projected_bytes);
    return true;
}

} // namespace

bool server_cache_checkpoint_thin_priced(
        server_cache_checkpoint_authority_context & context,
        int checkpoint_task_id,
        uint64_t max_replay_tokens,
        const common_prompt_checkpoint * seam_heuristic,
        bool capacity_mode,
        bool attempt_claimed) noexcept {
    if (!context.authority || !context.retention || !context.leases ||
        context.checkpoints.size() < 2) {
        return false;
    }
    if (!attempt_claimed &&
        !server_cache_checkpoint_thinning_attempt_begin(context, capacity_mode)) {
        return false;
    }
    context.thinning_refusal =
        common_cache_plan_destruction_reason::none;
    context.leases->lifecycle_point();
    struct local_candidate {
        server_cache_checkpoint_iterator victim;
        server_cache_checkpoint_iterator recovery;
        server_cache_checkpoint_trade_input price;
    };
    struct member_inventory {
        server_cache_checkpoint_iterator member;
        server_retention_checkpoint_inventory catalog;
        bool found = false;
    };
    std::vector<local_candidate> local;
    std::vector<server_cache_checkpoint_trade_input> prices;
    try {
        local.reserve(context.checkpoints.size());
        prices.reserve(context.checkpoints.size());
        std::vector<member_inventory> inventory;
        inventory.reserve(context.checkpoints.size());
        for (auto it = context.checkpoints.begin();
             it != context.checkpoints.end(); ++it) {
            member_inventory member;
            member.member = it;
            member.found = context.retention->checkpoint_inventory(
                server_retention_instance_key::for_checkpoint(context.slot_id, &*it),
                member.catalog);
            inventory.push_back(std::move(member));
        }

        size_t previous_index = 0;
        for (size_t index = 1; index < inventory.size(); ++index) {
            auto it = inventory[index].member;
            auto previous = inventory[previous_index].member;
            const bool close = it->n_tokens >= previous->n_tokens &&
                uint64_t(it->n_tokens - previous->n_tokens) <=
                    max_replay_tokens;
            if ((!capacity_mode && !close) ||
                it->id_task == checkpoint_task_id) {
                previous_index = index;
                continue;
            }

            local_candidate candidate;
            candidate.victim = it;
            candidate.recovery = previous;
            candidate.price.ordinal = uint32_t(index);
            candidate.price.recovery_ordinal =
                uint32_t(previous_index);
            candidate.price.payload_bytes = it->size();
            candidate.price.replay_tokens =
                it->n_tokens >= previous->n_tokens
                    ? uint64_t(it->n_tokens - previous->n_tokens)
                    : UINT64_MAX;
            candidate.price.seam_heuristic_protected =
                seam_heuristic == &*it;
            const bool same_replay_lineage =
                server_cache_checkpoint_bounded_replay(
                    *previous, *it, max_replay_tokens);
            candidate.price.recovery_available =
                same_replay_lineage &&
                inventory[previous_index].found &&
                inventory[previous_index].catalog.identity_known &&
                inventory[previous_index].catalog.release_owned;
            const auto & victim_catalog = inventory[index].catalog;
            if (inventory[index].found &&
                victim_catalog.identity_known &&
                victim_catalog.release_owned) {
                candidate.price.artifact =
                    victim_catalog.artifact_id;
                candidate.price.stable_id =
                    victim_catalog.stable_id;
                candidate.price.identity_known = true;
                candidate.price.mandatory_anchor =
                    victim_catalog.mandatory_anchor ||
                    victim_catalog.recovery_pinned;
                candidate.price.hard_leased = server_cache_lease_is_hard(
                    victim_catalog.lease);
                uint32_t weight = 0;
                GGML_ASSERT(server_cache_retention_weight_milli(
                    victim_catalog.lease.cls ==
                        server_cache_lease_class::soft,
                    context.main_family,
                    SERVER_CACHE_HOST_WEIGHT_SCALE, weight));
                candidate.price.weight_milli = weight;
            }
            local.push_back(std::move(candidate));
            prices.push_back(local.back().price);
            if (!close) {
                previous_index = index;
            }
        }
    } catch (...) {
        return false;
    }
    if (local.empty()) {
        return false;
    }

    const auto * calib = common_cache_plan_calib_find(
        context.authority->calibration_profile);
    while (!local.empty()) {
        const auto plan = server_cache_plan_checkpoint_thinning(
            prices, calib);
        if (!plan.selected) {
            context.thinning_refusal = plan.reason;
            if (!server_cache_checkpoint_refusal_state_changed(context, plan.reason)) {
                return false;
            }
            common_cache_plan_destruction_receipt receipt;
            receipt.state = common_cache_plan_destruction_state::refused;
            receipt.reason = plan.reason;
            receipt.effects = common_cache_plan_destruction_effect_bit(
                common_cache_plan_destruction_effect::
                    checkpoint_member_drop);
            receipt.admission_sequence =
                ++context.authority->destruction_quote_sequence;
            context.authority->observe_host_destruction(receipt, true);
            if (context.destruction) {
                context.destruction->note_checkpoint_thin_refused();
                if (plan.protection !=
                        server_cache_checkpoint_protection::none) {
                    switch (plan.protection) {
                        case server_cache_checkpoint_protection::
                                 seam_heuristic:
                            context.destruction->
                                note_checkpoint_thin_heuristic_refused();
                            break;
                        case server_cache_checkpoint_protection::
                                 mandatory_anchor:
                            context.destruction->
                                note_checkpoint_thin_mandatory_refused();
                            break;
                        case server_cache_checkpoint_protection::
                                 hard_lease:
                            context.destruction->
                                note_checkpoint_thin_hard_lease_refused();
                            break;
                        case server_cache_checkpoint_protection::none:
                        case server_cache_checkpoint_protection::_count:
                            break;
                    }
                }
            }
            emit_checkpoint_destruction(context,
                receipt, 0, 0,
                SERVER_CACHE_HOST_WEIGHT_SCALE, UINT32_MAX);
            return false;
        }
        const auto chosen = std::find_if(
            local.begin(), local.end(), [&](const auto & candidate) {
                return candidate.price.ordinal == plan.ordinal;
            });
        if (chosen == local.end()) {
            return false;
        }
        const auto chosen_index = size_t(chosen - local.begin());
        server_cache_checkpoint_iterator next;
        if (checkpoint_drop_certified(context,
                chosen->victim, chosen->recovery,
                capacity_mode
                    ? server_cache_destruction_reason::checkpoint_capacity
                    : server_cache_destruction_reason::checkpoint_thin,
                plan.price_us, plan.weight_milli,
                plan.ordinal, next)) {
            return true;
        }
        local.erase(chosen);
        prices.erase(prices.begin() + chosen_index);
    }
    return false;
}

bool server_cache_checkpoint_capacity_floor(
        server_cache_checkpoint_authority_context & context,
        int checkpoint_task_id,
        const common_prompt_checkpoint * seam_heuristic,
        server_cache_checkpoint_iterator & victim,
        common_cache_plan_destruction_reason & refusal) noexcept {
    victim = context.checkpoints.end();
    refusal = context.floor_refusal;
    if (!context.attempts.begin(
            server_cache_checkpoint_attempt_lane::capacity_floor)) {
        return false;
    }
    refusal = common_cache_plan_destruction_reason::mandatory_anchor;
    if (context.leases) {
        context.leases->lifecycle_point();
    }
    std::vector<server_cache_checkpoint_floor_input> inputs;
    std::vector<server_cache_checkpoint_iterator> members;
    try {
        inputs.reserve(context.checkpoints.size());
        members.reserve(context.checkpoints.size());
        uint32_t ordinal = 0;
        for (auto it = context.checkpoints.begin();
             it != context.checkpoints.end(); ++it, ++ordinal) {
            server_cache_checkpoint_floor_input input;
            input.ordinal = ordinal;
            const auto key =
                server_retention_instance_key::for_checkpoint(context.slot_id, &*it);
            server_retention_checkpoint_inventory catalog;
            const bool catalog_found = context.retention &&
                context.retention->checkpoint_inventory(key, catalog);
            input.recovery_pinned = catalog_found &&
                catalog.recovery_pinned;
            if (it->id_task == checkpoint_task_id ||
                input.recovery_pinned) {
                input.protection =
                    server_cache_checkpoint_protection::mandatory_anchor;
            } else if (seam_heuristic == &*it) {
                input.protection =
                    server_cache_checkpoint_protection::seam_heuristic;
            }
            if (catalog_found) {
                if (catalog.mandatory_anchor) {
                    input.protection =
                        server_cache_checkpoint_protection::
                            mandatory_anchor;
                }
                if (catalog.identity_known &&
                    server_cache_lease_is_hard(catalog.lease)) {
                    input.protection =
                        server_cache_checkpoint_protection::hard_lease;
                }
            }
            inputs.push_back(input);
            members.push_back(it);
        }
    } catch (...) {
        refusal = common_cache_plan_destruction_reason::internal_fault;
        context.floor_refusal = refusal;
        return false;
    }
    const auto plan = server_cache_plan_checkpoint_capacity_floor(inputs);
    refusal = plan.reason;
    context.floor_refusal = refusal;
    if (!plan.selected || plan.ordinal >= members.size()) {
        return false;
    }
    victim = members[plan.ordinal];
    return true;
}

void server_cache_checkpoint_publication_skipped(
        server_cache_checkpoint_authority_context & context,
        common_cache_plan_destruction_reason reason) noexcept {
    if (!context.authority ||
        !server_cache_checkpoint_refusal_state_changed(context, reason, true)) {
        return;
    }
    common_cache_plan_destruction_receipt receipt;
    receipt.state = common_cache_plan_destruction_state::refused;
    receipt.reason = reason;
    receipt.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::checkpoint_member_drop);
    receipt.admission_sequence =
        ++context.authority->destruction_quote_sequence;
    context.authority->observe_host_destruction(receipt, true);
    if (context.destruction) {
        context.destruction->note_checkpoint_publication_skip();
    }
    emit_checkpoint_destruction(context,
        receipt, 0, 0, SERVER_CACHE_HOST_WEIGHT_SCALE, UINT32_MAX);
}


namespace {

bool checkpoint_payload_equal(
        const common_prompt_checkpoint & a,
        const common_prompt_checkpoint & b) noexcept {
    return a.n_tokens == b.n_tokens &&
           a.id_task == b.id_task &&
           a.pos_min == b.pos_min &&
           a.pos_max == b.pos_max &&
           a.checkpoint_epoch == b.checkpoint_epoch &&
           a.checkpoint_epoch_swa == b.checkpoint_epoch_swa &&
           a.computation_frontier == b.computation_frontier &&
           a.data_tgt == b.data_tgt &&
           a.data_dft == b.data_dft &&
           a.data_dft_full_sequence == b.data_dft_full_sequence &&
           a.accel.ring == b.accel.ring &&
           a.accel.spec == b.accel.spec;
}

bool build_host_retention_artifact_uninspected(
        server_prompt_cache & cache,
        server_prompt_cache_state & state,
        server_cache_destruction_artifact & out,
        server_cache_lease_identity & identity) noexcept {
    out = {};
    identity = {};
    try {
        if (!cache.retention_obs || !cache.lease_obs ||
            !cache.lease_execution_identity) {
            return false;
        }
        server_retention_candidate catalog;
        const auto key = server_retention_instance_key::for_host_entry(&state);
        if (!cache.retention_obs->candidate_for_instance(key, catalog) ||
            catalog.artifact_id.v == 0 ||
            catalog.record.kind != common_retention_artifact_kind::host_entry) {
            return false;
        }
        if (!server_cache_lease_build_identity(
                *cache.lease_execution_identity,
                state.adapter_config_key,
                state.prompt.tokens,
                state.prompt.n_tokens(),
                identity)) {
            return false;
        }
        out.candidate.artifact_id = catalog.artifact_id;
        out.candidate.record = catalog.record;
        out.candidate.lineage = catalog.lineage;
        out.candidate.availability = catalog.avail;
        out.candidate.identity_known = true;
        out.kind = common_retention_artifact_kind::host_entry;
        out.host_source_id = state.cache_plan_source_id;
        out.pool = catalog.record.stamp.pool;
        out.mandatory_anchor = catalog.record.stamp.mandatory_anchor;
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

bool build_host_retention_artifact(
        server_prompt_cache & cache,
        server_prompt_cache_state & state,
        server_cache_destruction_artifact & out) noexcept {
    server_cache_lease_identity identity;
    if (!build_host_retention_artifact_uninspected(
            cache, state, out, identity)) {
        return false;
    }
    out.candidate.lease = cache.lease_obs->inspect(
        out.candidate.artifact_id, identity);
    return true;
}

bool build_host_destruction_artifact(
        server_prompt_cache & cache,
        server_prompt_cache_state & state,
        server_cache_destruction_artifact & out) noexcept {
    if (!state.accounting_complete ||
        !build_host_retention_artifact(cache, state, out)) {
        return false;
    }
    try {
        out.candidate.release_ops.reserve(state.release_ops().size());
        for (const auto op : state.release_ops()) {
            if (!op) {
                out = {};
                return false;
            }
            out.candidate.release_ops.push_back(op);
        }
        return true;
    } catch (...) {
        out = {};
        return false;
    }
}

void release_host_recovery_pin(void * context) noexcept {
    auto * state = static_cast<server_prompt_cache_state *>(context);
    GGML_ASSERT(state && state->recovery_pins > 0);
    state->recovery_pins--;
}

bool build_host_recovery_source(
        server_prompt_cache & cache,
        server_prompt_cache_state & state,
        std::vector<llama_cache_acct_artifact_id> & artifacts,
        std::vector<llama_cache_acct_op_id> & ops) noexcept {
    artifacts.clear();
    ops.clear();
    try {
        if (!cache.retention_obs) {
            return false;
        }
        server_retention_candidate catalog;
        const auto key = server_retention_instance_key::for_host_entry(&state);
        if (!cache.retention_obs->candidate_for_instance(key, catalog) ||
            catalog.artifact_id.v == 0 ||
            catalog.record.kind != common_retention_artifact_kind::host_entry ||
            catalog.avail != server_retention_candidate_availability::available) {
            return false;
        }
        artifacts.push_back(catalog.artifact_id);
        if (!state.accounting_complete) {
            return false;
        }
        ops.reserve(state.release_ops().size());
        for (const auto op : state.release_ops()) {
            if (!op) {
                return false;
            }
            ops.push_back(op);
        }
        std::sort(ops.begin(), ops.end());
        ops.erase(std::unique(ops.begin(), ops.end()), ops.end());
        return !ops.empty();
    } catch (...) {
        artifacts.clear();
        ops.clear();
        return false;
    }
}

server_cache_recovery_pin acquire_host_recovery_pin(
        server_prompt_cache_state & state,
        std::vector<llama_cache_acct_artifact_id> artifacts,
        std::vector<llama_cache_acct_op_id> ops) noexcept {
    if (state.recovery_pins == std::numeric_limits<uint32_t>::max()) {
        return {};
    }
    state.recovery_pins++;
    auto pin = server_cache_recovery_pin::acquire(
        &state,
        release_host_recovery_pin,
        std::move(artifacts),
        std::move(ops));
    if (!pin.valid()) {
        state.recovery_pins--;
    }
    return pin;
}

struct host_trade_ranking {
    bool price_known = false;
    uint64_t price_us = 0;
    uint32_t weight_milli = SERVER_CACHE_HOST_WEIGHT_SCALE;
    uint32_t ordinal = 0;
    int32_t source_id = -1;
    llama_cache_acct_artifact_id artifact_id;
    bool zero_destruction_known = false;
    bool zero_destruction = false;
    bool zero_destruction_tie_break = false;
    common_cache_family_role family_role = common_cache_family_role::_count;
    common_cache_plan_payload_kind payload_kind =
        common_cache_plan_payload_kind::unavailable;
};

struct host_destruction_certification {
    bool ready = false;
    server_cache_prepared_release_capability capability;
    server_cache_recovery_pin pin;
    common_cache_plan_destruction_quote quote;
    server_prompt_cache_retirement_manifest retirement;
    uint64_t projected_bytes = 0;
};

void server_prompt_cache_observe_host_destruction(
        server_prompt_cache & cache,
        const common_cache_plan_destruction_receipt & receipt,
        bool observe_classification,
        uint64_t projected_bytes,
        const host_trade_ranking * ranking = nullptr) noexcept {
    cache.publish_authority->observe_host_destruction(
        receipt, observe_classification);
    if (!cache.debug_observability) {
        return;
    }
    try {
        const json unavailable = common_cache_acct_known_name(
            llama_cache_acct_known::unavailable);
        json payload = server_json_from_ordered(server_cache_destruction_receipt_json(
            receipt, projected_bytes));
        payload["price_us"] = ranking && ranking->price_known
            ? json(ranking->price_us) : unavailable;
        payload["retention_weight_milli"] = ranking
            ? json(ranking->weight_milli) : unavailable;
        payload["rank_ordinal"] = ranking
            ? json(ranking->ordinal) : unavailable;
        payload["victim_source_id"] = ranking && ranking->source_id >= 0
            ? json(ranking->source_id) : unavailable;
        payload["victim_artifact_id"] = ranking &&
                ranking->artifact_id.v != 0
            ? json(ranking->artifact_id.v) : unavailable;
        payload["zero_destruction"] = ranking &&
                ranking->zero_destruction_known
            ? json(ranking->zero_destruction) : unavailable;
        payload["zero_destruction_tie_break"] = ranking
            ? json(ranking->zero_destruction_tie_break) : json(false);
        payload["declared_family_role"] = ranking &&
                ranking->family_role < common_cache_family_role::_count
            ? json(ranking->family_role == common_cache_family_role::main
                ? "main" : ranking->family_role ==
                    common_cache_family_role::branch
                    ? "branch" : "background")
            : json(nullptr);
        payload["legacy_fallbacks"] = cache.destruction_obs
            ? cache.destruction_obs->host_trade_legacy_fallbacks : uint64_t(0);
        payload["retention_capacity_executed"] = cache.destruction_obs
            ? cache.destruction_obs->host_trade_retention_capacity_executed : uint64_t(0);
        payload["publication_skips"] = cache.destruction_obs
            ? cache.destruction_obs->host_trade_publication_skips : uint64_t(0);
        cache.debug_destruction_emissions++;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n", payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb maintenance or destruction.
    }
}

llama_cache_acct_artifact_id host_entry_artifact_id(
        const server_prompt_cache & cache,
        const server_prompt_cache_state & state) noexcept {
    return cache.retention_obs
        ? cache.retention_obs->artifact_id(
              server_retention_instance_key::for_host_entry(&state))
        : llama_cache_acct_artifact_id {};
}

void emit_recovery_pin_excluded(
        server_prompt_cache & cache,
        const server_prompt_cache_state & state) noexcept {
    if (!cache.debug_observability) {
        return;
    }
    try {
        const auto artifact = host_entry_artifact_id(cache, state);
        common_cache_plan_destruction_receipt receipt;
        receipt.payload_kind =
            server_cache_plan_payload_kind(state.payload.kind());
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
        json payload = server_json_from_ordered(
            server_cache_destruction_receipt_json(receipt, 0));
        payload["evidence_event"] = "recovery_pin_excluded";
        payload["recovery_pin_excluded"] = {
            { "artifact_id", artifact.v },
            { "source_id", state.cache_plan_source_id },
            { "pin_count", state.recovery_pins },
        };
        payload["floor_outcome"] = "pending";
        cache.debug_recovery_pin_exclusions++;
        cache.debug_last_recovery_pin_excluded = artifact;
        cache.debug_destruction_emissions++;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n", payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb victim selection or pressure.
    }
}

void emit_host_pressure_floor_outcome(
        server_prompt_cache & cache,
        const char * outcome,
        llama_cache_acct_artifact_id victim_artifact,
        int32_t victim_source_id) noexcept {
    if (!cache.debug_observability) {
        return;
    }
    try {
        common_cache_plan_destruction_receipt receipt;
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
        json payload = server_json_from_ordered(
            server_cache_destruction_receipt_json(receipt, 0));
        payload["evidence_event"] = "floor_outcome";
        payload["recovery_pin_excluded"] = nullptr;
        payload["floor_outcome"] = outcome;
        payload["floor_victim_artifact_id"] = victim_artifact.v != 0
            ? json(victim_artifact.v)
            : json(common_cache_acct_known_name(
                  llama_cache_acct_known::unavailable));
        payload["floor_victim_source_id"] = victim_source_id >= 0
            ? json(victim_source_id)
            : json(common_cache_acct_known_name(
                  llama_cache_acct_known::unavailable));
        cache.debug_host_pressure_floor_outcomes++;
        cache.debug_destruction_emissions++;
        SRV_INF("CACHE_HOST_DESTRUCTION %s\n", payload.dump().c_str());
    } catch (...) {
        // Debug evidence must never perturb the already-chosen terminal.
    }
}

host_destruction_certification certify_host_destruction(
        server_prompt_cache & cache,
        server_prompt_cache::iterator victim_state,
        server_prompt_cache::iterator survivor_state,
        uint64_t admission_sequence,
        bool allow_authorized_recovery,
        bool count_redundant_refusals,
        const host_trade_ranking * ranking = nullptr) noexcept {
    host_destruction_certification out;
    auto & authority = *cache.publish_authority;

    const auto refuse_initial = [&](common_cache_plan_destruction_reason reason) {
        out.quote = {};
        auto & receipt = out.quote.receipt;
        receipt.effects = common_cache_plan_destruction_effect_bit(
            common_cache_plan_destruction_effect::
                different_host_source_consumption);
        receipt.state = common_cache_plan_destruction_state::refused;
        receipt.reason = reason;
        receipt.admission_sequence = admission_sequence;
        server_prompt_cache_observe_host_destruction(
            cache, receipt, true, 0, ranking);
        if (cache.destruction_obs && count_redundant_refusals) {
            cache.destruction_obs->note_redundant_host_refused(
                admission_sequence);
        }
    };
    const auto refuse_certified = [&](common_cache_plan_destruction_reason reason) {
        auto & receipt = out.quote.receipt;
        receipt.state = common_cache_plan_destruction_state::refused;
        receipt.reason = reason;
        server_prompt_cache_observe_host_destruction(
            cache, receipt, true, 0, ranking);
        if (cache.destruction_obs && count_redundant_refusals) {
            cache.destruction_obs->note_redundant_host_refused(
                admission_sequence);
        }
    };

    server_cache_destruction_artifact victim;
    std::vector<llama_cache_acct_artifact_id> recovery_ids;
    std::vector<llama_cache_acct_op_id> recovery_ops;
    llama_cache_acct_artifact_id recovery_artifact;
    common_cache_plan_displaced_fate recovery_fate =
        common_cache_plan_displaced_fate::unavailable;
    // recovery proof taxonomy: a named survivor that is not an exact three-payload
    // duplicate is recovery_unavailable even when the victim's artifact
    // manifest is independently incomplete. Redundancy is the outer proof.
    if (survivor_state != cache.states.end() &&
        !server_prompt_cache::exactly_redundant(
            *victim_state, *survivor_state)) {
        refuse_initial(
            common_cache_plan_destruction_reason::recovery_unavailable);
        return out;
    }
    if (!server_prompt_cache_capture_retirement(
            cache, victim_state, out.retirement)) {
        refuse_initial(common_cache_plan_destruction_reason::manifest_incomplete);
        return out;
    }
    if (!build_host_destruction_artifact(cache, *victim_state, victim)) {
        // Every host-consumption/redundancy effect needs its own by-host
        // catalog contribution. A partial displacement-only union is not
        // certifiable.
        refuse_initial(common_cache_plan_destruction_reason::manifest_incomplete);
        return out;
    }
    if (survivor_state != cache.states.end()) {
        if (!build_host_recovery_source(
                cache, *survivor_state, recovery_ids, recovery_ops)) {
            refuse_initial(
                common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }
        recovery_artifact = recovery_ids.front();
        out.pin = acquire_host_recovery_pin(
            *survivor_state, recovery_ids, recovery_ops);
        recovery_fate = common_cache_plan_displaced_fate::exact_duplicate;
    } else if (allow_authorized_recovery && authority.host_recovery) {
        server_cache_host_recovery_evidence evidence;
        if (!authority.host_recovery(
                authority.host_recovery_context, *victim_state, evidence) ||
            evidence.artifact.v == 0 || evidence.ops.empty() ||
            !evidence.pin.valid() ||
            !evidence.pin.binds_exact(evidence.artifact, evidence.ops) ||
            evidence.fate !=
                common_cache_plan_displaced_fate::retained_sealed_artifact) {
            refuse_initial(
                common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }
        recovery_artifact = evidence.artifact;
        recovery_ops = std::move(evidence.ops);
        try {
            recovery_ids.push_back(recovery_artifact);
        } catch (...) {
            refuse_initial(
                common_cache_plan_destruction_reason::internal_fault);
            return out;
        }
        out.pin = std::move(evidence.pin);
        recovery_fate = evidence.fate;
    } else {
        refuse_initial(common_cache_plan_destruction_reason::recovery_unavailable);
        return out;
    }

    try {
        if (!out.pin.valid()) {
            refuse_initial(common_cache_plan_destruction_reason::recovery_unavailable);
            return out;
        }

        const server_cache_destruction_preview_callback preview =
            [&](const auto & ops, uint64_t serial, auto & released) {
                return cache.acct->preview_release_set(ops, serial, released);
            };
        const server_cache_destruction_projection_callback project =
            [&](const auto & released, auto & domains) {
                return authority.project_release(released, domains);
            };
        const uint64_t accounting_serial = cache.acct->serial();
        out.quote = server_cache_destruction_quote_redundant_host(
            victim,
            accounting_serial,
            admission_sequence,
            preview,
            project);
        out.quote.receipt.payload_kind = ranking
            ? ranking->payload_kind
            : common_cache_plan_payload_kind::unavailable;
        server_prompt_cache_observe_host_destruction(
            cache,
            out.quote.receipt,
            out.quote.receipt.state !=
                common_cache_plan_destruction_state::quoted,
            0,
            ranking);
        if (out.quote.receipt.state !=
                common_cache_plan_destruction_state::quoted) {
            if (cache.destruction_obs && count_redundant_refusals) {
                cache.destruction_obs->note_redundant_host_refused(
                    admission_sequence);
            }
            return out;
        }

        std::vector<server_cache_destruction_artifact> current = {
            std::move(victim),
        };
        auto prepared = server_cache_prepare_release_set(
            out.quote,
            current,
            *cache.acct,
            cache.acct->serial(),
            project,
            std::move(out.pin));
        if (prepared.status !=
                server_cache_prepare_release_status::prepared) {
            refuse_certified(prepared.reason);
            return out;
        }
        if (!common_cache_plan_projected_release_bytes(
                out.quote.projected_domains, out.projected_bytes)) {
            refuse_certified(
                common_cache_plan_destruction_reason::accounting_unavailable);
            return out;
        }

        // The exact-duplicate fate and resolved citation become claims only
        // after every refusal conjunct, fresh effect check, and disjoint pin
        // has succeeded. Schema 6 retains the source identity after the pin
        // itself closes.
        server_cache_destruction_certify_receipt(
            out.quote.receipt, recovery_fate,
            recovery_artifact, recovery_ops);
        server_prompt_cache_observe_host_destruction(
            cache, out.quote.receipt, true, out.projected_bytes, ranking);
        out.capability = std::move(prepared.capability);
        out.ready = true;
        return out;
    } catch (...) {
        out.pin = {};
        if (out.quote.receipt.union_effect_digest.valid()) {
            refuse_certified(common_cache_plan_destruction_reason::internal_fault);
        } else {
            refuse_initial(common_cache_plan_destruction_reason::internal_fault);
        }
        return out;
    }
}

void commit_certified_host_destruction(
        server_prompt_cache & cache,
        host_destruction_certification & certified,
        const std::thread::id & scheduler_owner,
        const host_trade_ranking * ranking = nullptr) noexcept {
    GGML_ASSERT(certified.ready);
    GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
    const auto release_status =
        certified.capability.commit(certified.pin);
    GGML_ASSERT(release_status ==
                common_cache_plan_destruction_reason::none);
    server_prompt_cache_retire_manifest(cache, certified.retirement);
    certified.quote.receipt.state =
        common_cache_plan_destruction_state::executed;
    certified.quote.receipt.actual_accounting_serial =
        cache.acct->serial();
    server_prompt_cache_observe_host_destruction(
        cache,
        certified.quote.receipt,
        false,
        certified.projected_bytes,
        ranking);
    // The cited recovery source was held through accounting commit and
    // receipt publication; this local destruction dependency now closes.
    certified.pin = {};
}

struct host_trade_candidate {
    server_prompt_cache::iterator victim;
    server_prompt_cache::iterator recovery;
    host_trade_ranking ranking;
    uint64_t marginal_resident_bytes = 0;
    std::vector<llama_cache_acct_op_id> release_ops;
    bool marginal_resident_known = false;
    bool attempted = false;
    bool lease_known = false;
    bool main_family = false;
    bool soft_leased = false;
    bool hard_leased = false;
    bool mandatory_anchor = false;
    bool vbr = false;
    bool vbr_logical_alias = false;
    bool retirement_ready = false;
};

bool vbr_release_resident_bytes(
        const llama_cache_acct_release_set_preview & preview,
        uint64_t & bytes) noexcept {
    bytes = 0;
    if (preview.accounting_serial == 0) {
        return false;
    }
    for (const auto & row : preview.rows) {
        if (row.resident_allocated > UINT64_MAX - bytes) {
            bytes = 0;
            return false;
        }
        bytes += row.resident_allocated;
    }
    return true;
}

bool vbr_release_resident_bytes(
        const vbr_artifact_prepared_retire & prepared,
        uint64_t & bytes) noexcept {
    return prepared.ready() &&
           vbr_release_resident_bytes(prepared.preview(), bytes);
}

uint64_t server_prompt_cache_shadow_hash(uint64_t value) noexcept {
    value ^= value >> 30;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value *= UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

server_prompt_cache::iterator find_exact_host_recovery(
        server_prompt_cache & cache,
        server_prompt_cache::iterator victim) noexcept {
    for (auto it = cache.states.begin(); it != cache.states.end(); ++it) {
        if (it != victim && server_prompt_cache::exactly_redundant(
                *victim, *it)) {
            return it;
        }
    }
    return cache.states.end();
}

bool host_trade_price(
        server_prompt_cache & cache,
        server_prompt_cache::iterator victim,
        uint32_t ordinal,
        server_cache_destruction_reason reason,
        const common_cache_plan_calib * calib,
        host_trade_candidate & out,
        bool preview_vbr = true,
        const server_cache_destruction_artifact * prepared_artifact = nullptr) noexcept {
    out = {};
    out.victim = victim;
    out.ranking.ordinal = ordinal;
    out.ranking.source_id = victim->cache_plan_source_id;
    out.ranking.family_role = victim->cache_family.declared()
        ? victim->cache_family.role : common_cache_family_role::_count;
    out.main_family = victim->main_family;
    try {
        auto & authority = *cache.publish_authority;
        server_cache_destruction_artifact local_artifact;
        out.vbr = victim->payload.kind() ==
            server_prompt_cache_payload_kind::vbr_artifact;
        out.ranking.payload_kind =
            server_cache_plan_payload_kind(victim->payload.kind());
        if (!prepared_artifact && !(out.vbr
                ? build_host_retention_artifact(cache, *victim, local_artifact)
                : build_host_destruction_artifact(cache, *victim, local_artifact))) {
            return false;
        }
        const auto & artifact = prepared_artifact
            ? *prepared_artifact : local_artifact;
        out.ranking.artifact_id = artifact.candidate.artifact_id;
        out.mandatory_anchor = artifact.mandatory_anchor;
        if (artifact.candidate.lease.state !=
                server_cache_lease_eval_state::known) {
            return false;
        }
        out.lease_known = true;
        out.soft_leased = artifact.candidate.lease.cls ==
            server_cache_lease_class::soft;
        out.hard_leased = server_cache_lease_is_hard(
            artifact.candidate.lease);
        if (out.vbr) {
            out.vbr_logical_alias =
                victim->payload.vbr_logical_erase_only();
            if (out.vbr_logical_alias && !out.hard_leased) {
                out.retirement_ready = true;
                return true;
            }
            // Ordinary pressure owns only compact-current entries. A quality
            // anchor has a separate byte pool and parent-value competition;
            // it must never be hidden in compact victim bytes.
            if (out.hard_leased || !cache.acct ||
                victim->payload.vbr_has_quality_anchor()) {
                return false;
            }
            if (!victim->payload.vbr_retirement_exclusive()) {
                return false;
            }
            if (preview_vbr && reason ==
                    server_cache_destruction_reason::host_capacity) {
                llama_cache_acct_release_set_preview preview;
                if (!victim->payload.preview_vbr_retire(
                        cache.acct->serial(), preview) ||
                    !vbr_release_resident_bytes(
                        preview, out.marginal_resident_bytes)) {
                    return false;
                }
                out.marginal_resident_known = true;
            }
            out.retirement_ready = true;
            // VBR payloads use typed restore authority. They enter the
            // avoided-prefill projection below, but never the legacy fixed
            // calibrated exact-recovery ladder.
            return true;
        }
        if (reason == server_cache_destruction_reason::host_capacity) {
            out.release_ops = artifact.candidate.release_ops;
            // Retain a complete diagnostic proposal until the inventory-wide
            // one-lock marginal preview below replaces it with exact physical
            // release evidence. Unknown evidence never authorizes retention-capacity eviction.
            uint64_t snapshot_bytes = 0;
            uint64_t checkpoint_bytes = 0;
            uint64_t accelerator_bytes = 0;
            if (!server_prompt_cache::payload_bytes(
                    *victim, snapshot_bytes, checkpoint_bytes,
                    accelerator_bytes) ||
                checkpoint_bytes > UINT64_MAX - snapshot_bytes ||
                accelerator_bytes > UINT64_MAX - snapshot_bytes -
                    checkpoint_bytes) {
                return false;
            }
            out.marginal_resident_bytes = snapshot_bytes +
                checkpoint_bytes + accelerator_bytes;
        }
        out.retirement_ready = true;
        if (!calib) {
            return false;
        }

        out.recovery = find_exact_host_recovery(cache, victim);
        out.ranking.zero_destruction_known = true;
        out.ranking.zero_destruction = out.recovery != cache.states.end();
        if (out.hard_leased) {
            return false;
        }

        uint32_t additional_weight = SERVER_CACHE_HOST_WEIGHT_SCALE;
        if (common_cache_family_allows_additional_weight(
                victim->cache_family) && authority.host_retention_weight) {
            if (!authority.host_retention_weight(
                    authority.host_retention_weight_context,
                    *victim, additional_weight) ||
                additional_weight == 0) {
                return false;
            }
        }

        uint32_t weight = 0;
        uint64_t price = 0;
        if (!server_cache_host_retention_price_us(
                *calib, victim->size(), out.soft_leased,
                out.main_family, weight, price, additional_weight)) {
            return false;
        }
        out.ranking.weight_milli = weight;
        out.ranking.price_us = price;
        out.ranking.price_known = true;
        return true;
    } catch (...) {
        return false;
    }
}

bool populate_vbr_host_trade_marginals(
        server_prompt_cache & cache,
        std::vector<host_trade_candidate> & candidates) noexcept {
    try {
        std::vector<const server_prompt_cache_payload *> payloads;
        std::vector<size_t> indices;
        payloads.reserve(candidates.size());
        indices.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            auto & candidate = candidates[i];
            if (!candidate.vbr || !candidate.retirement_ready ||
                candidate.vbr_logical_alias || candidate.hard_leased ||
                candidate.mandatory_anchor) {
                continue;
            }
            payloads.push_back(&candidate.victim->payload);
            indices.push_back(i);
        }
        if (payloads.empty()) {
            return true;
        }
        std::vector<vbr_artifact_retire_resident_preview> marginals;
        if (!cache.acct ||
            !server_prompt_cache_payload::preview_vbr_retire_resident_batch(
                payloads, cache.acct->serial(), marginals) ||
            marginals.size() != indices.size()) {
            return false;
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            auto & candidate = candidates[indices[i]];
            candidate.marginal_resident_bytes = marginals[i].resident;
            candidate.marginal_resident_known = marginals[i].known;
            if (!marginals[i].known) {
                candidate.retirement_ready = false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

bool populate_vbr_host_trade_marginals_conditioned(
        server_prompt_cache & cache,
        std::vector<host_trade_candidate> & candidates,
        const host_trade_candidate & baseline) noexcept {
    try {
        std::vector<const server_prompt_cache_payload *> payloads;
        std::vector<size_t> indices;
        payloads.reserve(candidates.size());
        indices.reserve(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i) {
            auto & candidate = candidates[i];
            if (&candidate == &baseline || !candidate.vbr ||
                !candidate.retirement_ready ||
                candidate.vbr_logical_alias || candidate.hard_leased ||
                candidate.mandatory_anchor) {
                continue;
            }
            payloads.push_back(&candidate.victim->payload);
            indices.push_back(i);
        }
        if (payloads.empty()) {
            return false;
        }
        std::vector<vbr_artifact_retire_resident_preview> marginals;
        if (!cache.acct ||
            !server_prompt_cache_payload::
                preview_vbr_retire_resident_conditioned_batch(
                    &baseline.victim->payload, payloads,
                    cache.acct->serial(), marginals) ||
            marginals.size() != indices.size()) {
            return false;
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            auto & candidate = candidates[indices[i]];
            candidate.marginal_resident_bytes = marginals[i].resident;
            candidate.marginal_resident_known = marginals[i].known;
            if (!marginals[i].known) {
                candidate.retirement_ready = false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

struct host_trade_retention_capacity_projection {
    bool complete = false;
    bool release_evidence_complete = true;
    uint64_t candidate_count = 0;
    llama_cache_acct_artifact_id artifact;
    uint64_t lineage_id = 0;
    common_retention_pool pool = common_retention_pool::attention;
    uint64_t lost_work = 0;
    uint64_t resource = 0;
};

// Allocation-free singleton projection for the synchronous retention-capacity authority.
// Fixed checkpoint planes may be shared by host/live aliases, so each fixed
// candidate carries the exact ledger marginal computed during inventory
// construction. A zero singleton yield is not executable here and remains on
// the legacy floor. The broader counterfactual projector retains compound
// support for debug/model-free analysis.
host_trade_retention_capacity_projection project_host_trade_retention_capacity(
        server_prompt_cache & cache,
        server_cache_destruction_reason reason,
        server_prompt_cache::iterator incoming,
        const std::vector<host_trade_candidate> & candidates,
        server_prompt_cache_shadow_row * rows,
        server_prompt_cache_shadow_artifact_slot * artifacts,
        server_prompt_cache_shadow_lineage_slot * lineages,
        llama_cache_acct_artifact_id ignored_artifact = {},
        llama_cache_acct_artifact_id excluded_artifact = {}) noexcept {
    host_trade_retention_capacity_projection result;
    if (!rows || !artifacts || !lineages || !cache.retention_obs || !cache.acct ||
        candidates.size() > SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES) {
        return result;
    }

    static_assert((SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY &
                   (SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY - 1)) == 0);
    constexpr size_t index_mask =
        SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY - 1;
    std::fill_n(artifacts, SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY,
        server_prompt_cache_shadow_artifact_slot {});
    std::fill_n(lineages, SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY,
        server_prompt_cache_shadow_lineage_slot {});

    struct fill_context {
        server_prompt_cache_shadow_row * rows = nullptr;
        server_prompt_cache_shadow_artifact_slot * artifacts = nullptr;
        server_prompt_cache_shadow_lineage_slot * lineages = nullptr;
        size_t size = 0;
        size_t observed = 0;
        llama_cache_acct_artifact_id excluded;
    } fill { rows, artifacts, lineages, 0, 0, excluded_artifact };
    const auto fill_value = [](void * opaque,
            const server_retention_value_snapshot & value) noexcept {
        auto & context = *static_cast<fill_context *>(opaque);
        context.observed++;
        if (value.artifact_id == context.excluded) {
            return true;
        }
        if (context.size == SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES ||
            !value.artifact_id.v || !value.stamp.lineage_id) {
            return false;
        }
        context.rows[context.size] = {
            value.artifact_id,
            value.instance_key,
            value.kind,
            value.stamp,
            value.lineage,
            value.external_shared_coverage_tokens,
            0,
            false,
            false,
        };

        constexpr size_t mask =
            SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY - 1;
        size_t artifact_slot = size_t(server_prompt_cache_shadow_hash(
            value.artifact_id.v)) & mask;
        size_t probes = 0;
        while (context.artifacts[artifact_slot].artifact_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (context.artifacts[artifact_slot].artifact_id ==
                    value.artifact_id.v) {
                return false;
            }
            artifact_slot = (artifact_slot + 1) & mask;
        }
        if (context.artifacts[artifact_slot].artifact_id) {
            return false;
        }
        context.artifacts[artifact_slot] = {
            value.artifact_id.v, uint32_t(context.size) };

        const uint64_t lineage_key = value.stamp.lineage_id ^
            (uint64_t(uint8_t(value.stamp.pool)) << 56);
        size_t lineage_slot = size_t(server_prompt_cache_shadow_hash(
            lineage_key)) & mask;
        probes = 0;
        while (context.lineages[lineage_slot].lineage_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (context.lineages[lineage_slot].lineage_id ==
                    value.stamp.lineage_id &&
                context.lineages[lineage_slot].pool == value.stamp.pool) {
                break;
            }
            lineage_slot = (lineage_slot + 1) & mask;
        }
        auto & lineage = context.lineages[lineage_slot];
        if (lineage.lineage_id &&
            (lineage.lineage_id != value.stamp.lineage_id ||
             lineage.pool != value.stamp.pool)) {
            return false;
        }
        if (!lineage.lineage_id) {
            lineage.lineage_id = value.stamp.lineage_id;
            lineage.pool = value.stamp.pool;
        }
        const uint64_t coverage = value.stamp.coverage_tokens;
        if (coverage > lineage.maximum_coverage) {
            lineage.second_coverage = lineage.maximum_coverage;
            lineage.maximum_coverage = coverage;
            lineage.maximum_count = 1;
        } else if (coverage == lineage.maximum_coverage) {
            lineage.maximum_count++;
        } else {
            lineage.second_coverage = std::max(
                lineage.second_coverage, coverage);
        }
        context.size++;
        return true;
    };
    const auto inventory = cache.retention_obs->value_snapshots(
        &fill, fill_value, excluded_artifact);
    if (inventory.status !=
            server_retention_value_snapshot_status::complete ||
        inventory.size != fill.observed || fill.size == 0) {
        return result;
    }

    auto * begin = rows;
    auto * end = rows + fill.size;
    const auto find_artifact = [&](llama_cache_acct_artifact_id id) {
        if (!id.v) {
            return end;
        }
        size_t slot = size_t(server_prompt_cache_shadow_hash(id.v)) &
            index_mask;
        size_t probes = 0;
        while (artifacts[slot].artifact_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (artifacts[slot].artifact_id == id.v) {
                return begin + artifacts[slot].row_index;
            }
            slot = (slot + 1) & index_mask;
        }
        return end;
    };

    // The priced inventory already covers every ordinary physical host
    // entry. Join those artifacts directly instead of repeating a sidecar
    // association lookup for every state. Incoming publications and
    // recovery-pinned entries are deliberately absent from that inventory;
    // join only those exceptional retained providers afterward.
    for (const auto & candidate : candidates) {
        if (candidate.ranking.artifact_id == excluded_artifact) {
            continue;
        }
        if (reason == server_cache_destruction_reason::host_capacity &&
            !candidate.vbr && !candidate.marginal_resident_known) {
            result.release_evidence_complete = false;
        }
        auto * row = find_artifact(candidate.ranking.artifact_id);
        if (!candidate.ranking.artifact_id.v || row == end ||
            row->kind != common_retention_artifact_kind::host_entry ||
            row->backing_known || row->releasable ||
            !candidate.lease_known) {
            return {};
        }
        row->backing_known = true;
        row->releasable = !candidate.hard_leased &&
            candidate.retirement_ready &&
            candidate.victim != incoming &&
            candidate.victim->recovery_pins == 0;
        if (!row->releasable) {
            continue;
        }
        result.candidate_count++;
        if (reason == server_cache_destruction_reason::host_token_limit) {
            row->resource = candidate.victim->prompt.n_tokens();
        } else if (candidate.vbr) {
            row->resource = candidate.marginal_resident_bytes;
        } else {
            row->resource = candidate.marginal_resident_bytes;
        }
    }
    for (auto state = cache.states.begin(); state != cache.states.end();
            ++state) {
        if (state != incoming && state->recovery_pins == 0) {
            continue;
        }
        const auto artifact = cache.retention_obs->artifact_id(
            server_retention_instance_key::for_host_entry(&*state));
        auto * row = find_artifact(artifact);
        if (!artifact.v || row == end ||
            row->kind != common_retention_artifact_kind::host_entry ||
            row->backing_known) {
            return {};
        }
        row->backing_known = true;
    }
    if (ignored_artifact.v) {
        auto * row = find_artifact(ignored_artifact);
        if (row == end ||
            row->kind != common_retention_artifact_kind::host_entry ||
            row->backing_known) {
            return {};
        }
        row->backing_known = true;
    }
    for (const auto * row = begin; row != end; ++row) {
        if (row->kind == common_retention_artifact_kind::host_entry &&
            !row->backing_known) {
            return {};
        }
    }

    const auto find_lineage = [&](common_retention_pool pool,
                                  uint64_t lineage_id) {
        if (!lineage_id) {
            return static_cast<server_prompt_cache_shadow_lineage_slot *>(nullptr);
        }
        const uint64_t key = lineage_id ^
            (uint64_t(uint8_t(pool)) << 56);
        size_t slot = size_t(server_prompt_cache_shadow_hash(key)) &
            index_mask;
        size_t probes = 0;
        while (lineages[slot].lineage_id &&
               probes++ < SERVER_PROMPT_CACHE_SHADOW_INDEX_CAPACITY) {
            if (lineages[slot].lineage_id == lineage_id &&
                lineages[slot].pool == pool) {
                return &lineages[slot];
            }
            slot = (slot + 1) & index_mask;
        }
        return static_cast<server_prompt_cache_shadow_lineage_slot *>(nullptr);
    };

    bool have_best = false;
    common_retention_shadow_value best;
    const uint64_t competition_epoch =
        cache.retention_obs->competition_epoch_value();
    for (const auto * row = begin; row != end; ++row) {
        if (!row->releasable || row->resource == 0) {
            continue;
        }
        auto * lineage = find_lineage(
            row->stamp.pool, row->stamp.lineage_id);
        GGML_ASSERT(lineage != nullptr);
        server_retention_singleton_quote quote;
        if (!server_retention_quote_singleton(
                row->stamp, row->lineage,
                lineage->maximum_coverage, lineage->second_coverage,
                lineage->maximum_count,
                row->external_shared_coverage_tokens,
                row->resource, competition_epoch, {}, quote)) {
            return {};
        }
        const int comparison = have_best
            ? common_retention_shadow_compare(quote.value, best) : -1;
        if (!have_best || comparison < 0 || (comparison == 0 &&
                std::tie(row->stamp.pool, row->stamp.lineage_id,
                         row->artifact_id.v) <
                std::tie(result.pool, result.lineage_id,
                         result.artifact.v))) {
            have_best = true;
            best = quote.value;
            result.artifact = row->artifact_id;
            result.lineage_id = row->stamp.lineage_id;
            result.pool = row->stamp.pool;
            result.lost_work = quote.lost_work_units;
            result.resource = row->resource;
        }
    }
    result.complete = have_best;
    return result;
}

void observe_host_trade_refusal(
        server_prompt_cache & cache,
        uint64_t admission_sequence,
        common_cache_plan_destruction_reason reason,
        const host_trade_ranking * ranking = nullptr) noexcept {
    common_cache_plan_destruction_receipt receipt;
    receipt.effects = common_cache_plan_destruction_effect_bit(
        common_cache_plan_destruction_effect::
            different_host_source_consumption);
    receipt.state = common_cache_plan_destruction_state::refused;
    receipt.reason = reason;
    receipt.admission_sequence = admission_sequence;
    receipt.payload_kind = ranking
        ? ranking->payload_kind
        : common_cache_plan_payload_kind::unavailable;
    server_prompt_cache_observe_host_destruction(
        cache, receipt, true, 0, ranking);
}

} // namespace

static bool server_prompt_cache_revalidate_vbr_victim(
        server_prompt_cache & cache,
        server_prompt_cache_state * expected_incumbent,
        llama_cache_acct_artifact_id expected_artifact) noexcept {
    if (cache.states.empty() || !cache.acct || !cache.retention_obs ||
        !cache.lease_obs) {
        return false;
    }
    cache.lease_obs->lifecycle_point();
    if (!expected_incumbent || !expected_artifact.v ||
        cache.retention_obs->artifact_id(
            server_retention_instance_key::for_host_entry(
                expected_incumbent)) != expected_artifact ||
        expected_incumbent->recovery_pins != 0) {
        return false;
    }
    server_cache_destruction_artifact evidence;
    if (!build_host_retention_artifact(
            cache, *expected_incumbent, evidence) ||
        evidence.mandatory_anchor ||
        evidence.candidate.lease.state !=
            server_cache_lease_eval_state::known ||
        server_cache_lease_is_hard(evidence.candidate.lease) ||
        evidence.candidate.artifact_id != expected_artifact) {
        return false;
    }
    return true;
}

static bool server_prompt_cache_plan_vbr_pressure(
        server_prompt_cache & cache,
        size_t projected_bytes,
        size_t projected_tokens,
        size_t max_victims,
        bool begin_competition_wave,
        server_prompt_cache_vbr_pressure_plan & plan,
        server_prompt_cache_shadow_row * shadow_rows,
        server_prompt_cache_shadow_artifact_slot * shadow_artifacts,
        server_prompt_cache_shadow_lineage_slot * shadow_lineages,
        llama_cache_acct_artifact_id ignored_artifact) noexcept {
    plan = {};
    if (max_victims == 0 || max_victims > plan.victims.size() ||
        cache.states.empty() || !cache.acct || !cache.retention_obs ||
        !cache.lease_obs) {
        return false;
    }
    cache.lease_obs->lifecycle_point();

    if (cache.states.size() > 1) {
        const bool byte_pressure = cache.limit_size > 0 &&
            projected_bytes > cache.limit_size;
        const auto reason = byte_pressure
            ? server_cache_destruction_reason::host_capacity
            : server_cache_destruction_reason::host_token_limit;
        if (!shadow_rows || !shadow_artifacts || !shadow_lineages ||
            cache.states.size() >
                SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES ||
            (begin_competition_wave &&
             !cache.retention_obs->begin_competition_wave())) {
            return false;
        }
        std::vector<host_trade_candidate> candidates;
        try {
            std::vector<server_prompt_cache::iterator> victims;
            std::vector<uint32_t> ordinals;
            std::vector<server_cache_destruction_artifact> evidence;
            std::vector<server_cache_lease_identity> identities;
            std::vector<server_cache_lease_inspection_request> requests;
            victims.reserve(cache.states.size());
            ordinals.reserve(cache.states.size());
            evidence.reserve(cache.states.size());
            identities.reserve(cache.states.size());
            requests.reserve(cache.states.size());
            candidates.reserve(cache.states.size());
            uint32_t ordinal = 0;
            for (auto it = cache.states.begin(); it != cache.states.end();
                    ++it, ++ordinal) {
                const auto state_artifact = cache.retention_obs->artifact_id(
                    server_retention_instance_key::for_host_entry(&*it));
                if (state_artifact == ignored_artifact) {
                    continue;
                }
                if (it->payload.kind() !=
                        server_prompt_cache_payload_kind::vbr_artifact ||
                    it->payload.vbr_has_quality_anchor() ||
                    it->payload.vbr_logical_erase_only()) {
                    return false;
                }
                if (it->recovery_pins != 0) {
                    continue;
                }
                victims.push_back(it);
                ordinals.push_back(ordinal);
                evidence.emplace_back();
                identities.emplace_back();
                if (!build_host_retention_artifact_uninspected(
                        cache, *it, evidence.back(), identities.back())) {
                    return false;
                }
                requests.push_back({
                    evidence.back().candidate.artifact_id,
                    &identities.back(),
                });
            }
            std::vector<server_cache_lease_evaluation> leases;
            if (!cache.lease_obs->inspect_batch(requests, leases) ||
                leases.size() != evidence.size()) {
                return false;
            }
            for (size_t i = 0; i < evidence.size(); ++i) {
                evidence[i].candidate.lease = leases[i];
                host_trade_candidate candidate;
                (void) host_trade_price(
                    cache, victims[i], ordinals[i], reason,
                    nullptr, candidate, false, &evidence[i]);
                candidates.push_back(std::move(candidate));
            }
            if (candidates.empty() || (byte_pressure &&
                !populate_vbr_host_trade_marginals(cache, candidates))) {
                return false;
            }
        } catch (...) {
            return false;
        }
        const auto projection = project_host_trade_retention_capacity(
            cache, reason,
            cache.states.end(), candidates,
            shadow_rows, shadow_artifacts, shadow_lineages,
            ignored_artifact);
        if (!projection.complete || !projection.release_evidence_complete ||
            !projection.artifact.v) {
            return false;
        }
        const auto selected = std::find_if(
            candidates.begin(), candidates.end(), [&](const auto & value) {
                return value.ranking.artifact_id == projection.artifact &&
                    (!byte_pressure || value.marginal_resident_known) &&
                    value.retirement_ready && value.lease_known &&
                    !value.hard_leased && !value.mandatory_anchor &&
                    value.victim->recovery_pins == 0;
            });
        if (selected == candidates.end()) {
            return false;
        }
        const size_t first_tokens = size_t(std::max(
            0, selected->victim->prompt.n_tokens()));
        const size_t after_bytes = selected->marginal_resident_bytes >
                projected_bytes
            ? 0 : projected_bytes -
                size_t(selected->marginal_resident_bytes);
        const size_t after_tokens = first_tokens > projected_tokens
            ? 0 : projected_tokens - first_tokens;
        const auto fits = [&](size_t bytes, size_t tokens) {
            return (cache.limit_size == 0 || bytes <= cache.limit_size) &&
                (cache.limit_tokens == 0 ||
                 tokens <= server_prompt_cache_effective_token_limit(
                    cache.limit_size, cache.limit_tokens, bytes, tokens));
        };
        plan.victims[0] = &*selected->victim;
        plan.artifacts[0] = selected->ranking.artifact_id;
        plan.soft_leased[0] = selected->soft_leased;
        plan.count = 1;
        if (fits(after_bytes, after_tokens)) {
            return true;
        }
        // The bounded compound terminal follows two consecutive byte-pressure
        // decisions only. A token-only second step can have a different retention-capacity
        // order and remains an explicit unsupported shape.
        if (max_victims < 2 || !byte_pressure ||
            cache.limit_size == 0 || after_bytes <= cache.limit_size ||
            !selected->ranking.artifact_id.v) {
            plan = {};
            return false;
        }
        if (!populate_vbr_host_trade_marginals_conditioned(
                cache, candidates, *selected)) {
            plan = {};
            return false;
        }
        const auto second_projection = project_host_trade_retention_capacity(
            cache, reason, cache.states.end(), candidates,
            shadow_rows, shadow_artifacts, shadow_lineages,
            ignored_artifact, selected->ranking.artifact_id);
        if (!second_projection.complete ||
            !second_projection.release_evidence_complete ||
            !second_projection.artifact.v) {
            plan = {};
            return false;
        }
        const auto second = std::find_if(
            candidates.begin(), candidates.end(), [&](const auto & value) {
                return value.ranking.artifact_id ==
                        second_projection.artifact &&
                    value.marginal_resident_known &&
                    value.retirement_ready && value.lease_known &&
                    !value.hard_leased && !value.mandatory_anchor &&
                    value.victim->recovery_pins == 0;
            });
        if (second == candidates.end() || second == selected) {
            plan = {};
            return false;
        }
        std::vector<const server_prompt_cache_payload *> pair {
            &selected->victim->payload,
            &second->victim->payload,
        };
        llama_cache_acct_release_set_preview preview;
        uint64_t pair_bytes = 0;
        if (!server_prompt_cache_payload::preview_vbr_retire_union(
                pair, cache.acct->serial(), preview) ||
            !vbr_release_resident_bytes(preview, pair_bytes) ||
            pair_bytes > SIZE_MAX) {
            plan = {};
            return false;
        }
        const size_t second_tokens = size_t(std::max(
            0, second->victim->prompt.n_tokens()));
        if (second_tokens > after_tokens) {
            plan = {};
            return false;
        }
        const size_t pair_after_bytes = pair_bytes > projected_bytes
            ? 0 : projected_bytes - size_t(pair_bytes);
        const size_t pair_after_tokens = after_tokens - second_tokens;
        if (!fits(pair_after_bytes, pair_after_tokens)) {
            plan = {};
            return false;
        }
        plan.victims[1] = &*second->victim;
        plan.artifacts[1] = second->ranking.artifact_id;
        plan.soft_leased[1] = second->soft_leased;
        plan.count = 2;
        return true;
    }

    auto current = cache.states.begin();
    if (current->recovery_pins != 0) {
        return false;
    }
    host_trade_candidate candidate;
    (void) host_trade_price(
        cache, current, 0,
        server_cache_destruction_reason::host_capacity,
        nullptr, candidate);
    if (!candidate.retirement_ready || !candidate.lease_known ||
        candidate.hard_leased || candidate.mandatory_anchor ||
        !candidate.ranking.artifact_id.v) {
        return false;
    }

    uint64_t released_bytes = candidate.marginal_resident_bytes;
    if (!candidate.vbr) {
        llama_cache_acct_release_set_preview preview;
        if (candidate.release_ops.empty() ||
            !cache.acct->preview_release_set(
                candidate.release_ops, cache.acct->serial(), preview) ||
            !vbr_release_resident_bytes(preview, released_bytes)) {
            return false;
        }
    } else if (!candidate.marginal_resident_known ||
               candidate.vbr_logical_alias) {
        return false;
    }
    const size_t released_tokens =
        size_t(std::max(0, current->prompt.n_tokens()));
    const size_t after_bytes = released_bytes > projected_bytes
        ? 0 : projected_bytes - size_t(released_bytes);
    const size_t after_tokens = released_tokens > projected_tokens
        ? 0 : projected_tokens - released_tokens;
    if (cache.limit_size > 0 && after_bytes > cache.limit_size) {
        return false;
    }
    const size_t effective = server_prompt_cache_effective_token_limit(
        cache.limit_size, cache.limit_tokens,
        after_bytes, after_tokens);
    if (cache.limit_tokens != 0 && after_tokens > effective) {
        return false;
    }
    plan.victims[0] = &*current;
    plan.artifacts[0] = candidate.ranking.artifact_id;
    plan.soft_leased[0] = candidate.soft_leased;
    plan.count = 1;
    return true;
}

bool server_prompt_cache::acquire_durable_recovery(
        const server_tokens & tokens,
        const std::string & adapter_config_key,
        llama_cache_acct_artifact_id & artifact,
        std::vector<llama_cache_acct_op_id> & ops,
        server_cache_recovery_pin & pin) noexcept {
    return acquire_durable_recovery(
        find_state_exact(tokens, adapter_config_key), artifact, ops, pin);
}

bool server_prompt_cache::acquire_durable_recovery(
        iterator state,
        llama_cache_acct_artifact_id & artifact,
        std::vector<llama_cache_acct_op_id> & ops,
        server_cache_recovery_pin & pin) noexcept {
    artifact = {};
    ops.clear();
    pin = {};
    try {
        if (state == states.end()) {
            return false;
        }
        std::vector<llama_cache_acct_artifact_id> artifacts;
        if (!build_host_recovery_source(
                *this, *state, artifacts, ops) || artifacts.size() != 1) {
            return false;
        }
        artifact = artifacts.front();
        pin = acquire_host_recovery_pin(
            *state, std::move(artifacts), ops);
        if (!pin.valid() || !pin.binds_exact(artifact, ops)) {
            artifact = {};
            ops.clear();
            pin = {};
            return false;
        }
        return true;
    } catch (...) {
        artifact = {};
        ops.clear();
        pin = {};
    }
    return false;
}

server_cache_durable_fallback_proof
server_prompt_cache_host_fallback_proof(
        server_prompt_cache & cache,
        const server_cache_control_selector & selector) noexcept {
    if (selector.kind != server_cache_control_subject_kind::host_snapshot ||
        selector.retention_key.kind !=
            common_retention_artifact_kind::host_entry) {
        return {};
    }
    auto state = std::find_if(
        cache.states.begin(), cache.states.end(), [&](const auto & value) {
            return &value == reinterpret_cast<const server_prompt_cache_state *>(
                selector.retention_key.instance);
        });
    if (state == cache.states.end()) {
        return {};
    }
    llama_cache_acct_artifact_id artifact;
    std::vector<llama_cache_acct_op_id> ops;
    server_cache_recovery_pin pin;
    if (!cache.acquire_durable_recovery(state, artifact, ops, pin)) {
        return {};
    }
    return server_cache_retention_fallback_proof(std::move(pin));
}

bool server_prompt_cache::exactly_redundant(
        const server_prompt_cache_state & victim,
        const server_prompt_cache_state & survivor) noexcept {
    try {
        const bool vbr = victim.payload.kind() ==
            server_prompt_cache_payload_kind::vbr_artifact;
        if (&victim == &survivor ||
            victim.payload.kind() != survivor.payload.kind() ||
            victim.adapter_config_key != survivor.adapter_config_key ||
            victim.vbr_execution_identity !=
                survivor.vbr_execution_identity ||
            (vbr && !victim.payload.same_storage(survivor.payload)) ||
            victim.prompt.sequence_epoch != survivor.prompt.sequence_epoch ||
            victim.prompt.n_tokens() > survivor.prompt.n_tokens() ||
            victim.prompt.tokens.get_common_prefix(survivor.prompt.tokens) !=
                size_t(victim.prompt.n_tokens()) ||
            (!vbr && !victim.payload.same_storage(survivor.payload)) ||
            victim.prompt.checkpoints.size() !=
                survivor.prompt.checkpoints.size()) {
            return false;
        }
        std::string victim_media;
        std::string survivor_media;
        if (!victim.prompt.tokens.media_content_identity(
                victim.prompt.n_tokens(), victim_media) ||
            !survivor.prompt.tokens.media_content_identity(
                victim.prompt.n_tokens(), survivor_media) ||
            victim_media != survivor_media) {
            return false;
        }
        auto a = victim.prompt.checkpoints.begin();
        auto b = survivor.prompt.checkpoints.begin();
        for (; a != victim.prompt.checkpoints.end(); ++a, ++b) {
            if (!checkpoint_payload_equal(*a, *b)) {
                return false;
            }
        }
        return true;
    } catch (...) {
        return false;
    }
}

void server_prompt_cache::observe_retention_pressure_choice(
        server_cache_destruction_reason reason,
        iterator incoming,
        iterator incumbent,
        bool competition_wave_valid) noexcept {
    if (!retention_obs) {
        return;
    }
    const auto increment = [](uint64_t & value) noexcept {
        if (value != UINT64_MAX) {
            value++;
        }
    };
    increment(retention_shadow.choices);
    auto & event = retention_shadow.last;
    event = {};
    event.reason = reason;
    event.competition_epoch = retention_obs->competition_epoch_value();

    const auto unavailable = [&]() noexcept {
        increment(retention_shadow.unavailable);
        if (debug_observability) {
            SRV_INF(
                "CACHE_RETENTION_SHADOW status=unavailable reason=%u "
                "epoch=%" PRIu64 " candidates=%" PRIu64 "\n",
                unsigned(reason), event.competition_epoch,
                event.candidate_count);
        }
    };

    struct fill_context {
        server_prompt_cache_shadow_row * rows = nullptr;
        size_t size = 0;
    };
    const auto fill_value = [](void * opaque,
            const server_retention_value_snapshot & value) noexcept {
        auto & context = *static_cast<fill_context *>(opaque);
        if (context.size == SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES) {
            return false;
        }
        context.rows[context.size++] = {
            value.artifact_id,
            value.instance_key,
            value.kind,
            value.stamp,
            value.lineage,
            value.external_shared_coverage_tokens,
            0,
            false,
            false,
        };
        return true;
    };

    // Lifecycle shadow uses the same immutable catalog, evaluated leases and
    // serial-bound accounting release preview as the certified authority. It
    // remains counterfactual: the already-selected lifecycle victim is the
    // incumbent and no projection result can authorize mutation during observation.
    if (publish_authority) {
        if (!competition_wave_valid || !retention_shadow_rows || !acct ||
            !lease_obs || !lease_execution_identity ||
            reason != server_cache_destruction_reason::host_capacity ||
            states.size() > SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES ||
            incumbent == states.end()) {
            unavailable();
            return;
        }
        try {
            fill_context fill { retention_shadow_rows.get(), 0 };
            const auto inventory = retention_obs->value_snapshots(
                &fill, fill_value);
            if (inventory.status !=
                    server_retention_value_snapshot_status::complete ||
                inventory.size != fill.size || fill.size == 0) {
                unavailable();
                return;
            }
            auto * begin = retention_shadow_rows.get();
            auto * end = begin + fill.size;
            std::sort(begin, end, [](const auto & a, const auto & b) {
                return a.artifact_id.v < b.artifact_id.v;
            });
            const auto find_artifact = [&](llama_cache_acct_artifact_id id) {
                const auto found = std::lower_bound(
                    begin, end, id.v, [](const auto & row, uint64_t value) {
                        return row.artifact_id.v < value;
                    });
                return found != end && found->artifact_id == id ? found : end;
            };

            std::vector<server_cache_yield_candidate> candidates;
            candidates.reserve(fill.size);
            for (auto * row = begin; row != end; ++row) {
                server_retention_candidate catalog;
                if (!retention_obs->candidate_for_instance(
                        row->instance_key, catalog) ||
                    catalog.artifact_id != row->artifact_id ||
                    catalog.record.kind != row->kind ||
                    catalog.record.stamp.stable_id != row->stamp.stable_id ||
                    catalog.record.stamp.lineage_id !=
                        row->stamp.lineage_id ||
                    catalog.lineage != row->lineage) {
                    unavailable();
                    return;
                }
                server_cache_yield_candidate candidate;
                candidate.artifact_id = catalog.artifact_id;
                candidate.record = catalog.record;
                candidate.lineage = catalog.lineage;
                candidate.availability =
                    server_retention_candidate_availability::
                        in_flight_mutation;
                candidate.lease = {
                    server_cache_lease_eval_state::known,
                    server_cache_lease_class::none,
                    server_cache_lease_eligibility::eligible,
                };
                candidate.identity_known = true;
                candidate.external_shared_coverage_tokens =
                    row->external_shared_coverage_tokens;
                candidates.push_back(std::move(candidate));
            }

            uint64_t n_candidates = 0;
            for (auto it = states.begin(); it != states.end(); ++it) {
                server_cache_destruction_artifact artifact;
                if (!build_host_destruction_artifact(*this, *it, artifact)) {
                    unavailable();
                    return;
                }
                auto candidate = std::move(artifact.candidate);
                auto * row = find_artifact(candidate.artifact_id);
                if (row == end || row->kind !=
                        common_retention_artifact_kind::host_entry ||
                    row->backing_known) {
                    unavailable();
                    return;
                }
                row->backing_known = true;
                candidate.external_shared_coverage_tokens =
                    row->external_shared_coverage_tokens;
                if (it == incoming || it->recovery_pins != 0) {
                    candidate.availability =
                        server_retention_candidate_availability::
                            in_flight_mutation;
                }
                if (candidate.availability ==
                        server_retention_candidate_availability::available &&
                    candidate.lease.eligibility ==
                        server_cache_lease_eligibility::eligible &&
                    !server_cache_lease_is_hard(candidate.lease) &&
                    !candidate.release_ops.empty()) {
                    n_candidates++;
                }
                candidates[size_t(row - begin)] = std::move(candidate);
            }
            for (const auto * row = begin; row != end; ++row) {
                if (row->kind == common_retention_artifact_kind::host_entry &&
                    !row->backing_known) {
                    unavailable();
                    return;
                }
            }
            event.candidate_count = n_candidates;
            event.incumbent_artifact = host_entry_artifact_id(
                *this, *incumbent);
            if (!event.incumbent_artifact.v) {
                unavailable();
                return;
            }

            const uint64_t accounting_serial = acct->serial();
            const auto host_domain =
                llama_cache_acct_resource_domain::non_device(
                    llama_cache_acct_residency::pageable_host);
            const auto projection = server_retention_shadow_project(
                candidates,
                event.competition_epoch,
                host_domain,
                accounting_serial,
                [this](const std::vector<llama_cache_acct_op_id> & ops,
                       uint64_t serial,
                       llama_cache_acct_release_set_preview & out) {
                    return acct->preview_release_set(ops, serial, out);
                });
            if (!projection.complete || projection.alternatives.empty() ||
                projection.alternatives.front().artifact_ids.size() != 1) {
                unavailable();
                return;
            }
            const auto & alternative = projection.alternatives.front();
            event.proposed_artifact = alternative.artifact_ids.front();
            event.proposed_lineage = alternative.lineage_id;
            event.proposed_pool = alternative.pool;
            event.proposed_lost_work = alternative.lost_work_units;
            event.proposed_resource = alternative.value.marginal_resource;
            event.status = server_prompt_cache_shadow_status::complete;
            event.agrees =
                event.incumbent_artifact == event.proposed_artifact;
            increment(retention_shadow.complete);
            increment(event.agrees
                ? retention_shadow.agreements
                : retention_shadow.disagreements);
            if (debug_observability) {
                SRV_INF(
                    "CACHE_RETENTION_SHADOW status=complete reason=%u "
                    "epoch=%" PRIu64 " candidates=%" PRIu64
                    " incumbent=%" PRIu64 " proposed=%" PRIu64
                    " agrees=%s lost_work=%" PRIu64
                    " resource=%" PRIu64 "\n",
                    unsigned(reason), event.competition_epoch,
                    event.candidate_count, event.incumbent_artifact.v,
                    event.proposed_artifact.v,
                    event.agrees ? "true" : "false",
                    event.proposed_lost_work, event.proposed_resource);
            }
        } catch (...) {
            unavailable();
        }
        return;
    }

    // The lifecycle route returned above after lowering its evaluated lease
    // and accounting evidence. This branch observes only the ordinary
    // fixed-cache FIFO route; never mix the two evidence shapes.
    if (!competition_wave_valid ||
        !retention_shadow_rows || incumbent == states.end() ||
        states.size() > SERVER_PROMPT_CACHE_SHADOW_MAX_CANDIDATES) {
        unavailable();
        return;
    }

    fill_context fill { retention_shadow_rows.get(), 0 };
    const auto inventory = retention_obs->value_snapshots(
        &fill, fill_value);
    if (inventory.status !=
            server_retention_value_snapshot_status::complete ||
        inventory.size != fill.size) {
        unavailable();
        return;
    }
    const size_t n_rows = fill.size;
    if (n_rows == 0) {
        unavailable();
        return;
    }

    auto * begin = retention_shadow_rows.get();
    auto * end = begin + n_rows;
    std::sort(begin, end, [](const auto & a, const auto & b) {
        return a.artifact_id.v < b.artifact_id.v;
    });
    const auto find_artifact = [&](llama_cache_acct_artifact_id id) {
        const auto found = std::lower_bound(
            begin, end, id.v, [](const auto & row, uint64_t value) {
                return row.artifact_id.v < value;
            });
        return found != end && found->artifact_id == id ? found : end;
    };
    uint64_t n_candidates = 0;
    for (auto it = states.begin(); it != states.end(); ++it) {
        const auto artifact = retention_obs->artifact_id(
            server_retention_instance_key::for_host_entry(&*it));
        auto * row = find_artifact(artifact);
        const uint64_t resource =
            reason == server_cache_destruction_reason::host_token_limit
                ? uint64_t(it->prompt.n_tokens())
                : uint64_t(it->size());
        if (!artifact.v || row == end ||
            row->kind != common_retention_artifact_kind::host_entry ||
            row->backing_known || resource == 0) {
            unavailable();
            return;
        }
        row->backing_known = true;
        row->resource = resource;
        row->releasable = it != incoming && it->recovery_pins == 0;
        if (row->releasable) {
            n_candidates++;
        }
    }
    for (const auto * row = begin; row != end; ++row) {
        if (row->kind == common_retention_artifact_kind::host_entry &&
            !row->backing_known) {
            unavailable();
            return;
        }
    }
    event.candidate_count = n_candidates;
    const auto incumbent_artifact = retention_obs->artifact_id(
        server_retention_instance_key::for_host_entry(&*incumbent));
    const auto * incumbent_row = find_artifact(incumbent_artifact);
    if (!incumbent_artifact.v || incumbent_row == end ||
        incumbent_row->kind != common_retention_artifact_kind::host_entry ||
        !incumbent_row->backing_known || !incumbent_row->releasable) {
        unavailable();
        return;
    }
    event.incumbent_artifact = incumbent_artifact;
    event.incumbent_lineage = incumbent_row->lineage.lineage_id;

    std::sort(begin, end, [](const auto & a, const auto & b) {
        return std::tie(
                   a.stamp.pool, a.stamp.lineage_id,
                   a.stamp.coverage_tokens, a.stamp.recency_ordinal,
                   a.artifact_id.v) <
               std::tie(
                   b.stamp.pool, b.stamp.lineage_id,
                   b.stamp.coverage_tokens, b.stamp.recency_ordinal,
                   b.artifact_id.v);
    });

    bool have_proposed = false;
    common_retention_shadow_value proposed_value;
    for (size_t first = 0; first < n_rows;) {
        size_t last = first + 1;
        while (last < n_rows &&
               begin[last].stamp.pool == begin[first].stamp.pool &&
               begin[last].stamp.lineage_id ==
                   begin[first].stamp.lineage_id) {
            if (begin[last].lineage != begin[first].lineage) {
                unavailable();
                return;
            }
            last++;
        }

        uint64_t maximum = 0;
        uint64_t second = 0;
        size_t n_maximum = 0;
        for (size_t i = first; i < last; ++i) {
            const uint64_t coverage = begin[i].stamp.coverage_tokens;
            if (coverage > maximum) {
                second = maximum;
                maximum = coverage;
                n_maximum = 1;
            } else if (coverage == maximum) {
                n_maximum++;
            } else if (coverage > second) {
                second = coverage;
            }
        }

        for (size_t i = first; i < last; ++i) {
            if (!begin[i].releasable) {
                continue;
            }
            server_retention_singleton_quote quote;
            if (!server_retention_quote_singleton(
                    begin[i].stamp, begin[i].lineage,
                    maximum, second, uint32_t(n_maximum),
                    begin[i].external_shared_coverage_tokens,
                    begin[i].resource, event.competition_epoch,
                    {}, quote)) {
                unavailable();
                return;
            }
            const bool lower = !have_proposed ||
                common_retention_shadow_compare(
                    quote.value, proposed_value) < 0;
            const bool tied = have_proposed &&
                common_retention_shadow_compare(
                    quote.value, proposed_value) == 0;
            if (lower || (tied &&
                    std::tie(begin[i].stamp.pool,
                             begin[i].stamp.lineage_id,
                             begin[i].artifact_id.v) <
                    std::tie(event.proposed_pool,
                             event.proposed_lineage,
                             event.proposed_artifact.v))) {
                have_proposed = true;
                proposed_value = quote.value;
                event.proposed_artifact = begin[i].artifact_id;
                event.proposed_lineage = begin[i].stamp.lineage_id;
                event.proposed_pool = begin[i].stamp.pool;
                event.proposed_lost_work = quote.lost_work_units;
                event.proposed_resource = begin[i].resource;
            }
        }
        first = last;
    }

    if (!have_proposed) {
        unavailable();
        return;
    }
    event.status = server_prompt_cache_shadow_status::complete;
    event.agrees = event.incumbent_artifact == event.proposed_artifact;
    increment(retention_shadow.complete);
    if (event.agrees) {
        increment(retention_shadow.agreements);
    } else {
        increment(retention_shadow.disagreements);
    }
    if (debug_observability) {
        SRV_INF(
            "CACHE_RETENTION_SHADOW status=complete reason=%u "
            "epoch=%" PRIu64 " candidates=%" PRIu64
            " incumbent=%" PRIu64 " proposed=%" PRIu64
            " agrees=%s lost_work=%" PRIu64
            " resource=%" PRIu64 "\n",
            unsigned(reason), event.competition_epoch,
            event.candidate_count, event.incumbent_artifact.v,
            event.proposed_artifact.v,
            event.agrees ? "true" : "false",
            event.proposed_lost_work, event.proposed_resource);
    }
}

bool server_prompt_cache::destroy_priced_host_entry(
        server_cache_destruction_reason reason,
        iterator incoming,
        iterator & legacy_floor,
        common_cache_plan_destruction_reason & floor_reason,
        bool & recovery_pin_excluded,
        bool competition_wave_valid,
        bool & observe_retention_shadow,
        uint64_t & released_bytes,
        size_t & released_tokens,
        llama_cache_acct_artifact_id required_victim) {
    released_bytes = 0;
    released_tokens = 0;
    legacy_floor = states.end();
    floor_reason = common_cache_plan_destruction_reason::capacity_refused;
    recovery_pin_excluded = false;
    if (!publish_authority ||
        (reason != server_cache_destruction_reason::host_capacity &&
         reason != server_cache_destruction_reason::host_token_limit)) {
        return false;
    }
    if (!acct || !retention_obs || !lease_obs || !lease_execution_identity) {
        floor_reason = common_cache_plan_destruction_reason::lease_unavailable;
        if (destruction_obs) {
            destruction_obs->note_host_trade_substrate_fault();
            destruction_obs->host_trade_legacy_fallbacks++;
        }
        if (!host_trade_substrate_warned) {
            host_trade_substrate_warned = true;
            SRV_WRN("%s\n",
                    "host retention pricing unavailable: lifecycle lease/accounting substrate is incomplete");
        }
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (it == incoming) {
                continue;
            }
            // Without the complete lease/accounting substrate, no learned
            // authority may inspect or retire a VBR capability. Keep it as
            // durable retained coverage instead of bypassing a potentially
            // hard lease through the fixed-state terminal.
            if (!it->payload.fixed_state_restorable()) {
                continue;
            }
            if (it->recovery_pins != 0) {
                recovery_pin_excluded = true;
                emit_recovery_pin_excluded(*this, *it);
                continue;
            }
            legacy_floor = it;
            break;
        }
        const uint64_t sequence =
            ++publish_authority->destruction_quote_sequence;
        observe_host_trade_refusal(*this, sequence, floor_reason);
        return false;
    }

    // lease boundary is an execution-time lease boundary. Expire first, then inspect
    // each immutable host artifact once for pricing. Soft protection raises
    // price; only a hard lease makes a candidate ineligible. If every priced
    // candidate fails certification, the caller deliberately executes the
    // historical FIFO victim so the user's configured bound remains real.
    lease_obs->lifecycle_point();
    // The retained calibration profile prices pageable-host bytes. It has no
    // lawful token-yield currency, so token pressure proceeds directly to
    // The retention-capacity policy's exact token denominator (or the deterministic FIFO floor).
    const auto * calib = reason ==
            server_cache_destruction_reason::host_capacity
        ? common_cache_plan_calib_find(
              publish_authority->calibration_profile)
        : nullptr;
    std::vector<host_trade_candidate> candidates;
    try {
        candidates.reserve(states.size());
        uint32_t ordinal = 0;
        for (auto it = states.begin(); it != states.end(); ++it, ++ordinal) {
            if (it == incoming) {
                continue;
            }
            if (it->recovery_pins != 0) {
                recovery_pin_excluded = true;
                emit_recovery_pin_excluded(*this, *it);
                continue;
            }
            host_trade_candidate candidate;
            (void) host_trade_price(
                *this, it, ordinal, reason, calib, candidate, false);
            if (candidate.hard_leased) {
                candidate.attempted = true;
                const uint64_t quote_sequence =
                    ++publish_authority->destruction_quote_sequence;
                observe_host_trade_refusal(
                    *this,
                    quote_sequence,
                    common_cache_plan_destruction_reason::hard_lease_blocked,
                    &candidate.ranking);
                if (destruction_obs) {
                    destruction_obs->note_host_trade_veto();
                }
            }
            candidates.push_back(std::move(candidate));
        }
        if (reason == server_cache_destruction_reason::host_capacity &&
            !populate_vbr_host_trade_marginals(*this, candidates)) {
            if (destruction_obs) {
                destruction_obs->host_trade_legacy_fallbacks++;
            }
            return false;
        }
        if (reason == server_cache_destruction_reason::host_capacity) {
            std::vector<llama_cache_acct_release_set_view> fixed_sets;
            std::vector<size_t> fixed_indices;
            fixed_sets.reserve(candidates.size());
            fixed_indices.reserve(candidates.size());
            for (size_t i = 0; i < candidates.size(); ++i) {
                auto & candidate = candidates[i];
                if (candidate.vbr || candidate.release_ops.empty()) {
                    continue;
                }
                fixed_sets.push_back({
                    candidate.release_ops.data(),
                    candidate.release_ops.size(),
                });
                fixed_indices.push_back(i);
            }
            std::vector<uint64_t> marginals;
            if (!fixed_sets.empty() &&
                acct->preview_release_set_resident_batch(
                    fixed_sets, acct->serial(), marginals) &&
                marginals.size() == fixed_indices.size()) {
                for (size_t i = 0; i < fixed_indices.size(); ++i) {
                    auto & candidate = candidates[fixed_indices[i]];
                    candidate.marginal_resident_bytes = marginals[i];
                    candidate.marginal_resident_known = true;
                }
            }
        }
    } catch (...) {
        if (destruction_obs) {
            destruction_obs->host_trade_legacy_fallbacks++;
        }
        return false;
    }

    // Exact shared-owner aliases are zero-value logical cleanup, not a
    // positive-byte retention-capacity alternative. Execute the first lawful alias before
    // calibrated or learned victim selection; the outer pressure loop then
    // remeasures and prices the now-exclusive survivor. This preserves pins,
    // hard leases, and incoming protection while ensuring redundant aliases
    // cannot mask a more valuable unique entry.
    const auto alias = std::find_if(
        candidates.begin(), candidates.end(), [](const auto & candidate) {
            return candidate.vbr_logical_alias && candidate.lease_known &&
                   !candidate.hard_leased &&
                   candidate.victim->recovery_pins == 0;
        });
    if (alias != candidates.end()) {
        if (required_victim.v &&
            alias->ranking.artifact_id != required_victim) {
            return false;
        }
        if (observe_retention_shadow) {
            observe_retention_pressure_choice(
                reason, incoming, alias->victim, competition_wave_valid);
        }
        SRV_WRN(
            " - removing redundant VBR host alias source_id=%d\n",
            alias->victim->cache_plan_source_id);
        return destroy_retention_capacity_entry(
            alias->victim, reason, nullptr,
            &released_bytes, &released_tokens);
    }

    const auto stable_key = [](const host_trade_candidate & candidate) {
        // Keep B's planner-key shape explicit even though this inventory has
        // only the host provider; later mixed-provider trades retain ordering.
        return std::make_tuple(
            uint8_t(common_cache_plan_provider::host_cache_entry),
            candidate.victim->cache_plan_source_id,
            candidate.ranking.ordinal);
    };
    while (true) {
        uint64_t minimum = std::numeric_limits<uint64_t>::max();
        for (const auto & candidate : candidates) {
            if (!candidate.attempted && candidate.ranking.price_known) {
                minimum = std::min(minimum, candidate.ranking.price_us);
            }
        }
        if (minimum == std::numeric_limits<uint64_t>::max()) {
            break;
        }
        const long double floor = std::max<long double>(
            (long double) minimum * COMMON_CACHE_PLAN_TIE_REL_FLOOR,
            COMMON_CACHE_PLAN_TIE_ABS_FLOOR_US);
        host_trade_candidate * chosen = nullptr;
        bool saw_zero = false;
        bool saw_destructive = false;
        for (auto & candidate : candidates) {
            if (candidate.attempted || !candidate.ranking.price_known ||
                (long double) candidate.ranking.price_us >
                    (long double) minimum + floor) {
                continue;
            }
            saw_zero |= candidate.ranking.zero_destruction;
            saw_destructive |= !candidate.ranking.zero_destruction;
            if (!chosen ||
                std::make_tuple(!candidate.ranking.zero_destruction,
                                stable_key(candidate)) <
                std::make_tuple(!chosen->ranking.zero_destruction,
                                stable_key(*chosen))) {
                chosen = &candidate;
            }
        }
        const bool mixed_destruction_tie = saw_zero && saw_destructive;
        GGML_ASSERT(chosen != nullptr);
        chosen->attempted = true;
        chosen->ranking.zero_destruction_tie_break =
            mixed_destruction_tie && chosen->ranking.zero_destruction;
        if (required_victim.v &&
            chosen->ranking.artifact_id != required_victim) {
            return false;
        }

        const uint64_t quote_sequence =
            ++publish_authority->destruction_quote_sequence;
        auto certified = certify_host_destruction(
            *this,
            chosen->victim,
            chosen->recovery,
            quote_sequence,
            true,
            false,
            &chosen->ranking);
        if (!certified.ready) {
            if (destruction_obs) {
                if (certified.quote.receipt.reason ==
                        common_cache_plan_destruction_reason::
                            hard_lease_blocked) {
                    destruction_obs->note_host_trade_veto();
                } else {
                    destruction_obs->note_host_trade_refused();
                }
            }
            if (certified.quote.receipt.reason ==
                    common_cache_plan_destruction_reason::
                        hard_lease_blocked) {
                chosen->hard_leased = true;
            } else if (certified.quote.receipt.reason ==
                    common_cache_plan_destruction_reason::
                        lease_unavailable) {
                chosen->lease_known = false;
            }
            continue;
        }

        if (destruction_obs) {
            destruction_obs->host_trade_attempted++;
        }

        const auto admission = server_prompt_cache_observe_drop(
            *this, *chosen->victim, reason);
        const uint64_t victim_bytes = chosen->victim->size();
        const size_t victim_tokens =
            size_t(chosen->victim->prompt.n_tokens());
        const std::thread::id scheduler_owner = std::this_thread::get_id();
        if (observe_retention_shadow) {
            observe_retention_pressure_choice(
                reason, incoming, chosen->victim, competition_wave_valid);
        }
        SRV_WRN(
            " - removing priced host entry source_id=%d (size = %.3f MiB)\n",
            chosen->victim->cache_plan_source_id,
            chosen->victim->size() / (1024.0 * 1024.0));
        server_prompt_cache_destroy_entry_impl(*this, chosen->victim);
        // lease boundary uses the same no-interleaving terminal as recovery proof. Pricing and all
        // fallible recovery work completed before the physical erase; the raw
        // list mutation has no callback/C writer, and capability commit is the
        // immediately following operation on update_slots' owner thread.
        commit_certified_host_destruction(
            *this, certified, scheduler_owner, &chosen->ranking);
        if (destruction_obs) {
            destruction_obs->note_host_trade_executed(
                admission.sequence,
                certified.projected_bytes,
                chosen->main_family,
                chosen->soft_leased,
                chosen->ranking.zero_destruction_tie_break);
        }
        if (recovery_pin_excluded) {
            emit_host_pressure_floor_outcome(
                *this, "priced_evicted", chosen->ranking.artifact_id,
                chosen->ranking.source_id);
        }
        released_bytes = reason ==
                server_cache_destruction_reason::host_capacity
            ? certified.projected_bytes : victim_bytes;
        released_tokens = victim_tokens;
        return true;
    }

    // Candidates without a fitted/complete price never join a partial
    // optimum. Emit one typed refusal per skipped victim, then retain the
    // exact historical FIFO terminal. No new request is refused merely
    // because destruction evidence is incomplete.
    if (reason == server_cache_destruction_reason::host_capacity) {
        for (auto & candidate : candidates) {
            if (candidate.vbr || candidate.attempted ||
                candidate.ranking.price_known) {
                continue;
            }
            candidate.attempted = true;
            const uint64_t quote_sequence =
                ++publish_authority->destruction_quote_sequence;
            observe_host_trade_refusal(
                *this,
                quote_sequence,
                common_cache_plan_destruction_reason::capacity_refused,
                &candidate.ranking);
            if (destruction_obs) {
                destruction_obs->note_host_trade_unpriced();
            }
        }
    }
    for (const auto & candidate : candidates) {
        if (candidate.lease_known && !candidate.hard_leased &&
            candidate.retirement_ready &&
            candidate.victim->recovery_pins == 0) {
            legacy_floor = candidate.victim;
            break;
        }
    }
    if (legacy_floor == states.end() && std::any_of(
            candidates.begin(), candidates.end(), [](const auto & candidate) {
                return candidate.hard_leased;
            })) {
        floor_reason =
            common_cache_plan_destruction_reason::hard_lease_blocked;
    }

    // Retention capacity replaces only the lawful lifecycle fallback. Host capacity uses
    // exact resident payload bytes; token pressure uses exact prompt tokens.
    // The calibrated/certified byte ladder above, hard leases, pins, and
    // incoming publication retain precedence. Reproject on every victim;
    // record only the first decision in a multi-removal competition wave.
    if (retention_capacity_authority &&
        (reason == server_cache_destruction_reason::host_capacity ||
         reason == server_cache_destruction_reason::host_token_limit) &&
        legacy_floor != states.end()) {
        const auto projection = competition_wave_valid
            ? project_host_trade_retention_capacity(
                  *this, reason, incoming, candidates,
                  retention_shadow_rows.get(),
                  retention_shadow_artifacts.get(),
                  retention_shadow_lineages.get())
            : host_trade_retention_capacity_projection {};
        const auto proposed = projection.artifact;
        if (observe_retention_shadow) {
            const auto increment = [](uint64_t & value) noexcept {
                if (value != UINT64_MAX) {
                    value++;
                }
            };
            increment(retention_shadow.choices);
            auto & event = retention_shadow.last;
            event = {};
            event.reason = reason;
            event.competition_epoch =
                retention_obs->competition_epoch_value();
            event.candidate_count = projection.candidate_count;
            event.incumbent_artifact = host_entry_artifact_id(
                *this, *legacy_floor);
            if (projection.complete && proposed.v != 0 &&
                event.incumbent_artifact.v != 0) {
                event.proposed_artifact = proposed;
                event.proposed_lineage = projection.lineage_id;
                event.proposed_pool = projection.pool;
                event.proposed_lost_work = projection.lost_work;
                event.proposed_resource = projection.resource;
                event.status = server_prompt_cache_shadow_status::complete;
                event.agrees = event.incumbent_artifact == proposed;
                increment(retention_shadow.complete);
                increment(event.agrees
                    ? retention_shadow.agreements
                    : retention_shadow.disagreements);
                if (debug_observability) {
                    SRV_INF(
                        "CACHE_RETENTION_SHADOW status=complete reason=%u "
                        "epoch=%" PRIu64 " candidates=%" PRIu64
                        " incumbent=%" PRIu64 " proposed=%" PRIu64
                        " agrees=%s lost_work=%" PRIu64
                        " resource=%" PRIu64 "\n",
                        unsigned(reason), event.competition_epoch,
                        event.candidate_count,
                        event.incumbent_artifact.v,
                        event.proposed_artifact.v,
                        event.agrees ? "true" : "false",
                        event.proposed_lost_work,
                        event.proposed_resource);
                }
            } else {
                increment(retention_shadow.unavailable);
                if (debug_observability) {
                    SRV_INF(
                        "CACHE_RETENTION_SHADOW status=unavailable "
                        "reason=%u epoch=%" PRIu64
                        " candidates=%" PRIu64 "\n",
                        unsigned(reason), event.competition_epoch,
                        event.candidate_count);
                }
            }
        }
        observe_retention_shadow = false;
        if (projection.release_evidence_complete && proposed.v != 0) {
            const auto selected = std::find_if(
                candidates.begin(), candidates.end(), [&](const auto & value) {
                    return value.ranking.artifact_id == proposed &&
                        value.lease_known && !value.hard_leased &&
                        value.victim != incoming &&
                        value.victim->recovery_pins == 0;
            });
            if (selected != candidates.end()) {
                if (required_victim.v &&
                    selected->ranking.artifact_id != required_victim) {
                    return false;
                }
                const int32_t source_id =
                    selected->victim->cache_plan_source_id;
                const size_t victim_bytes = selected->victim->size();
                vbr_artifact_prepared_retire selected_retire;
                vbr_artifact_prepared_retire * selected_retire_ptr = nullptr;
                if (selected->vbr && !selected->vbr_logical_alias) {
                    if (!acct ||
                        !selected->victim->payload.prepare_vbr_retire(
                            acct->serial(), selected_retire)) {
                        return false;
                    }
                    selected_retire_ptr = &selected_retire;
                }
                if (destroy_retention_capacity_entry(
                        selected->victim, reason,
                        selected_retire_ptr,
                        &released_bytes, &released_tokens)) {
                    if (destruction_obs) {
                        destruction_obs->host_trade_retention_capacity_executed++;
                    }
                    SRV_WRN(
                        " - removing retention-capacity host entry source_id=%d (size = %.3f MiB)\n",
                        source_id, victim_bytes / (1024.0 * 1024.0));
                    return true;
                }
            }
        }
    }
    if (destruction_obs) {
        destruction_obs->host_trade_legacy_fallbacks++;
    }
    return false;
}

void server_prompt_cache::refuse_incoming_under_pressure(
        iterator incoming,
        server_cache_destruction_reason reason) {
    if (incoming == states.end()) {
        return;
    }
    if (incoming->payload.kind() ==
            server_prompt_cache_payload_kind::vbr_artifact) {
        vbr_artifact_prepared_retire prepared;
        const bool alias_only =
            incoming->payload.vbr_logical_erase_only();
        if (!acct ||
            (!alias_only && !incoming->payload.prepare_vbr_retire(
                acct->serial(), prepared)) ||
            !destroy_retention_capacity_entry(
                incoming, reason,
                alias_only ? nullptr : &prepared)) {
            destroy_entry(incoming, reason);
        }
    } else {
        destroy_entry(incoming, reason);
    }
}

bool server_prompt_cache::evict_front_under_pressure(
        server_cache_destruction_reason reason,
        iterator incoming,
        bool competition_wave_valid,
        bool observe_retention_shadow,
        uint64_t & released_bytes,
        size_t & released_tokens,
        llama_cache_acct_artifact_id required_victim) {
    released_bytes = 0;
    released_tokens = 0;
    GGML_ASSERT(!states.empty());
    iterator legacy_floor = states.end();
    common_cache_plan_destruction_reason floor_reason =
        common_cache_plan_destruction_reason::capacity_refused;
    bool recovery_pin_excluded = false;
    if (destroy_priced_host_entry(
            reason, incoming, legacy_floor, floor_reason,
            recovery_pin_excluded, competition_wave_valid,
            observe_retention_shadow, released_bytes,
            released_tokens, required_victim)) {
        return true;
    }

    if (required_victim.v) {
        floor_reason =
            common_cache_plan_destruction_reason::capacity_refused;
        legacy_floor = states.end();
    }

    // Lifecycle-off is the byte-identical historical FIFO floor. Once
    // lifecycle authority exists, the floor skips recovery pins and admits
    // only entries whose inspected lease is known non-hard.
    if (!publish_authority) {
        for (auto it = states.begin(); it != states.end(); ++it) {
            if (it != incoming && it->recovery_pins == 0 &&
                it->payload.vbr_logical_erase_only()) {
                legacy_floor = it;
                break;
            }
        }
        if (legacy_floor == states.end()) {
            for (auto it = states.begin(); it != states.end(); ++it) {
                // Fixed state retains its historical FIFO order. An owned
                // VBR node may join that floor only with an exact catalog
                // retirement capability.
                if (it == incoming || it->recovery_pins != 0) {
                    continue;
                }
                if (it->payload.fixed_state_restorable()) {
                    legacy_floor = it;
                    break;
                }
                vbr_artifact_prepared_retire prepared;
                if (acct && !it->payload.vbr_has_quality_anchor() &&
                    it->payload.prepare_vbr_retire(
                        acct->serial(), prepared)) {
                    legacy_floor = it;
                    break;
                }
            }
        }
    }
    if (legacy_floor != states.end()) {
        const auto floor_artifact = host_entry_artifact_id(
            *this, *legacy_floor);
        const int32_t floor_source_id = legacy_floor->cache_plan_source_id;
        if (observe_retention_shadow) {
            observe_retention_pressure_choice(
                reason, incoming, legacy_floor, competition_wave_valid);
        }
        SRV_WRN(
            " - removing fallback host entry source_id=%d (size = %.3f MiB)\n",
            legacy_floor->cache_plan_source_id,
            legacy_floor->size() / (1024.0 * 1024.0));
        bool floor_evicted = false;
        if (legacy_floor->payload.kind() ==
                server_prompt_cache_payload_kind::vbr_artifact) {
            vbr_artifact_prepared_retire prepared;
            const bool alias_only =
                legacy_floor->payload.vbr_logical_erase_only();
            floor_evicted = acct &&
                (alias_only ||
                 legacy_floor->payload.prepare_vbr_retire(
                     acct->serial(), prepared)) &&
                destroy_retention_capacity_entry(
                    legacy_floor, reason,
                    alias_only ? nullptr : &prepared,
                    &released_bytes, &released_tokens);
        } else {
            if (publish_authority && acct) {
                floor_evicted = destroy_retention_capacity_entry(
                    legacy_floor, reason, nullptr,
                    &released_bytes, &released_tokens);
            } else {
                released_bytes = legacy_floor->size();
                released_tokens =
                    size_t(legacy_floor->prompt.n_tokens());
                destroy_entry(legacy_floor, reason);
                floor_evicted = true;
            }
        }
        if (!floor_evicted) {
            // The VBR quote became unavailable before mutation. Preserve the
            // incumbent and execute the ordinary incoming-refusal terminal.
        } else {
            if (recovery_pin_excluded) {
                emit_host_pressure_floor_outcome(
                    *this, "legacy_evicted", floor_artifact, floor_source_id);
            }
            return true;
        }
    }

    // Hard leases are proof-backed guarantees. If no unpinned, known-nonhard
    // retained entry exists, publication—not an existing protected entry—is
    // the refused operation. Public maintenance without an incoming save
    // simply leaves the configured pressure visible for a later retry.
    if (destruction_obs && incoming != states.end()) {
        destruction_obs->note_host_trade_publication_skip();
    }
    if (publish_authority) {
        const uint64_t sequence =
            ++publish_authority->destruction_quote_sequence;
        observe_host_trade_refusal(*this, sequence, floor_reason);
    }
    if (incoming != states.end()) {
        if (observe_retention_shadow) {
            observe_retention_pressure_choice(
                reason, incoming, incoming, competition_wave_valid);
        }
        refuse_incoming_under_pressure(incoming, reason);
    }
    if (recovery_pin_excluded) {
        emit_host_pressure_floor_outcome(
            *this, "publication_skipped", {}, -1);
    }
    return false;
}

bool server_prompt_cache::destroy_retention_capacity_entry(
        iterator it,
        server_cache_destruction_reason reason,
        vbr_artifact_prepared_retire * vbr_retire,
        uint64_t * released_bytes,
        size_t * released_tokens) {
    if (released_bytes) {
        *released_bytes = 0;
    }
    if (released_tokens) {
        *released_tokens = 0;
    }
    if (!acct || it == states.end()) {
        return false;
    }

    if (it->payload.kind() ==
            server_prompt_cache_payload_kind::vbr_artifact) {
        uint64_t bytes = 0;
        server_prompt_cache_retirement_manifest retirement;
        const bool alias_only = vbr_retire == nullptr &&
            it->payload.vbr_logical_erase_only();
        if ((!alias_only &&
             (!vbr_retire ||
              !vbr_release_resident_bytes(*vbr_retire, bytes))) ||
            bytes > SIZE_MAX ||
            !server_prompt_cache_capture_retirement(
                *this, it, retirement)) {
            return false;
        }
        const size_t tokens = size_t(it->prompt.n_tokens());
        const auto admission =
            server_prompt_cache_observe_drop(*this, *it, reason);
        if (!alias_only && acct->serial() !=
                vbr_retire->preview().accounting_serial) {
            return false;
        }

        const std::thread::id scheduler_owner = std::this_thread::get_id();
        server_prompt_cache_destroy_entry_impl(*this, it);
        GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
        const auto status = alias_only
            ? vbr_artifact_prepared_retire_status::retired
            : vbr_retire->commit();
        // Once the logical owner is gone, commit owns an unconditional
        // current-serial cleanup fallback. `unavailable` would therefore be
        // an internal invariant violation, never a reason to touch a second
        // cache victim after this iterator was erased.
        GGML_ASSERT(status !=
                    vbr_artifact_prepared_retire_status::unavailable);
        server_prompt_cache_retire_manifest(*this, retirement);
        if (destruction_obs && !alias_only) {
            destruction_obs->note_prepared_release(
                admission.sequence, status !=
                    vbr_artifact_prepared_retire_status::unavailable);
        }
        if (released_bytes) {
            *released_bytes = alias_only || status !=
                    vbr_artifact_prepared_retire_status::retired
                ? UINT64_MAX : bytes;
        }
        if (released_tokens) {
            *released_tokens = tokens;
        }
        return true;
    }

    if (!publish_authority) {
        return false;
    }

    server_prompt_cache_retirement_manifest retirement;
    if (!server_prompt_cache_capture_retirement(*this, it, retirement)) {
        return false;
    }
    std::vector<llama_cache_acct_op_id> ops;
    try {
        const auto & release_ops = it->release_ops();
        ops.reserve(release_ops.size());
        for (const auto op : release_ops) {
            if (!op) {
                return false;
            }
            ops.push_back(op);
        }
    } catch (...) {
        return false;
    }

    const uint64_t serial = acct->serial();
    auto prepared = llama_cache_prepare_release_set(*acct, ops, serial);
    if (!prepared.ready()) {
        return false;
    }
    uint64_t exact_bytes = 0;
    if (!vbr_release_resident_bytes(prepared.preview(), exact_bytes)) {
        return false;
    }
    const auto admission =
        server_prompt_cache_observe_drop(*this, *it, reason);
    if (acct->serial() != prepared.accounting_serial()) {
        return false;
    }

    const size_t tokens = size_t(it->prompt.n_tokens());
    const std::thread::id scheduler_owner = std::this_thread::get_id();
    server_prompt_cache_destroy_entry_impl(*this, it);
    GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
    const auto release_status = prepared.commit();
    GGML_ASSERT(release_status ==
                llama_cache_conditional_release_status::released);
    server_prompt_cache_retire_manifest(*this, retirement);
    if (destruction_obs) {
        destruction_obs->note_prepared_release(admission.sequence, true);
    }
    if (released_bytes) {
        *released_bytes = exact_bytes;
    }
    if (released_tokens) {
        *released_tokens = tokens;
    }
    return true;
}

bool server_prompt_cache::destroy_vbr_pair(
        iterator first,
        iterator second,
        server_cache_destruction_reason reason,
        bool first_soft_leased,
        bool second_soft_leased,
        uint64_t & released_bytes,
        size_t & released_tokens) {
    released_bytes = 0;
    released_tokens = 0;
    if (!acct || first == states.end() || second == states.end() ||
        first == second || first->recovery_pins != 0 ||
        second->recovery_pins != 0 ||
        first->payload.kind() !=
            server_prompt_cache_payload_kind::vbr_artifact ||
        second->payload.kind() !=
            server_prompt_cache_payload_kind::vbr_artifact ||
        first->payload.vbr_has_quality_anchor() ||
        second->payload.vbr_has_quality_anchor() ||
        !first->payload.vbr_retirement_exclusive() ||
        !second->payload.vbr_retirement_exclusive() ||
        first->prompt.n_tokens() < 0 || second->prompt.n_tokens() < 0) {
        return false;
    }

    server_prompt_cache_retirement_manifest retirements[2];
    std::vector<const server_prompt_cache_payload *> payloads;
    try {
        payloads.reserve(2);
        payloads.push_back(&first->payload);
        payloads.push_back(&second->payload);
    } catch (...) {
        return false;
    }
    vbr_artifact_prepared_retire prepared;
    uint64_t bytes = 0;
    const size_t first_tokens = size_t(first->prompt.n_tokens());
    const size_t second_tokens = size_t(second->prompt.n_tokens());
    if (second_tokens > SIZE_MAX - first_tokens ||
        !server_prompt_cache_capture_retirement(
            *this, first, retirements[0]) ||
        !server_prompt_cache_capture_retirement(
            *this, second, retirements[1]) ||
        server_fault("vbr_prompt_cache_pair_prepare_fail") ||
        !server_prompt_cache_payload::prepare_vbr_retire_union(
            payloads, acct->serial(), prepared) ||
        !vbr_release_resident_bytes(prepared, bytes) ||
        bytes > SIZE_MAX ||
        acct->serial() != prepared.preview().accounting_serial) {
        return false;
    }

    const auto admission = server_prompt_cache_observe_drop_pair(
        *this, *first, *second, reason, prepared.preview());
    const bool first_main_family = first->main_family;
    const bool second_main_family = second->main_family;
    if (destruction_obs) {
        destruction_obs->host_trade_attempted += 2;
    }
    const std::thread::id scheduler_owner = std::this_thread::get_id();
    server_prompt_cache_destroy_entry_impl(*this, first);
    server_prompt_cache_destroy_entry_impl(*this, second);
    GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
    const auto status = prepared.commit();
    GGML_ASSERT(status !=
                vbr_artifact_prepared_retire_status::unavailable);
    server_prompt_cache_retire_manifest(*this, retirements[0]);
    server_prompt_cache_retire_manifest(*this, retirements[1]);
    if (destruction_obs) {
        const uint64_t sequences[] = { admission.sequence };
        destruction_obs->note_host_trade_executed(
            admission.sequence,
            status == vbr_artifact_prepared_retire_status::retired
                ? bytes : 0,
            first_main_family, first_soft_leased, false);
        destruction_obs->note_host_trade_executed(
            admission.sequence, 0,
            second_main_family, second_soft_leased, false);
        destruction_obs->note_prepared_release_batch(
            sequences, 1, status !=
                vbr_artifact_prepared_retire_status::unavailable);
    }
    released_bytes = status == vbr_artifact_prepared_retire_status::retired
        ? bytes : UINT64_MAX;
    released_tokens = first_tokens + second_tokens;
    return true;
}

server_prompt_cache::iterator server_prompt_cache::destroy_entry_impl(
        iterator it,
        server_cache_destruction_reason reason,
        iterator recovery) {
    const auto admission = server_prompt_cache_observe_drop(*this, *it, reason);
    // This pass-through owns exactly one accounting terminal outside the raw
    // physical primitive. Victim ordering belongs to the caller (historical
    // lifecycle floor or retention-capacity authority); either route executes the
    // same exact terminal through a freshly prepared capability.
    const auto & release_ops = it->release_ops();
    const std::thread::id scheduler_owner = std::this_thread::get_id();

    llama_cache_prepared_release_set prepared;
    server_prompt_cache_retirement_manifest retirement;
    // The legacy-fallback manifest is captured independently from recovery proof's
    // certified manifest: a refused exact-redundancy proof must still execute
    // the historical retirement terminal. Both are read-only snapshots; only
    // the selected terminal retires them after the physical erase.
    const bool retirement_ready = publish_authority && acct &&
        server_prompt_cache_capture_retirement(*this, it, retirement);
    bool capability_ready = false;
    host_destruction_certification redundant;
    if (publish_authority && acct &&
        reason == server_cache_destruction_reason::host_dedup &&
        recovery != states.end()) {
        redundant = certify_host_destruction(
            *this, it, recovery, admission.sequence, false, true);
    }
    if (publish_authority && acct) {
        std::vector<llama_cache_acct_op_id> ops;
        bool setup_ok = retirement_ready;
        try {
            ops.reserve(release_ops.size());
            for (const auto op : release_ops) {
                if (op) {
                    ops.push_back(op);
                }
            }
        } catch (...) {
            setup_ok = false;
        }

        if (!redundant.ready && setup_ok && !ops.empty()) {
            const uint64_t serial = acct->serial();
            prepared = llama_cache_prepare_release_set(
                *acct, ops, serial);
            capability_ready = prepared.ready();
        }
    }

    if (!capability_ready && !redundant.ready) {
        server_prompt_cache_retire_entry(*this, it);
    }
    auto next = server_prompt_cache_destroy_entry_impl(*this, it);
    if (redundant.ready) {
        // recovery proof certify→mutate→commit boundary. Like immutable host restore, publication and
        // dedup run synchronously on update_slots. The physical list erase has
        // no callback or C producer; the immediate capability commit is the
        // only ledger terminal, so no ledger write can interleave. The
        // recovery pin remains live across both operations and prevents the
        // cited survivor from entering this raw primitive.
        commit_certified_host_destruction(
            *this, redundant, scheduler_owner);
        if (destruction_obs) {
            destruction_obs->note_redundant_host_executed(
                admission.sequence, redundant.projected_bytes);
        }
    } else if (capability_ready) {
        GGML_ASSERT(scheduler_owner == std::this_thread::get_id());
        // immutable host restore prepare→mutate→commit boundary. destroy_entry() is called
        // synchronously by update_slots-owned prompt-cache publication/load
        // maintenance. The raw erase only destroys value storage: it has no
        // callback and no C producer. The very next operation commits the
        // prepared release, so no scheduler-owned ledger write can interleave.
        // A future asynchronous producer invalidates this proof and must add
        // a real claim/lock before this authority remains enabled.
        // The same-frame assertion above is a machine-checked refactor
        // tripwire: the CMake contract scan deliberately string-matches it.
        const auto release_status = prepared.commit();
        GGML_ASSERT(release_status ==
                    llama_cache_conditional_release_status::released);
        server_prompt_cache_retire_manifest(*this, retirement);
        if (destruction_obs) {
            destruction_obs->note_prepared_release(
                admission.sequence, true);
        }
    } else if (acct) {
        // Preparation is fail-closed with respect to the new capability, but
        // immutable host restore does not yet own host victim selection: retain the bounded
        // legacy FIFO/dedup behavior and its exactly-one release terminal.
        for (const auto op : release_ops) {
            if (op) {
                (void) acct->release(op);
            }
        }
        if (publish_authority && destruction_obs) {
            destruction_obs->note_prepared_release(
                admission.sequence, false);
        }
    }
    return next;
}

// Release symmetry for whole-cache destruction/replacement (model reload swaps the cache
// object while the observer ledger survives): every charged entry releases before the
// container dies, or the next snapshot would carry phantom host-cache bytes.
void server_prompt_cache::clear_accounting() {
    if (!acct && !destruction_obs && !retention_obs) {
        return;
    }
    for (auto & st : states) {
        server_prompt_cache_observe_drop(
            *this, st, server_cache_destruction_reason::host_shutdown);
        if (acct) {
            acct_release_entry(st);
        }
        if (retention_obs) {
            retention_obs->retire(
                server_retention_instance_key::for_host_entry(&st));
            for (auto & checkpoint : st.prompt.checkpoints) {
                retention_obs->retire(
                    server_retention_instance_key::for_checkpoint(
                        -1, &checkpoint));
            }
        }
    }
}

bool server_prompt_cache::retention_sources_available(
        const server_prompt & source_prompt,
        int32_t source_slot) const noexcept {
    if (!retention_obs || source_slot < 0) {
        return true;
    }
    if (!retention_obs->clone_source_available(
            server_retention_instance_key::for_slot(source_slot))) {
        return false;
    }
    for (const auto & checkpoint : source_prompt.checkpoints) {
        if (!retention_obs->clone_source_available(
                server_retention_instance_key::for_checkpoint(
                    source_slot, &checkpoint))) {
            return false;
        }
    }
    return true;
}

bool server_prompt_cache::vbr_retention_source_available(
        int32_t source_slot) const noexcept {
    return !retention_obs || source_slot < 0 ||
        retention_obs->clone_source_available(
            server_retention_instance_key::for_slot(source_slot));
}

bool server_prompt_cache::publish(
        std::list<server_prompt_cache_state> entry,
        const server_prompt * source_prompt,
        int32_t source_slot,
        iterator * published) {
    return publish_impl(
        std::move(entry), source_prompt, source_slot, published, false, {});
}

bool server_prompt_cache::publish_impl(
        std::list<server_prompt_cache_state> entry,
        const server_prompt * source_prompt,
        int32_t source_slot,
        iterator * published,
        bool vbr_retention_prepared,
        server_prompt_cache_vbr_pressure_citation required_victims,
        int64_t source_vbr_coverage_tokens) {
    if (published) {
        *published = states.end();
    }
    // stage() produces exactly one detached node. Enforce that transaction
    // shape here as well: admission/accounting below bind one logical host
    // entry, so accepting a longer list would let later nodes bypass both.
    if (entry.size() != 1 || !entry.front().payload.publishable()) {
        return false;
    }
    auto & staged = entry.front();
    const bool is_vbr = staged.payload.kind() ==
        server_prompt_cache_payload_kind::vbr_artifact;
    const auto source_matches_staged_vbr = [&]() noexcept {
        if (!source_prompt) {
            return false;
        }
        if (source_vbr_coverage_tokens < 0) {
            return server_prompt_cache_vbr_frontier_matches(
                *source_prompt, staged.payload,
                staged.vbr_execution_identity,
                staged.adapter_config_key);
        }
        try {
            return staged.prompt.n_tokens() == source_vbr_coverage_tokens &&
                   source_prompt->n_tokens() >= source_vbr_coverage_tokens &&
                   source_prompt->sequence_epoch ==
                       staged.prompt.sequence_epoch &&
                   source_prompt->tokens.get_common_prefix(
                       staged.prompt.tokens) ==
                       size_t(source_vbr_coverage_tokens);
        } catch (...) {
            return false;
        }
    };

    // A sealed VBR capability is already charged in the cache's canonical
    // ledger. Borrowed/manual views remain publishable only when they fit
    // without pressure; cache-owned views may enter the canonical pressure
    // transaction, whose prepared retirement quote supplies exact marginal
    // physical bytes.
    if (is_vbr) {
        if (!server_prompt_cache_vbr_frontier_matches(
                staged.prompt, staged.payload,
                staged.vbr_execution_identity,
                staged.adapter_config_key) ||
            !source_matches_staged_vbr() ||
            !acct || !staged.payload.accounted_by(acct) ||
            !retention_obs || source_slot < 0 ||
            !retention_obs->prefix_tracking_enabled() ||
            !retention_obs->prefix_tracking_available() ||
            !staged.prompt.checkpoints.empty() ||
            !retention_obs->clone_source_available(
                server_retention_instance_key::for_slot(source_slot))) {
            return false;
        }
        const uint64_t staged_anchor =
            staged.payload.vbr_anchor_resident_bytes();
        if (staged_anchor > staged.size()) {
            return false;
        }
        const size_t staged_compact =
            staged.size() - size_t(staged_anchor);
        const size_t staged_budget_bytes = quality_anchor_budget_enabled
            ? staged_compact : staged.size();
        if ((limit_size > 0 && staged_budget_bytes > limit_size) ||
            (limit_size == 0 && limit_tokens > 0 &&
             size_t(staged.prompt.n_tokens()) > limit_tokens)) {
            return false;
        }
        size_t total_bytes = staged_budget_bytes;
        size_t total_anchor_bytes = quality_anchor_budget_enabled
            ? size_t(staged_anchor) : 0;
        size_t fixed_bytes = 0;
        size_t total_tokens = size_t(staged.prompt.n_tokens());
        std::vector<const server_prompt_cache_payload *> payloads;
        std::vector<const server_prompt_cache_state *> fixed_states;
        for (const auto & state : states) {
            // An unpinned exact alias is replaced later in this same
            // scheduler transaction. Excluding it makes republication
            // idempotent at an exact byte/token limit; a pinned alias remains
            // a separate logical node and keeps its token charge.
            if (state.recovery_pins == 0 &&
                exactly_redundant(state, staged)) {
                continue;
            }
            if (size_t(state.prompt.n_tokens()) > SIZE_MAX - total_tokens) {
                return false;
            }
            total_tokens += size_t(state.prompt.n_tokens());
            if (!quality_anchor_budget_enabled) {
                if (state.size() > SIZE_MAX - total_bytes) {
                    return false;
                }
                total_bytes += state.size();
            } else {
                const uint64_t anchor = state.payload.kind() ==
                        server_prompt_cache_payload_kind::vbr_artifact
                    ? state.payload.vbr_anchor_resident_bytes() : 0;
                const size_t state_bytes = state.size();
                if (anchor > state_bytes ||
                    size_t(anchor) > SIZE_MAX - total_anchor_bytes ||
                    state_bytes - size_t(anchor) >
                        SIZE_MAX - total_bytes) {
                    return false;
                }
                total_anchor_bytes += size_t(anchor);
                total_bytes += state_bytes - size_t(anchor);
            }
        }
        const bool anchor_exact_needed = quality_anchor_budget_enabled &&
            ((limit_size > 0 && total_bytes > limit_size) ||
             total_anchor_bytes > limit_anchor_size);
        if (anchor_exact_needed) {
            try {
                payloads.reserve(states.size() + 1);
                fixed_states.reserve(states.size());
                payloads.push_back(&staged.payload);
                for (const auto & state : states) {
                    if (state.recovery_pins == 0 &&
                        exactly_redundant(state, staged)) {
                        continue;
                    }
                    if (state.payload.kind() ==
                            server_prompt_cache_payload_kind::vbr_artifact) {
                        payloads.push_back(&state.payload);
                    } else {
                        fixed_states.push_back(&state);
                    }
                }
            } catch (...) {
                return false;
            }
            server_prompt_cache_vbr_budget_summary budgets;
            if (!server_prompt_cache_payload::summarize_vbr_budgets(
                    payloads, budgets) ||
                !server_prompt_cache_measure_fixed_states(
                    fixed_states, acct, fixed_bytes) ||
                budgets.compact_resident_bytes > SIZE_MAX - fixed_bytes) {
                return false;
            }
            total_bytes = fixed_bytes +
                size_t(budgets.compact_resident_bytes);
            total_anchor_bytes = size_t(budgets.anchor_resident_bytes);
        } else if (limit_size > 0 && total_bytes > limit_size && acct) {
            try {
                payloads.reserve(states.size() + 1);
                fixed_states.clear();
                fixed_states.reserve(states.size());
                payloads.push_back(&staged.payload);
                fixed_bytes = 0;
                for (const auto & state : states) {
                    if (state.recovery_pins == 0 &&
                        exactly_redundant(state, staged)) {
                        continue;
                    }
                    if (state.payload.kind() ==
                            server_prompt_cache_payload_kind::vbr_artifact) {
                        payloads.push_back(&state.payload);
                    } else {
                        fixed_states.push_back(&state);
                    }
                }
                server_prompt_cache_vbr_budget_summary budgets;
                if (server_prompt_cache_payload::summarize_vbr_budgets(
                        payloads, budgets) &&
                    server_prompt_cache_measure_fixed_states(
                        fixed_states, acct, fixed_bytes) &&
                    budgets.compact_resident_bytes <=
                        SIZE_MAX - fixed_bytes &&
                    budgets.anchor_resident_bytes <=
                        SIZE_MAX - fixed_bytes -
                        size_t(budgets.compact_resident_bytes)) {
                    total_bytes = fixed_bytes +
                        size_t(budgets.compact_resident_bytes) +
                        size_t(budgets.anchor_resident_bytes);
                }
            } catch (...) {
                // Keep the conservative allocation-free upper bound.
            }
        }
        const size_t effective_token_limit =
            server_prompt_cache_effective_token_limit(
                limit_size, limit_tokens, total_bytes, total_tokens);
        const bool requires_pressure =
            (limit_size > 0 && total_bytes > limit_size) ||
            (limit_tokens > 0 && total_tokens > effective_token_limit);
        if (requires_pressure &&
            !staged.payload.vbr_retirement_exclusive() &&
            !staged.payload.vbr_logical_erase_only()) {
            return false;
        }
    }

    // A host save is one compound payload: the host entry plus every copied
    // checkpoint. Validate all lineage sources before the payload admission
    // and allocation-free list splice below. A restored/trimmed ring can
    // legitimately reach this seam with a checkpoint whose optional sidecar
    // publication was refused; that makes this save a local soft miss, not a
    // reason to poison the complete retention/accounting catalog. The bound
    // is the configured checkpoint-ring size (default 32), and this runs only
    // on the host-save path.
    if (!is_vbr && retention_obs && source_prompt && source_slot >= 0) {
        if (source_prompt->checkpoints.size() !=
                entry.front().prompt.checkpoints.size() ||
            !retention_sources_available(*source_prompt, source_slot)) {
            return false;
        }
    }

    bool fixed_retention_prepared = false;
    const auto retire_fixed_preparation = [&]() noexcept {
        if (!retention_obs) {
            return;
        }
        retention_obs->retire(
            server_retention_instance_key::for_host_entry(&staged));
        for (auto & checkpoint : staged.prompt.checkpoints) {
            retention_obs->retire(
                server_retention_instance_key::for_checkpoint(
                    -1, &checkpoint));
        }
    };
    if (!is_vbr && publish_authority && retention_obs &&
        source_prompt && source_slot >= 0) {
        const auto source_key =
            server_retention_instance_key::for_slot(source_slot);
        const auto destination_key =
            server_retention_instance_key::for_host_entry(&staged);
        fixed_retention_prepared = retention_obs->clone(
            source_key, destination_key) &&
            server_prompt_retention_publish_exact_prefix(
                *retention_obs, destination_key, staged.prompt,
                staged.adapter_config_key, staged.prompt.n_tokens());
        auto source_checkpoint = source_prompt->checkpoints.begin();
        auto host_checkpoint = staged.prompt.checkpoints.begin();
        for (; fixed_retention_prepared &&
               source_checkpoint != source_prompt->checkpoints.end() &&
               host_checkpoint != staged.prompt.checkpoints.end();
               ++source_checkpoint, ++host_checkpoint) {
            fixed_retention_prepared = retention_obs->clone(
                server_retention_instance_key::for_checkpoint(
                    source_slot, &*source_checkpoint),
                server_retention_instance_key::for_checkpoint(
                    -1, &*host_checkpoint));
        }
        if (!fixed_retention_prepared) {
            retire_fixed_preparation();
            return false;
        }
    }

    // Publication-authority boundary: the detached entry is complete, but no shipped cache state has
    // changed yet. Refusal drops only this detached node; the live slot remains the sole copy.
    // The callback commits all accounting leaves before returning true, and states.splice below
    // is allocation-free/noexcept, so accounting can never lag a published entry.
    if (!is_vbr && publish_authority &&
        !publish_authority->admit_host_entry(entry.front())) {
        retire_fixed_preparation();
        return false;
    }

    // The detached list node has its final stable address before splice.
    // Prepare the logical lineage/prefix first so every fallible VBR metadata
    // operation precedes the allocation-free cache publication. On refusal,
    // retire the provisional destination; the live source and sealed catalog
    // capability remain valid. The shared prefix index may separately latch
    // unavailable under its canonical fail-closed contract.
    if (is_vbr && !vbr_retention_prepared) {
        const auto source_key =
            server_retention_instance_key::for_slot(source_slot);
        const auto destination_key =
            server_retention_instance_key::for_host_entry(&staged);
        const bool cloned = retention_obs->clone(
            source_key, destination_key);
        const bool indexed = cloned &&
            !server_fault("vbr_prompt_cache_prefix_fail") &&
            server_prompt_retention_publish_exact_prefix(
                *retention_obs, destination_key, staged.prompt,
                staged.adapter_config_key, staged.prompt.n_tokens());
        if (!indexed) {
            // A real prefix-index insertion failure deliberately makes the
            // shared policy evidence unavailable (the sidecar's canonical
            // fail-closed terminal). Either way, no logical cache node is
            // published and this provisional association is retired.
            retention_obs->retire(destination_key);
            return false;
        }
        const server_cache_lease_subject source {
            retention_obs->artifact_id(source_key),
            common_retention_artifact_kind::live_slot,
            source_slot,
        };
        const server_cache_lease_subject destination {
            retention_obs->artifact_id(destination_key),
            common_retention_artifact_kind::host_entry,
            -1,
        };
        server_prompt_cache_mirror_lease(
            *this, true, &source, destination, staged.prompt,
            staged.adapter_config_key, staged.prompt.n_tokens());
    } else if (is_vbr) {
        // The move-only metadata capability installed this exact association
        // and prefix before D2H. Its detached std::list node retains the same
        // address through this by-value move and the final splice.
        const auto destination_key =
            server_retention_instance_key::for_host_entry(&staged);
        if (!retention_obs ||
            retention_obs->artifact_id(destination_key).v == 0) {
            return false;
        }
    }

    // Splice the pre-allocated node in FIRST (no allocation, no throw) so the new entry is durably
    // linked before any potentially-throwing comparison below. Then remove cached prompts of the
    // same adapter identity fully contained in the new (larger) prompt: a contained entry
    // under a different adapter config is a distinct valid state and is kept. If a comparison throws
    // (OOM) mid-loop, the new entry is already safely in `states`; at worst a few obsolete entries
    // remain (benign, FIFO-evicted later) -- never a lost node or partial corruption.
    states.splice(states.end(), entry);
    const auto self = std::prev(states.end());

    if (acct && !publish_authority) {
        acct_charge_entry(*self);
    }
    if (!is_vbr && retention_obs && source_prompt && source_slot >= 0) {
        const auto source_key =
            server_retention_instance_key::for_slot(source_slot);
        const auto destination_key =
            server_retention_instance_key::for_host_entry(&*self);
        if (!fixed_retention_prepared) {
            (void) server_prompt_cache_mirror_artifact_clone(
                *this,
                source_key, common_retention_artifact_kind::live_slot,
                source_slot,
                destination_key, common_retention_artifact_kind::host_entry,
                -1,
                self->prompt, self->adapter_config_key,
                self->prompt.n_tokens());
        } else {
            const server_cache_lease_subject source_subject {
                retention_obs->artifact_id(source_key),
                common_retention_artifact_kind::live_slot,
                source_slot,
            };
            const server_cache_lease_subject destination_subject {
                retention_obs->artifact_id(destination_key),
                common_retention_artifact_kind::host_entry,
                -1,
            };
            server_prompt_cache_mirror_lease(
                *this, true, &source_subject, destination_subject,
                self->prompt, self->adapter_config_key,
                self->prompt.n_tokens());
        }
        auto source_checkpoint = source_prompt->checkpoints.begin();
        auto host_checkpoint = self->prompt.checkpoints.begin();
        for (; source_checkpoint != source_prompt->checkpoints.end() &&
               host_checkpoint != self->prompt.checkpoints.end();
               ++source_checkpoint, ++host_checkpoint) {
            const auto source_checkpoint_key =
                server_retention_instance_key::for_checkpoint(
                    source_slot, &*source_checkpoint);
            const auto host_checkpoint_key =
                server_retention_instance_key::for_checkpoint(
                    -1, &*host_checkpoint);
            if (!fixed_retention_prepared) {
                (void) server_prompt_cache_mirror_artifact_clone(
                    *this,
                    source_checkpoint_key,
                    common_retention_artifact_kind::checkpoint,
                    source_slot,
                    host_checkpoint_key,
                    common_retention_artifact_kind::checkpoint,
                    -1,
                    self->prompt, self->adapter_config_key,
                    host_checkpoint->n_tokens);
            } else {
                const server_cache_lease_subject source_subject {
                    retention_obs->artifact_id(source_checkpoint_key),
                    common_retention_artifact_kind::checkpoint,
                    source_slot,
                };
                const server_cache_lease_subject destination_subject {
                    retention_obs->artifact_id(host_checkpoint_key),
                    common_retention_artifact_kind::checkpoint,
                    -1,
                };
                server_prompt_cache_mirror_lease(
                    *this, true, &source_subject, destination_subject,
                    self->prompt, self->adapter_config_key,
                    host_checkpoint->n_tokens);
            }
        }
    }

    for (auto it = states.begin(); it != states.end();) {
        if (it != self && it->adapter_config_key == self->adapter_config_key) {
            int len = -1;
            bool obsolete = false;
            if (is_vbr) {
                // Immutable owner identity rejects distinct VBR packages in
                // O(1) before exactly_redundant examines their long prefix.
                obsolete = exactly_redundant(*it, *self);
                if (obsolete) {
                    len = it->prompt.n_tokens();
                }
            } else if (it->payload.fixed_state_restorable()) {
                len = it->prompt.tokens.get_common_prefix(self->prompt.tokens);
                obsolete = len == (int) it->prompt.tokens.size();
            }
            if (obsolete) {
                // A recovery citation outlives its destruction commit
                // through the dependent B execution. Dedup is another victim
                // enumerator: it must leave the cited physical host node in
                // place rather than reaching the raw eraser's invariant
                // assert while that non-policy pin is live.
                if (it->recovery_pins != 0) {
                    ++it;
                    continue;
                }
                SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);
                it = destroy_entry_impl(
                    it, server_cache_destruction_reason::host_dedup,
                    self);
                continue;
            }
        }
        ++it;
    }

    // enforce the cache limits through the single canonical eviction primitive. The entry's bytes
    // are already committed, so a local make-room loop would prevent no memory spike and, being
    // size-only, would skip the token limit. update() enforces both and evicts oldest-first,
    // preserving the just-added entry.
    if (!update_impl(self, required_victims)) {
        return false;
    }
    if (published) {
        *published = self;
    }
    return true;
}

bool server_prompt_cache::prepare_restore_delivery(
        iterator source,
        server_prompt_cache_restore_delivery & delivery) const noexcept {
    delivery = {};
    delivery.cache_family = source->cache_family;
    if (!publish_authority) {
        return true;
    }
    try {
        if (server_fault("load_clone_fail")) {
            return false;
        }
        delivery.prompt = source->prompt.clone();
        delivery.retains_source = true;
        return true;
    } catch (...) {
        delivery = {};
        return false;
    }
}

static void server_prompt_cache_mirror_restore_retention(
        server_prompt_cache & cache,
        server_prompt_cache_state * source,
        server_prompt & destination,
        int32_t id_slot,
        bool retained_source,
        bool continues_lineage) {
    if (!cache.retention_obs) {
        return;
    }
    const auto host_key =
        server_retention_instance_key::for_host_entry(source);
    const auto live_key =
        server_retention_instance_key::for_slot(id_slot);
    if (continues_lineage) {
        server_prompt_cache_mirror_artifact_clone(
            cache,
            host_key, common_retention_artifact_kind::host_entry, -1,
            live_key, common_retention_artifact_kind::live_slot, id_slot,
            destination, source->adapter_config_key,
            destination.n_tokens());
    } else {
        const bool branched = cache.retention_obs->branch(
            host_key, live_key, nullptr, true);
        if (branched) {
            server_prompt_cache_mirror_prefix(
                cache, live_key, destination,
                source->adapter_config_key, destination.n_tokens());
        }
    }
    // Selection does not count as reuse. Carry the immutable host source in a
    // scheduler-consumed transition receipt so successful launch credits it
    // once before admitting a divergent destination; every unlaunched path
    // abandons the receipt and provisional branch.
    (void) cache.retention_obs->prepare_for_launch(host_key, live_key);

    const auto admit_restored_checkpoints = [&]() -> bool {
        if (!cache.publish_authority || !cache.retention_obs) {
            return true;
        }
        try {
            std::vector<server_retention_instance_key> keys;
            std::vector<server_cache_live_checkpoint_admission> batch;
            keys.reserve(destination.checkpoints.size());
            batch.reserve(destination.checkpoints.size());
            size_t checkpoint_ordinal = 0;
            for (const auto & checkpoint : destination.checkpoints) {
                const auto key =
                    server_retention_instance_key::for_checkpoint(
                        id_slot, &checkpoint);
                llama_cache_acct_artifact_id artifact;
                if ((server_fault("restore_checkpoint_artifact_missing") &&
                     checkpoint_ordinal == 1) ||
                    !cache.retention_obs->checkpoint_admission_artifact(
                        key, artifact)) {
                    return false;
                }
                checkpoint_ordinal++;
                server_cache_live_checkpoint_admission member;
                member.artifact = artifact;
                member.checkpoint = &checkpoint;
                keys.push_back(key);
                batch.push_back(std::move(member));
            }
            if (batch.size() != destination.checkpoints.size()) {
                return false;
            }
            if (batch.empty()) {
                return destination.checkpoints.empty();
            }
            if (!cache.publish_authority->admit_live_checkpoints(batch)) {
                SRV_WRN(
                    "restored checkpoint ownership batch admission failed; "
                    "%zu members remain fail-closed\n", batch.size());
                return false;
            }
            GGML_ASSERT(keys.size() == batch.size());
            bool attached = true;
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!cache.retention_obs->attach_release_ops(
                        keys[i], std::move(batch[i].committed))) {
                    SRV_WRN("%s\n",
                        "restored checkpoint ownership attach failed; member remains fail-closed");
                    attached = false;
                }
            }
            return attached;
        } catch (...) {
            SRV_WRN("%s\n",
                "restored checkpoint ownership batch setup failed; ring remains fail-closed");
            return false;
        }
    };

    const auto discard_unowned_checkpoints = [&]() noexcept {
        if (cache.retention_obs) {
            for (const auto & checkpoint : destination.checkpoints) {
                cache.retention_obs->retire(
                    server_retention_instance_key::for_checkpoint(
                        id_slot, &checkpoint));
            }
        }
        destination.checkpoints.clear();
    };

    if (!retained_source) {
        // std::list's equal allocator move transfers the original nodes, so
        // destination checkpoint addresses are the historical host keys.
        for (const auto & checkpoint : destination.checkpoints) {
            const auto host_checkpoint =
                server_retention_instance_key::for_checkpoint(
                    -1, &checkpoint);
            const auto live_checkpoint =
                server_retention_instance_key::for_checkpoint(
                    id_slot, &checkpoint);
            const bool rebound = continues_lineage
                ? cache.retention_obs->rebind(
                    host_checkpoint, live_checkpoint)
                : cache.retention_obs->branch(
                    host_checkpoint, live_checkpoint, &live_key);
            const auto artifact =
                cache.retention_obs->artifact_id(live_checkpoint);
            const server_cache_lease_subject checkpoint_destination {
                artifact,
                common_retention_artifact_kind::checkpoint,
                id_slot,
            };
            server_prompt_cache_mirror_lease(
                cache, rebound, nullptr, checkpoint_destination,
                destination, source->adapter_config_key,
                checkpoint.n_tokens);
        }
        if (!admit_restored_checkpoints()) {
            discard_unowned_checkpoints();
        }
        return;
    }

    auto source_checkpoint = source->prompt.checkpoints.begin();
    auto destination_checkpoint = destination.checkpoints.begin();
    for (; source_checkpoint != source->prompt.checkpoints.end() &&
           destination_checkpoint != destination.checkpoints.end();
           ++source_checkpoint, ++destination_checkpoint) {
        const auto source_key =
            server_retention_instance_key::for_checkpoint(
                -1, &*source_checkpoint);
        const auto destination_key =
            server_retention_instance_key::for_checkpoint(
                id_slot, &*destination_checkpoint);
        if (continues_lineage) {
            server_prompt_cache_mirror_artifact_clone(
                cache,
                source_key, common_retention_artifact_kind::checkpoint, -1,
                destination_key, common_retention_artifact_kind::checkpoint,
                id_slot,
                destination, source->adapter_config_key,
                destination_checkpoint->n_tokens);
        } else {
            (void) cache.retention_obs->branch(
                source_key, destination_key, &live_key);
        }
    }
    GGML_ASSERT(source_checkpoint == source->prompt.checkpoints.end());
    GGML_ASSERT(destination_checkpoint == destination.checkpoints.end());
    if (!admit_restored_checkpoints()) {
        discard_unowned_checkpoints();
    }
}

void server_prompt_cache::release_vbr_occupied_replacement(
        server_prompt_cache_vbr_replacement_ticket & ticket) noexcept {
    if (ticket.cache_ != this) {
        return;
    }
    if (retention_obs && ticket.provisional_key_.instance != 0) {
        retention_obs->abandon_prepared_launch(ticket.provisional_key_);
        retention_obs->retire(ticket.provisional_key_);
    }
    ticket.recovery_pin_.reset();
    ticket.incoming_ = {};
    ticket.clear();
}

bool server_prompt_cache::prepare_vbr_occupied_replacement(
        server_prompt_cache_vbr_restore_candidate && incoming,
        server_prompt & incumbent,
        common_cache_family_binding & incumbent_family,
        const common_cache_family_binding & incoming_family,
        int32_t id_slot,
        const std::string & execution_identity,
        const std::string & adapter_config_key,
        server_prompt_cache_vbr_replacement_ticket & ticket,
        server_prompt_cache_vbr_replacement_diagnostics * diagnostics) noexcept {
    ticket = {};
    if (diagnostics) {
        *diagnostics = {};
    }
    if (!incoming.ready() || incoming.cache_ != this || id_slot < 0 ||
        incoming.prepared_slot_ >= 0 || incoming.prepared_destination_ ||
        incoming.adopted_destination_ || incoming.prepared_prompt_ ||
        !incoming.source_ || !retention_obs || !lease_obs ||
        execution_identity.empty() || adapter_config_key.empty() ||
        !lease_execution_identity ||
        *lease_execution_identity != execution_identity ||
        incumbent.tokens.empty() || incumbent.tokens.has_media() ||
        incumbent.sequence_epoch == 0 ||
        incoming.source_->prompt.tokens.has_media() ||
        !incoming.source_->prompt.checkpoints.empty() ||
        uint64_t(incoming.source_->prompt.n_tokens()) !=
            incoming.source_tokens_ ||
        incoming.prefix_tokens_ > incoming.source_tokens_) {
        return false;
    }
    const uint64_t incumbent_tokens = uint64_t(incumbent.n_tokens());
    const uint64_t live_lcp = incumbent.tokens.get_common_prefix(
        incoming.source_->prompt.tokens);
    if (incoming.prefix_tokens_ <= live_lcp) {
        return false;
    }

    const auto incumbent_key =
        server_retention_instance_key::for_slot(id_slot);
    server_retention_candidate incumbent_retention;
    if (!retention_obs->candidate_for_instance(
            incumbent_key, incumbent_retention) ||
        incumbent_retention.avail !=
            server_retention_candidate_availability::available ||
        incumbent_retention.record.kind !=
            common_retention_artifact_kind::live_slot ||
        incumbent_retention.artifact_id.v == 0 ||
        incumbent_retention.lineage.lineage_id == 0) {
        return false;
    }
    const auto incumbent_artifact = incumbent_retention.artifact_id;

    std::array<uint8_t, 32> incumbent_digest = {};
    std::array<uint8_t, 32> incoming_digest = {};
    server_cache_lease_identity incumbent_lease_identity;
    if (!incumbent.tokens.retention_token_digest(incumbent_digest) ||
        !server_cache_lease_build_identity(
            execution_identity, adapter_config_key, incumbent.tokens,
            int64_t(incumbent_tokens), incumbent_lease_identity)) {
        return false;
    }
    const auto incumbent_lease = lease_obs->inspect_range(
        incumbent_artifact, incumbent_lease_identity,
        incumbent.sequence_epoch, 0, incumbent_tokens);
    if (incumbent_lease.state != server_cache_lease_eval_state::known ||
        server_cache_lease_is_hard(incumbent_lease)) {
        return false;
    }

    iterator recovery = states.end();
    for (auto current = states.begin(); current != states.end(); ++current) {
        if (diagnostics) {
            ++diagnostics->recovery_states_visited;
        }
        if (current->adapter_config_key != adapter_config_key ||
            current->vbr_execution_identity != execution_identity ||
            current->cache_family != incumbent_family ||
            current->prompt.sequence_epoch != incumbent.sequence_epoch ||
            current->prompt.n_tokens() != incumbent.n_tokens() ||
            current->prompt.tokens.has_media() ||
            !current->prompt.checkpoints.empty() ||
            !current->payload.vbr_artifact()) {
            continue;
        }
        std::array<uint8_t, 32> current_digest = {};
        if (!current->prompt.tokens.retention_token_digest(current_digest) ||
            current_digest != incumbent_digest) {
            continue;
        }
        if (diagnostics) {
            ++diagnostics->recovery_digest_matches;
        }
        if (recovery != states.end()) {
            return false;
        }
        recovery = current;
    }
    if (recovery == states.end() ||
        recovery->recovery_pins == std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    const auto recovery_owner = recovery->payload.vbr_compact_owner();
    const auto incoming_owner_artifact =
        incoming.payload_->reference_artifact();
    const auto recovery_owner_artifact =
        recovery_owner ? recovery_owner->reference_artifact()
                       : llama_cache_acct_artifact_id {};
    if (!recovery_owner || incoming_owner_artifact.v == 0 ||
        recovery_owner_artifact.v == 0) {
        return false;
    }
    bool raw_token_comparison = false;
    const bool frontier_matches = server_prompt_cache_vbr_frontier_matches(
            recovery->prompt, recovery->payload,
            execution_identity, adapter_config_key,
            &raw_token_comparison);
    if (diagnostics) {
        diagnostics->recovery_raw_token_comparisons +=
            raw_token_comparison ? 1 : 0;
    }
    if (!frontier_matches) {
        return false;
    }

    server_retention_candidate recovery_retention;
    const auto recovery_key =
        server_retention_instance_key::for_host_entry(&*recovery);
    if (!retention_obs->candidate_for_instance(
            recovery_key, recovery_retention) ||
        recovery_retention.avail !=
            server_retention_candidate_availability::available ||
        recovery_retention.artifact_id.v == 0 ||
        !recovery_retention.provenance_op) {
        return false;
    }
    std::vector<llama_cache_acct_op_id> recovery_ops;
    std::unique_ptr<server_cache_recovery_pin> recovery_pin;
    std::unique_ptr<server_prompt> replacement;
    try {
        recovery_ops.push_back(recovery_retention.provenance_op);
        auto pin = acquire_host_recovery_pin(
            *recovery, { recovery_retention.artifact_id }, recovery_ops);
        if (!pin.valid() || !pin.binds_exact(
                recovery_retention.artifact_id, recovery_ops)) {
            return false;
        }
        recovery_pin = std::make_unique<server_cache_recovery_pin>(
            std::move(pin));
        replacement = std::make_unique<server_prompt>();
        if (incoming.requires_prefix_projection_) {
            replacement->tokens =
                incoming.source_->prompt.tokens.clone_text_prefix(
                    size_t(incoming.prefix_tokens_));
            replacement->sequence_epoch =
                incoming.source_->prompt.sequence_epoch;
        } else {
            *replacement = incoming.source_->prompt.clone();
        }
    } catch (...) {
        return false;
    }
    if (!replacement || replacement->tokens.has_media() ||
        !replacement->checkpoints.empty() ||
        uint64_t(replacement->n_tokens()) != incoming.prefix_tokens_ ||
        replacement->tokens.pos_next() != incoming.selected_next_position_ ||
        !replacement->tokens.retention_token_digest(incoming_digest)) {
        return false;
    }

    const auto source_key =
        server_retention_instance_key::for_host_entry(incoming.source_);
    const server_retention_instance_key provisional_key {
        common_retention_artifact_kind::live_slot,
        id_slot,
        reinterpret_cast<uintptr_t>(replacement.get()),
    };
    if (provisional_key.instance == 0 ||
        retention_obs->artifact_id(provisional_key).v != 0 ||
        !server_prompt_cache_mirror_artifact_clone(
            *this,
            source_key, common_retention_artifact_kind::host_entry, -1,
            provisional_key, common_retention_artifact_kind::live_slot,
            id_slot, *replacement, adapter_config_key,
            int64_t(incoming.prefix_tokens_),
            incoming.requires_prefix_projection_
                ? server_prompt_cache_prefix_clone_mode::branch_prefix
                : server_prompt_cache_prefix_clone_mode::share_source) ||
        !retention_obs->prepare_for_launch(source_key, provisional_key) ||
        retention_obs->artifact_id(incumbent_key) != incumbent_artifact) {
        retention_obs->abandon_prepared_launch(provisional_key);
        retention_obs->retire(provisional_key);
        return false;
    }

    server_prompt_cache_vbr_replacement_ticket prepared;
    try {
        prepared.cache_ = this;
        prepared.incoming_ = std::move(incoming);
        prepared.recovery_source_ = &*recovery;
        prepared.recovery_owner_ = recovery_owner;
        prepared.recovery_pin_ = std::move(recovery_pin);
        prepared.recovery_ops_ = std::move(recovery_ops);
        prepared.incumbent_ = &incumbent;
        prepared.incumbent_family_current_ = &incumbent_family;
        prepared.incumbent_family_ = incumbent_family;
        prepared.incoming_family_ = incoming_family;
        prepared.execution_identity_ = execution_identity;
        prepared.adapter_config_key_ = adapter_config_key;
        prepared.replacement_prompt_ = std::move(replacement);
        prepared.provisional_key_ = provisional_key;
        prepared.destination_slot_ = id_slot;
        prepared.incoming_prefix_tokens_ =
            prepared.incoming_.prefix_tokens_;
        prepared.incumbent_tokens_ = incumbent_tokens;
        prepared.incumbent_live_lcp_ = live_lcp;
        prepared.incumbent_sequence_epoch_ = incumbent.sequence_epoch;
        prepared.incoming_token_digest_ = incoming_digest;
        prepared.incumbent_token_digest_ = incumbent_digest;
        prepared.incumbent_lease_identity_ =
            std::move(incumbent_lease_identity);
        prepared.incumbent_artifact_ = incumbent_artifact;
        prepared.incumbent_lineage_ =
            incumbent_retention.lineage.lineage_id;
        prepared.incoming_owner_artifact_ = incoming_owner_artifact;
        prepared.recovery_owner_artifact_ = recovery_owner_artifact;
        prepared.recovery_host_artifact_ =
            recovery_retention.artifact_id;
        prepared.provisional_artifact_ =
            retention_obs->artifact_id(provisional_key);
    } catch (...) {
        retention_obs->abandon_prepared_launch(provisional_key);
        retention_obs->retire(provisional_key);
        return false;
    }
    if (!prepared.ready()) {
        return false;
    }
    ticket = std::move(prepared);
    return ticket.ready();
}

bool server_prompt_cache::prepare_vbr_occupied_replacement_publish(
        server_prompt_cache_vbr_replacement_ticket & ticket) noexcept {
    if (!ticket.ready() || ticket.cache_ != this || ticket.published_ ||
        !retention_obs) {
        return false;
    }
    const auto occupied_key =
        server_retention_instance_key::for_slot(ticket.destination_slot_);
    if (!retention_obs->prepared_launch_destination_swappable(
            ticket.provisional_key_, occupied_key)) {
        return false;
    }
    ticket.publish_prepared_ = true;
    return true;
}

void server_prompt_cache::publish_vbr_occupied_replacement(
        server_prompt_cache_vbr_replacement_ticket & ticket) noexcept {
    static_assert(std::is_nothrow_swappable_v<server_prompt>);
    GGML_ASSERT(ticket.cache_ == this);
    GGML_ASSERT(ticket.publish_prepared_);
    GGML_ASSERT(!ticket.published_);
    GGML_ASSERT(ticket.ready());
    GGML_ASSERT(ticket.incumbent_ != nullptr);
    GGML_ASSERT(ticket.replacement_prompt_ != nullptr);
    const auto occupied_key =
        server_retention_instance_key::for_slot(ticket.destination_slot_);
    // The scheduler resolved every refusal before entering the adopter's
    // no-fail region. This consumes two existing hash nodes and cannot allocate.
    const bool rebound = retention_obs->swap_prepared_launch_destination(
        ticket.provisional_key_, occupied_key);
    GGML_ASSERT(rebound);
    using std::swap;
    swap(*ticket.incumbent_, *ticket.replacement_prompt_);
    *ticket.incumbent_family_current_ = ticket.incoming_family_;
    ticket.publish_prepared_ = false;
    ticket.published_ = true;
}

void server_prompt_cache::commit_vbr_occupied_replacement(
        server_prompt_cache_vbr_replacement_ticket & ticket,
        server_prompt & destination,
        common_cache_family_binding & destination_family,
        int32_t id_slot) noexcept {
    const bool valid =
        ticket.cache_ == this && ticket.published_ && !ticket.publish_prepared_ &&
        ticket.incumbent_ == &destination && ticket.destination_slot_ == id_slot &&
        retention_obs &&
        retention_obs->artifact_id(
            server_retention_instance_key::for_slot(id_slot)) ==
                ticket.provisional_artifact_ &&
        ticket.incoming_.source_ && ticket.incoming_.payload_ &&
        ticket.incoming_.payload_->reference_artifact() ==
            ticket.incoming_owner_artifact_ &&
        ticket.recovery_owner_ &&
        ticket.recovery_owner_->reference_artifact() ==
            ticket.recovery_owner_artifact_ &&
        ticket.recovery_pin_ &&
        ticket.recovery_pin_->binds_exact(
            ticket.recovery_host_artifact_, ticket.recovery_ops_) &&
        !destination.tokens.has_media() && destination.checkpoints.empty() &&
        uint64_t(destination.n_tokens()) == ticket.incoming_prefix_tokens_ &&
        destination.sequence_epoch ==
            ticket.incoming_.source_->prompt.sequence_epoch;
    GGML_ASSERT(valid);
    std::array<uint8_t, 32> destination_digest = {};
    GGML_ASSERT(destination.tokens.retention_token_digest(destination_digest));
    GGML_ASSERT(destination_digest == ticket.incoming_token_digest_);
    GGML_ASSERT(&destination_family == ticket.incumbent_family_current_);
    GGML_ASSERT(destination_family == ticket.incoming_family_);
    if (destruction_obs) {
        destruction_obs->note_host_restore(true);
    }
    // Both durable host owners and the old prompt image remain held until this
    // post-adopt terminal. Releasing them cannot affect the installed slot.
    ticket.recovery_pin_.reset();
    ticket.recovery_owner_.reset();
    ticket.incoming_ = {};
    ticket.clear();
}

bool server_prompt_cache::prepare_vbr_restore_destination(
        server_prompt_cache_vbr_restore_candidate & candidate,
        server_prompt & destination,
        int32_t id_slot) noexcept {
    if (!candidate.ready() || candidate.cache_ != this || id_slot < 0 ||
        !retention_obs || !destination.tokens.empty() ||
        !destination.checkpoints.empty() || destination.sequence_epoch != 0) {
        return false;
    }
    if (candidate.prepared_slot_ >= 0) {
        return candidate.prepared_slot_ == id_slot &&
            candidate.prepared_destination_ == &destination;
    }
    auto * source = candidate.source_;
    const auto source_key =
        server_retention_instance_key::for_host_entry(source);
    const auto destination_key =
        server_retention_instance_key::for_slot(id_slot);
    if (retention_obs->artifact_id(destination_key).v != 0) {
        return false;
    }
    try {
        if (candidate.requires_prefix_projection_) {
            if (source->prompt.tokens.has_media() ||
                !source->prompt.checkpoints.empty() ||
                uint64_t(source->prompt.n_tokens()) !=
                    candidate.source_tokens_ ||
                candidate.prefix_tokens_ >= candidate.source_tokens_) {
                return false;
            }
            candidate.prepared_prompt_ = std::make_unique<server_prompt>();
            candidate.prepared_prompt_->tokens =
                source->prompt.tokens.clone_text_prefix(
                    size_t(candidate.prefix_tokens_));
            candidate.prepared_prompt_->sequence_epoch =
                source->prompt.sequence_epoch;
            if (!candidate.prepared_prompt_->checkpoints.empty() ||
                candidate.prepared_prompt_->tokens.has_media() ||
                candidate.prepared_prompt_->tokens.pos_next() !=
                    candidate.selected_next_position_) {
                candidate.prepared_prompt_.reset();
                return false;
            }
        } else {
            candidate.prepared_prompt_ =
                std::make_unique<server_prompt>(source->prompt.clone());
        }
    } catch (...) {
        candidate.prepared_prompt_.reset();
        return false;
    }
    bool mirrored = false;
    if (candidate.requires_prefix_projection_) {
        // Divergence inside the parent is a new probationary lineage. Keep
        // source credit in the prepared-launch ticket, publish only the
        // shortened destination prefix, and admit it only when the scheduler
        // consumes the successful launch terminal.
        mirrored = server_prompt_cache_mirror_artifact_clone(
            *this,
            source_key, common_retention_artifact_kind::host_entry, -1,
            destination_key, common_retention_artifact_kind::live_slot,
            id_slot, *candidate.prepared_prompt_,
            source->adapter_config_key,
            int64_t(candidate.prefix_tokens_),
            server_prompt_cache_prefix_clone_mode::branch_prefix);
    } else {
        mirrored = server_prompt_cache_mirror_artifact_clone(
            *this,
            source_key, common_retention_artifact_kind::host_entry, -1,
            destination_key, common_retention_artifact_kind::live_slot,
            id_slot, *candidate.prepared_prompt_,
            source->adapter_config_key,
            int64_t(candidate.prefix_tokens_),
            server_prompt_cache_prefix_clone_mode::share_source);
    }
    if (!mirrored ||
        !retention_obs->prepare_for_launch(source_key, destination_key)) {
        retention_obs->abandon_prepared_launch(destination_key);
        retention_obs->retire(destination_key);
        candidate.prepared_prompt_.reset();
        return false;
    }
    candidate.prepared_slot_ = id_slot;
    candidate.prepared_destination_ = &destination;
    return true;
}

bool server_prompt_cache::publish_vbr_restore(
        server_prompt_cache_vbr_restore_candidate & candidate) noexcept {
    if (!candidate.ready() || candidate.cache_ != this ||
        candidate.prepared_slot_ < 0 || !candidate.prepared_destination_ ||
        !candidate.prepared_prompt_ ||
        candidate.adopted_destination_ || !retention_obs ||
        !retention_obs->prepared_for_launch(
            server_retention_instance_key::for_slot(
                candidate.prepared_slot_))) {
        return false;
    }
    auto & destination = *candidate.prepared_destination_;
    if (!destination.tokens.empty() || !destination.checkpoints.empty() ||
        destination.sequence_epoch != 0) {
        return false;
    }
    destination = std::move(*candidate.prepared_prompt_);
    candidate.adopted_destination_ = &destination;
    return true;
}

bool server_prompt_cache::commit_vbr_restore(
        server_prompt_cache_vbr_restore_candidate & candidate,
        server_prompt & destination,
        common_cache_family_binding & destination_family,
        int32_t id_slot) noexcept {
    if (!candidate.ready() || candidate.cache_ != this ||
        candidate.prepared_slot_ != id_slot || !retention_obs ||
        candidate.adopted_destination_ != &destination) {
        return false;
    }
    auto * source = candidate.source_;
    const auto * variants = source ? source->payload.vbr_variants() : nullptr;
    const bool selected_owner_matches = variants &&
        (variants->compact_current() == candidate.payload_ ||
         variants->quality_anchor() == candidate.payload_);
    if (!source || source->recovery_pins == 0 ||
        !selected_owner_matches ||
        !retention_obs->prepared_for_launch(
            server_retention_instance_key::for_slot(id_slot))) {
        release_vbr_restore(candidate);
        return false;
    }
    // Exact continuation restores the parent's family. A projected prefix is
    // a new request branch; its caller-provided incoming family must survive
    // the restore instead of inheriting the parent's cache provenance.
    if (!candidate.requires_prefix_projection_) {
        destination_family = candidate.cache_family_;
    }
    if (destruction_obs) {
        destruction_obs->note_host_restore(true);
    }
    if (debug_observability) {
        try {
            debug_lifecycle_emissions++;
            SRV_INF(
                "CACHE_HOST_LIFECYCLE {\"mode\":\"vbr_non_consuming\","
                "\"payload_kind\":\"vbr_artifact\","
                "\"source_id\":%d,\"host_entries\":%zu,"
                "\"host_bytes\":%zu,\"prefix_tokens\":%" PRIu64 "}\n",
                candidate.source_id_, states.size(), size(),
                candidate.prefix_tokens_);
        } catch (...) {
            // Observability cannot change a committed restore terminal.
        }
    }
    --source->recovery_pins;
    candidate.clear();
    return true;
}

void server_prompt_cache::commit_restore_delivery(
        iterator source,
        server_prompt_cache_restore_delivery && delivery,
        server_prompt & destination,
        int32_t id_slot,
        int32_t debug_source_id,
        uint64_t reused_prefix_tokens,
        bool continues_lineage) {
    if (delivery.retains_source) {
        GGML_ASSERT(publish_authority != nullptr);
        destination = std::move(delivery.prompt);
        server_prompt_cache_mirror_restore_retention(
            *this, &*source, destination, id_slot, true,
            continues_lineage);
        if (retention_obs && reused_prefix_tokens == 0) {
            retention_obs->abandon_prepared_launch(
                server_retention_instance_key::for_slot(id_slot));
        }
        if (destruction_obs) {
            destruction_obs->note_host_restore(true);
        }
        if (debug_observability) {
            debug_lifecycle_emissions++;
            SRV_INF(
                "CACHE_HOST_LIFECYCLE {\"mode\":\"non_consuming\","
                "\"payload_kind\":\"fixed_state\","
                "\"source_id\":%d,\"host_entries\":%zu,"
                "\"host_bytes\":%zu,\"retained_restores\":%" PRIu64 "}\n",
                debug_source_id, states.size(), size(),
                destruction_obs
                    ? destruction_obs->host_restores_retained
                    : uint64_t(0));
        }
        return;
    }

    // Lifecycle-off is the historical move/rebind/erase terminal verbatim.
    destination = std::move(source->prompt);
    server_prompt_cache_mirror_restore_retention(
        *this, &*source, destination, id_slot, false,
        continues_lineage);
    if (retention_obs && reused_prefix_tokens == 0) {
        retention_obs->abandon_prepared_launch(
            server_retention_instance_key::for_slot(id_slot));
    }
    if (destruction_obs) {
        destruction_obs->note_host_restore(false);
    }
    destroy_entry(
        source, server_cache_destruction_reason::host_consumed_restore);
}

// The observed/unobserved split is a compile-time instantiation: with the observer
// off, load() runs the pre-cache-plan observer candidate loop with zero observer branches. Single source —
// every `if constexpr (Observed)` block vanishes from the <false> instantiation.
template <bool Observed>
bool server_prompt_cache::load_impl(
        server_prompt & prompt, const server_tokens & tokens_new,
        llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
        const std::string & adapter_config_key, common_cache_plan_record * rec,
        int32_t required_source_id,
        common_cache_family_binding * restored_family,
        server_prompt_cache_restore_shape & restore_shape) {
    restore_shape = server_prompt_cache_restore_shape::none;
    if constexpr (!Observed) {
        (void) rec;
        (void) required_source_id;
    }
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // Observer tallies exist only in the observed instantiation and only carry
    // values this selection computes anyway
    [[maybe_unused]] int32_t obs_source_best = -1;
    [[maybe_unused]] int     obs_lcp_sel  = 0;
    int reuse_lcp_best = 0;

    // find the most similar cached prompt, that would also preserve the most context.
    // Observer transport [observer, noexcept]: ONE row per visited entry, keyed by its
    // request-local immutable source id; every evaluated survivor starts as a cost loser and
    // the shipped winner is promoted to accepted after the scan. find_or_add returning
    // nullptr = inventory overflow — the provider's state latches and rows stop, the
    // shipped scan is untouched.
    for (auto it = states.begin(); it != states.end(); ++it) {
        // Keep fixed-state inventory byte-identical to the planner. VBR nodes
        // use the typed restore selector; reject them before assigning a
        // bounded fixed source ID or scanning their long prefix here.
        if (!it->payload.fixed_state_restorable()) {
            continue;
        }
        [[maybe_unused]] common_cache_plan_candidate * row = nullptr;
        [[maybe_unused]] int32_t obs_source = -1;
        if constexpr (Observed) {
            if (cache_plan_get_source_id(*it, obs_source)) {
                // Required-provider authority evaluated every host row before
                // mutation. Save-before-load may deduplicate the list, but a
                // surviving node keeps its request-local id; skip non-selected
                // states before their O(context) token LCP.
                if (required_source_id >= 0 &&
                    obs_source != required_source_id) {
                    continue;
                }
                row = rec->find_or_add(
                    common_cache_plan_provider::host_cache_entry,
                    obs_source, COMMON_CACHE_PLAN_PHASE_HOST_SCAN,
                    rec->id_slot, rec->selection);
                if (row) {
                    row->payload_kind =
                        common_cache_plan_payload_kind::fixed_state;
                }
            } else {
                rec->inventory_states[size_t(
                    common_cache_plan_provider::host_cache_entry)] =
                    common_cache_plan_inventory_state::overflowed;
                if (required_source_id >= 0) {
                    continue;
                }
            }
        }

        int lcp_cur = 0;
        if constexpr (Observed) {
            lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
            server_cache_plan_apply_host(row, server_cache_plan_evaluate_host(
                it->payload.fixed_state_restorable(),
                it->adapter_config_key == adapter_config_key,
                lcp_cur, tokens_new.size(), it->prompt.tokens.size(),
                it->payload.size()));
        }

        // never serve state built under a different adapter configuration: token LCP alone
        // would hand adapter A's KV to a base/adapter-B request. Live-slot rebinds are caught at
        // launch, but a host-cache entry is restored here during prefill, after that check.
        if (it->adapter_config_key != adapter_config_key) {
            continue;
        }

        if constexpr (!Observed) {
            // Preserve the shipped instantiation: the potentially O(n)
            // LCP scan occurs only after both O(1) reject guards.
            lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
        }

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if constexpr (Observed) {
            if (required_source_id >= 0) {
                // primary cache planner exact-provider authority: the planner already selected
                // this complete host plan. Preserve all structural/identity
                // guards above, but do not re-run the legacy two-axis choice.
                it_best = it;
                f_keep_best = f_keep_cur;
                f_sim_best = f_sim_cur;
                obs_source_best = obs_source;
                obs_lcp_sel = lcp_cur;
                reuse_lcp_best = lcp_cur;
                continue;
            }
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
            reuse_lcp_best = lcp_cur;
            if constexpr (Observed) {
                obs_source_best = obs_source;
                obs_lcp_sel  = lcp_cur; // the winner's exact LCP, from the shipped computation
            }
        }
    }

    if constexpr (Observed) {
        // the scan visits every entry (no short-circuit): the declared domain is complete
        // even when it is empty
        rec->note_inventory_complete(common_cache_plan_provider::host_cache_entry);
        if (it_best != states.end() && obs_source_best >= 0) {
            auto * win = rec->find_or_add(common_cache_plan_provider::host_cache_entry,
                                          obs_source_best, COMMON_CACHE_PLAN_PHASE_HOST_SCAN,
                                          rec->id_slot, rec->selection);
            if (win) {
                win->payload_kind =
                    common_cache_plan_payload_kind::fixed_state;
                win->accept(); // shipped winner: promote over the scan-time cost-loser default
                win->lcp_tokens    = llama_cache_acct_value::measured((uint64_t) obs_lcp_sel);
                // bytes the restore actually installs (main+draft state) — NOT entry
                // size(), which also sums every retained checkpoint.
                win->payload_bytes = llama_cache_acct_value::measured(
                    (uint64_t) it_best->payload.size());
                rec->select(common_cache_plan_provider::host_cache_entry, win);
            }
        }
    }

    if (it_best == states.end()) {
        if constexpr (Observed) {
            if (required_source_id >= 0) {
                return false;
            }
        }
        // nothing better than the slot's current state; leave the slot as-is
        return true;
    }

    SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    // immutable host restore stages the immutable source copy before either main or draft
    // context is touched. Allocation failure therefore leaves the source and
    // both target contexts at the caller-owned pre-restore boundary.
    server_prompt_cache_restore_delivery delivery;
    if (!prepare_restore_delivery(it_best, delivery)) {
        SRV_ERR("%s\n", "failed to stage non-consuming host restore");
        return false;
    }

    // Source bytes remain immutable throughout both restores. Lifecycle mode
    // keeps the entry after success too; legacy mode consumes it only after
    // BOTH sides succeed. On any failure the source remains fully intact and
    // the caller resets both target sequences, never leaving a half-restore.
    {
        const auto * fixed = it_best->payload.fixed_state();
        GGML_ASSERT(fixed != nullptr);
        const size_t size_tgt = fixed->main.size();
        size_t n_tgt = llama_state_seq_set_data_ext(
            ctx_tgt, fixed->main.data(), size_tgt,
            id_slot, 0);
        if (server_fault("load_fail")) { n_tgt = size_tgt > 0 ? size_tgt - 1 : 0; } // Restore fault seam.
        if (n_tgt != size_tgt) {
            SRV_ERR("failed to restore target state (%zu != %zu bytes)\n", n_tgt, size_tgt);
            if constexpr (Observed) {
                // the accepted row was ELIGIBILITY; the attempted restore failed short —
                // note_reject demotes the disposition and records the structural reason
                if (auto * sel = rec->selected_row(common_cache_plan_provider::host_cache_entry)) {
                    sel->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
                }
            }
            return false;
        }
    }

    const auto * fixed = it_best->payload.fixed_state();
    GGML_ASSERT(fixed != nullptr);
    bool draft_image_restored = false;
    if (ctx_dft && !fixed->drft.empty()) {
        const size_t size_dft = fixed->drft.size();
        const size_t n_dft = llama_state_seq_set_data_ext(
            ctx_dft, fixed->drft.data(), size_dft,
            id_slot, 0);
        if (n_dft != size_dft) {
            SRV_WRN("failed to restore draft state (%zu != %zu bytes)\n", n_dft, size_dft);
            if constexpr (Observed) {
                if (auto * sel = rec->selected_row(common_cache_plan_provider::host_cache_entry)) {
                    sel->note_reject(COMMON_CACHE_PLAN_REASON_PAYLOAD_SHORT);
                }
            }
            return false;
        }
        draft_image_restored = true;
    }

    // Both sides restored: atomically select the lifecycle retain terminal or
    // the historical move+release+erase terminal.
    if constexpr (Observed) {
        if (auto * sel = rec->selected_row(common_cache_plan_provider::host_cache_entry)) {
            sel->delivered = true; // recorded at the delivery point, never inferred [cache-plan observer]
        }
    }
    if (restored_family) {
        *restored_family = delivery.cache_family;
    }
    restore_shape = draft_image_restored
        ? server_prompt_cache_restore_shape::target_and_draft
        : server_prompt_cache_restore_shape::target_only;
    commit_restore_delivery(
        it_best, std::move(delivery), prompt, id_slot, obs_source_best,
        uint64_t(std::max(reuse_lcp_best, 0)),
        reuse_lcp_best == int(it_best->prompt.tokens.size()));

    return true;
}

template bool server_prompt_cache::load_impl<false>(server_prompt &, const server_tokens &, llama_context *, llama_context *, int32_t, const std::string &, common_cache_plan_record *, int32_t, common_cache_family_binding *, server_prompt_cache_restore_shape &);
template bool server_prompt_cache::load_impl<true>(server_prompt &, const server_tokens &, llama_context *, llama_context *, int32_t, const std::string &, common_cache_plan_record *, int32_t, common_cache_family_binding *, server_prompt_cache_restore_shape &);

bool server_prompt_cache::load(
        server_prompt & prompt, const server_tokens & tokens_new,
        llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
        const std::string & adapter_config_key,
        server_prompt_cache_restore_shape & restore_shape,
        common_cache_plan_record * rec, int32_t required_source_id,
        common_cache_family_binding * restored_family) {
    GGML_ASSERT(rec != nullptr || required_source_id < 0);
    // One dispatch outside every loop: the off path is the original loop.
    return rec != nullptr
        ? load_impl<true>(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot, adapter_config_key, rec, required_source_id, restored_family, restore_shape)
        : load_impl<false>(prompt, tokens_new, ctx_tgt, ctx_dft, id_slot, adapter_config_key, nullptr, required_source_id, restored_family, restore_shape);
}

void server_prompt_cache::update() {
    (void) update_impl(states.end());
}

bool server_prompt_cache::enforce_quality_anchor_budget(
        iterator incoming,
        uint64_t competition_epoch,
        size_t & anchor_bytes) {
    static_assert(SERVER_PROMPT_CACHE_VBR_ANCHOR_MAX_CANDIDATES ==
        SERVER_RETENTION_MAX_CANDIDATES);
    if (anchor_bytes <= limit_anchor_size) {
        return true;
    }
    // An anchor is optional acceleration state. If complete parent-value
    // evidence is unavailable during publication, discard only the incoming
    // anchor and preserve its already-published compact checkpoint. This is
    // also the no-ranking fast terminal for a zero anchor budget.
    const auto discard_incoming_anchor = [&]() noexcept {
        if (incoming == states.end() ||
            !incoming->payload.vbr_has_quality_anchor()) {
            return false;
        }
        server_prompt_cache_payload compact;
        if (!incoming->payload.prepare_vbr_compact_only(compact)) {
            return false;
        }
        if (!acct) {
            return false;
        }
        std::vector<const server_prompt_cache_payload *> selected;
        try {
            selected.push_back(&incoming->payload);
        } catch (...) {
            return false;
        }
        std::vector<vbr_artifact_prepared_retire> prepared_retires;
        if (!server_prompt_cache_payload::prepare_vbr_anchor_retire_batch(
                selected, acct->serial(), prepared_retires)) {
            return false;
        }
        incoming->payload = std::move(compact);
        for (auto & prepared : prepared_retires) {
            const auto status = prepared.commit();
            GGML_ASSERT(status !=
                vbr_artifact_prepared_retire_status::unavailable);
        }
        quality_anchor_retires++;
        const auto measured = server_prompt_cache_measure_budgets(states, acct);
        if (!measured.exact) {
            return false;
        }
        anchor_bytes = measured.anchor;
        return true;
    };
    const auto refuse_or_discard_incoming = [&]() noexcept {
        if (discard_incoming_anchor() &&
            anchor_bytes <= limit_anchor_size) {
            return true;
        }
        quality_anchor_refusals++;
        return false;
    };
    if (!publish_authority || !acct || !retention_obs || !lease_obs ||
        !lease_execution_identity || competition_epoch == 0) {
        return refuse_or_discard_incoming();
    }

    struct anchor_binding {
        llama_cache_acct_artifact_id artifact;
        iterator state;
    };
    try {
        std::vector<server_live_retention_candidate> rows(
            SERVER_RETENTION_MAX_CANDIDATES);
        struct fill_context {
            server_live_retention_candidate * rows = nullptr;
            size_t capacity = 0;
            size_t size = 0;
        } fill { rows.data(), rows.size(), 0 };
        const auto visitor = [](void * opaque,
                const server_retention_value_snapshot & value) noexcept {
            auto & context = *static_cast<fill_context *>(opaque);
            if (context.size == context.capacity ||
                context.size > size_t(INT32_MAX)) {
                return false;
            }
            auto & row = context.rows[context.size];
            row.slot_id = int32_t(context.size);
            row.artifact_id = value.artifact_id;
            row.stamp = value.stamp;
            row.lineage = value.lineage;
            row.external_shared_coverage_tokens =
                value.external_shared_coverage_tokens;
            row.present = true;
            row.eligible = false;
            context.size++;
            return true;
        };
        const auto inventory = retention_obs->value_snapshots(
            &fill, visitor);
        if (inventory.status !=
                server_retention_value_snapshot_status::complete ||
            inventory.size != fill.size || fill.size == 0) {
            return refuse_or_discard_incoming();
        }
        rows.resize(fill.size);

        std::vector<std::pair<uint64_t, size_t>> row_index;
        row_index.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            row_index.push_back({ rows[i].artifact_id.v, i });
        }
        std::sort(row_index.begin(), row_index.end());
        if (std::adjacent_find(
                row_index.begin(), row_index.end(),
                [](const auto & a, const auto & b) {
                    return a.first == b.first;
                }) != row_index.end()) {
            return refuse_or_discard_incoming();
        }

        lease_obs->lifecycle_point();
        std::vector<anchor_binding> bindings;
        bindings.reserve(states.size());
        bool have_anchor_binding = false;
        uint32_t ordinal = 0;
        for (auto it = states.begin(); it != states.end(); ++it, ++ordinal) {
            if (it->payload.kind() !=
                    server_prompt_cache_payload_kind::vbr_artifact) {
                continue;
            }
            host_trade_candidate priced;
            (void) host_trade_price(
                *this, it, ordinal,
                server_cache_destruction_reason::host_capacity,
                nullptr, priced);
            const auto found = std::lower_bound(
                row_index.begin(), row_index.end(),
                std::pair<uint64_t, size_t> {
                    priced.ranking.artifact_id.v, 0 });
            if (!priced.ranking.artifact_id.v ||
                found == row_index.end() ||
                found->first != priced.ranking.artifact_id.v ||
                !priced.lease_known) {
                return refuse_or_discard_incoming();
            }
            auto & row = rows[found->second];
            if (row.eligible) {
                return refuse_or_discard_incoming();
            }
            if (it->payload.vbr_has_quality_anchor()) {
                row.eligible =
                    it->payload.vbr_anchor_resident_bytes() != 0 &&
                    !priced.hard_leased && !priced.mandatory_anchor &&
                    it->recovery_pins == 0;
                have_anchor_binding = true;
            }
            bindings.push_back({ priced.ranking.artifact_id, it });
        }
        if (!have_anchor_binding || !server_live_retention_prepare(
                rows.data(), rows.size(), competition_epoch)) {
            return refuse_or_discard_incoming();
        }
        std::vector<server_anchor_parent_rank> parent_values;
        if (!server_anchor_parent_values_prepared(
                rows.data(), rows.size(), competition_epoch, parent_values)) {
            return refuse_or_discard_incoming();
        }
        std::sort(bindings.begin(), bindings.end(), [](const auto & a,
                                                       const auto & b) {
            return a.artifact.v < b.artifact.v;
        });
        std::sort(parent_values.begin(), parent_values.end(), [](const auto & a,
                                                                 const auto & b) {
            return a.artifact_id.v < b.artifact_id.v;
        });

        std::vector<server_prompt_cache_vbr_anchor_plan_candidate>
            plan_candidates;
        plan_candidates.reserve(bindings.size());
        for (const auto & binding : bindings) {
            const auto ranked = std::lower_bound(
                parent_values.begin(), parent_values.end(), binding.artifact.v,
                [](const auto & value, uint64_t artifact) {
                    return value.artifact_id.v < artifact;
                });
            server_prompt_cache_vbr_anchor_plan_candidate candidate;
            candidate.payload = &binding.state->payload;
            candidate.artifact_id = binding.artifact;
            candidate.eligible = ranked != parent_values.end() &&
                ranked->artifact_id == binding.artifact;
            if (candidate.eligible) {
                candidate.parent_value_q = ranked->parent_value.lost_value_q;
                candidate.recency_ordinal =
                    ranked->parent_value.recency_ordinal;
                candidate.pool = uint8_t(ranked->pool);
                candidate.lineage_id = ranked->lineage_id;
            }
            plan_candidates.push_back(candidate);
        }
        std::vector<llama_cache_acct_artifact_id> selected;
        if (!server_prompt_cache_plan_vbr_anchor_releases(
                plan_candidates, anchor_bytes, limit_anchor_size,
                selected)) {
            return refuse_or_discard_incoming();
        }

        struct anchor_terminal {
            iterator state;
            server_prompt_cache_payload compact;
        };
        std::vector<anchor_terminal> terminals;
        terminals.reserve(selected.size());
        for (const auto artifact : selected) {
            const auto binding = std::lower_bound(
                bindings.begin(), bindings.end(), artifact.v,
                [](const auto & value, uint64_t id) {
                    return value.artifact.v < id;
                });
            if (binding == bindings.end() ||
                binding->artifact != artifact ||
                binding->state == states.end() ||
                !binding->state->payload.vbr_has_quality_anchor() ||
                binding->state->recovery_pins != 0) {
                terminals.clear();
                return refuse_or_discard_incoming();
            }
            host_trade_candidate current;
            (void) host_trade_price(
                *this, binding->state, 0,
                server_cache_destruction_reason::host_capacity,
                nullptr, current);
            if (!current.lease_known || current.hard_leased ||
                current.mandatory_anchor ||
                current.ranking.artifact_id != artifact) {
                terminals.clear();
                return refuse_or_discard_incoming();
            }
            anchor_terminal terminal;
            terminal.state = binding->state;
            if (!binding->state->payload.prepare_vbr_compact_only(
                    terminal.compact)) {
                terminals.clear();
                return refuse_or_discard_incoming();
            }
            terminals.push_back(std::move(terminal));
        }
        std::vector<const server_prompt_cache_payload *> selected_payloads;
        selected_payloads.reserve(terminals.size());
        for (const auto & terminal : terminals) {
            selected_payloads.push_back(&terminal.state->payload);
        }
        std::vector<vbr_artifact_prepared_retire> prepared_retires;
        if (server_fault("vbr_anchor_prepare_fail") ||
            !server_prompt_cache_payload::prepare_vbr_anchor_retire_batch(
                selected_payloads, acct->serial(), prepared_retires)) {
            terminals.clear();
            return refuse_or_discard_incoming();
        }
        // Every allocation, lease, replacement, and last-owner retirement
        // capability is ready before the first node changes variant.
        for (auto & terminal : terminals) {
            terminal.state->payload = std::move(terminal.compact);
            quality_anchor_retires++;
        }
        for (auto & prepared : prepared_retires) {
            const auto status = prepared.commit();
            GGML_ASSERT(status !=
                vbr_artifact_prepared_retire_status::unavailable);
        }
        const auto measured = server_prompt_cache_measure_budgets(states, acct);
        if (!measured.exact) {
            return refuse_or_discard_incoming();
        }
        anchor_bytes = measured.anchor;
        if (anchor_bytes > limit_anchor_size) {
            return refuse_or_discard_incoming();
        }
        return true;
    } catch (...) {
        return refuse_or_discard_incoming();
    }
}

bool server_prompt_cache::update_impl(
        iterator incoming,
        server_prompt_cache_vbr_pressure_citation required_victims) {
    if (limit_size == 0 && limit_tokens == 0 &&
        !quality_anchor_budget_enabled) {
        return true;
    }
    bool pressure_wave_started = false;
    bool competition_wave_valid = true;
    bool retention_shadow_observed = false;
    size_t cache_bytes = 0;
    size_t cache_tokens = 0;
    size_t anchor_bytes = 0;
    const auto measure_cache = [&]() noexcept {
        cache_bytes = 0;
        cache_tokens = 0;
        anchor_bytes = 0;
        bool has_vbr = false;
        for (const auto & state : states) {
            const bool vbr = state.payload.kind() ==
                server_prompt_cache_payload_kind::vbr_artifact;
            has_vbr |= vbr;
            if (vbr && quality_anchor_budget_enabled) {
                const uint64_t anchor =
                    state.payload.vbr_anchor_resident_bytes();
                const size_t total = state.size();
                if (anchor > total || anchor > SIZE_MAX - anchor_bytes) {
                    cache_bytes = SIZE_MAX;
                    anchor_bytes = SIZE_MAX;
                } else {
                    cache_bytes += total - size_t(anchor);
                    anchor_bytes += size_t(anchor);
                }
            } else {
                cache_bytes += state.size();
            }
            cache_tokens += state.prompt.n_tokens();
        }

        // The per-entry sum is an allocation-free upper bound. Exact union
        // work is needed only close enough to a byte boundary that shared VBR
        // allocations or fixed checkpoint planes can change the decision.
        const bool pressure_plausible =
            (limit_size > 0 && cache_bytes > limit_size) ||
            (quality_anchor_budget_enabled &&
             anchor_bytes > limit_anchor_size);
        if (!pressure_plausible) {
            return;
        }
        bool has_shared_fixed_plane = false;
        if (!has_vbr) {
            for (const auto & state : states) {
                for (const auto & checkpoint : state.prompt.checkpoints) {
                    has_shared_fixed_plane |=
                        checkpoint.data_tgt.storage_use_count() > 1 ||
                        checkpoint.data_dft.storage_use_count() > 1 ||
                        checkpoint.accel.ring.storage_use_count() > 1 ||
                        checkpoint.accel.spec.storage_use_count() > 1;
                }
            }
        }
        if (!has_vbr && !has_shared_fixed_plane) {
            return;
        }
        const auto measured = server_prompt_cache_measure_budgets(states, acct);
        if (measured.exact) {
            cache_bytes = measured.compact;
            anchor_bytes = quality_anchor_budget_enabled
                ? measured.anchor : 0;
            if (!quality_anchor_budget_enabled) {
                cache_bytes += measured.anchor;
            }
        }
    };
    measure_cache();
    const auto begin_pressure_wave = [&]() noexcept {
        if (pressure_wave_started) {
            return;
        }
        pressure_wave_started = true;
        if (!retention_obs) {
            return;
        }
        if (retention_shadow.pressure_waves != UINT64_MAX) {
            retention_shadow.pressure_waves++;
        }
        competition_wave_valid =
            retention_obs->begin_competition_wave();
    };
    if (quality_anchor_budget_enabled &&
        anchor_bytes > limit_anchor_size) {
        begin_pressure_wave();
        SRV_WRN(
            " - quality-anchor limit reached (size = %.3f MiB, cap = %.3f MiB)\n",
            anchor_bytes / (1024.0 * 1024.0),
            limit_anchor_size / (1024.0 * 1024.0));
        if (!enforce_quality_anchor_budget(
                incoming,
                retention_obs ? retention_obs->competition_epoch_value() : 0,
                anchor_bytes)) {
            if (incoming != states.end()) {
                if (destruction_obs) {
                    destruction_obs->note_host_trade_publication_skip();
                }
                if (publish_authority) {
                    const uint64_t sequence =
                        ++publish_authority->destruction_quote_sequence;
                    observe_host_trade_refusal(
                        *this, sequence,
                        common_cache_plan_destruction_reason::
                            capacity_refused);
                }
                refuse_incoming_under_pressure(
                    incoming,
                    server_cache_destruction_reason::host_capacity);
            }
            return false;
        }
        measure_cache();
    }
    if (limit_size > 0) {
        while (!states.empty() && cache_bytes > limit_size) {
            begin_pressure_wave();
            SRV_WRN(" - cache size limit reached (size = %.3f MiB)\n",
                    cache_bytes / (1024.0 * 1024.0));

            const bool observe_shadow = !retention_shadow_observed;
            uint64_t released_bytes = 0;
            size_t released_tokens = 0;
            if (required_victims.count == 2) {
                const auto incoming_artifact = incoming != states.end() &&
                        retention_obs
                    ? retention_obs->artifact_id(
                        server_retention_instance_key::for_host_entry(
                            &*incoming))
                    : llama_cache_acct_artifact_id {};
                server_prompt_cache_vbr_pressure_plan current;
                const bool planned = incoming_artifact.v != 0 &&
                    server_prompt_cache_plan_vbr_pressure(
                        *this, cache_bytes, cache_tokens, 2, false,
                        current,
                        retention_shadow_rows.get(),
                        retention_shadow_artifacts.get(),
                        retention_shadow_lineages.get(),
                        incoming_artifact);
                bool matches = planned && current.count != 0 &&
                    current.count <= required_victims.count;
                for (size_t i = 0; matches && i < current.count; ++i) {
                    matches = current.artifacts[i] ==
                        required_victims.artifacts[i];
                }
                if (!matches) {
                    refuse_incoming_under_pressure(
                        incoming,
                        server_cache_destruction_reason::host_capacity);
                    return false;
                }
                if (current.count == 1) {
                    required_victims.count = 1;
                } else {
                    const auto first = std::find_if(
                        states.begin(), states.end(), [&](const auto & value) {
                            return &value == current.victims[0];
                        });
                    const auto second = std::find_if(
                        states.begin(), states.end(), [&](const auto & value) {
                            return &value == current.victims[1];
                        });
                    if (first == states.end() || second == states.end()) {
                        refuse_incoming_under_pressure(
                            incoming,
                            server_cache_destruction_reason::host_capacity);
                        return false;
                    }
                    if (observe_shadow) {
                        observe_retention_pressure_choice(
                            server_cache_destruction_reason::host_capacity,
                            incoming, first,
                            competition_wave_valid);
                    }
                    if (!destroy_vbr_pair(
                            first, second,
                            server_cache_destruction_reason::host_capacity,
                            current.soft_leased[0],
                            current.soft_leased[1],
                            released_bytes, released_tokens)) {
                        refuse_incoming_under_pressure(
                            incoming,
                            server_cache_destruction_reason::host_capacity);
                        return false;
                    }
                    if (destruction_obs) {
                        destruction_obs->host_trade_retention_capacity_executed += 2;
                    }
                    required_victims = {};
                    retention_shadow_observed |= observe_shadow;
                    if (released_bytes == UINT64_MAX ||
                        released_bytes > cache_bytes ||
                        released_tokens > cache_tokens) {
                        measure_cache();
                    } else {
                        cache_bytes -= size_t(released_bytes);
                        cache_tokens -= released_tokens;
                    }
                    continue;
                }
            }
            if (!evict_front_under_pressure(
                    server_cache_destruction_reason::host_capacity,
                    incoming, competition_wave_valid, observe_shadow,
                    released_bytes, released_tokens,
                    required_victims.count == 1
                        ? required_victims.artifacts[0]
                        : llama_cache_acct_artifact_id {})) {
                return false;
            }
            required_victims = {};
            retention_shadow_observed |= observe_shadow;
            if (released_bytes == UINT64_MAX ||
                released_bytes > cache_bytes ||
                released_tokens > cache_tokens) {
                measure_cache();
            } else {
                cache_bytes -= size_t(released_bytes);
                cache_tokens -= released_tokens;
            }
        }
    }

    // average size per token
    // Dynamically increase the token limit if the measured physical payload
    // fits in the byte limit. Publication preflight uses this exact helper.
    const size_t limit_tokens_cur =
        server_prompt_cache_effective_token_limit(
            limit_size, limit_tokens, cache_bytes, cache_tokens);

    if (limit_tokens > 0) {
        while (!states.empty() && cache_tokens > limit_tokens_cur) {
            begin_pressure_wave();
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur,
                    cache_bytes / (1024.0 * 1024.0));

            const bool observe_shadow = !retention_shadow_observed;
            uint64_t released_bytes = 0;
            size_t released_tokens = 0;
            if (required_victims.count > 1) {
                refuse_incoming_under_pressure(
                    incoming,
                    server_cache_destruction_reason::host_token_limit);
                return false;
            }
            if (!evict_front_under_pressure(
                    server_cache_destruction_reason::host_token_limit,
                    incoming, competition_wave_valid, observe_shadow,
                    released_bytes, released_tokens,
                    required_victims.count == 1
                        ? required_victims.artifacts[0]
                        : llama_cache_acct_artifact_id {})) {
                return false;
            }
            required_victims = {};
            retention_shadow_observed |= observe_shadow;
            if (released_bytes == UINT64_MAX ||
                released_bytes > cache_bytes ||
                released_tokens > cache_tokens) {
                measure_cache();
            } else {
                cache_bytes -= size_t(released_bytes);
                cache_tokens -= released_tokens;
            }
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), cache_bytes / (1024.0 * 1024.0),
            limit_size / (1024.0 * 1024.0), limit_tokens,
            limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
    return true;
}
