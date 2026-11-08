#include "llama-safetensors-deepseek4.h"

#include "llama-impl.h"
#include "llama-safetensors-templates.h"
#include "llama-safetensors-metadata.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-tensor.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

#include "llama-safetensors-deepseek41-engram.inc"

enum class target_kind {
    TENSOR,
    EXPERTS,
    HASH_TABLE,
    EH_WEIGHT,
    EH_SCALE,
    STACK,
    ENGRAM_GATE,  // V4.1 q_weight * k_weight, folded to F32 as the GGUF converter does
};

struct target_spec {
    target_kind kind = target_kind::TENSOR;
    std::string source;
    std::optional<llama_safetensors_quant_binding> quant;
    uint32_t layer = 0;
    char expert_projection = 0;
    // STACK: same-shape bindings concatenated along a new outer dimension
    // (EXL3 wo_a group slices, per-expert EXL3 svh/suh vectors)
    std::vector<llama_safetensors_quant_binding> stack;
};

const llama_safetensors_tensor & require_tensor(
        const llama_safetensors_registry & registry, const std::string & name) {
    const auto * tensor = registry.find(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing DeepSeek-V4 source tensor '" + name + "'");
    }
    return *tensor;
}

std::vector<int64_t> reverse_shape(const llama_safetensors_tensor & tensor) {
    std::vector<int64_t> result;
    result.reserve(tensor.shape.size());
    for (auto it = tensor.shape.rbegin(); it != tensor.shape.rend(); ++it) {
        if (*it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("DeepSeek-V4 tensor dimension exceeds runtime limits: '" + tensor.name + "'");
        }
        result.push_back(static_cast<int64_t>(*it));
    }
    return result;
}

std::optional<llama_safetensors_quant_binding> bind_quant(
        const llama_safetensors_quant_adapters & quant,
        const std::string & module,
        llama_safetensors_quant_role role) {
    return quant.bind(module, role);
}

target_spec projection(
        const llama_safetensors_quant_adapters & quant,
        std::string module,
        llama_safetensors_quant_role role = llama_safetensors_quant_role::WEIGHT) {
    target_spec result;
    result.quant = bind_quant(quant, module, role);
    result.source = result.quant ? result.quant->primary : module +
        (role == llama_safetensors_quant_role::WEIGHT ? ".weight" : ".scale");
    return result;
}

// EXL3 input signs exist only for quantized projections; no plain fallback source
target_spec input_scale(const llama_safetensors_quant_adapters & quant, const std::string & module) {
    target_spec result;
    result.quant = bind_quant(quant, module, llama_safetensors_quant_role::INPUT_SCALE);
    if (result.quant) result.source = result.quant->primary;
    return result;
}

// exllamav3 stores the grouped wo_a as one EXL3 tensor per group ("wo_a.slice.N")
target_spec slice_stack(
        const llama_safetensors_quant_adapters & quant,
        const llama_safetensors_registry & registry,
        const std::string & module,
        llama_safetensors_quant_role role) {
    target_spec result;
    for (uint32_t slice = 0;; ++slice) {
        const std::string name = module + ".slice." + std::to_string(slice);
        if (registry.find(name + ".trellis") == nullptr) break;
        auto binding = bind_quant(quant, name, role);
        if (!binding) throw std::runtime_error("missing DeepSeek-V4 EXL3 slice binding for '" + name + "'");
        result.stack.push_back(std::move(*binding));
    }
    if (!result.stack.empty()) {
        result.kind = target_kind::STACK;
        result.source = result.stack.front().primary;
    }
    return result;
}

target_spec map_target(
        const llama_safetensors_quant_adapters & quant,
        const llama_safetensors_registry & registry,
        uint32_t n_layer,
        uint32_t n_mtp,
        uint32_t n_expert,
        const std::string & target) {
    if (target == "token_embd.weight") return { target_kind::TENSOR, "embed.weight" };
    if (target == "output_norm.weight") return { target_kind::TENSOR, "norm.weight" };
    if (target == "output.weight") return projection(quant, "head");
    if (target == "output.scale") return projection(quant, "head", llama_safetensors_quant_role::WEIGHT_SCALE);
    if (target == "output.input_scale") return input_scale(quant, "head");
    if (target == "output_hc_fn.weight") return { target_kind::TENSOR, "hc_head_fn" };
    if (target == "output_hc_base.weight") return { target_kind::TENSOR, "hc_head_base" };
    if (target == "output_hc_scale.weight") return { target_kind::TENSOR, "hc_head_scale" };

    static const std::regex block(R"(^blk\.([0-9]+)\.(.+)$)");
    std::smatch match;
    if (!std::regex_match(target, match, block)) return {};
    const uint64_t parsed = std::stoull(match[1].str());
    if (parsed >= static_cast<uint64_t>(n_layer) + n_mtp) return {};
    const uint32_t layer = static_cast<uint32_t>(parsed);
    const bool mtp = layer >= n_layer;
    const std::string prefix = mtp ? "mtp." + std::to_string(layer - n_layer) + "." :
        "layers." + std::to_string(layer) + ".";
    const std::string suffix = match[2].str();

    if (mtp && suffix == "nextn.eh_proj.weight") return { target_kind::EH_WEIGHT, prefix, {}, layer };
    if (mtp && suffix == "nextn.eh_proj.scale") return { target_kind::EH_SCALE, prefix, {}, layer };
    if (mtp && suffix == "nextn.enorm.weight") return { target_kind::TENSOR, prefix + "enorm.weight" };
    if (mtp && suffix == "nextn.hnorm.weight") return { target_kind::TENSOR, prefix + "hnorm.weight" };
    if (mtp && suffix == "nextn.shared_head_norm.weight") return { target_kind::TENSOR, prefix + "norm.weight" };
    if (suffix == "engram_embd.weight") return { target_kind::TENSOR, prefix + "engram.embed.weight" };
    if (suffix == "engram_embd.scale") return { target_kind::TENSOR, prefix + "engram.embed.scale" };
    if (suffix == "engram_gate.weight") return { target_kind::ENGRAM_GATE, prefix + "engram." };

    struct mapping { const char * target; const char * source; bool projected; };
    static constexpr mapping mappings[] = {
        { "attn_norm.weight",             "attn_norm.weight",                    false },
        { "attn_sinks.weight",            "attn.attn_sink",                     false },
        { "attn_q_a.weight",              "attn.wq_a",                          true  },
        { "attn_q_a.scale",               "attn.wq_a",                          true  },
        { "attn_q_a_norm.weight",         "attn.q_norm.weight",                 false },
        { "attn_q_b.weight",              "attn.wq_b",                          true  },
        { "attn_q_b.scale",               "attn.wq_b",                          true  },
        { "attn_kv.weight",               "attn.wkv",                           true  },
        { "attn_kv.scale",                "attn.wkv",                           true  },
        { "attn_kv_a_norm.weight",        "attn.kv_norm.weight",                false },
        { "attn_output_a.weight",         "attn.wo_a",                          true  },
        { "attn_output_a.scale",          "attn.wo_a",                          true  },
        { "attn_output_b.weight",         "attn.wo_b",                          true  },
        { "attn_output_b.scale",          "attn.wo_b",                          true  },
        { "hc_attn_fn.weight",            "hc_attn_fn",                         false },
        { "hc_attn_base.weight",          "hc_attn_base",                       false },
        { "hc_attn_scale.weight",         "hc_attn_scale",                      false },
        { "hc_ffn_fn.weight",             "hc_ffn_fn",                          false },
        { "hc_ffn_base.weight",           "hc_ffn_base",                        false },
        { "hc_ffn_scale.weight",          "hc_ffn_scale",                       false },
        { "attn_compressor_kv.weight",    "attn.compressor.wkv",                true  },
        { "attn_compressor_kv.scale",     "attn.compressor.wkv",                true  },
        { "attn_compressor_gate.weight",  "attn.compressor.wgate",              true  },
        { "attn_compressor_gate.scale",   "attn.compressor.wgate",              true  },
        { "attn_compressor_ape.weight",   "attn.compressor.ape",                false },
        { "attn_compressor_norm.weight",  "attn.compressor.norm.weight",        false },
        { "indexer.proj.weight",          "attn.indexer.weights_proj",          true  },
        { "indexer.proj.scale",           "attn.indexer.weights_proj",          true  },
        { "indexer.attn_q_b.weight",      "attn.indexer.wq_b",                  true  },
        { "indexer.attn_q_b.scale",       "attn.indexer.wq_b",                  true  },
        { "indexer_compressor_kv.weight", "attn.indexer.compressor.wkv",        true  },
        { "indexer_compressor_kv.scale",  "attn.indexer.compressor.wkv",        true  },
        { "indexer_compressor_gate.weight", "attn.indexer.compressor.wgate",    true  },
        { "indexer_compressor_gate.scale",  "attn.indexer.compressor.wgate",    true  },
        { "indexer_compressor_ape.weight",  "attn.indexer.compressor.ape",      false },
        { "indexer_compressor_norm.weight", "attn.indexer.compressor.norm.weight", false },
        { "ffn_gate_inp.weight",          "ffn.gate.weight",                    false },
        { "exp_probs_b.bias",             "ffn.gate.bias",                      false },
        { "ffn_norm.weight",              "ffn_norm.weight",                    false },
        { "ffn_gate_shexp.weight",        "ffn.shared_experts.w1",              true  },
        { "ffn_gate_shexp.scale",         "ffn.shared_experts.w1",              true  },
        { "ffn_down_shexp.weight",        "ffn.shared_experts.w2",              true  },
        { "ffn_down_shexp.scale",         "ffn.shared_experts.w2",              true  },
        { "ffn_up_shexp.weight",          "ffn.shared_experts.w3",              true  },
        { "ffn_up_shexp.scale",           "ffn.shared_experts.w3",              true  },
        // V4.1
        { "indexer.attn_k.weight",        "attn.indexer.wk",                    true  },
        { "indexer.attn_k.scale",         "attn.indexer.wk",                    true  },
        { "indexer.k_norm.weight",        "attn.indexer.k_norm.weight",         false },
        { "engram_kv.weight",             "engram.wkv",                         true  },
        { "engram_kv.scale",              "engram.wkv",                         true  },
        { "exp_probs_b_vl.bias",          "ffn.gate.bias_vl",                   false },
    };
    static const std::string input_suffix = ".input_scale";
    const bool input = suffix.size() > input_suffix.size() &&
        suffix.compare(suffix.size() - input_suffix.size(), input_suffix.size(), input_suffix) == 0;
    const std::string weight_suffix = input ? suffix.substr(0, suffix.size() - input_suffix.size()) + ".weight" : suffix;
    for (const mapping & item : mappings) {
        if (weight_suffix != item.target) continue;
        const std::string source = prefix + item.source;
        if (!item.projected) return input ? target_spec{} : target_spec{ target_kind::TENSOR, source };
        const bool scale = suffix.size() >= 6 && suffix.compare(suffix.size() - 6, 6, ".scale") == 0;
        const auto role = input ? llama_safetensors_quant_role::INPUT_SCALE :
            scale ? llama_safetensors_quant_role::WEIGHT_SCALE : llama_safetensors_quant_role::WEIGHT;
        if (registry.find(source + ".slice.0.trellis") != nullptr) return slice_stack(quant, registry, source, role);
        return input ? input_scale(quant, source) : projection(quant, source, role);
    }

    if (suffix == "ffn_gate_tid2eid.weight") return { target_kind::HASH_TABLE, prefix + "ffn.gate.tid2eid" };
    static constexpr const char * expert_targets[] = { "ffn_gate_exps", "ffn_down_exps", "ffn_up_exps" };
    for (size_t projection_index = 0; projection_index < 3; ++projection_index) {
        const std::string base = expert_targets[projection_index];
        const char expert_projection = char('1' + projection_index);
        if (suffix == base + ".weight") {
            target_spec result;
            result.kind = target_kind::EXPERTS;
            result.source = prefix;
            result.layer = layer;
            result.expert_projection = expert_projection;
            return result;
        }
        if (suffix != base + ".scale" && suffix != base + input_suffix) continue;
        // EXL3 experts: per-expert svh / suh vectors stack to [n, n_expert] / [k, n_expert]
        const auto role = suffix == base + ".scale" ? llama_safetensors_quant_role::WEIGHT_SCALE :
            llama_safetensors_quant_role::INPUT_SCALE;
        const auto module_of = [&](uint32_t expert) {
            return prefix + "ffn.experts." + std::to_string(expert) + ".w" + expert_projection;
        };
        // expert 0 stands for the bank, as in describe()
        const auto weight = bind_quant(quant, module_of(0), llama_safetensors_quant_role::WEIGHT);
        if (!weight || !ggml_type_is_exl3(weight->target_type)) return {};
        target_spec result;
        for (uint32_t expert = 0; expert < n_expert; ++expert) {
            const std::string module = module_of(expert);
            auto side = bind_quant(quant, module, role);
            if (!side) throw std::runtime_error("missing DeepSeek-V4 EXL3 expert side vector for '" + module + "'");
            result.stack.push_back(std::move(*side));
        }
        result.kind = target_kind::STACK;
        result.source = result.stack.front().primary;
        return result;
    }
    return {};
}

std::string expert_module(const target_spec & spec, uint32_t expert) {
    return spec.source + "ffn.experts." + std::to_string(expert) + ".w" + spec.expert_projection;
}

bool expert_type_supported(ggml_type type) {
    return type == GGML_TYPE_MXFP4 || ggml_type_is_exl3(type);
}

std::vector<uint8_t> concat_rows(
        const llama_safetensors_tensor & lhs_desc, std::vector<uint8_t> lhs,
        const llama_safetensors_tensor & rhs_desc, std::vector<uint8_t> rhs) {
    if (lhs_desc.shape.size() != 2 || rhs_desc.shape != lhs_desc.shape || lhs_desc.shape[0] == 0 ||
        lhs.size() != rhs.size() || lhs.size() % lhs_desc.shape[0] != 0) {
        throw std::runtime_error("DeepSeek-V4 MTP projection pair has incompatible shapes");
    }
    const size_t row_bytes = lhs.size() / lhs_desc.shape[0];
    std::vector<uint8_t> result(lhs.size() + rhs.size());
    for (size_t row = 0; row < lhs_desc.shape[0]; ++row) {
        std::memcpy(result.data() + row * 2 * row_bytes, lhs.data() + row * row_bytes, row_bytes);
        std::memcpy(result.data() + row * 2 * row_bytes + row_bytes, rhs.data() + row * row_bytes, row_bytes);
    }
    return result;
}

bool is_prime(uint64_t n) {
    if (n < 2 || n % 2 == 0) return n == 2;
    for (uint64_t d = 3; d * d <= n; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

// EngramLayout.from_args: per layer, per n-gram order, per head, the next prime above
// vocab_size - 1 not used before - i.e. consecutive primes from vocab_size on
std::vector<uint64_t> engram_primes(size_t count, uint64_t vocab_size) {
    std::vector<uint64_t> result;
    result.reserve(count);
    for (uint64_t candidate = vocab_size; result.size() < count; ++candidate) {
        if (is_prime(candidate)) result.push_back(candidate);
    }
    return result;
}

// EXL3 packs leave out the Engram tables: they come from the base checkpoint's shards
std::filesystem::path engram_base_dir(const llama_safetensors_json & config) {
    std::vector<std::filesystem::path> dirs;
    if (const char * env = std::getenv("LLAMA_SAFETENSORS_BASE_DIR")) dirs.emplace_back(env);
    if (config.contains("engram_table_dir") && config.at("engram_table_dir").is_string()) {
        dirs.emplace_back(config.at("engram_table_dir").get<std::string>());
    }
    for (const auto & dir : dirs) {
        if (std::filesystem::is_regular_file(dir / "model.safetensors.index.json")) return dir;
    }
    throw std::runtime_error(
        "the DeepSeek-V4.1 Engram tables are not in this checkpoint: set LLAMA_SAFETENSORS_BASE_DIR to a directory with the "
        "base model's model.safetensors.index.json and the shards holding layers.N.engram.embed.*");
}

} // namespace

llama_safetensors_deepseek4_importer::llama_safetensors_deepseek4_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_json config,
        llama_safetensors_io_mode io_mode) :
    model_dir_(model_dir), config_(std::move(config)), v41_(probe_v41(config_)),
    hparams_(v41_ ? config_.at("text_config") : config_) {
    if (!probe(config_)) throw std::runtime_error("native DeepSeek-V4 importer received the wrong architecture");
    n_layer_ = hparams_.at("num_hidden_layers").get<uint32_t>();
    n_mtp_ = v41_ ? 0 : hparams_.value("num_nextn_predict_layers", 0U);
    n_expert_ = hparams_.at("n_routed_experts").get<uint32_t>();
    if (v41_) {
        // the FP8 base checkpoint keeps block scales the V4.1 model does not load; it goes through the GGUF converter
        const auto & quant = config_.value("quantization_config", llama_safetensors_json::object());
        if (n_layer_ != 40 || n_expert_ != 384 || quant.value("quant_method", std::string()) != "exl3") {
            throw std::runtime_error("native DeepSeek-V4.1 importer supports the 40-layer, 384-expert EXL3 checkpoints");
        }
    } else if (n_layer_ != 43 || n_mtp_ > 1 || n_expert_ != 256 ||
        config_.at("expert_dtype").get<std::string>() != "fp4") {
        throw std::runtime_error("native DeepSeek-V4 importer does not support this tensor geometry");
    }
    const auto generation_path = model_dir_ / "generation_config.json";
    generation_ = std::filesystem::is_regular_file(generation_path) ? llama_safetensors_read_json(generation_path) :
        llama_safetensors_json::object();
    tokenizer_ = llama_safetensors_read_tokenizer_json(model_dir_ / "tokenizer.json");
    const auto tokenizer_config_path = model_dir_ / "tokenizer_config.json";
    if (std::filesystem::is_regular_file(tokenizer_config_path)) {
        const auto tokenizer_config = llama_safetensors_read_json(tokenizer_config_path);
        if (tokenizer_config.contains("chat_template") && tokenizer_config.at("chat_template").is_string()) {
            chat_template_ = tokenizer_config.at("chat_template").get<std::string>();
        }
    }
    if (!chat_template_) chat_template_ = llama_safetensors_read_optional_text(model_dir_ / "chat_template.jinja");
    if (!chat_template_) {
        const std::string model_id_hint = config_.value("_name_or_path", model_dir_.filename().string());
        const std::string_view fallback = v41_ ? LLAMA_SAFETENSORS_DEEPSEEK41_CHAT_TEMPLATE :
            model_id_hint.find("0731") != std::string::npos ?
            LLAMA_SAFETENSORS_DEEPSEEK4_0731_CHAT_TEMPLATE : LLAMA_SAFETENSORS_DEEPSEEK4_CHAT_TEMPLATE;
        chat_template_ = std::string(fallback);
    }
    registry_ = llama_safetensors_registry::load(model_dir_, io_mode);
    const auto engram_layers = hparams_.value("engram_layer_ids", std::vector<uint32_t>());
    if (v41_ && !engram_layers.empty() &&
        registry_.find("layers." + std::to_string(engram_layers.front()) + ".engram.embed.weight") == nullptr) {
        const auto dir = engram_base_dir(config_);
        std::vector<std::string> tables;
        for (uint32_t layer : engram_layers) {
            tables.push_back("layers." + std::to_string(layer) + ".engram.embed.weight");
            tables.push_back("layers." + std::to_string(layer) + ".engram.embed.scale");
        }
        LLAMA_LOG_INFO("%s: DeepSeek-V4.1 Engram tables from %s\n", __func__, dir.string().c_str());
        registry_.add_from(dir, tables);
    }
    quant_ = std::make_unique<llama_safetensors_quant_adapters>(config_, registry_);
    if (n_mtp_ != 0 && registry_.find("mtp.0.attn_norm.weight") == nullptr) n_mtp_ = 0;
    // exllamav3 0731 ships a DSpark block drafter (main_proj over target layers) instead of NextN e/h_proj
    if (n_mtp_ != 0 && registry_.find("mtp.0.e_proj.weight") == nullptr && registry_.find("mtp.0.e_proj.trellis") == nullptr) {
        LLAMA_LOG_WARN("%s: DeepSeek-V4 MTP block is not NextN (DSpark drafter?); loading without MTP\n", __func__);
        n_mtp_ = 0;
    }
}

bool llama_safetensors_deepseek4_importer::probe(const llama_safetensors_json & config) {
    return config.value("model_type", std::string()) == "deepseek_v4" || probe_v41(config);
}

bool llama_safetensors_deepseek4_importer::probe_v41(const llama_safetensors_json & config) {
    return config.value("model_type", std::string()) == "deepseek_v41" && config.contains("text_config");
}

gguf_context * llama_safetensors_deepseek4_importer::build_metadata() const {
    llama_safetensors_metadata_sink sink;
    const std::string arch = v41_ ? "deepseek41" : "deepseek4";
    sink.set_string("general.architecture", arch);
    sink.set_string("general.type", "model");
    sink.set_string("general.name", model_dir_.filename().empty() ? "DeepSeek-V4 Safetensors" : model_dir_.filename().string());
    sink.set_u32("general.file_type", quant_->file_type());
    sink.set_u32("general.quantization_version", 2);
    llama_safetensors_emit_sampling_defaults(sink, generation_);
    sink.set_u32(arch + ".block_count", n_layer_ + n_mtp_);
    sink.set_u32(arch + ".context_length", hparams_.at("max_position_embeddings").get<uint32_t>());
    sink.set_u32(arch + ".embedding_length", hparams_.at("hidden_size").get<uint32_t>());
    sink.set_u32(arch + ".embedding_length_out", hparams_.at("hidden_size").get<uint32_t>() * hparams_.at("hc_mult").get<uint32_t>());
    sink.set_u32(arch + ".expert_feed_forward_length", hparams_.at("moe_intermediate_size").get<uint32_t>());
    sink.set_u32(arch + ".expert_count", n_expert_);
    sink.set_u32(arch + ".expert_used_count", hparams_.at("num_experts_per_tok").get<uint32_t>());
    sink.set_u32(arch + ".expert_shared_count", hparams_.at("n_shared_experts").get<uint32_t>());
    sink.set_f32(arch + ".expert_weights_scale", hparams_.at("routed_scaling_factor").get<float>());
    sink.set_bool(arch + ".expert_weights_norm", hparams_.at("norm_topk_prob").get<bool>());
    sink.set_u32(arch + ".expert_gating_func", 4);
    std::vector<float> clamp(n_layer_ + n_mtp_, hparams_.at("swiglu_limit").get<float>());
    sink.set_f32_array(arch + ".swiglu_clamp_exp", clamp.data(), clamp.size());
    sink.set_f32_array(arch + ".swiglu_clamp_shexp", clamp.data(), clamp.size());
    sink.set_u32(arch + ".attention.head_count", hparams_.at("num_attention_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.head_count_kv", hparams_.at("num_key_value_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.key_length", hparams_.at("head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.value_length", hparams_.at("head_dim").get<uint32_t>());
    sink.set_f32(arch + ".attention.layer_norm_rms_epsilon", hparams_.at("rms_norm_eps").get<float>());
    sink.set_u32(arch + ".attention.q_lora_rank", hparams_.at("q_lora_rank").get<uint32_t>());
    sink.set_u32(arch + ".attention.sliding_window", hparams_.at("sliding_window").get<uint32_t>());
    sink.set_u32(arch + ".attention.output_group_count", hparams_.at("o_groups").get<uint32_t>());
    sink.set_u32(arch + ".attention.output_lora_rank", hparams_.at("o_lora_rank").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.head_count", hparams_.at("index_n_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.key_length", hparams_.at("index_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.top_k", hparams_.at("index_topk").get<uint32_t>());
    sink.set_f32(arch + ".attention.compress_rope_freq_base", hparams_.at("compress_rope_theta").get<float>());
    auto ratios = hparams_.at("compress_ratios").get<std::vector<uint32_t>>();
    if (ratios.size() == n_layer_ && n_mtp_ != 0) ratios.push_back(0);
    sink.set_u32_array(arch + ".attention.compress_ratios", ratios.data(), ratios.size());
    sink.set_u32(arch + ".hyper_connection.count", hparams_.at("hc_mult").get<uint32_t>());
    sink.set_u32(arch + ".hyper_connection.sinkhorn_iterations", hparams_.at("hc_sinkhorn_iters").get<uint32_t>());
    sink.set_f32(arch + ".hyper_connection.epsilon", hparams_.at("hc_eps").get<float>());
    sink.set_u32(arch + ".hash_layer_count", hparams_.value("num_hash_layers", 0U)); // none in V4.1
    sink.set_u32(arch + ".rope.dimension_count", hparams_.at("qk_rope_head_dim").get<uint32_t>());
    sink.set_f32(arch + ".rope.freq_base", hparams_.at("rope_theta").get<float>());
    const auto & rope_scaling = hparams_.at("rope_scaling");
    // V4.1 spells the YaRN type HF-style ("rope_type")
    if (!rope_scaling.is_object() ||
        rope_scaling.value("type", rope_scaling.value("rope_type", std::string())) != "yarn") {
        throw std::runtime_error("native DeepSeek-V4 importer requires the published YaRN RoPE scaling contract");
    }
    sink.set_string(arch + ".rope.scaling.type", "yarn");
    sink.set_f32(arch + ".rope.scaling.factor", rope_scaling.at("factor").get<float>());
    sink.set_u32(
        arch + ".rope.scaling.original_context_length",
        rope_scaling.at("original_max_position_embeddings").get<uint32_t>());
    sink.set_f32(arch + ".rope.scaling.yarn_beta_fast", rope_scaling.at("beta_fast").get<float>());
    sink.set_f32(arch + ".rope.scaling.yarn_beta_slow", rope_scaling.at("beta_slow").get<float>());
    if (n_mtp_ != 0) sink.set_u32(arch + ".nextn_predict_layers", n_mtp_);
    if (v41_) emit_v41_metadata(sink, arch);

    // V4.1 EXL3 packs ship no generation_config: the token ids are in config.json
    const auto & ids = generation_.contains("bos_token_id") ? generation_ : config_;
    const uint32_t bos = llama_safetensors_first_token_id(ids.at("bos_token_id"), "bos_token_id");
    const uint32_t eos = llama_safetensors_first_token_id(ids.at("eos_token_id"), "eos_token_id");
    llama_safetensors_emit_bpe_tokenizer(sink, tokenizer_, {
        "joyai-llm", hparams_.at("vocab_size").get<uint32_t>(), bos, eos, std::string("<｜▁pad▁｜>"), false,
    }, chat_template_);
    gguf_context * ctx = sink.release();
    if (v41_) emit_v41_token_map(ctx);
    return ctx;
}

void llama_safetensors_deepseek4_importer::emit_v41_metadata(
        llama_safetensors_metadata_sink & sink, const std::string & arch) const {
    const auto kv_src = hparams_.at("kv_source_layer_ids").get<std::vector<uint32_t>>();
    const auto idx_src = hparams_.at("index_source_layer_ids").get<std::vector<uint32_t>>();
    sink.set_u32_array(arch + ".attention.kv_source_layers", kv_src.data(), kv_src.size());
    sink.set_u32_array(arch + ".attention.indexer.source_layers", idx_src.data(), idx_src.size());
    const int32_t cand_src = hparams_.value("candidate_source_layer_id", -1);
    if (cand_src >= 0) {
        sink.set_i32(arch + ".attention.indexer.candidate_source_layer", cand_src);
        sink.set_u32(arch + ".attention.indexer.candidate_top_k_blocks", hparams_.at("candidate_topk_blocks").get<uint32_t>());
        sink.set_u32(arch + ".attention.indexer.candidate_block_size", hparams_.at("candidate_block_size").get<uint32_t>());
    }

    // Engram: one table per layer (local head offsets). The multipliers and the token map come
    // from numpy's PCG64 and the HF normalizers, so they are baked for the published layout.
    const auto layers = hparams_.value("engram_layer_ids", std::vector<uint32_t>());
    if (layers.empty()) return;
    const uint32_t n_gram = hparams_.at("engram_max_ngram_size").get<uint32_t>();
    const uint32_t n_head = hparams_.at("engram_n_heads").get<uint32_t>();
    const auto rows = hparams_.at("engram_num_embeddings").get<std::vector<uint64_t>>();
    if (!std::equal(layers.begin(), layers.end(), std::begin(DSV41_ENGRAM_LAYERS), std::end(DSV41_ENGRAM_LAYERS)) ||
        n_gram * std::size(DSV41_ENGRAM_LAYERS) != std::size(DSV41_ENGRAM_MULTIPLIERS) || rows.size() != layers.size() ||
        hparams_.at("engram_compressed_vocab_size").get<uint32_t>() != DSV41_ENGRAM_COMPRESSED_VOCAB_SIZE ||
        hparams_.at("vocab_size").get<uint32_t>() != DSV41_ENGRAM_VOCAB_SIZE) {
        throw std::runtime_error("DeepSeek-V4.1 Engram layout differs from DeepSeek-V4.1-Flash-1017; convert it with convert_hf_to_gguf.py");
    }
    const size_t n_cols = size_t(n_gram - 1) * n_head;
    const auto primes = engram_primes(layers.size() * n_cols, hparams_.at("engram_vocab_size").get<uint64_t>());
    std::vector<uint64_t> offsets(primes.size());
    for (size_t e = 0; e < layers.size(); ++e) {
        uint64_t offset = 0;
        for (size_t c = e * n_cols; c < (e + 1) * n_cols; ++c) {
            offsets[c] = offset;
            offset += primes[c];
        }
        if (offset != rows[e]) {
            throw std::runtime_error("DeepSeek-V4.1 Engram layer " + std::to_string(layers[e]) + ": bucket counts do not sum to engram_num_embeddings");
        }
    }
    sink.set_u32_array(arch + ".engram.layers", layers.data(), layers.size());
    sink.set_u32(arch + ".engram.max_ngram_size", n_gram);
    sink.set_u32(arch + ".engram.head_count", n_head);
    sink.set_u32(arch + ".engram.head_dim", hparams_.at("engram_head_dim").get<uint32_t>());
    sink.set_u64_array(arch + ".engram.multipliers", DSV41_ENGRAM_MULTIPLIERS, std::size(DSV41_ENGRAM_MULTIPLIERS));
    sink.set_u64_array(arch + ".engram.offsets", offsets.data(), offsets.size());
    sink.set_u64_array(arch + ".engram.primes", primes.data(), primes.size());
    sink.set_u32(arch + ".engram.pad_token_id", hparams_.at("engram_pad_token_id").get<uint32_t>());
}

// The baked token map is only valid for the vocabulary it was built from: check the emitted tokens
void llama_safetensors_deepseek4_importer::emit_v41_token_map(gguf_context * ctx) const {
    if (gguf_find_key(ctx, "deepseek41.engram.layers") < 0) return;
    const int64_t tokens = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    const size_t n_vocab = tokens < 0 ? 0 : gguf_get_arr_n(ctx, tokens);
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n_vocab; ++i) {
        const char * token = gguf_get_arr_str(ctx, tokens, i);
        for (size_t j = 0, n = std::strlen(token) + 1; j < n; ++j) {
            hash = (hash ^ uint8_t(token[j])) * 0x100000001b3ull;
        }
    }
    if (n_vocab != DSV41_ENGRAM_VOCAB_SIZE || hash != DSV41_ENGRAM_VOCAB_FNV1A) {
        gguf_free(ctx);
        throw std::runtime_error("DeepSeek-V4.1 tokenizer differs from DeepSeek-V4.1-Flash-1017, the baked Engram token map does not apply");
    }
    std::vector<uint32_t> token_map(n_vocab);
    uint32_t next = 0;
    size_t collision = 0;
    for (uint32_t token = 0; token < n_vocab; ++token) {
        if (collision < std::size(DSV41_ENGRAM_TOKEN_COLLISIONS) && DSV41_ENGRAM_TOKEN_COLLISIONS[collision][0] == token) {
            token_map[token] = DSV41_ENGRAM_TOKEN_COLLISIONS[collision++][1];
        } else {
            token_map[token] = next++;
        }
    }
    GGML_ASSERT(next == DSV41_ENGRAM_COMPRESSED_VOCAB_SIZE && collision == std::size(DSV41_ENGRAM_TOKEN_COLLISIONS));
    gguf_set_arr_data(ctx, "deepseek41.engram.token_map", GGUF_TYPE_UINT32, token_map.data(), token_map.size());
}

bool llama_safetensors_deepseek4_importer::describe(
        const std::string & target, ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const {
    const target_spec spec = map_target(*quant_, registry_, n_layer_, n_mtp_, n_expert_, target);
    std::vector<int64_t> shape;
    if (spec.kind == target_kind::EXPERTS) {
        const auto binding = quant_->bind(expert_module(spec, 0), llama_safetensors_quant_role::WEIGHT);
        if (!binding || !expert_type_supported(binding->target_type) || binding->target_shape.size() != 2) return false;
        type = binding->target_type;
        shape = binding->target_shape;
        shape.push_back(n_expert_);
    } else if (spec.kind == target_kind::STACK) {
        const auto & first = spec.stack.front();
        for (const auto & binding : spec.stack) {
            if (binding.target_type != first.target_type || binding.target_shape != first.target_shape) return false;
        }
        type = first.target_type;
        shape = first.target_shape;
        shape.push_back(static_cast<int64_t>(spec.stack.size()));
    } else if (spec.kind == target_kind::HASH_TABLE) {
        const auto * source = registry_.find(spec.source);
        if (source == nullptr) return false;
        type = GGML_TYPE_I32;
        shape = reverse_shape(*source);
    } else if (spec.kind == target_kind::ENGRAM_GATE) {
        const auto * q = registry_.find(spec.source + "q_weight");
        const auto * k = registry_.find(spec.source + "k_weight");
        if (q == nullptr || k == nullptr || q->shape != k->shape) return false;
        type = GGML_TYPE_F32;
        shape = reverse_shape(*q);
    } else if (spec.kind == target_kind::EH_WEIGHT || spec.kind == target_kind::EH_SCALE) {
        const auto e = quant_->bind(spec.source + "e_proj", spec.kind == target_kind::EH_WEIGHT ?
            llama_safetensors_quant_role::WEIGHT : llama_safetensors_quant_role::WEIGHT_SCALE);
        const auto h = quant_->bind(spec.source + "h_proj", spec.kind == target_kind::EH_WEIGHT ?
            llama_safetensors_quant_role::WEIGHT : llama_safetensors_quant_role::WEIGHT_SCALE);
        if (!e || !h || e->target_type != h->target_type || e->target_shape != h->target_shape || e->target_shape.size() != 2) return false;
        type = e->target_type;
        shape = e->target_shape;
        shape[spec.kind == target_kind::EH_WEIGHT ? 0 : 1] *= 2;
    } else {
        if (spec.source.empty()) return false;
        const auto * source = registry_.find(spec.source);
        if (source == nullptr) return false;
        type = spec.quant ? spec.quant->target_type : llama_safetensors_plain_type(*source);
        shape = spec.quant ? spec.quant->target_shape : reverse_shape(*source);
    }
    if (shape.empty() || shape.size() > GGML_MAX_DIMS) return false;
    ne.fill(1);
    std::copy(shape.begin(), shape.end(), ne.begin());
    return true;
}

size_t llama_safetensors_deepseek4_importer::tensor_capacity_hint() const {
    return std::max<size_t>(1024, registry_.tensors().size() / 8);
}

void llama_safetensors_deepseek4_importer::bind(const std::string & target) const {
    const target_spec spec = map_target(*quant_, registry_, n_layer_, n_mtp_, n_expert_, target);
    if (spec.kind == target_kind::EXPERTS) {
        for (uint32_t expert = 0; expert < n_expert_; ++expert) {
            const auto binding = quant_->bind(expert_module(spec, expert), llama_safetensors_quant_role::WEIGHT);
            if (!binding) throw std::runtime_error("missing DeepSeek-V4 routed expert quantization binding");
            quant_->consume(*binding);
        }
    } else if (spec.kind == target_kind::EH_WEIGHT || spec.kind == target_kind::EH_SCALE) {
        const auto role = spec.kind == target_kind::EH_WEIGHT ? llama_safetensors_quant_role::WEIGHT :
            llama_safetensors_quant_role::WEIGHT_SCALE;
        const auto e = quant_->bind(spec.source + "e_proj", role);
        const auto h = quant_->bind(spec.source + "h_proj", role);
        if (!e || !h) throw std::runtime_error("missing DeepSeek-V4 MTP projection quantization binding");
        quant_->consume(*e); quant_->consume(*h);
    } else if (spec.kind == target_kind::STACK) {
        for (const auto & binding : spec.stack) quant_->consume(binding);
    } else if (spec.quant) {
        quant_->consume(*spec.quant);
    }
}

bool llama_safetensors_deepseek4_importer::load(
        const std::string & target, ggml_tensor * destination, bool check_tensor) const {
    const target_spec spec = map_target(*quant_, registry_, n_layer_, n_mtp_, n_expert_, target);
    if (spec.kind == target_kind::EXPERTS) {
        const size_t plane = ggml_nbytes(destination) / n_expert_;
        if (plane * n_expert_ != ggml_nbytes(destination)) throw std::runtime_error("invalid DeepSeek-V4 expert destination");
        const auto build_plane = [&](uint32_t expert) {
            const auto binding = quant_->bind(expert_module(spec, expert), llama_safetensors_quant_role::WEIGHT);
            if (!binding) throw std::runtime_error("missing DeepSeek-V4 expert binding during upload");
            auto bytes = quant_->finalize(*binding, quant_->read(*binding));
            if (bytes.size() != plane) throw std::runtime_error("DeepSeek-V4 expert plane has the wrong byte size");
            if (check_tensor && binding->target_type == GGML_TYPE_MXFP4 &&
                !ggml_validate_row_data(GGML_TYPE_MXFP4, bytes.data(), bytes.size()))
                throw std::runtime_error("DeepSeek-V4 expert contains invalid MXFP4 data");
            return bytes;
        };

        const auto first = quant_->bind(expert_module(spec, 0), llama_safetensors_quant_role::WEIGHT);
        bool mapped = first.has_value();
        if (first) {
            const auto * primary = registry_.find(first->primary);
            mapped = primary != nullptr && registry_.data(*primary) != nullptr;
            for (const std::string & auxiliary : first->auxiliaries) {
                const auto * desc = registry_.find(auxiliary);
                mapped = mapped && desc != nullptr && registry_.data(*desc) != nullptr;
            }
        }

        // Buffered and direct-I/O readers share file cursors and must remain
        // serial. Mapped shards are immutable, so bounded batches can repack
        // independently before their non-overlapping destination uploads.
        constexpr uint32_t batch_size = 64;
        for (uint32_t begin = 0; begin < n_expert_; begin += mapped ? batch_size : 1) {
            const uint32_t count = std::min<uint32_t>(mapped ? batch_size : 1, n_expert_ - begin);
            std::vector<std::vector<uint8_t>> planes(count);
            if (mapped && count > 1) {
                std::atomic<uint32_t> next { 0 };
                std::exception_ptr failure;
                std::mutex failure_mutex;
                const uint32_t detected = std::max(1U, std::thread::hardware_concurrency());
                const uint32_t workers = std::min<uint32_t>(count, std::min(32U, detected));
                std::vector<std::thread> threads;
                threads.reserve(workers);
                for (uint32_t worker = 0; worker < workers; ++worker) {
                    threads.emplace_back([&]() {
                        while (true) {
                            const uint32_t local = next.fetch_add(1, std::memory_order_relaxed);
                            if (local >= count) break;
                            try {
                                planes[local] = build_plane(begin + local);
                            } catch (...) {
                                std::lock_guard<std::mutex> lock(failure_mutex);
                                if (!failure) failure = std::current_exception();
                                next.store(count, std::memory_order_relaxed);
                                break;
                            }
                        }
                    });
                }
                for (std::thread & thread : threads) thread.join();
                if (failure) std::rethrow_exception(failure);
            } else {
                planes[0] = build_plane(begin);
            }
            for (uint32_t local = 0; local < count; ++local) {
                ggml_backend_tensor_set(destination, planes[local].data(), (begin + local) * plane, plane);
            }
        }
        return true;
    }
    if (spec.kind == target_kind::TENSOR && !spec.source.empty()) {
        return llama_safetensors_load_tensor_direct(registry_, { spec.source, spec.quant }, destination, check_tensor);
    }
    return false;
}

// The V4.1 Engram tables (~100 GB per layer) are read lazily, straight from the shards
std::optional<llama_model_tensor_file_region> llama_safetensors_deepseek4_importer::file_region(
        const ggml_tensor * destination) const {
    if (std::strstr(destination->name, ".engram_embd.") == nullptr) return std::nullopt;
    const target_spec spec = map_target(*quant_, registry_, n_layer_, n_mtp_, n_expert_, destination->name);
    return llama_safetensors_tensor_file_region(registry_, { spec.source, std::nullopt }, destination);
}

std::vector<uint8_t> llama_safetensors_deepseek4_importer::materialize(
        const std::string & target, ggml_type target_type, size_t target_size) const {
    try {
        const target_spec spec = map_target(*quant_, registry_, n_layer_, n_mtp_, n_expert_, target);
        std::vector<uint8_t> result;
        if (spec.kind == target_kind::HASH_TABLE) {
            const auto & source = require_tensor(registry_, spec.source);
            const auto bytes = registry_.read(source);
            if (source.dtype != llama_safetensors_dtype::I64 || bytes.size() % sizeof(int64_t) != 0 || target_type != GGML_TYPE_I32)
                throw std::runtime_error("invalid DeepSeek-V4 hash table contract");
            result.resize(bytes.size() / 2);
            for (size_t i = 0; i < bytes.size() / sizeof(int64_t); ++i) {
                int64_t value; std::memcpy(&value, bytes.data() + i * sizeof(value), sizeof(value));
                if (value < INT32_MIN || value > INT32_MAX) throw std::runtime_error("DeepSeek-V4 hash table value exceeds I32");
                const int32_t narrowed = static_cast<int32_t>(value);
                std::memcpy(result.data() + i * sizeof(narrowed), &narrowed, sizeof(narrowed));
            }
        } else if (spec.kind == target_kind::EH_WEIGHT || spec.kind == target_kind::EH_SCALE) {
            const auto role = spec.kind == target_kind::EH_WEIGHT ? llama_safetensors_quant_role::WEIGHT :
                llama_safetensors_quant_role::WEIGHT_SCALE;
            const auto e = quant_->bind(spec.source + "e_proj", role);
            const auto h = quant_->bind(spec.source + "h_proj", role);
            if (!e || !h) throw std::runtime_error("missing DeepSeek-V4 MTP projection pair");
            auto eb = quant_->finalize(*e, quant_->read(*e));
            auto hb = quant_->finalize(*h, quant_->read(*h));
            if (spec.kind == target_kind::EH_WEIGHT) {
                result = concat_rows(require_tensor(registry_, e->primary), std::move(eb),
                                     require_tensor(registry_, h->primary), std::move(hb));
            } else {
                result.reserve(eb.size() + hb.size());
                result.insert(result.end(), eb.begin(), eb.end());
                result.insert(result.end(), hb.begin(), hb.end());
            }
        } else if (spec.kind == target_kind::ENGRAM_GATE) {
            const auto & q = require_tensor(registry_, spec.source + "q_weight");
            const auto & k = require_tensor(registry_, spec.source + "k_weight");
            if (q.dtype != llama_safetensors_dtype::BF16 || k.dtype != llama_safetensors_dtype::BF16 || target_type != GGML_TYPE_F32)
                throw std::runtime_error("invalid DeepSeek-V4.1 Engram gate contract");
            const auto qf = llama_safetensors_bf16_to_f32(registry_.read(q));
            const auto kf = llama_safetensors_bf16_to_f32(registry_.read(k));
            if (qf.size() != kf.size()) throw std::runtime_error("DeepSeek-V4.1 Engram gate q/k size mismatch");
            result.resize(qf.size());
            for (size_t i = 0; i < qf.size(); i += sizeof(float)) {
                float a, b;
                std::memcpy(&a, qf.data() + i, sizeof(a));
                std::memcpy(&b, kf.data() + i, sizeof(b));
                a *= b;
                std::memcpy(result.data() + i, &a, sizeof(a));
            }
        } else if (spec.kind == target_kind::TENSOR) {
            return llama_safetensors_materialize_tensor(registry_, *quant_, { spec.source, spec.quant }, target_type, target_size);
        } else if (spec.kind == target_kind::STACK) {
            result.reserve(target_size);
            for (const auto & binding : spec.stack) {
                const auto bytes = quant_->finalize(binding, quant_->read(binding));
                result.insert(result.end(), bytes.begin(), bytes.end());
            }
        } else {
            throw std::runtime_error("DeepSeek-V4 routed experts must use bounded upload");
        }
        if (result.size() != target_size) throw std::runtime_error("produced " + std::to_string(result.size()) + " bytes, expected " + std::to_string(target_size));
        return result;
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to materialize DeepSeek-V4 tensor '" + target + "': " + error.what());
    }
}

void llama_safetensors_deepseek4_importer::validate_complete() const {
    quant_->validate_complete();
}
