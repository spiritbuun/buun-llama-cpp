#include "llama-safetensors-deepseek4.h"

#include "llama-safetensors-templates.h"
#include "llama-safetensors-metadata.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-tensor.h"

#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

enum class target_kind {
    TENSOR,
    EXPERTS,
    HASH_TABLE,
    EH_WEIGHT,
    EH_SCALE,
};

struct target_spec {
    target_kind kind = target_kind::TENSOR;
    std::string source;
    std::optional<llama_safetensors_quant_binding> quant;
    uint32_t layer = 0;
    char expert_projection = 0;
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

ggml_type plain_type(const llama_safetensors_tensor & tensor) {
    switch (tensor.dtype) {
        case llama_safetensors_dtype::BF16: return GGML_TYPE_BF16;
        case llama_safetensors_dtype::F16:  return GGML_TYPE_F16;
        case llama_safetensors_dtype::F32:  return GGML_TYPE_F32;
        case llama_safetensors_dtype::I32:  return GGML_TYPE_I32;
        default:
            throw std::runtime_error("unsupported plain DeepSeek-V4 dtype for '" + tensor.name + "'");
    }
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

target_spec map_target(
        const llama_safetensors_quant_adapters & quant,
        uint32_t n_layer,
        uint32_t n_mtp,
        const std::string & target) {
    if (target == "token_embd.weight") return { target_kind::TENSOR, "embed.weight" };
    if (target == "output_norm.weight") return { target_kind::TENSOR, "norm.weight" };
    if (target == "output.weight") return projection(quant, "head");
    if (target == "output.scale") return projection(quant, "head", llama_safetensors_quant_role::WEIGHT_SCALE);
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
    };
    for (const mapping & item : mappings) {
        if (suffix != item.target) continue;
        const std::string source = prefix + item.source;
        if (!item.projected) return { target_kind::TENSOR, source };
        const bool scale = suffix.size() >= 6 && suffix.compare(suffix.size() - 6, 6, ".scale") == 0;
        return projection(quant, source, scale ? llama_safetensors_quant_role::WEIGHT_SCALE :
            llama_safetensors_quant_role::WEIGHT);
    }

    if (suffix == "ffn_gate_tid2eid.weight") return { target_kind::HASH_TABLE, prefix + "ffn.gate.tid2eid" };
    if (suffix == "ffn_gate_exps.weight" || suffix == "ffn_down_exps.weight" || suffix == "ffn_up_exps.weight") {
        target_spec result;
        result.kind = target_kind::EXPERTS;
        result.source = prefix;
        result.layer = layer;
        result.expert_projection = suffix == "ffn_gate_exps.weight" ? '1' :
            suffix == "ffn_down_exps.weight" ? '2' : '3';
        return result;
    }
    return {};
}

std::string expert_module(const target_spec & spec, uint32_t expert) {
    return spec.source + "ffn.experts." + std::to_string(expert) + ".w" + spec.expert_projection;
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

} // namespace

llama_safetensors_deepseek4_importer::llama_safetensors_deepseek4_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_json config,
        llama_safetensors_io_mode io_mode) :
    model_dir_(model_dir), config_(std::move(config)) {
    if (!probe(config_)) throw std::runtime_error("native DeepSeek-V4 importer received the wrong architecture");
    n_layer_ = config_.at("num_hidden_layers").get<uint32_t>();
    n_mtp_ = config_.value("num_nextn_predict_layers", 0U);
    n_expert_ = config_.at("n_routed_experts").get<uint32_t>();
    if (n_layer_ != 43 || n_mtp_ > 1 || n_expert_ != 256 ||
        config_.at("expert_dtype").get<std::string>() != "fp4") {
        throw std::runtime_error("native DeepSeek-V4 importer does not support this tensor geometry");
    }
    generation_ = llama_safetensors_read_json(model_dir_ / "generation_config.json");
    tokenizer_ = llama_safetensors_read_json(model_dir_ / "tokenizer.json");
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
        const std::string_view fallback = model_id_hint.find("0731") != std::string::npos ?
            LLAMA_SAFETENSORS_DEEPSEEK4_0731_CHAT_TEMPLATE : LLAMA_SAFETENSORS_DEEPSEEK4_CHAT_TEMPLATE;
        chat_template_ = std::string(fallback);
    }
    registry_ = llama_safetensors_registry::load(model_dir_, io_mode);
    quant_ = std::make_unique<llama_safetensors_quant_adapters>(config_, registry_);
    if (n_mtp_ != 0 && registry_.find("mtp.0.attn_norm.weight") == nullptr) n_mtp_ = 0;
}

bool llama_safetensors_deepseek4_importer::probe(const llama_safetensors_json & config) {
    return config.value("model_type", std::string()) == "deepseek_v4";
}

gguf_context * llama_safetensors_deepseek4_importer::build_metadata() const {
    llama_safetensors_metadata_sink sink;
    const std::string arch = "deepseek4";
    sink.set_string("general.architecture", arch);
    sink.set_string("general.type", "model");
    sink.set_string("general.name", model_dir_.filename().empty() ? "DeepSeek-V4 Safetensors" : model_dir_.filename().string());
    sink.set_u32("general.file_type", quant_->file_type());
    sink.set_u32("general.quantization_version", 2);
    llama_safetensors_emit_sampling_defaults(sink, generation_);
    sink.set_u32(arch + ".block_count", n_layer_ + n_mtp_);
    sink.set_u32(arch + ".context_length", config_.at("max_position_embeddings").get<uint32_t>());
    sink.set_u32(arch + ".embedding_length", config_.at("hidden_size").get<uint32_t>());
    sink.set_u32(arch + ".embedding_length_out", config_.at("hidden_size").get<uint32_t>() * config_.at("hc_mult").get<uint32_t>());
    sink.set_u32(arch + ".expert_feed_forward_length", config_.at("moe_intermediate_size").get<uint32_t>());
    sink.set_u32(arch + ".expert_count", n_expert_);
    sink.set_u32(arch + ".expert_used_count", config_.at("num_experts_per_tok").get<uint32_t>());
    sink.set_u32(arch + ".expert_shared_count", config_.at("n_shared_experts").get<uint32_t>());
    sink.set_f32(arch + ".expert_weights_scale", config_.at("routed_scaling_factor").get<float>());
    sink.set_bool(arch + ".expert_weights_norm", config_.at("norm_topk_prob").get<bool>());
    sink.set_u32(arch + ".expert_gating_func", 4);
    std::vector<float> clamp(n_layer_ + n_mtp_, config_.at("swiglu_limit").get<float>());
    sink.set_f32_array(arch + ".swiglu_clamp_exp", clamp.data(), clamp.size());
    sink.set_f32_array(arch + ".swiglu_clamp_shexp", clamp.data(), clamp.size());
    sink.set_u32(arch + ".attention.head_count", config_.at("num_attention_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.head_count_kv", config_.at("num_key_value_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.key_length", config_.at("head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.value_length", config_.at("head_dim").get<uint32_t>());
    sink.set_f32(arch + ".attention.layer_norm_rms_epsilon", config_.at("rms_norm_eps").get<float>());
    sink.set_u32(arch + ".attention.q_lora_rank", config_.at("q_lora_rank").get<uint32_t>());
    sink.set_u32(arch + ".attention.sliding_window", config_.at("sliding_window").get<uint32_t>());
    sink.set_u32(arch + ".attention.output_group_count", config_.at("o_groups").get<uint32_t>());
    sink.set_u32(arch + ".attention.output_lora_rank", config_.at("o_lora_rank").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.head_count", config_.at("index_n_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.key_length", config_.at("index_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.top_k", config_.at("index_topk").get<uint32_t>());
    sink.set_f32(arch + ".attention.compress_rope_freq_base", config_.at("compress_rope_theta").get<float>());
    auto ratios = config_.at("compress_ratios").get<std::vector<uint32_t>>();
    if (ratios.size() == n_layer_ && n_mtp_ != 0) ratios.push_back(0);
    sink.set_u32_array(arch + ".attention.compress_ratios", ratios.data(), ratios.size());
    sink.set_u32(arch + ".hyper_connection.count", config_.at("hc_mult").get<uint32_t>());
    sink.set_u32(arch + ".hyper_connection.sinkhorn_iterations", config_.at("hc_sinkhorn_iters").get<uint32_t>());
    sink.set_f32(arch + ".hyper_connection.epsilon", config_.at("hc_eps").get<float>());
    sink.set_u32(arch + ".hash_layer_count", config_.at("num_hash_layers").get<uint32_t>());
    sink.set_u32(arch + ".rope.dimension_count", config_.at("qk_rope_head_dim").get<uint32_t>());
    sink.set_f32(arch + ".rope.freq_base", config_.at("rope_theta").get<float>());
    const auto & rope_scaling = config_.at("rope_scaling");
    if (!rope_scaling.is_object() || rope_scaling.value("type", std::string()) != "yarn") {
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

    const uint32_t bos = llama_safetensors_first_token_id(generation_.at("bos_token_id"), "bos_token_id");
    const uint32_t eos = llama_safetensors_first_token_id(generation_.at("eos_token_id"), "eos_token_id");
    llama_safetensors_emit_bpe_tokenizer(sink, tokenizer_, {
        "joyai-llm", config_.at("vocab_size").get<uint32_t>(), bos, eos, std::string("<｜▁pad▁｜>"), false,
    }, chat_template_);
    return sink.release();
}

bool llama_safetensors_deepseek4_importer::describe(
        const std::string & target, ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const {
    const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
    std::vector<int64_t> shape;
    if (spec.kind == target_kind::EXPERTS) {
        const auto binding = quant_->bind(expert_module(spec, 0), llama_safetensors_quant_role::WEIGHT);
        if (!binding || binding->target_type != GGML_TYPE_MXFP4 || binding->target_shape.size() != 2) return false;
        type = GGML_TYPE_MXFP4;
        shape = binding->target_shape;
        shape.push_back(n_expert_);
    } else if (spec.kind == target_kind::HASH_TABLE) {
        const auto * source = registry_.find(spec.source);
        if (source == nullptr) return false;
        type = GGML_TYPE_I32;
        shape = reverse_shape(*source);
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
        type = spec.quant ? spec.quant->target_type : plain_type(*source);
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
    const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
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
    } else if (spec.quant) {
        quant_->consume(*spec.quant);
    }
}

bool llama_safetensors_deepseek4_importer::load(
        const std::string & target, ggml_tensor * destination, bool check_tensor) const {
    const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
    if (spec.kind == target_kind::EXPERTS) {
        const size_t plane = ggml_nbytes(destination) / n_expert_;
        if (plane * n_expert_ != ggml_nbytes(destination)) throw std::runtime_error("invalid DeepSeek-V4 expert destination");
        const auto build_plane = [&](uint32_t expert) {
            const auto binding = quant_->bind(expert_module(spec, expert), llama_safetensors_quant_role::WEIGHT);
            if (!binding) throw std::runtime_error("missing DeepSeek-V4 expert binding during upload");
            auto bytes = quant_->finalize(*binding, quant_->read(*binding));
            if (bytes.size() != plane) throw std::runtime_error("DeepSeek-V4 expert plane has the wrong byte size");
            if (check_tensor && !ggml_validate_row_data(GGML_TYPE_MXFP4, bytes.data(), bytes.size()))
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

std::vector<uint8_t> llama_safetensors_deepseek4_importer::materialize(
        const std::string & target, ggml_type target_type, size_t target_size) const {
    try {
        const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
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
        } else if (spec.kind == target_kind::TENSOR) {
            const auto & source = require_tensor(registry_, spec.source);
            result = spec.quant ? quant_->finalize(*spec.quant, quant_->read(*spec.quant)) : registry_.read(source);
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
