#include "llama-safetensors-glm5next.h"

#include "llama-safetensors-metadata.h"
#include "llama-safetensors-names.h"
#include "llama-safetensors-tensor.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <utility>

namespace {

enum class target_kind {
    NONE,
    PLAIN,        // registry source copied verbatim, BF16/F16 -> F32 when the target asks for it
    PROJECTION,   // single quantized (EXL3) projection: weight / scale / input_scale
    EXPERTS,      // per-expert quantized projections stacked along the expert dimension
    EXPERT_MAP,   // original routing ID -> precision-group-local ID (or -1)
    CONV_THIRD,   // one of the three contiguous thirds of the fused KDA conv1d weight
    KV_B_K,       // the k component of the split MLA kv_b_proj
    KV_B_V,       // the v component of the split MLA kv_b_proj
    A_LOG,        // ssm_a = -exp(A_log)
};

struct target_spec {
    target_kind kind = target_kind::NONE;
    std::string source;
    std::optional<llama_safetensors_quant_binding> quant;
    llama_safetensors_quant_role role = llama_safetensors_quant_role::WEIGHT;
    std::string expert_projection;
    int conv_third = 0;
    bool force_f32 = false;
    int group = -1;
};

constexpr const char * model_prefix = "model.language_model";

const llama_safetensors_tensor & require_tensor(
        const llama_safetensors_registry & registry, const std::string & name) {
    const auto * tensor = registry.find(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing GLM5-Next source tensor '" + name + "'");
    }
    return *tensor;
}

bool ends_with(const std::string & value, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return value.size() >= n && value.compare(value.size() - n, n, suffix) == 0;
}

std::vector<int64_t> reverse_shape(const llama_safetensors_tensor & tensor) {
    std::vector<int64_t> result;
    result.reserve(tensor.shape.size());
    for (auto it = tensor.shape.rbegin(); it != tensor.shape.rend(); ++it) {
        if (*it > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::runtime_error("GLM5-Next tensor dimension exceeds runtime limits: '" + tensor.name + "'");
        }
        result.push_back(static_cast<int64_t>(*it));
    }
    return result;
}

// Mirrors llama_safetensors_plain_type: rank-1 half-precision vectors (norms, biases) are
// promoted to F32 so the graph reads them directly; rank-2+ stays in its native precision.
ggml_type plain_type(const llama_safetensors_tensor & tensor, bool force_f32) {
    if (force_f32) {
        return GGML_TYPE_F32;
    }
    switch (tensor.dtype) {
        case llama_safetensors_dtype::BF16: return tensor.shape.size() >= 2 ? GGML_TYPE_BF16 : GGML_TYPE_F32;
        case llama_safetensors_dtype::F16:  return tensor.shape.size() >= 2 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        case llama_safetensors_dtype::F32:  return GGML_TYPE_F32;
        case llama_safetensors_dtype::I32:  return GGML_TYPE_I32;
        default:
            throw std::runtime_error("unsupported plain GLM5-Next dtype for '" + tensor.name + "'");
    }
}

target_spec make_projection(
        const llama_safetensors_quant_adapters & quant,
        const std::string & module,
        llama_safetensors_quant_role role) {
    target_spec result;
    result.quant = quant.bind(module, role);
    if (!result.quant) {
        return {};
    }
    result.kind = target_kind::PROJECTION;
    result.source = result.quant->primary;
    return result;
}

// The generic side-tensor pass requests "<base>.scale" / "<base>.input_scale" for EXL3 projections;
// everything else arrives as "<base>.weight", "<base>.expert_map" or a bare / ".bias" name.
llama_safetensors_quant_role split_role(const std::string & name, std::string & base) {
    static const struct { const char * suffix; llama_safetensors_quant_role role; } roles[] = {
        { ".expert_map",  llama_safetensors_quant_role::WEIGHT },
        { ".input_scale", llama_safetensors_quant_role::INPUT_SCALE },
        { ".scale",       llama_safetensors_quant_role::WEIGHT_SCALE },
        { ".weight",      llama_safetensors_quant_role::WEIGHT },
    };
    for (const auto & r : roles) {
        if (ends_with(name, r.suffix)) {
            base = name.substr(0, name.size() - std::strlen(r.suffix));
            return r.role;
        }
    }
    base = name;
    return llama_safetensors_quant_role::WEIGHT;
}

target_spec map_target(
        const llama_safetensors_quant_adapters & quant,
        uint32_t n_layer,
        uint32_t n_mtp,
        const std::string & target) {
    if (target == "token_embd.weight") {
        target_spec s; s.kind = target_kind::PLAIN; s.source = std::string(model_prefix) + ".embed_tokens.weight"; return s;
    }
    if (target == "output_norm.weight") {
        target_spec s; s.kind = target_kind::PLAIN; s.source = std::string(model_prefix) + ".norm.weight"; return s;
    }
    if (target == "output.weight" || target == "output.scale" || target == "output.input_scale") {
        std::string base;
        return make_projection(quant, "lm_head", split_role(target, base));
    }

    static const std::regex block(R"(^blk\.([0-9]+)\.(.+)$)");
    std::smatch match;
    if (!std::regex_match(target, match, block)) return {};
    const uint64_t parsed = std::stoull(match[1].str());
    if (parsed >= static_cast<uint64_t>(n_layer) + n_mtp) return {};
    const uint32_t layer = static_cast<uint32_t>(parsed);
    const std::string prefix = std::string(model_prefix) + ".layers." + std::to_string(layer) + ".";
    std::string suffix = match[2].str();
    int group = -1;
    static const std::regex grouped(R"(^(ffn_(?:gate|up|down)_exps)\.g([0-9]+)\.(weight|scale|input_scale|expert_map)$)");
    std::smatch group_match;
    if (std::regex_match(suffix, group_match, grouped)) {
        group = std::stoi(group_match[2].str());
        suffix = group_match[1].str() + "." + group_match[3].str();
    }

    std::string base;
    const llama_safetensors_quant_role role = split_role(suffix, base);

    // EXL3 single projections: GGUF base -> checkpoint module (relative to the layer prefix).
    static const struct { const char * base; const char * module; } projections[] = {
        { "attn_qkv",       "self_attn.qkv_proj" },            // fused KDA q|k|v
        { "attn_output",    "self_attn.o_proj" },              // KDA + MLA output
        { "attn_q_a",       "self_attn.q_a_proj" },
        { "attn_q_b",       "self_attn.q_b_proj" },
        { "attn_kv_a_mqa",  "self_attn.kv_a_proj_with_mqa" },
        { "indexer.attn_q_b", "self_attn.indexer.wq_b" },
        { "ffn_gate",       "mlp.gate_proj" },                 // dense layers 0..2
        { "ffn_up",         "mlp.up_proj" },
        { "ffn_down",       "mlp.down_proj" },
        { "ffn_gate_shexp", "mlp.shared_experts.gate_proj" },
        { "ffn_up_shexp",   "mlp.shared_experts.up_proj" },
        { "ffn_down_shexp", "mlp.shared_experts.down_proj" },
        { "nextn.eh_proj",  "eh_proj" },
    };
    for (const auto & projection : projections) {
        if (base == projection.base) {
            auto spec = make_projection(quant, prefix + projection.module, role);
            if (spec.quant) return spec;
            break;
        }
    }

    // EXL3 routed experts stacked across mlp.experts.{0..n_expert-1}.
    if (base == "ffn_gate_exps" || base == "ffn_up_exps" || base == "ffn_down_exps") {
        target_spec result;
        result.kind = ends_with(suffix, ".expert_map") ? target_kind::EXPERT_MAP : target_kind::EXPERTS;
        result.group = group;
        result.source = prefix;
        result.role = role;
        result.expert_projection = base == "ffn_gate_exps" ? "gate_proj" :
            base == "ffn_up_exps" ? "up_proj" : "down_proj";
        return result;
    }

    // Architecture-specific repacks keyed by the full suffix (plain F16/F32 sources, never EXL3).
    if (suffix == "ssm_a") {
        target_spec s; s.kind = target_kind::A_LOG; s.source = prefix + "self_attn.A_log"; return s;
    }
    if (suffix == "ssm_conv1d_q.weight" || suffix == "ssm_conv1d_k.weight" || suffix == "ssm_conv1d_v.weight") {
        target_spec s; s.kind = target_kind::CONV_THIRD; s.source = prefix + "self_attn.conv1d.weight";
        s.conv_third = suffix == "ssm_conv1d_q.weight" ? 0 : suffix == "ssm_conv1d_k.weight" ? 1 : 2;
        return s;
    }
    if (suffix == "attn_k_b.weight") {
        target_spec s; s.kind = target_kind::KV_B_K; s.source = prefix + "self_attn.kv_b_proj.weight"; return s;
    }
    if (suffix == "attn_v_b.weight") {
        target_spec s; s.kind = target_kind::KV_B_V; s.source = prefix + "self_attn.kv_b_proj.weight"; return s;
    }

    // Plain tensors keyed by the full GGUF suffix -> checkpoint name (relative to the layer prefix).
    static const struct { const char * suffix; const char * source; } plains[] = {
        { "attn_norm.weight",               "input_layernorm.weight" },
        { "ffn_norm.weight",                "post_attention_layernorm.weight" },
        { "hc_attn_fn.weight",              "hc_attn_fn" },
        { "hc_attn_base.weight",            "hc_attn_base" },
        { "hc_attn_scale.weight",           "hc_attn_scale" },
        { "hc_ffn_fn.weight",               "hc_ffn_fn" },
        { "hc_ffn_base.weight",             "hc_ffn_base" },
        { "hc_ffn_scale.weight",            "hc_ffn_scale" },
        { "ssm_f_a.weight",                 "self_attn.f_a_proj.weight" },
        { "ssm_f_b.weight",                 "self_attn.f_b_proj.weight" },
        { "ssm_g_a.weight",                 "self_attn.g_a_proj.weight" },
        { "ssm_g_b.weight",                 "self_attn.g_b_proj.weight" },
        { "ssm_beta.weight",                "self_attn.b_proj.weight" },
        { "ssm_dt.bias",                    "self_attn.dt_bias" },
        { "ssm_norm.weight",                "self_attn.o_norm.weight" },
        { "attn_q_a_norm.weight",           "self_attn.q_a_layernorm.weight" },
        { "attn_kv_a_norm.weight",          "self_attn.kv_a_layernorm.weight" },
        { "indexer.k_norm.weight",          "self_attn.indexer.k_norm.weight" },
        { "indexer.k_norm.bias",            "self_attn.indexer.k_norm.bias" },
        { "indexer.proj.weight",            "self_attn.indexer.weights_proj.weight" },
        { "indexer.attn_k.weight",          "self_attn.indexer.wk.weight" },
        { "indexer.attn_q_b.weight",        "self_attn.indexer.wq_b.weight" },
        { "indexer_compressor_gate.weight", "self_attn.indexer.index_kpool_compress_gate" },
        { "indexer_compressor_ape.weight",  "self_attn.indexer.index_kpool_compress_ape" },
        { "ffn_gate_inp.weight",            "mlp.gate.weight" },
        { "exp_probs_b.bias",               "mlp.gate.e_score_correction_bias" },
        { "nextn.enorm.weight",             "enorm.weight" },
        { "nextn.hnorm.weight",             "hnorm.weight" },
        { "nextn.shared_head_norm.weight",  "shared_head.norm.weight" },
    };
    for (const auto & plain : plains) {
        if (suffix == plain.suffix) {
            target_spec s; s.kind = target_kind::PLAIN; s.source = prefix + plain.source;
            // the one rank-2 F16 source the GGUF converter promotes to F32
            s.force_f32 = suffix == "indexer_compressor_gate.weight";
            return s;
        }
    }

    return {};
}

std::string expert_module(const target_spec & spec, uint32_t expert) {
    return spec.source + "mlp.experts." + std::to_string(expert) + "." + spec.expert_projection;
}

std::vector<uint8_t> promote_to_f32(const llama_safetensors_tensor & source, std::vector<uint8_t> bytes) {
    if (source.dtype == llama_safetensors_dtype::BF16) return llama_safetensors_bf16_to_f32(bytes);
    if (source.dtype == llama_safetensors_dtype::F16)  return llama_safetensors_f16_to_f32(bytes);
    return bytes;
}

// conv1d.weight is [3*d_inner, 1, d_conv] row-major; each third (q/k/v) is a contiguous block that
// lands directly into the ggml [d_conv, 1, d_inner] tensor without reordering.
std::vector<uint8_t> conv_third(const llama_safetensors_tensor & source,
        const std::vector<uint8_t> & bytes, int third, size_t target_size) {
    if (source.dtype != llama_safetensors_dtype::BF16 && source.dtype != llama_safetensors_dtype::F16 &&
            source.dtype != llama_safetensors_dtype::F32) {
        throw std::runtime_error("unsupported GLM5-Next conv1d dtype");
    }
    const size_t third_bytes = bytes.size() / 3;
    if (third_bytes * 3 != bytes.size()) {
        throw std::runtime_error("GLM5-Next conv1d weight is not divisible into three channels");
    }
    std::vector<uint8_t> slice(bytes.begin() + third * third_bytes, bytes.begin() + (third + 1) * third_bytes);
    std::vector<uint8_t> result = promote_to_f32(source, std::move(slice));
    if (result.size() != target_size) {
        throw std::runtime_error("GLM5-Next conv1d third has the wrong size");
    }
    return result;
}

// kv_b_proj is F16 [n_head*(qk_nope+v_head), kv_lora] laid out row-major as [h, m, l] with
// m = qk_nope+v_head rows per head and l = kv_lora latent columns. The leading qk_nope rows per
// head are the K component, the trailing v_head rows are the V component. qk_nope, v_head and
// kv_lora are independent (all 256/256/512 for GLM-5.3-Flash), so the split sizes come from the
// config, never from assuming any two of them are equal.
std::vector<uint8_t> kv_b_split(
        const llama_safetensors_tensor & source, const std::vector<uint8_t> & bytes,
        bool want_v, int64_t n_head, int64_t qk_nope, int64_t v_head, int64_t kv_lora, size_t target_size) {
    if (source.dtype != llama_safetensors_dtype::F16) {
        throw std::runtime_error("GLM5-Next kv_b_proj must be F16");
    }
    const int64_t m_rows = qk_nope + v_head;            // rows per head in the fused projection
    const int64_t l_cols = kv_lora;                     // shared latent columns per row
    const int64_t v_dim = want_v ? v_head : qk_nope;    // rows this split keeps
    if (bytes.size() != static_cast<size_t>(n_head * m_rows * l_cols) * sizeof(uint16_t)) {
        throw std::runtime_error("GLM5-Next kv_b_proj has an unexpected element count");
    }
    const uint16_t * src = reinterpret_cast<const uint16_t *>(bytes.data());
    std::vector<uint16_t> out(static_cast<size_t>(n_head) * v_dim * l_cols);
    for (int64_t h = 0; h < n_head; ++h) {
        for (int64_t r = 0; r < v_dim; ++r) {
            const int64_t m = want_v ? (qk_nope + r) : r;
            for (int64_t l = 0; l < l_cols; ++l) {
                const int64_t src_index = h * m_rows * l_cols + m * l_cols + l;
                // k: ggml [qk_nope, kv_lora, n_head] -> (r, l, h); v: [kv_lora, v_head, n_head] -> (l, r, h)
                const int64_t dst_index = want_v ?
                    (h * v_dim * l_cols + r * l_cols + l) :
                    (h * v_dim * l_cols + l * v_dim + r);
                out[dst_index] = src[src_index];
            }
        }
    }
    std::vector<uint8_t> result(out.size() * sizeof(uint16_t));
    std::memcpy(result.data(), out.data(), result.size());
    if (result.size() != target_size) {
        throw std::runtime_error("GLM5-Next kv_b split has the wrong size");
    }
    return result;
}

std::vector<uint8_t> a_log(const llama_safetensors_tensor & source, std::vector<uint8_t> bytes, size_t target_size) {
    if (source.dtype != llama_safetensors_dtype::F32 || bytes.size() != target_size) {
        throw std::runtime_error("GLM5-Next A_log must be F32");
    }
    float * values = reinterpret_cast<float *>(bytes.data());
    for (size_t i = 0; i < bytes.size() / sizeof(float); ++i) {
        values[i] = -std::exp(values[i]);
    }
    return bytes;
}

const llama_safetensors_json & text_config(const llama_safetensors_json & config) {
    return config.contains("text_config") ? config.at("text_config") : config;
}

// MLA split dimensions for kv_b_proj. These mirror the glm5-next graph builder: n_head from
// num_attention_heads, qk_nope = qk_head_dim - qk_rope_head_dim, v_head = v_head_dim, latent =
// kv_lora_rank. They are the authority for the attn_k_b/attn_v_b shapes and for the row split.
struct kv_b_dims {
    int64_t n_head;
    int64_t qk_nope;
    int64_t v_head;
    int64_t kv_lora;
};

kv_b_dims mla_kv_b_dims(const llama_safetensors_json & config) {
    const auto & text = text_config(config);
    kv_b_dims d;
    d.n_head  = text.at("num_attention_heads").get<int64_t>();
    d.qk_nope = text.at("qk_head_dim").get<int64_t>() - text.at("qk_rope_head_dim").get<int64_t>();
    d.v_head  = text.at("v_head_dim").get<int64_t>();
    d.kv_lora = text.at("kv_lora_rank").get<int64_t>();
    return d;
}

} // namespace

llama_safetensors_glm5next_importer::llama_safetensors_glm5next_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_json config,
        llama_safetensors_io_mode io_mode) :
    model_dir_(model_dir), config_(std::move(config)), generation_(llama_safetensors_json::object()) {
    if (!probe(config_)) throw std::runtime_error("native GLM5-Next importer received the wrong architecture");
    const auto & text = text_config(config_);
    n_layer_ = text.at("num_hidden_layers").get<uint32_t>();
    n_mtp_ = text.value("num_nextn_predict_layers", 0U);
    n_expert_ = text.at("n_routed_experts").get<uint32_t>();
    if (n_mtp_ > 1 || n_expert_ == 0) {
        throw std::runtime_error("native GLM5-Next importer does not support this tensor geometry");
    }

    const auto generation_path = model_dir_ / "generation_config.json";
    if (std::filesystem::is_regular_file(generation_path)) {
        generation_ = llama_safetensors_read_json(generation_path);
    }
    tokenizer_ = llama_safetensors_read_tokenizer_json(model_dir_ / "tokenizer.json");
    const auto tokenizer_config_path = model_dir_ / "tokenizer_config.json";
    if (std::filesystem::is_regular_file(tokenizer_config_path)) {
        const auto tokenizer_config = llama_safetensors_read_json(tokenizer_config_path);
        if (tokenizer_config.contains("chat_template") && tokenizer_config.at("chat_template").is_string()) {
            chat_template_ = tokenizer_config.at("chat_template").get<std::string>();
        }
    }
    if (!chat_template_) chat_template_ = llama_safetensors_read_optional_text(model_dir_ / "chat_template.jinja");

    registry_ = llama_safetensors_registry::load(model_dir_, io_mode);
    quant_ = std::make_unique<llama_safetensors_quant_adapters>(config_, registry_);
    if (n_mtp_ != 0 && registry_.find(std::string(model_prefix) + ".layers." + std::to_string(n_layer_) + ".enorm.weight") == nullptr) {
        n_mtp_ = 0;
    }
}

bool llama_safetensors_glm5next_importer::probe(const llama_safetensors_json & config) {
    const std::string model_type = config.value("model_type", std::string());
    return model_type == "glm5_next" || model_type == "glm5_next_text";
}

gguf_context * llama_safetensors_glm5next_importer::build_metadata() const {
    const auto & text = text_config(config_);
    llama_safetensors_metadata_sink sink;
    const std::string arch = "glm5-next";
    const uint32_t n_layer_all = n_layer_ + n_mtp_;

    sink.set_string("general.architecture", arch);
    sink.set_string("general.type", "model");
    sink.set_string("general.name", model_dir_.filename().empty() ? "GLM5-Next Safetensors" : model_dir_.filename().string());
    sink.set_u32("general.file_type", quant_->file_type());
    sink.set_u32("general.quantization_version", 2);
    llama_safetensors_emit_sampling_defaults(sink, generation_);

    sink.set_u32(arch + ".block_count", n_layer_all);
    sink.set_u32(arch + ".context_length", text.at("max_position_embeddings").get<uint32_t>());
    sink.set_u32(arch + ".embedding_length", text.at("hidden_size").get<uint32_t>());
    sink.set_u32(arch + ".feed_forward_length", text.at("intermediate_size").get<uint32_t>());
    sink.set_u32(arch + ".expert_feed_forward_length", text.at("moe_intermediate_size").get<uint32_t>());
    sink.set_u32(arch + ".leading_dense_block_count", text.at("first_k_dense_replace").get<uint32_t>());
    sink.set_u32(arch + ".expert_count", n_expert_);
    sink.set_u32(arch + ".expert_used_count", text.at("num_experts_per_tok").get<uint32_t>());
    sink.set_u32(arch + ".expert_shared_count", text.at("n_shared_experts").get<uint32_t>());
    sink.set_f32(arch + ".expert_weights_scale", text.at("routed_scaling_factor").get<float>());
    sink.set_bool(arch + ".expert_weights_norm", text.at("norm_topk_prob").get<bool>());
    sink.set_u32(arch + ".expert_gating_func", 2);   // sigmoid
    {
        std::vector<float> clamp(n_layer_all, text.at("swiglu_limit").get<float>());
        sink.set_f32_array(arch + ".swiglu_clamp_exp", clamp.data(), clamp.size());
    }

    sink.set_u32(arch + ".attention.head_count", text.at("num_attention_heads").get<uint32_t>());
    {
        // KDA uses no KV heads; MLA and trailing NextN blocks cache one latent per token.
        const auto layers = text.at("layer_types").get<std::vector<std::string>>();
        if (layers.size() != n_layer_) throw std::runtime_error("GLM5-Next layer_types length mismatch");
        std::vector<uint32_t> head_count_kv(n_layer_all, 1);
        for (uint32_t i = 0; i < n_layer_; ++i) {
            if (layers[i] != "linear_attention" && layers[i] != "deepseek_sparse_attention") {
                throw std::runtime_error("unsupported GLM5-Next attention type: " + layers[i]);
            }
            head_count_kv[i] = layers[i] != "linear_attention";
        }
        sink.set_u32_array(arch + ".attention.head_count_kv", head_count_kv.data(), head_count_kv.size());
    }
    sink.set_f32(arch + ".attention.layer_norm_rms_epsilon", text.at("rms_norm_eps").get<float>());
    sink.set_u32(arch + ".attention.q_lora_rank", text.at("q_lora_rank").get<uint32_t>());
    const uint32_t kv_lora = text.at("kv_lora_rank").get<uint32_t>();
    const uint32_t qk_rope = text.at("qk_rope_head_dim").get<uint32_t>();
    sink.set_u32(arch + ".attention.kv_lora_rank", kv_lora);
    // MLA stores the compressed latent (+rope) in the KV cache; key_length/value_length size that
    // cache (n_embd_k_gqa = key_length * head_count_kv), while the *_mla pair carries the real head
    // geometry used for the absorbed-q projection.
    sink.set_u32(arch + ".attention.key_length", kv_lora + qk_rope);
    sink.set_u32(arch + ".attention.value_length", kv_lora);
    sink.set_u32(arch + ".attention.key_length_mla", text.at("qk_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.value_length_mla", text.at("v_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".rope.dimension_count", qk_rope);   // 0 == nope-only

    sink.set_u32(arch + ".attention.indexer.head_count", text.at("index_n_heads").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.key_length", text.at("index_head_dim").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.top_k", text.at("index_topk").get<uint32_t>());
    sink.set_u32(arch + ".attention.indexer.kpool", text.at("index_kpool").get<uint32_t>());
    sink.set_bool(arch + ".attention.indexer.kpool_select_tail", text.at("index_kpool_always_select_tail").get<bool>());
    if (text.contains("indexer_types")) {
        const auto types = text.at("indexer_types").get<std::vector<std::string>>();
        if (types.size() != n_layer_) throw std::runtime_error("GLM5-Next indexer_types length mismatch");
        std::vector<uint32_t> full;
        for (const auto & type : types) full.push_back(type == "full");
        sink.set_u32_array(arch + ".attention.indexer.types", full.data(), full.size());
    }

    const auto & linear = text.at("linear_attn_config");
    sink.set_u32(arch + ".ssm.conv_kernel", linear.at("short_conv_kernel_size").get<uint32_t>());
    sink.set_u32(arch + ".kda.head_dim", linear.at("head_dim").get<uint32_t>());
    sink.set_f32(arch + ".kda.gate_lower_bound", linear.at("gate_lower_bound").get<float>());

    sink.set_u32(arch + ".hyper_connection.count", text.at("hc_mult").get<uint32_t>());
    sink.set_u32(arch + ".hyper_connection.sinkhorn_iterations", text.at("hc_sinkhorn_iters").get<uint32_t>());
    sink.set_f32(arch + ".hyper_connection.epsilon", text.at("hc_eps").get<float>());
    if (n_mtp_ != 0) sink.set_u32(arch + ".nextn_predict_layers", n_mtp_);

    const auto & eos_source = generation_.contains("eos_token_id") ? generation_.at("eos_token_id") : text.at("eos_token_id");
    const uint32_t eos = llama_safetensors_first_token_id(eos_source, "GLM5-Next eos_token_id");
    const auto special_id = [&](const char * content) {
        for (const auto & token : tokenizer_.at("added_tokens")) {
            if (token.at("content") == content) return token.at("id").get<uint32_t>();
        }
        throw std::runtime_error(std::string("missing GLM5-Next special token: ") + content);
    };
    llama_safetensors_emit_bpe_tokenizer(sink, tokenizer_, {
        "glm5", text.at("vocab_size").get<uint32_t>(), special_id("[gMASK]"), eos,
        std::nullopt, true, {}, false,
    }, chat_template_);
    // Match the GGUF converter's _set_vocab_glm. GLM terminates assistant
    // turns with the next role marker, not just <|endoftext|>.
    sink.set_u32("tokenizer.ggml.eot_token_id", special_id("<|user|>"));
    sink.set_u32("tokenizer.ggml.eom_token_id", special_id("<|observation|>"));
    sink.set_u32("tokenizer.ggml.unknown_token_id", special_id("<|endoftext|>"));
    return sink.release();
}

const std::vector<std::vector<uint32_t>> & llama_safetensors_glm5next_importer::partitions(
        const std::string & prefix, const std::string & projection) const {
    const std::string key = prefix + projection;
    std::lock_guard<std::mutex> lock(partitions_mutex_);
    auto found = partitions_.find(key);
    if (found != partitions_.end()) return found->second;
    target_spec spec;
    spec.source = prefix;
    spec.expert_projection = projection;
    std::map<ggml_type, std::vector<uint32_t>> by_type;
    std::vector<int64_t> shape;
    for (uint32_t i = 0; i < n_expert_; ++i) {
        const auto binding = quant_->bind(expert_module(spec, i), llama_safetensors_quant_role::WEIGHT);
        if (!binding || !ggml_type_is_exl3(binding->target_type)) {
            throw std::runtime_error("expected EXL3 expert: " + expert_module(spec, i));
        }
        if (i == 0) shape = binding->target_shape;
        if (shape != binding->target_shape) throw std::runtime_error("expert shapes differ: " + key);
        by_type[binding->target_type].push_back(i);
    }
    std::vector<std::vector<uint32_t>> result;
    for (auto & item : by_type) result.push_back(std::move(item.second));
    return partitions_.emplace(key, std::move(result)).first->second;
}

const std::vector<uint32_t> & llama_safetensors_glm5next_importer::expert_ids(
        const std::string & prefix, const std::string & projection, int group) const {
    const auto & groups = partitions(prefix, projection);
    if (group >= 0) return groups.at(group);
    if (groups.size() != 1) throw std::runtime_error("mixed expert bank requires precision groups: " + prefix + projection);
    return groups.front();
}

std::vector<int64_t> llama_safetensors_glm5next_importer::expert_group_sizes(const std::string & target) const {
    const auto spec = map_target(*quant_, n_layer_, n_mtp_, target);
    if (spec.kind != target_kind::EXPERTS) return {};
    const auto & groups = partitions(spec.source, spec.expert_projection);
    if (groups.size() == 1) return {};
    std::vector<int64_t> sizes;
    for (const auto & group : groups) sizes.push_back(group.size());
    return sizes;
}

bool llama_safetensors_glm5next_importer::describe(
        const std::string & target, ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const {
    const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
    std::vector<int64_t> shape;
    switch (spec.kind) {
        case target_kind::NONE:
            return false;
        case target_kind::PROJECTION:
            return llama_safetensors_describe_tensor(registry_, { spec.source, spec.quant }, type, ne);
        case target_kind::EXPERTS: {
            const auto & ids = expert_ids(spec.source, spec.expert_projection, spec.group);
            const auto binding = quant_->bind(expert_module(spec, ids.front()), spec.role);
            if (!binding) return false;
            type = binding->target_type;
            shape = binding->target_shape;
            if (shape.size() == 1 && shape[0] == 1) {
                shape = { static_cast<int64_t>(ids.size()) };   // scalar per-expert activation scale
            } else {
                shape.push_back(static_cast<int64_t>(ids.size()));
            }
            break;
        }
        case target_kind::EXPERT_MAP:
            (void) expert_ids(spec.source, spec.expert_projection, spec.group);
            type = GGML_TYPE_I32;
            shape = {1, n_expert_};
            break;
        case target_kind::CONV_THIRD: {
            const auto & source = require_tensor(registry_, spec.source);   // [3*d_inner, 1, d_conv]
            const std::vector<int64_t> full = reverse_shape(source);        // ggml [d_conv, 1, 3*d_inner]
            if (full.size() != 3 || full[2] % 3 != 0) return false;
            type = GGML_TYPE_F32;
            shape = { full[0], full[1], full[2] / 3 };
            break;
        }
        case target_kind::KV_B_K:
        case target_kind::KV_B_V: {
            const auto & source = require_tensor(registry_, spec.source);   // F16 [n_head*(qk_nope+v_head), kv_lora]
            if (source.shape.size() != 2) return false;
            const kv_b_dims d = mla_kv_b_dims(config_);
            type = GGML_TYPE_F16;
            shape = spec.kind == target_kind::KV_B_V ?
                std::vector<int64_t>{ d.kv_lora, d.v_head,  d.n_head } :     // [kv_lora, v_head, n_head]
                std::vector<int64_t>{ d.qk_nope, d.kv_lora, d.n_head };      // [qk_nope, kv_lora, n_head]
            break;
        }
        case target_kind::A_LOG: {
            const auto & source = require_tensor(registry_, spec.source);
            type = GGML_TYPE_F32;
            shape = reverse_shape(source);
            break;
        }
        case target_kind::PLAIN: {
            const auto * source = registry_.find(spec.source);
            if (source == nullptr) return false;
            type = plain_type(*source, spec.force_f32);
            shape = reverse_shape(*source);
            break;
        }
    }
    if (shape.empty() || shape.size() > GGML_MAX_DIMS) return false;
    ne.fill(1);
    std::copy(shape.begin(), shape.end(), ne.begin());
    return true;
}

size_t llama_safetensors_glm5next_importer::tensor_capacity_hint() const {
    return std::max<size_t>(1024, registry_.tensors().size() / 8);
}

void llama_safetensors_glm5next_importer::bind(const std::string & target) const {
    const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
    if (spec.kind == target_kind::EXPERTS) {
        for (uint32_t expert : expert_ids(spec.source, spec.expert_projection, spec.group)) {
            const auto binding = quant_->bind(expert_module(spec, expert), spec.role);
            if (!binding) throw std::runtime_error("missing GLM5-Next routed expert binding '" + expert_module(spec, expert) + "'");
            quant_->consume(*binding);
        }
    } else if (spec.quant) {
        quant_->consume(*spec.quant);
    }
}

bool llama_safetensors_glm5next_importer::load(
        const std::string & target, ggml_tensor * destination, bool check_tensor) const {
    const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
    // RAW sources (unquantized weights whose on-disk type already matches) upload straight from the
    // mapped shard; everything else (EXL3 repacks, type promotions, architecture repacks) is built
    // by materialize().
    if (spec.kind == target_kind::PLAIN || spec.kind == target_kind::PROJECTION) {
        return llama_safetensors_load_tensor_direct(registry_, { spec.source, spec.quant }, destination, check_tensor);
    }
    if (spec.kind == target_kind::EXPERTS) {
        size_t offset = 0;
        stream(target, [&](const void * data, size_t size) {
            if (size > ggml_nbytes(destination) - offset) throw std::runtime_error("expert upload exceeds destination");
            ggml_backend_tensor_set(destination, data, offset, size);
            offset += size;
        });
        if (offset != ggml_nbytes(destination)) throw std::runtime_error("incomplete expert upload");
        return true;
    }
    return false;
}

std::vector<uint8_t> llama_safetensors_glm5next_importer::materialize(
        const std::string & target, ggml_type target_type, size_t target_size) const {
    try {
        const target_spec spec = map_target(*quant_, n_layer_, n_mtp_, target);
        std::vector<uint8_t> result;
        switch (spec.kind) {
            case target_kind::NONE:
                throw std::runtime_error("unmapped GLM5-Next target");
            case target_kind::PROJECTION:
            case target_kind::PLAIN:
                return llama_safetensors_materialize_tensor(registry_, *quant_, { spec.source, spec.quant },
                                                            target_type, target_size);
            case target_kind::EXPERTS:
                result.reserve(target_size);
                stream(target, [&](const void * data, size_t size) {
                    const auto * bytes = static_cast<const uint8_t *>(data);
                    result.insert(result.end(), bytes, bytes + size);
                });
                break;
            case target_kind::EXPERT_MAP: {
                std::vector<int32_t> map(n_expert_, -1);
                const auto & ids = expert_ids(spec.source, spec.expert_projection, spec.group);
                for (size_t i = 0; i < ids.size(); ++i) map[ids[i]] = i;
                result.resize(map.size() * sizeof(int32_t));
                std::memcpy(result.data(), map.data(), result.size());
                break;
            }
            case target_kind::CONV_THIRD: {
                const auto & source = require_tensor(registry_, spec.source);
                result = conv_third(source, registry_.read(source), spec.conv_third, target_size);
                break;
            }
            case target_kind::KV_B_K:
            case target_kind::KV_B_V: {
                const auto & source = require_tensor(registry_, spec.source);
                const kv_b_dims d = mla_kv_b_dims(config_);
                result = kv_b_split(source, registry_.read(source), spec.kind == target_kind::KV_B_V,
                                    d.n_head, d.qk_nope, d.v_head, d.kv_lora, target_size);
                break;
            }
            case target_kind::A_LOG: {
                const auto & source = require_tensor(registry_, spec.source);
                result = a_log(source, registry_.read(source), target_size);
                break;
            }
        }
        if (result.size() != target_size) {
            throw std::runtime_error("produced " + std::to_string(result.size()) + " bytes, expected " + std::to_string(target_size));
        }
        return result;
    } catch (const std::exception & error) {
        throw std::runtime_error("failed to materialize GLM5-Next tensor '" + target + "': " + error.what());
    }
}

bool llama_safetensors_glm5next_importer::can_stream(const std::string & target) const {
    const auto spec = map_target(*quant_, n_layer_, n_mtp_, target);
    if (spec.quant) return quant_->can_stream(*spec.quant);
    if (spec.kind != target_kind::EXPERTS) return false;
    for (auto id : expert_ids(spec.source, spec.expert_projection, spec.group)) {
        const auto binding = quant_->bind(expert_module(spec, id), spec.role);
        if (!binding || !quant_->can_stream(*binding)) return false;
    }
    return true;
}

void llama_safetensors_glm5next_importer::stream(
        const std::string & target, const std::function<void(const void *, size_t)> & write) const {
    const auto spec = map_target(*quant_, n_layer_, n_mtp_, target);
    if (spec.quant) {
        quant_->stream(*spec.quant, write);
    } else if (spec.kind == target_kind::EXPERTS) {
        for (auto id : expert_ids(spec.source, spec.expert_projection, spec.group)) {
            const auto binding = quant_->bind(expert_module(spec, id), spec.role);
            if (!binding) throw std::runtime_error("missing expert: " + expert_module(spec, id));
            if (quant_->can_stream(*binding)) {
                quant_->stream(*binding, write);
            } else {
                const auto bytes = quant_->finalize(*binding, quant_->read(*binding));
                write(bytes.data(), bytes.size());
            }
        }
    } else {
        throw std::runtime_error("unsupported streaming tensor: " + target);
    }
}

void llama_safetensors_glm5next_importer::validate_complete() const {
    quant_->validate_complete();
}
