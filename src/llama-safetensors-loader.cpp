#include "llama-safetensors.h"

#include "llama-safetensors-importer.h"
#include "llama-safetensors-qwen3.h"
#include "llama-safetensors-qwen35.h"
#include "llama-model-source.h"
#include "llama.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using importer_probe = bool (*)(const llama_safetensors_json &);
using importer_create = std::unique_ptr<llama_safetensors_importer> (*)(
    const std::filesystem::path &, const llama_safetensors_json &, llama_safetensors_io_mode);

struct importer_registration {
    const char *    name;
    importer_probe  probe;
    importer_create create;
};

std::unique_ptr<llama_safetensors_importer> create_qwen35_importer(
        const std::filesystem::path & model_dir,
        const llama_safetensors_json & config,
        llama_safetensors_io_mode io_mode) {
    return std::make_unique<llama_safetensors_qwen35_importer>(model_dir, config, io_mode);
}

std::unique_ptr<llama_safetensors_importer> create_qwen3_importer(
        const std::filesystem::path & model_dir,
        const llama_safetensors_json & config,
        llama_safetensors_io_mode io_mode) {
    return std::make_unique<llama_safetensors_qwen3_importer>(model_dir, config, io_mode);
}

std::unique_ptr<llama_safetensors_importer> select_importer(
        const std::filesystem::path & model_dir,
        llama_safetensors_io_mode io_mode) {
    const llama_safetensors_json config = llama_safetensors_read_model_config(model_dir);
    static constexpr std::array<importer_registration, 2> importers = { {
        { "qwen3", llama_safetensors_qwen3_importer::probe, create_qwen3_importer },
        { "qwen3_5", llama_safetensors_qwen35_importer::probe, create_qwen35_importer },
    } };

    const importer_registration * match = nullptr;
    for (const importer_registration & candidate : importers) {
        if (!candidate.probe(config)) {
            continue;
        }
        if (match != nullptr) {
            throw std::runtime_error(
                "ambiguous native safetensors architecture: both '" + std::string(match->name) +
                "' and '" + candidate.name + "' matched");
        }
        match = &candidate;
    }
    if (match == nullptr) {
        const std::string model_type = config.value("model_type", std::string("<missing>"));
        const std::string architectures = config.contains("architectures") ?
            config.at("architectures").dump() : "<missing>";
        const std::string quant_method = config.contains("quantization_config") &&
                config.at("quantization_config").is_object() ?
            config.at("quantization_config").value("quant_method", std::string("<missing>")) : "<missing>";
        throw std::runtime_error(
            "unsupported native safetensors architecture in '" + model_dir.string() +
            "' (model_type=" + model_type + ", architectures=" + architectures +
            ", quant_method=" + quant_method + ")");
    }
    return match->create(model_dir, config, io_mode);
}

llama_safetensors_io_mode source_io_mode(llama_load_mode load_mode) {
    switch (load_mode) {
        case LLAMA_LOAD_MODE_MMAP:
        case LLAMA_LOAD_MODE_MMAP_MLOCK:
            return llama_safetensors_io_mode::MMAP;
        case LLAMA_LOAD_MODE_DIRECT_IO:
            return llama_safetensors_io_mode::DIRECT;
        case LLAMA_LOAD_MODE_NONE:
        case LLAMA_LOAD_MODE_MLOCK:
            return llama_safetensors_io_mode::BUFFERED;
    }
    throw std::runtime_error("unknown model load mode");
}

}  // namespace

llama_safetensors_json llama_safetensors_read_model_config(const std::filesystem::path & model_dir) {
    llama_safetensors_json config = llama_safetensors_read_json(model_dir / "config.json");
    const auto inc_quant_path = model_dir / "quantization_config.json";
    if (!config.contains("quantization_config") && std::filesystem::is_regular_file(inc_quant_path)) {
        llama_safetensors_json quant = llama_safetensors_read_json(inc_quant_path);
        if (!quant.is_object() || quant.empty()) {
            throw std::runtime_error("quantization_config.json must contain a non-empty object");
        }
        config["quantization_config"] = std::move(quant);
    }
    const auto hf_quant_path = model_dir / "hf_quant_config.json";
    if (!config.contains("quantization_config") && std::filesystem::is_regular_file(hf_quant_path)) {
        const llama_safetensors_json hf_quant = llama_safetensors_read_json(hf_quant_path);
        if (hf_quant.contains("quantization") && hf_quant.at("quantization").is_object()) {
            llama_safetensors_json quant = hf_quant.at("quantization");
            quant["quant_method"] = "modelopt";
            if (hf_quant.contains("producer")) {
                quant["producer"] = hf_quant.at("producer");
            }
            config["quantization_config"] = std::move(quant);
        }
    }
    const auto quanto_map_path = model_dir / "quanto_qmap.json";
    if (std::filesystem::is_regular_file(quanto_map_path)) {
        if (config.contains("quantization_config")) {
            throw std::runtime_error(
                "native safetensors model contains both quantization_config and quanto_qmap.json");
        }
        llama_safetensors_json quanto_map = llama_safetensors_read_json(quanto_map_path);
        if (!quanto_map.is_object() || quanto_map.empty()) {
            throw std::runtime_error("quanto_qmap.json must contain a non-empty object");
        }
        config["quantization_config"] = {
            { "quant_method", "quanto" },
            { "quantization_map", std::move(quanto_map) },
        };
    }
    return config;
}

llama_model * llama_model_load_from_safetensors_dir(
        const std::filesystem::path & model_dir, llama_model_params params) {
    std::unique_ptr<llama_safetensors_importer> importer = select_importer(model_dir, source_io_mode(params.load_mode));
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(importer->build_metadata(), gguf_free);

    class safetensors_source final : public llama_model_tensor_source {
      public:
        safetensors_source(
                std::unique_ptr<llama_safetensors_importer> importer,
                bool check_tensors) :
            importer_(std::move(importer)), check_tensors_(check_tensors) {}

        bool describe(
                const std::string & canonical_name,
                ggml_type & type,
                std::array<int64_t, GGML_MAX_DIMS> & ne) const override {
            return importer_->describe(canonical_name, type, ne);
        }

        size_t tensor_capacity_hint() const override {
            return importer_->tensor_capacity_hint();
        }

        void bind(const std::string & canonical_name) const override {
            ++targets_[canonical_name].bound;
            importer_->bind(canonical_name);
        }

        void load(ggml_tensor * tensor) const override {
            const std::string canonical_name = tensor->name;
            auto target = targets_.find(canonical_name);
            if (target == targets_.end()) {
                throw std::runtime_error("loading unbound safetensors target '" + canonical_name + "'");
            }
            if (target->second.loaded >= target->second.bound) {
                throw std::runtime_error("excess safetensors target load for '" + canonical_name + "'");
            }
            ++target->second.loaded;
            if (importer_->load(canonical_name, tensor, check_tensors_)) {
                return;
            }
            std::vector<uint8_t> data = importer_->materialize(
                tensor->name, tensor->type, ggml_nbytes(tensor));
            if (check_tensors_ && !ggml_validate_row_data(tensor->type, data.data(), data.size())) {
                throw std::runtime_error(std::string("tensor '") + tensor->name + "' has invalid data");
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
        }

        void validate_complete() const override {
            for (const auto & [name, target] : targets_) {
                if (target.loaded != target.bound) {
                    throw std::runtime_error("described safetensors target was not loaded: '" + name + "'");
                }
            }
            importer_->validate_complete();
        }

      private:
        struct target_state {
            size_t bound  = 0;
            size_t loaded = 0;
        };

        std::unique_ptr<llama_safetensors_importer> importer_;
        bool check_tensors_;
        mutable std::unordered_map<std::string, target_state> targets_;
    };
    safetensors_source source(std::move(importer), params.check_tensors);

    return llama_model_init_from_source(metadata.get(), &source, params);
}
