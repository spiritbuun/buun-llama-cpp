#include "llama-safetensors.h"

#include "llama-safetensors-importer.h"
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

using importer_probe = bool (*)(const std::filesystem::path &);
using importer_create = std::unique_ptr<llama_safetensors_importer> (*)(const std::filesystem::path &);

struct importer_registration {
    const char *    name;
    importer_probe  probe;
    importer_create create;
};

std::unique_ptr<llama_safetensors_importer> create_qwen35_importer(
        const std::filesystem::path & model_dir) {
    return std::make_unique<llama_safetensors_qwen35_importer>(model_dir);
}

std::unique_ptr<llama_safetensors_importer> select_importer(
        const std::filesystem::path & model_dir) {
    static constexpr std::array<importer_registration, 1> importers = { {
        { "qwen3_5", llama_safetensors_qwen35_importer::probe, create_qwen35_importer },
    } };

    const importer_registration * match = nullptr;
    for (const importer_registration & candidate : importers) {
        if (!candidate.probe(model_dir)) {
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
        throw std::runtime_error(
            "unsupported native safetensors architecture in '" + model_dir.string() + "'");
    }
    return match->create(model_dir);
}

}  // namespace

llama_model * llama_model_load_from_safetensors_dir(
        const std::filesystem::path & model_dir, llama_model_params params) {
    std::unique_ptr<llama_safetensors_importer> importer = select_importer(model_dir);
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
            if (!targets_.emplace(canonical_name, false).second) {
                throw std::runtime_error("duplicate safetensors target binding for '" + canonical_name + "'");
            }
            importer_->bind(canonical_name);
        }

        void load(ggml_tensor * tensor) const override {
            const std::string canonical_name = tensor->name;
            auto target = targets_.find(canonical_name);
            if (target == targets_.end()) {
                throw std::runtime_error("loading unbound safetensors target '" + canonical_name + "'");
            }
            if (target->second) {
                throw std::runtime_error("duplicate safetensors target load for '" + canonical_name + "'");
            }
            target->second = true;
            std::vector<uint8_t> data = importer_->materialize(
                tensor->name, tensor->type, ggml_nbytes(tensor));
            if (check_tensors_ && !ggml_validate_row_data(tensor->type, data.data(), data.size())) {
                throw std::runtime_error(std::string("tensor '") + tensor->name + "' has invalid data");
            }
            ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
        }

        void validate_complete() const override {
            for (const auto & [name, loaded] : targets_) {
                if (!loaded) {
                    throw std::runtime_error("described safetensors target was not loaded: '" + name + "'");
                }
            }
            importer_->validate_complete();
        }

      private:
        std::unique_ptr<llama_safetensors_importer> importer_;
        bool check_tensors_;
        mutable std::unordered_map<std::string, bool> targets_;
    };
    safetensors_source source(std::move(importer), params.check_tensors);

    return llama_model_init_from_source(metadata.get(), &source, params);
}
