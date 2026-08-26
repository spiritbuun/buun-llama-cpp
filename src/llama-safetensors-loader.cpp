#include "llama-safetensors.h"

#include "llama-safetensors-qwen35.h"
#include "llama.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

llama_model * llama_model_load_from_safetensors_dir(
        const std::filesystem::path & model_dir, llama_model_params params) {
    // P0 has one supported importer. Add architecture/format probes here as
    // support expands; callers should not need architecture-specific changes.
    llama_safetensors_qwen35_importer importer(model_dir);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(importer.build_metadata(), gguf_free);

    struct callback_state {
        llama_safetensors_qwen35_importer * importer;
        bool                                check_tensors;
    } state { &importer, params.check_tensors };

    const auto set_tensor_data = [](ggml_tensor * tensor, void * userdata) {
        auto * state = static_cast<callback_state *>(userdata);
        std::vector<uint8_t> data = state->importer->materialize(
            tensor->name, tensor->type, ggml_nbytes(tensor));
        if (state->check_tensors && !ggml_validate_row_data(tensor->type, data.data(), data.size())) {
            throw std::runtime_error(std::string("tensor '") + tensor->name + "' has invalid data");
        }
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size());
    };

    return llama_model_init_from_user(metadata.get(), set_tensor_data, &state, params);
}
