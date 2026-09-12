#pragma once

#include "ggml.h"
#include "llama-mmap.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <functional>
#include <stdexcept>

struct gguf_context;
struct llama_model;
struct llama_model_params;

// Canonical, untransformed bytes already present in a file. The model loader
// opens and owns the mapping; no importer-owned pointer escapes model loading.
struct llama_model_tensor_file_region {
    std::string path;
    size_t offset;
};

// Internal model-weight source seam. Model implementations request canonical
// llama.cpp tensors; a source describes and fills those tensors without
// exposing its container format to model graphs or backend dispatch.
class llama_model_tensor_source {
  public:
    virtual bool describe(
        const std::string & canonical_name,
        ggml_type & type,
        std::array<int64_t, GGML_MAX_DIMS> & ne) const = 0;

    // Upper bound used only to reserve ggml metadata space. It does not
    // enumerate or authorize runtime tensors; model code remains authoritative.
    virtual size_t tensor_capacity_hint() const = 0;

    // Records that model tensor creation committed this canonical target.
    // A successful describe() alone may only be an optional capability probe.
    virtual void bind(const std::string & canonical_name) const = 0;

    virtual std::optional<llama_model_tensor_file_region> file_region(const ggml_tensor *) const {
        return std::nullopt;
    }

    // Pure placement/fit probe; preparation is deferred until allocation.
    virtual bool can_prepare_file(const ggml_tensor *) const { return false; }
    virtual std::unique_ptr<llama_file> prepare_file(
            const ggml_tensor *, const std::function<void()> &) const {
        throw std::runtime_error("source cannot prepare file-backed weights");
    }

    // mapped=true means the common loader bound raw or prepared file bytes.
    // The source still accounts for the binding and validates them if requested.
    virtual void load(ggml_tensor * destination, bool mapped = false) const = 0;

    // Called after all destination buffers have been populated. Sources use
    // this to reject incomplete or duplicate canonical bindings.
    virtual void validate_complete() const = 0;

    virtual ~llama_model_tensor_source() = default;
};

// Synchronous internal entry point. The source must remain alive until this
// function returns; all source reads and uploads complete before then.
llama_model * llama_model_init_from_source(
    gguf_context * metadata,
    const llama_model_tensor_source * source,
    llama_model_params params);
