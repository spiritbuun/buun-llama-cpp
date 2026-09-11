#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <string>

struct gguf_context;
struct llama_model;
struct llama_model_params;

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

    virtual void load(ggml_tensor * destination) const = 0;

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
