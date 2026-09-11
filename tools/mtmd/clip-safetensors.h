#pragma once

#include <cstddef>
#include <memory>
#include <string>

struct ggml_context;
struct ggml_tensor;
struct gguf_context;

// Private mtmd adapter for vision towers embedded in a native safetensors
// model directory. The text model and vision model keep their existing runtime
// implementations; this object only supplies the metadata, canonical tensor
// descriptors, and bounded uploads that a GGUF mmproj normally provides.
class clip_safetensors_source {
  public:
    static bool supports(const std::string & path);

    explicit clip_safetensors_source(const std::string & path);
    ~clip_safetensors_source();

    clip_safetensors_source(const clip_safetensors_source &) = delete;
    clip_safetensors_source & operator=(const clip_safetensors_source &) = delete;

    gguf_context * build_metadata() const;
    ggml_context * build_tensor_metadata() const;

    size_t tensor_count() const;
    size_t model_size() const;
    bool   load(const std::string & canonical_name, ggml_tensor * destination) const;

  private:
    struct impl;
    std::unique_ptr<impl> impl_;
};
