#pragma once

#include "ggml-backend.h"

// Private prefix-export experiment, resolved dynamically; no CUDA dependency
// in libllama. Tensor ranges come from the existing state serializer.
struct ggml_prefix_export_range {
    ggml_tensor * tensor;
    size_t offset;
    size_t size;
    size_t destination;
};

struct ggml_prefix_export_iface {
    bool (*prepare)(ggml_backend_t, const ggml_prefix_export_range *, size_t);
    bool (*read)(ggml_backend_t, unsigned char *, size_t);
    void (*cancel)(ggml_backend_t);
};

using ggml_prefix_export_get_iface = const ggml_prefix_export_iface * (*)();
