// Private experiment: relax one projection type while holding all others at MMVQ.
// No production option; only used with the fixed-attention diagnostic preload.
#include "ggml.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

bool ggml_cuda_should_use_mmvq(ggml_type type, int cc, int64_t n) {
    using fn = bool (*)(ggml_type, int, int64_t);
    static auto original = reinterpret_cast<fn>(dlsym(RTLD_NEXT,
        "_Z25ggml_cuda_should_use_mmvq9ggml_typeil"));
    static const int native_type = [] {
        const char * s = std::getenv("MTP_NATIVE_PROJECTION_TYPE");
        const int value = s ? std::atoi(s) : -1;
        std::fprintf(stderr, "PRIVATE native projection type=%d\n", value);
        return value;
    }();
    if (!original) { std::abort(); }
    if (cc == 860 && n <= 4 && ggml_is_quantized(type) && int(type) != native_type) {
        return true;
    }
    return original(type, cc, n);
}
