// Private diagnostic only: reuse an existing MMA specialization at small widths.
// No tensor tracing or synchronization; controls are not production options.
#include "ggml.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
struct ggml_backend_cuda_context;
void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    using fn = void (*)(ggml_backend_cuda_context &, ggml_tensor *);
    static auto original = reinterpret_cast<fn>(dlsym(RTLD_NEXT,
        "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor"));
    static const int cols = std::getenv("MTP_FIXED_FA_COLS") ? std::atoi(std::getenv("MTP_FIXED_FA_COLS")) : 0;
    static const int heads = std::getenv("MTP_FIXED_FA_HEADS") ? std::atoi(std::getenv("MTP_FIXED_FA_HEADS")) : 8;
    static auto fixed = []() -> fn {
        if (!cols) { return nullptr; }
        char symbol[200];
        std::snprintf(symbol, sizeof(symbol),
            "_Z37ggml_cuda_flash_attn_ext_mma_f16_caseILi256ELi256ELi%dELi%dELb0EEvR25ggml_backend_cuda_contextP11ggml_tensor", cols, heads);
        auto result = reinterpret_cast<fn>(dlsym(RTLD_NEXT, symbol));
        if (!result) { std::fprintf(stderr, "missing %s\n", symbol); std::abort(); }
        std::fprintf(stderr, "PRIVATE fixed F16 FA tile=%dx%d\n", cols, heads);
        return result;
    }();
    if (!original) { std::abort(); }
    auto * q = dst->src[0];
    if (fixed && q->ne[0] == 256 && q->ne[1] <= 4 && q->ne[3] == 1 &&
            dst->src[1]->type == GGML_TYPE_F16 && dst->src[2]->type == GGML_TYPE_F16 &&
            dst->src[3] && q->ne[2] == 24 && dst->src[1]->ne[2] == 4) {
        fixed(ctx, dst);
    } else {
        original(ctx, dst);
    }
}
