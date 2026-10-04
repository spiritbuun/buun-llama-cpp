// turbo fused MMA instance for matched GGML_TYPE_TURBO4_0 K / GGML_TYPE_TURBO4_0 V, D=256
// (both sides WHT-rotated, Q pre-rotated at the dispatch site; the (8,8) eight-row tile for
// MTP/DFlash verify widths 5..8 — without it those rounds fall to the O(n_kv) f16
// materialize route, the long-context decode regression vs llamAmpere)

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(256, 256, 8, 8, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0);
