#pragma once

#include "common.cuh"

// EXL3 (exllamav3) trellis-coded weights.  See docs/development/exl3-format-plan.md.
// Tensor: [k, n] with type GGML_TYPE_EXL3_{K}; data = 16x16 tiles in n-tile-major order
// [n/16][k/16][32*K bytes].  MUL_MAT sources: src[2] = svh (F16 [n]), src[3] = suh (F16 [k]).

static inline bool ggml_cuda_is_exl3(ggml_type type) {
    return type >= GGML_TYPE_EXL3_1 && type <= GGML_TYPE_EXL3_8;
}

static inline int ggml_cuda_exl3_bits(ggml_type type) {
    return int(type) - int(GGML_TYPE_EXL3_1) + 1;
}

bool ggml_cuda_exl3_supports_mul_mat(const ggml_tensor * dst);

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Dequantize the whole tensor to F16 rows W[n][k] (row n contiguous over k); rows [n0, n1).
void ggml_cuda_exl3_reconstruct_rows(const ggml_tensor * src0, int64_t n0, int64_t n1, half * dst, cudaStream_t stream);
