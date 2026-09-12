#pragma once

#include "common.cuh"

// EXL3 (exllamav3) trellis-coded weights.  See docs/development/exl3-format-plan.md.
// Tensor: [k, n] with type GGML_TYPE_EXL3_{K}; data = 16x16 tiles in n-tile-major order
// [n/16][k/16][32*K bytes].  MUL_MAT sources: src[2] = svh (F16 [n]), src[3] = suh (F16 [k]).

static inline bool ggml_cuda_is_exl3(ggml_type type) {
    return ggml_type_is_exl3(type);
}

static inline int ggml_cuda_exl3_bits(ggml_type type) {
    return ggml_exl3_bits(type);
}

// 2 = mul1, 1 = mcg, 0 = 3inst
static inline int ggml_cuda_exl3_codebook(ggml_type type) {
    return ggml_exl3_codebook(type);
}

bool ggml_cuda_exl3_supports_mul_mat(const ggml_tensor * dst);

void ggml_cuda_mul_mat_exl3(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// MUL_MAT_ID with EXL3 experts: grouped kernel (no host sync) for decode shapes; src[3] = svh
// [n, n_expert], src[4] = suh [k, n_expert].
bool ggml_cuda_exl3_mul_mat_id_fast(const ggml_tensor * dst);
void ggml_cuda_mul_mat_id_exl3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// CPU MoE-cache bridge: dot products of packed weights and pre-transformed
// F16-rounded F32 activations. The caller owns both Hadamard transforms.
void ggml_cuda_exl3_cache_mmv(const void * weights, ggml_type type,
    const float * activations, const int32_t * slots, const int32_t * activation_ids,
    float * output, int k, int n, size_t expert_bytes, int rows, int activation_rows,
    cudaStream_t stream);

// Dequantize the whole tensor to F16 rows W[n][k] (row n contiguous over k); rows [n0, n1).
void ggml_cuda_exl3_reconstruct_rows(const ggml_tensor * src0, int64_t n0, int64_t n1, half * dst, cudaStream_t stream);
