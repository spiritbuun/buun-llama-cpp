#pragma once

#include "common.cuh"

// TurboQuant CUDA kernel declarations
// SRHT (Subsampled Randomized Hadamard Transform) + Lloyd-Max quantization

void quantize_row_tbq3_0_cuda(const float * x, void * y, int64_t k, cudaStream_t stream);
void quantize_row_tbq4_0_cuda(const float * x, void * y, int64_t k, cudaStream_t stream);

void dequantize_row_tbq3_0_cuda(const void * x, float * y, int64_t k, cudaStream_t stream);
void dequantize_row_tbq4_0_cuda(const void * x, float * y, int64_t k, cudaStream_t stream);

// Wrappers matching to_fp32_cuda_t signature: (const void*, float*, int64_t nrows, int64_t n_per_row, cudaStream_t)
void dequantize_row_tbq3_0_cuda_fp32(const void * x, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
void dequantize_row_tbq4_0_cuda_fp32(const void * x, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);
