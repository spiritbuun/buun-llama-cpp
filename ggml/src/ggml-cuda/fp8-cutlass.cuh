#pragma once
#include "common.cuh"

// Workspace query (nullptr), then execution using caller-owned storage/stream.
extern "C" int buun_fp8_cutlass(const void *, const void *, const float *, const float *, float *,
    int, int, int, int, int, int, void *, size_t, size_t *, cudaStream_t);
extern "C" int buun_fp8_cutlass_ffn(const void *, const void *, const void *, const void *,
    const float *, const float *, const float *, const float *, float *, const int32_t *, const int32_t *,
    int, int, int, int, int, int, void *, size_t, size_t *, cudaStream_t);
