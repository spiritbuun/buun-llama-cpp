#pragma once

#include <cstddef>

// Host-to-device copies split at host-registration edges.
cudaError_t moe_zc_memcpy_h2d_async(void * dst, const void * src, size_t bytes, cudaStream_t stream);
// Register a cache fill source in place for direct DMA.
void moe_zc_register_fill_source(const void * src, size_t bytes);
