//
// TurboQuant CUDA kernels for KV cache quantization
// Implements SRHT (random rotation) + Lloyd-Max codebook quantization
// Reference: Zandieh et al., "TurboQuant", ICLR 2026
//

#include "tbq-quant.cuh"

// ============================================================================
// Lloyd-Max codebook values for standard Gaussian N(0,1)
// Computed via iterative centroid optimization
// ============================================================================

// 3-bit (8 levels)
static __constant__ float tbq3_centroids[8] = {
    -2.1519478649f, -1.3439114671f, -0.7560068854f, -0.2450947664f,
     0.2450947664f,  0.7560068854f,  1.3439114671f,  2.1519478649f
};
static __constant__ float tbq3_boundaries[7] = {
    -1.7479296660f, -1.0499591762f, -0.5005508259f, 0.0000000000f,
     0.5005508259f,  1.0499591762f,  1.7479296660f
};

// 4-bit (16 levels)
static __constant__ float tbq4_centroids[16] = {
    -2.7643471169f, -2.1048021157f, -1.6544546703f, -1.2904430627f,
    -0.9718584055f, -0.6794737713f, -0.4023510241f, -0.1332771696f,
     0.1332771696f,  0.4023510241f,  0.6794737713f,  0.9718584055f,
     1.2904430627f,  1.6544546703f,  2.1048021157f,  2.7643471169f
};
static __constant__ float tbq4_boundaries[15] = {
    -2.4345746163f, -1.8796283930f, -1.4724488665f, -1.1311507341f,
    -0.8256660884f, -0.5409123977f, -0.2678140968f,  0.0000000000f,
     0.2678140968f,  0.5409123977f,  0.8256660884f,  1.1311507341f,
     1.4724488665f,  1.8796283930f,  2.4345746163f
};

// Fixed Rademacher sign vector (128 bits = 4 uint32s)
// Generated from seed=42 for reproducibility
static __constant__ uint32_t tbq_rademacher[4] = {
    0xa3b1c6d9u, 0x7e4f2a85u, 0xd1936cf0u, 0x5b8e47a2u
};

// ============================================================================
// Device helper: Hadamard transform in shared memory (128-point)
// Self-inverse (up to scaling), uses butterfly pattern
// ============================================================================
static __device__ void hadamard_128_inplace(float * smem, int tid) {
    // First stage: pairs
    {
        int partner = tid ^ 1;
        float a = smem[tid], b = smem[partner];
        __syncthreads();
        if (tid < partner) {
            smem[tid]     = a + b;
            smem[partner] = a - b;
        }
        __syncthreads();
    }

    // Remaining stages
    #pragma unroll
    for (int step = 2; step < 128; step <<= 1) {
        int partner = tid ^ step;
        float a = smem[tid], b = smem[partner];
        __syncthreads();
        if (tid < partner) {
            smem[tid]     = a + b;
            smem[partner] = a - b;
        }
        __syncthreads();
    }

    // Normalize: 1/sqrt(128) = sqrt(2)/16 ~= 0.0883883f
    smem[tid] *= 0.0883883f;
    __syncthreads();
}

// ============================================================================
// Device helper: Apply random sign flips (Rademacher diagonal)
// ============================================================================
static __device__ void apply_sign_flips(float * smem, int tid) {
    int word = tid / 32;
    int bit  = tid % 32;
    float sign = (tbq_rademacher[word] >> bit) & 1 ? -1.0f : 1.0f;
    smem[tid] *= sign;
    __syncthreads();
}

// ============================================================================
// Device helper: Lloyd-Max 3-bit quantization (binary search)
// Returns index 0-7
// ============================================================================
static __device__ __forceinline__ int quantize_lloyd_max_3bit(float val) {
    int idx = 0;
    if (val > tbq3_boundaries[3]) {  // > 0
        if (val > tbq3_boundaries[5]) {
            idx = val > tbq3_boundaries[6] ? 7 : 6;
        } else {
            idx = val > tbq3_boundaries[4] ? 5 : 4;
        }
    } else {
        if (val > tbq3_boundaries[1]) {
            idx = val > tbq3_boundaries[2] ? 3 : 2;
        } else {
            idx = val > tbq3_boundaries[0] ? 1 : 0;
        }
    }
    return idx;
}

// ============================================================================
// Device helper: Lloyd-Max 4-bit quantization (binary search)
// Returns index 0-15
// ============================================================================
static __device__ __forceinline__ int quantize_lloyd_max_4bit(float val) {
    int idx = 0;
    if (val > tbq4_boundaries[7]) {  // > 0
        if (val > tbq4_boundaries[11]) {
            if (val > tbq4_boundaries[13]) {
                idx = val > tbq4_boundaries[14] ? 15 : 14;
            } else {
                idx = val > tbq4_boundaries[12] ? 13 : 12;
            }
        } else {
            if (val > tbq4_boundaries[9]) {
                idx = val > tbq4_boundaries[10] ? 11 : 10;
            } else {
                idx = val > tbq4_boundaries[8] ? 9 : 8;
            }
        }
    } else {
        if (val > tbq4_boundaries[3]) {
            if (val > tbq4_boundaries[5]) {
                idx = val > tbq4_boundaries[6] ? 7 : 6;
            } else {
                idx = val > tbq4_boundaries[4] ? 5 : 4;
            }
        } else {
            if (val > tbq4_boundaries[1]) {
                idx = val > tbq4_boundaries[2] ? 3 : 2;
            } else {
                idx = val > tbq4_boundaries[0] ? 1 : 0;
            }
        }
    }
    return idx;
}

// ============================================================================
// Device helper: 3-bit index extraction from packed bytes
// Each thread extracts its own 3-bit value from the packed qs[] array
// ============================================================================
static __device__ __forceinline__ int unpack_3bit(const uint8_t * qs, int tid) {
    int bit_offset = tid * 3;
    int byte_idx   = bit_offset / 8;
    int bit_pos    = bit_offset % 8;

    // Read one or two bytes depending on whether value spans a byte boundary
    int val = (qs[byte_idx] >> bit_pos);
    if (bit_pos > 5) {
        // Spans into next byte
        val |= ((int)qs[byte_idx + 1]) << (8 - bit_pos);
    }
    return val & 0x7;
}

// ============================================================================
// Quantization kernel: TBQ3_0
// Grid: ceil(k/128), Block: 128 threads
// ============================================================================
static __global__ void quantize_tbq3_0_kernel(const float * __restrict__ x,
                                                block_tbq3_0 * __restrict__ y,
                                                int64_t k) {
    const int64_t block_idx = blockIdx.x;
    const int     tid       = threadIdx.x;
    const int64_t offset    = block_idx * 128;

    if (offset + tid >= k) return;

    __shared__ float smem[128];
    __shared__ uint8_t indices[128];

    // 1. Load data into shared memory
    smem[tid] = x[offset + tid];
    __syncthreads();

    // 2. Compute L2 norm via parallel reduction
    __shared__ float norm_shared;
    {
        float val = smem[tid] * smem[tid];
        // Warp-level reduction
        for (int s = 16; s > 0; s >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, s);
        }

        // First thread of each warp writes to shared
        __shared__ float warp_sums[4];
        if (tid % 32 == 0) {
            warp_sums[tid / 32] = val;
        }
        __syncthreads();

        // Final reduction by thread 0
        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < 4; i++) {
                total += warp_sums[i];
            }
            norm_shared = sqrtf(total);
        }
        __syncthreads();
    }

    float norm = norm_shared;
    if (norm < 1e-12f) norm = 1e-12f;

    // 3. Normalize
    smem[tid] /= norm;
    __syncthreads();

    // 4. Apply random sign flips
    apply_sign_flips(smem, tid);

    // 5. Hadamard transform
    hadamard_128_inplace(smem, tid);

    // 6. Lloyd-Max quantize
    indices[tid] = (uint8_t)quantize_lloyd_max_3bit(smem[tid]);
    __syncthreads();

    // 7. Pack 3-bit indices into output bytes
    // 128 * 3 = 384 bits = 48 bytes
    // Each thread handles packing if tid < 48
    if (tid < 48) {
        int byte_idx = tid;
        int bit_start = byte_idx * 8;
        uint8_t packed = 0;

        // Each byte contains parts of 2-3 indices
        for (int b = 0; b < 8; b++) {
            int bit_pos = bit_start + b;
            int idx_num = bit_pos / 3;   // which index (0-127)
            int idx_bit = bit_pos % 3;   // which bit of that index (0-2)
            if (idx_num < 128) {
                packed |= (((indices[idx_num] >> idx_bit) & 1) << b);
            }
        }
        y[block_idx].qs[byte_idx] = packed;
    }

    // 8. Write norm
    if (tid == 0) {
        y[block_idx].norm = __float2half(norm);
    }
}

// ============================================================================
// Quantization kernel: TBQ4_0
// Grid: ceil(k/128), Block: 128 threads
// ============================================================================
static __global__ void quantize_tbq4_0_kernel(const float * __restrict__ x,
                                                block_tbq4_0 * __restrict__ y,
                                                int64_t k) {
    const int64_t block_idx = blockIdx.x;
    const int     tid       = threadIdx.x;
    const int64_t offset    = block_idx * 128;

    if (offset + tid >= k) return;

    __shared__ float smem[128];

    // 1. Load data
    smem[tid] = x[offset + tid];
    __syncthreads();

    // 2. Compute L2 norm
    __shared__ float norm_shared;
    {
        float val = smem[tid] * smem[tid];
        for (int s = 16; s > 0; s >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, s);
        }
        __shared__ float warp_sums[4];
        if (tid % 32 == 0) {
            warp_sums[tid / 32] = val;
        }
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (int i = 0; i < 4; i++) total += warp_sums[i];
            norm_shared = sqrtf(total);
        }
        __syncthreads();
    }

    float norm = norm_shared;
    if (norm < 1e-12f) norm = 1e-12f;

    // 3. Normalize
    smem[tid] /= norm;
    __syncthreads();

    // 4. Apply random sign flips
    apply_sign_flips(smem, tid);

    // 5. Hadamard transform
    hadamard_128_inplace(smem, tid);

    // 6. Lloyd-Max quantize (4-bit)
    // 7. Pack 4-bit indices as nibble pairs
    // 64 bytes for 128 nibbles: tid < 64 handles pairs
    if (tid < 64) {
        int lo = quantize_lloyd_max_4bit(smem[tid * 2 + 0]);
        int hi = quantize_lloyd_max_4bit(smem[tid * 2 + 1]);
        y[block_idx].qs[tid] = (uint8_t)((hi << 4) | (lo & 0xf));
    }

    // 8. Write norm
    if (tid == 0) {
        y[block_idx].norm = __float2half(norm);
    }
}

// ============================================================================
// Dequantization kernel: TBQ3_0
// Grid: ceil(k/128), Block: 128 threads
// ============================================================================
static __global__ void dequantize_tbq3_0_kernel(const block_tbq3_0 * __restrict__ x,
                                                  float * __restrict__ y,
                                                  int64_t k) {
    const int64_t block_idx = blockIdx.x;
    const int     tid       = threadIdx.x;
    const int64_t offset    = block_idx * 128;

    if (offset + tid >= k) return;

    __shared__ float smem[128];

    // 1. Unpack 3-bit index and codebook lookup
    int idx = unpack_3bit(x[block_idx].qs, tid);
    smem[tid] = tbq3_centroids[idx];
    __syncthreads();

    // 2. Inverse Hadamard transform (self-inverse up to scaling)
    hadamard_128_inplace(smem, tid);

    // 3. Inverse sign flips
    apply_sign_flips(smem, tid);

    // 4. Rescale by norm
    float norm = __half2float(x[block_idx].norm);
    y[offset + tid] = smem[tid] * norm;
}

// ============================================================================
// Dequantization kernel: TBQ4_0
// Grid: ceil(k/128), Block: 128 threads
// ============================================================================
static __global__ void dequantize_tbq4_0_kernel(const block_tbq4_0 * __restrict__ x,
                                                  float * __restrict__ y,
                                                  int64_t k) {
    const int64_t block_idx = blockIdx.x;
    const int     tid       = threadIdx.x;
    const int64_t offset    = block_idx * 128;

    if (offset + tid >= k) return;

    __shared__ float smem[128];

    // 1. Unpack 4-bit nibble and codebook lookup
    // tid < 128: byte_idx = tid / 2, covers 64 bytes (128 nibbles)
    int byte_idx = tid / 2;
    uint8_t packed = x[block_idx].qs[byte_idx];
    int idx = (tid & 1) ? ((packed >> 4) & 0xf) : (packed & 0xf);
    smem[tid] = tbq4_centroids[idx];
    __syncthreads();

    // 2. Inverse Hadamard transform
    hadamard_128_inplace(smem, tid);

    // 3. Inverse sign flips
    apply_sign_flips(smem, tid);

    // 4. Rescale by norm
    float norm = __half2float(x[block_idx].norm);
    y[offset + tid] = smem[tid] * norm;
}

// ============================================================================
// Host wrapper functions
// ============================================================================

void quantize_row_tbq3_0_cuda(const float * x, void * y, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TBQ3 == 0);
    const int64_t nblocks = k / QK_TBQ3;
    quantize_tbq3_0_kernel<<<nblocks, 128, 0, stream>>>(x, (block_tbq3_0 *)y, k);
}

void quantize_row_tbq4_0_cuda(const float * x, void * y, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TBQ4 == 0);
    const int64_t nblocks = k / QK_TBQ4;
    quantize_tbq4_0_kernel<<<nblocks, 128, 0, stream>>>(x, (block_tbq4_0 *)y, k);
}

void dequantize_row_tbq3_0_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TBQ3 == 0);
    const int64_t nblocks = k / QK_TBQ3;
    dequantize_tbq3_0_kernel<<<nblocks, 128, 0, stream>>>((const block_tbq3_0 *)x, y, k);
}

void dequantize_row_tbq4_0_cuda(const void * x, float * y, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % QK_TBQ4 == 0);
    const int64_t nblocks = k / QK_TBQ4;
    dequantize_tbq4_0_kernel<<<nblocks, 128, 0, stream>>>((const block_tbq4_0 *)x, y, k);
}

// Wrappers matching to_fp32_cuda_t: (const void*, float*, nrows, n_per_row, stream)
void dequantize_row_tbq3_0_cuda_fp32(const void * x, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_tbq3_0_cuda(x, y, nrows * n_per_row, stream);
}

void dequantize_row_tbq4_0_cuda_fp32(const void * x, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_tbq4_0_cuda(x, y, nrows * n_per_row, stream);
}
