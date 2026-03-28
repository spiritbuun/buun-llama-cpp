/*
 * TBQ (TurboQuant-B): KV cache compression via SRHT + Lloyd-Max codebook
 * Based on: Zandieh et al., "TurboQuant", ICLR 2026
 *
 * Implements GGML_TYPE_TBQ3_0 (3-bit) and GGML_TYPE_TBQ4_0 (4-bit)
 * for use as --cache-type-k tbq3 --cache-type-v tbq3 in llama-server.
 *
 * Key difference from TURBO types: TBQ uses SRHT (Subsampled Randomized
 * Hadamard Transform) + Lloyd-Max codebook quantization with 128-element
 * blocks and packed codebook indices (no separate signs array).
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>

// ============================================================================
// TBQ CPU reference implementation
// SRHT (random rotation) + Lloyd-Max codebook quantization
// ============================================================================

// Lloyd-Max codebook values for standard Gaussian N(0,1)
static const float tbq3_centroids_cpu[8] = {
    -2.1519478649f, -1.3439114671f, -0.7560068854f, -0.2450947664f,
     0.2450947664f,  0.7560068854f,  1.3439114671f,  2.1519478649f
};
static const float tbq3_boundaries_cpu[7] = {
    -1.7479296660f, -1.0499591762f, -0.5005508259f, 0.0000000000f,
     0.5005508259f,  1.0499591762f,  1.7479296660f
};
static const float tbq4_centroids_cpu[16] = {
    -2.7643471169f, -2.1048021157f, -1.6544546703f, -1.2904430627f,
    -0.9718584055f, -0.6794737713f, -0.4023510241f, -0.1332771696f,
     0.1332771696f,  0.4023510241f,  0.6794737713f,  0.9718584055f,
     1.2904430627f,  1.6544546703f,  2.1048021157f,  2.7643471169f
};
static const float tbq4_boundaries_cpu[15] = {
    -2.4345746163f, -1.8796283930f, -1.4724488665f, -1.1311507341f,
    -0.8256660884f, -0.5409123977f, -0.2678140968f,  0.0000000000f,
     0.2678140968f,  0.5409123977f,  0.8256660884f,  1.1311507341f,
     1.4724488665f,  1.8796283930f,  2.4345746163f
};

// Fixed Rademacher sign vector (128 bits), same as CUDA version
static const uint32_t tbq_rademacher_cpu[4] = {
    0xa3b1c6d9u, 0x7e4f2a85u, 0xd1936cf0u, 0x5b8e47a2u
};

static inline float tbq_get_sign(int idx) {
    return (tbq_rademacher_cpu[idx / 32] >> (idx % 32)) & 1 ? -1.0f : 1.0f;
}

// In-place Hadamard transform (size 128)
static void tbq_hadamard_128(float * data) {
    for (int step = 1; step < 128; step <<= 1) {
        for (int i = 0; i < 128; i++) {
            int partner = i ^ step;
            if (i < partner) {
                float a = data[i], b = data[partner];
                data[i]       = a + b;
                data[partner] = a - b;
            }
        }
    }
    // 1/sqrt(128) = sqrt(2)/16 ~= 0.0883883f
    for (int i = 0; i < 128; i++) {
        data[i] *= 0.0883883f;
    }
}

static inline int tbq_quantize_3bit(float val) {
    if (val > tbq3_boundaries_cpu[3]) {
        if (val > tbq3_boundaries_cpu[5]) return val > tbq3_boundaries_cpu[6] ? 7 : 6;
        return val > tbq3_boundaries_cpu[4] ? 5 : 4;
    }
    if (val > tbq3_boundaries_cpu[1]) return val > tbq3_boundaries_cpu[2] ? 3 : 2;
    return val > tbq3_boundaries_cpu[0] ? 1 : 0;
}

static inline int tbq_quantize_4bit(float val) {
    if (val > tbq4_boundaries_cpu[7]) {
        if (val > tbq4_boundaries_cpu[11]) {
            if (val > tbq4_boundaries_cpu[13]) return val > tbq4_boundaries_cpu[14] ? 15 : 14;
            return val > tbq4_boundaries_cpu[12] ? 13 : 12;
        }
        if (val > tbq4_boundaries_cpu[9]) return val > tbq4_boundaries_cpu[10] ? 11 : 10;
        return val > tbq4_boundaries_cpu[8] ? 9 : 8;
    }
    if (val > tbq4_boundaries_cpu[3]) {
        if (val > tbq4_boundaries_cpu[5]) return val > tbq4_boundaries_cpu[6] ? 7 : 6;
        return val > tbq4_boundaries_cpu[4] ? 5 : 4;
    }
    if (val > tbq4_boundaries_cpu[1]) return val > tbq4_boundaries_cpu[2] ? 3 : 2;
    return val > tbq4_boundaries_cpu[0] ? 1 : 0;
}

void quantize_row_tbq3_0_ref(const float * restrict x, block_tbq3_0 * restrict y, int64_t k) {
    assert(k % QK_TBQ3 == 0);
    const int64_t nb = k / QK_TBQ3;
    float tmp[128];
    uint8_t indices[128];

    for (int64_t i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < 128; j++) norm += x[i * 128 + j] * x[i * 128 + j];
        norm = sqrtf(norm);
        if (norm < 1e-12f) norm = 1e-12f;

        for (int j = 0; j < 128; j++) tmp[j] = x[i * 128 + j] / norm;
        for (int j = 0; j < 128; j++) tmp[j] *= tbq_get_sign(j);
        tbq_hadamard_128(tmp);
        for (int j = 0; j < 128; j++) indices[j] = (uint8_t)tbq_quantize_3bit(tmp[j]);

        memset(y[i].qs, 0, 48);
        for (int j = 0; j < 128; j++) {
            int bit_offset = j * 3;
            int byte_idx = bit_offset / 8;
            int bit_pos  = bit_offset % 8;
            y[i].qs[byte_idx] |= (indices[j] << bit_pos) & 0xFF;
            if (bit_pos > 5) y[i].qs[byte_idx + 1] |= indices[j] >> (8 - bit_pos);
        }
        y[i].norm = GGML_FP32_TO_FP16(norm);
    }
}

void quantize_row_tbq4_0_ref(const float * restrict x, block_tbq4_0 * restrict y, int64_t k) {
    assert(k % QK_TBQ4 == 0);
    const int64_t nb = k / QK_TBQ4;
    float tmp[128];

    for (int64_t i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < 128; j++) norm += x[i * 128 + j] * x[i * 128 + j];
        norm = sqrtf(norm);
        if (norm < 1e-12f) norm = 1e-12f;

        for (int j = 0; j < 128; j++) tmp[j] = x[i * 128 + j] / norm;
        for (int j = 0; j < 128; j++) tmp[j] *= tbq_get_sign(j);
        tbq_hadamard_128(tmp);

        for (int j = 0; j < 64; j++) {
            int lo = tbq_quantize_4bit(tmp[j * 2 + 0]);
            int hi = tbq_quantize_4bit(tmp[j * 2 + 1]);
            y[i].qs[j] = (uint8_t)((hi << 4) | (lo & 0xf));
        }
        y[i].norm = GGML_FP32_TO_FP16(norm);
    }
}

void quantize_row_tbq3_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_tbq3_0_ref(x, (block_tbq3_0 *)y, k);
}

void quantize_row_tbq4_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_tbq4_0_ref(x, (block_tbq4_0 *)y, k);
}

void dequantize_row_tbq3_0(const block_tbq3_0 * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_TBQ3 == 0);
    const int64_t nb = k / QK_TBQ3;
    float tmp[128];

    for (int64_t i = 0; i < nb; i++) {
        for (int j = 0; j < 128; j++) {
            int bit_offset = j * 3;
            int byte_idx = bit_offset / 8;
            int bit_pos  = bit_offset % 8;
            int idx = (x[i].qs[byte_idx] >> bit_pos);
            if (bit_pos > 5) idx |= ((int)x[i].qs[byte_idx + 1]) << (8 - bit_pos);
            tmp[j] = tbq3_centroids_cpu[idx & 0x7];
        }
        tbq_hadamard_128(tmp);
        for (int j = 0; j < 128; j++) tmp[j] *= tbq_get_sign(j);
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        for (int j = 0; j < 128; j++) y[i * 128 + j] = tmp[j] * norm;
    }
}

void dequantize_row_tbq4_0(const block_tbq4_0 * restrict x, float * restrict y, int64_t k) {
    assert(k % QK_TBQ4 == 0);
    const int64_t nb = k / QK_TBQ4;
    float tmp[128];

    for (int64_t i = 0; i < nb; i++) {
        for (int j = 0; j < 64; j++) {
            uint8_t packed = x[i].qs[j];
            tmp[j * 2 + 0] = tbq4_centroids_cpu[packed & 0xf];
            tmp[j * 2 + 1] = tbq4_centroids_cpu[(packed >> 4) & 0xf];
        }
        tbq_hadamard_128(tmp);
        for (int j = 0; j < 128; j++) tmp[j] *= tbq_get_sign(j);
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        for (int j = 0; j < 128; j++) y[i * 128 + j] = tmp[j] * norm;
    }
}

void ggml_vec_dot_tbq3_0_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc); GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by);

    const int nb = n / QK_TBQ3;
    float tmp[128];
    float sumf = 0.0f;
    const block_tbq3_0 * x = (const block_tbq3_0 *)vx;
    const block_q8_K    * y = (const block_q8_K *)vy;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < 128; j++) {
            int bit_offset = j * 3, byte_idx = bit_offset / 8, bit_pos = bit_offset % 8;
            int idx = (x[i].qs[byte_idx] >> bit_pos);
            if (bit_pos > 5) idx |= ((int)x[i].qs[byte_idx + 1]) << (8 - bit_pos);
            tmp[j] = tbq3_centroids_cpu[idx & 0x7];
        }
        tbq_hadamard_128(tmp);
        for (int j = 0; j < 128; j++) tmp[j] *= tbq_get_sign(j);
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float sum = 0.0f;
        for (int j = 0; j < 128; j++) sum += (tmp[j] * norm) * (y[i].qs[j] * y[i].d);
        sumf += sum;
    }
    *s = sumf;
}

void ggml_vec_dot_tbq4_0_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc); GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by);

    const int nb = n / QK_TBQ4;
    float tmp[128];
    float sumf = 0.0f;
    const block_tbq4_0 * x = (const block_tbq4_0 *)vx;
    const block_q8_K    * y = (const block_q8_K *)vy;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < 64; j++) {
            uint8_t packed = x[i].qs[j];
            tmp[j * 2 + 0] = tbq4_centroids_cpu[packed & 0xf];
            tmp[j * 2 + 1] = tbq4_centroids_cpu[(packed >> 4) & 0xf];
        }
        tbq_hadamard_128(tmp);
        for (int j = 0; j < 128; j++) tmp[j] *= tbq_get_sign(j);
        float norm = GGML_FP16_TO_FP32(x[i].norm);
        float sum = 0.0f;
        for (int j = 0; j < 128; j++) sum += (tmp[j] * norm) * (y[i].qs[j] * y[i].d);
        sumf += sum;
    }
    *s = sumf;
}
