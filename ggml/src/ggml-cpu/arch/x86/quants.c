#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "simd-mappings.h"

#include "../../quants.h"
#include "../../ggml-cpu-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#define UNUSED GGML_UNUSED

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVXVNNI__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}
#elif defined(__AVX__)
static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m128i axl = _mm256_castsi256_si128(ax);
    const __m128i axh = _mm256_extractf128_si256(ax, 1);
    const __m128i syl = _mm256_castsi256_si128(sy);
    const __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
static inline __m256 mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0, const __m128i x_2_1,
                                           const __m128i y_1_0, const __m128i y_1_1, const __m128i y_2_0, const __m128i y_2_1) {
    const __m128i mone = _mm_set1_epi16(1);

    const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
    const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
    const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
    const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
    const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
    const __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
    const __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
    return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
static inline __m256 quad_fp16_delta_float(const float x0, const float y0, const float x1, const float y1) {
    // GGML_CPU_FP16_TO_FP32 is faster than Intel F16C
    return _mm256_set_m128(_mm_set1_ps(GGML_CPU_FP16_TO_FP32(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}

static inline __m256 quad_mx_delta_float(const uint8_t x0, const float y0, const uint8_t x1, const float y1) {
    return _mm256_set_m128(_mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                           _mm_set1_ps(GGML_CPU_E8M0_TO_FP32_HALF(x0) * GGML_CPU_FP16_TO_FP32(y0)));
}
#endif
#elif defined(__SSSE3__)
static inline __m128i bytes_from_bits_16(const uint8_t * x) {
    uint16_t x16;
    memcpy(&x16, x, sizeof(uint16_t));

    const __m128i shuf_mask = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    __m128i bytes = _mm_shuffle_epi8(_mm_set1_epi16((short) x16), shuf_mask);
    const __m128i bit_mask = _mm_set_epi64x(0x7fbfdfeff7fbfdfe, 0x7fbfdfeff7fbfdfe);
    bytes = _mm_or_si128(bytes, bit_mask);

    return _mm_cmpeq_epi8(bytes, _mm_set1_epi64x(-1));
}

// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

void quantize_row_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float maxScalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_CPU_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

void quantize_row_q8_1(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * GGML_RESTRICT y = vy;
#if defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_CPU_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

        // Round to nearest integer
        v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
        v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
        v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
        v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

        // Convert floats to integers
        __m256i i0 = _mm256_cvtps_epi32( v0 );
        __m256i i1 = _mm256_cvtps_epi32( v1 );
        __m256i i2 = _mm256_cvtps_epi32( v2 );
        __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
        // Compute the sum of the quants and set y[i].s
        y[i].s = GGML_CPU_FP32_TO_FP16(d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
#else
        // Since we don't have in AVX some necessary functions,
        // we split the registers in half and call AVX2 analogs from SSE
        __m128i ni0 = _mm256_castsi256_si128( i0 );
        __m128i ni1 = _mm256_extractf128_si256( i0, 1);
        __m128i ni2 = _mm256_castsi256_si128( i1 );
        __m128i ni3 = _mm256_extractf128_si256( i1, 1);
        __m128i ni4 = _mm256_castsi256_si128( i2 );
        __m128i ni5 = _mm256_extractf128_si256( i2, 1);
        __m128i ni6 = _mm256_castsi256_si128( i3 );
        __m128i ni7 = _mm256_extractf128_si256( i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
        const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
        y[i].s = GGML_CPU_FP32_TO_FP16(d * hsum_i32_4(_mm_add_epi32(s0, s1)));

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

// placeholder implementation for Apple targets
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

//
// Helper functions
//

#if __AVX__ || __AVX2__ || __AVX512F__

// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#endif

void ggml_vec_dot_q1_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK1_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q1_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__AVX2__)
    const __m256i ones_8 = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    const __m256i byte_shuf = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit_masks = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128,
            1, 2, 4, 8, 16, 32, 64, (char) -128, 1, 2, 4, 8, 16, 32, 64, (char) -128);
    const __m256i zero = _mm256_setzero_si256();
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d);
        const uint32_t * GGML_RESTRICT qs32 = (const uint32_t *) x[ib].qs;
        const block_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];

        __m256 acc_block;
        {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) y_ptr[0].qs);
            const __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[0]), byte_shuf), bit_masks), zero);
            const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[0].d)), _mm256_cvtepi32_ps(s32));
        }
        for (int K = 1; K < 4; ++K) {
            const __m256i qy = _mm256_loadu_si256((const __m256i *) y_ptr[K].qs);
            const __m256i sm = _mm256_cmpeq_epi8(
                    _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int) qs32[K]), byte_shuf), bit_masks), zero);
            const __m256i sy = _mm256_sub_epi8(_mm256_xor_si256(qy, sm), sm);
            const __m256i s32 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_8, sy), ones_16);
            acc_block = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[K].d)), _mm256_cvtepi32_ps(s32), acc_block);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d0), acc_block, acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__AVX__)
    const __m128i ones_8 = _mm_set1_epi8(1);
    const __m128i ones_16 = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();
    __m256 acc = _mm256_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d);
        const block_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];
        __m256 acc_block;
        {
            const __m256i bit_mask = bytes_from_bits_32(&x[ib].qs[0]);
            const __m128i bit_mask_0 = _mm256_castsi256_si128(bit_mask);
            const __m128i bit_mask_1 = _mm256_extractf128_si256(bit_mask, 1);
            const __m128i qy_0 = _mm_loadu_si128((const __m128i *) &y_ptr[0].qs[0]);
            const __m128i qy_1 = _mm_loadu_si128((const __m128i *) &y_ptr[0].qs[16]);
            const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, zero);
            const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, zero);
            const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0);
            const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1);
            const __m128i sum16_0 = _mm_maddubs_epi16(ones_8, sy_0);
            const __m128i sum16_1 = _mm_maddubs_epi16(ones_8, sy_1);
            const __m128i sum32_0 = _mm_madd_epi16(sum16_0, ones_16);
            const __m128i sum32_1 = _mm_madd_epi16(sum16_1, ones_16);
            const __m256 q = _mm256_cvtepi32_ps(MM256_SET_M128I(sum32_1, sum32_0));
            acc_block = _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[0].d)), q);
        }
        for(int K = 1; K < 4; ++K) {
            const __m256i bit_mask = bytes_from_bits_32(&x[ib].qs[(K) * 4]);
            const __m128i bit_mask_0 = _mm256_castsi256_si128(bit_mask);
            const __m128i bit_mask_1 = _mm256_extractf128_si256(bit_mask, 1);
            const __m128i qy_0 = _mm_loadu_si128((const __m128i *) &y_ptr[(K)].qs[0]);
            const __m128i qy_1 = _mm_loadu_si128((const __m128i *) &y_ptr[(K)].qs[16]);
            const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, zero);
            const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, zero);
            const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0);
            const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1);
            const __m128i sum16_0 = _mm_maddubs_epi16(ones_8, sy_0);
            const __m128i sum16_1 = _mm_maddubs_epi16(ones_8, sy_1);
            const __m128i sum32_0 = _mm_madd_epi16(sum16_0, ones_16);
            const __m128i sum32_1 = _mm_madd_epi16(sum16_1, ones_16);
            const __m256 q = _mm256_cvtepi32_ps(MM256_SET_M128I(sum32_1, sum32_0));
            acc_block = _mm256_add_ps(acc_block, _mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[(K)].d)), q));
        }
#undef Q1_AVX_BLOCK

        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(d0), acc_block));
    }

    *s = hsum_float_8(acc);
#elif defined(__SSSE3__)
    const __m128i ones_8 = _mm_set1_epi8(1);
    const __m128i ones_16 = _mm_set1_epi16(1);
    const __m128i zero = _mm_setzero_si128();
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    for (int ib = 0; ib < nb; ++ib) {
        const __m128 d0 = _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d));
        const block_q8_0 * GGML_RESTRICT y_ptr = &y[ib * 4];

#define Q1_SSSE3_BLOCK(QS_OFF, Y_IDX, ACC) \
        { \
            const __m128i bit_mask_0 = bytes_from_bits_16(&x[ib].qs[(QS_OFF) + 0]); \
            const __m128i bit_mask_1 = bytes_from_bits_16(&x[ib].qs[(QS_OFF) + 2]); \
            const __m128i qy_0 = _mm_loadu_si128((const __m128i *) &y_ptr[(Y_IDX)].qs[0]); \
            const __m128i qy_1 = _mm_loadu_si128((const __m128i *) &y_ptr[(Y_IDX)].qs[16]); \
            const __m128i sign_mask_0 = _mm_cmpeq_epi8(bit_mask_0, zero); \
            const __m128i sign_mask_1 = _mm_cmpeq_epi8(bit_mask_1, zero); \
            const __m128i sy_0 = _mm_sub_epi8(_mm_xor_si128(qy_0, sign_mask_0), sign_mask_0); \
            const __m128i sy_1 = _mm_sub_epi8(_mm_xor_si128(qy_1, sign_mask_1), sign_mask_1); \
            const __m128i sum_0 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy_0), ones_16); \
            const __m128i sum_1 = _mm_madd_epi16(_mm_maddubs_epi16(ones_8, sy_1), ones_16); \
            const __m128 q = _mm_cvtepi32_ps(_mm_add_epi32(sum_0, sum_1)); \
            (ACC) = _mm_add_ps((ACC), _mm_mul_ps(_mm_mul_ps(d0, _mm_set1_ps(GGML_CPU_FP16_TO_FP32(y_ptr[(Y_IDX)].d))), q)); \
        }
        Q1_SSSE3_BLOCK(0,  0, acc_0)
        Q1_SSSE3_BLOCK(4,  1, acc_1)
        Q1_SSSE3_BLOCK(8,  2, acc_2)
        Q1_SSSE3_BLOCK(12, 3, acc_3)
#undef Q1_SSSE3_BLOCK
    }

    *s = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#else
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q1_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined(__AVX2__)
static inline __m256i q2_0_unpack32(const uint8_t * packed_bytes) {
    const __m256i mask = _mm256_set1_epi8(3);
#if defined(__AVX512VBMI__) && defined(__AVX512VL__)
    // Each 16-bit lane holds eight two-bit codes. Extract those codes
    // directly into bytes when the CPU supports byte multishift.
    const __m256i words = _mm256_cvtepu16_epi64(
            _mm_loadl_epi64((const __m128i *) packed_bytes));
    const __m256i shifts = _mm256_set1_epi64x(INT64_C(0x0e0c0a0806040200));
    const __m256i codes = _mm256_and_si256(mask,
            _mm256_multishift_epi64_epi8(shifts, words));
#else
    // Expand each packed byte into a dword, then place its four codes
    // into consecutive bytes. No weight repacking or padding is needed.
    const __m256i packed = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) packed_bytes));
    const __m256i codes = _mm256_and_si256(mask, _mm256_or_si256(
        _mm256_or_si256(packed, _mm256_slli_epi32(packed, 6)),
        _mm256_or_si256(_mm256_slli_epi32(packed, 12), _mm256_slli_epi32(packed, 18))));
#endif
    return codes;
}

static inline __m256i q2_0_dot32_codes(__m256i codes, const int8_t * act) {
    const __m256i one8 = _mm256_set1_epi8(1);
    const __m256i qy = _mm256_loadu_si256((const __m256i *) act);
    // dot(codes - 1, qy): unsigned codes avoid negating -128 in qy.
    // Neither 16-bit pair sum can saturate for codes in [0, 3].
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_sub_epi32(
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), codes, qy),
        _mm256_dpbusd_epi32(_mm256_setzero_si256(), one8, qy));
#else
    const __m256i one16 = _mm256_set1_epi16(1);
    const __m256i pairs = _mm256_sub_epi16(_mm256_maddubs_epi16(codes, qy),
                                          _mm256_maddubs_epi16(one8, qy));
    return _mm256_madd_epi16(pairs, one16);
#endif
}

static inline __m256i q2_0_dot32(const uint8_t * packed_bytes, const int8_t * act) {
    return q2_0_dot32_codes(q2_0_unpack32(packed_bytes), act);
}
#endif

void ggml_vec_dot_q2_0_g128_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK2_0_G128 == 0 && nrc == 1);
    UNUSED(nrc); UNUSED(bs); UNUSED(bx); UNUSED(by);
    const block_q2_0_g128 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    float sumf = 0.0f;
    for (int i = 0; i < n / QK2_0_G128; ++i) {
        float sumi = 0.0f;
        for (int k = 0; k < QK2_0_G128 / QK8_0; ++k) {
            const block_q8_0 * yb = &y[i * (QK2_0_G128 / QK8_0) + k];
            const int dot = hsum_i32_8(q2_0_dot32(x[i].qs + k * 8, yb->qs));
            // PQ2 shares the packed codes with Q2_0, but one weight scale
            // covers four activation scales. Preserve the scalar FP order;
            // only the exact integer products/reduction use vector lanes.
            sumi += GGML_CPU_FP16_TO_FP32(yb->d) * dot;
        }
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * sumi;
    }
    *s = sumf;
#else
    ggml_vec_dot_q2_0_g128_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q2_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
#if defined(__AVX2__)
    assert(n % QK2_0 == 0 && nrc == 1);
    UNUSED(nrc); UNUSED(bs); UNUSED(bx); UNUSED(by);
    const block_q2_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;
    float sumf = 0.0f;
    int i = 0;
#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__) && defined(__AVX512VL__)
    for (; i + 4 <= n / QK2_0; i += 4) {
        __m128 sumi = _mm_setzero_ps();
        for (int k = 0; k < 2; ++k) {
            const __m256i d0 = q2_0_dot32(x[i+0].qs + k*8, y[2*(i+0)+k].qs);
            const __m256i d1 = q2_0_dot32(x[i+1].qs + k*8, y[2*(i+1)+k].qs);
            const __m256i d2 = q2_0_dot32(x[i+2].qs + k*8, y[2*(i+2)+k].qs);
            const __m256i d3 = q2_0_dot32(x[i+3].qs + k*8, y[2*(i+3)+k].qs);
            const __m256i halves = _mm256_hadd_epi32(
                    _mm256_hadd_epi32(d0, d1), _mm256_hadd_epi32(d2, d3));
            const __m128i dots = _mm_add_epi32(_mm256_castsi256_si128(halves),
                    _mm256_extracti128_si256(halves, 1));
            const __m128 scales = _mm_setr_ps(
                    GGML_CPU_FP16_TO_FP32(y[2*(i+0)+k].d),
                    GGML_CPU_FP16_TO_FP32(y[2*(i+1)+k].d),
                    GGML_CPU_FP16_TO_FP32(y[2*(i+2)+k].d),
                    GGML_CPU_FP16_TO_FP32(y[2*(i+3)+k].d));
            sumi = _mm_add_ps(sumi, _mm_mul_ps(scales, _mm_cvtepi32_ps(dots)));
        }
        float blocks[4];
        _mm_storeu_ps(blocks, sumi);
        // Only integer reduction is grouped; keep FP32 block accumulation in
        // its original order, including the two independently scaled halves.
        for (int j = 0; j < 4; ++j) {
            sumf += GGML_CPU_FP16_TO_FP32(x[i+j].d) * blocks[j];
        }
    }
#endif
    for (; i < n / QK2_0; ++i) {
        float sumi = 0.0f;
        for (int k = 0; k < 2; ++k) {
            const block_q8_0 * yb = &y[2*i+k];
            const int dot = hsum_i32_8(q2_0_dot32(x[i].qs+k*8, yb->qs));
            // Keep the generic path's two activation scales and FP reduction
            // order; only the exactly representable integer dot is vectorized.
            sumi += GGML_CPU_FP16_TO_FP32(yb->d) * dot;
        }
        sumf += GGML_CPU_FP16_TO_FP32(x[i].d) * sumi;
    }
    *s = sumf;
#else
    ggml_vec_dot_q2_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined(__AVX2__)
static inline __m256i q2_0_codes_dot(__m256i codes, const int8_t * act) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), codes,
            _mm256_loadu_si256((const __m256i *) act));
#else
    // Codes are in [0, 3], so the signed 16-bit pair sums cannot saturate.
    return _mm256_madd_epi16(_mm256_maddubs_epi16(codes,
            _mm256_loadu_si256((const __m256i *) act)), _mm256_set1_epi16(1));
#endif
}

struct q2_0_prepared_act {
    int16_t sums[512];
    float scales[512];
};

// Q2_0 and PQ2 (Q2_0_G128) share the packed codes; a weight block of qk
// codes holds one fp16 scale followed by qk/32 activation-block slices.
static inline const uint8_t * q2_0_block(const void * vx, int qk, int i) {
    return (const uint8_t *) vx + (size_t) i * (sizeof(ggml_half) + qk/4);
}

static inline float q2_0_block_d(const uint8_t * block) {
    return GGML_CPU_FP16_TO_FP32(*(const ggml_half *) block);
}

static inline const uint8_t * q2_0_block_qs(const uint8_t * block) {
    return block + sizeof(ggml_half);
}

// Forced on every *_batch_impl so each literal (nr, qk) instance folds to
// constants; GCC otherwise clones the kernel with a runtime nr or qk.
#if defined(_MSC_VER)
#define BATCH_INLINE __forceinline
#else
#define BATCH_INLINE inline __attribute__((always_inline))
#endif

// Reuse unpacked weights across tokens selecting the same expert, optionally
// sharing activation metadata across output columns. Preserve FP32 block order.
static BATCH_INLINE void q2_0_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr,
                                 const struct q2_0_prepared_act * prepared, int qk) {
    const int kq = qk / QK8_0;
    const int nb = n / qk;
    for (int r = 0; r < nr; ++r) s[r] = 0.0f;
    int i = 0;
    for (; i + 4 <= nb; i += 4) {
        const uint8_t * x0 = q2_0_block(vx, qk, i+0);
        const uint8_t * x1 = q2_0_block(vx, qk, i+1);
        const uint8_t * x2 = q2_0_block(vx, qk, i+2);
        const uint8_t * x3 = q2_0_block(vx, qk, i+3);
        __m128 sums[4];
        for (int r = 0; r < nr; ++r) sums[r] = _mm_setzero_ps();
        for (int k = 0; k < kq; ++k) {
            const __m256i c0 = q2_0_unpack32(q2_0_block_qs(x0) + k*8);
            const __m256i c1 = q2_0_unpack32(q2_0_block_qs(x1) + k*8);
            const __m256i c2 = q2_0_unpack32(q2_0_block_qs(x2) + k*8);
            const __m256i c3 = q2_0_unpack32(q2_0_block_qs(x3) + k*8);
            for (int r = 0; r < nr; ++r) {
                const block_q8_0 * y = vy[r];
                const __m256i d0 = prepared ? q2_0_codes_dot(c0, y[kq*(i+0)+k].qs) : q2_0_dot32_codes(c0, y[kq*(i+0)+k].qs);
                const __m256i d1 = prepared ? q2_0_codes_dot(c1, y[kq*(i+1)+k].qs) : q2_0_dot32_codes(c1, y[kq*(i+1)+k].qs);
                const __m256i d2 = prepared ? q2_0_codes_dot(c2, y[kq*(i+2)+k].qs) : q2_0_dot32_codes(c2, y[kq*(i+2)+k].qs);
                const __m256i d3 = prepared ? q2_0_codes_dot(c3, y[kq*(i+3)+k].qs) : q2_0_dot32_codes(c3, y[kq*(i+3)+k].qs);
                const __m256i halves = _mm256_hadd_epi32(
                        _mm256_hadd_epi32(d0, d1), _mm256_hadd_epi32(d2, d3));
                __m128i dots = _mm_add_epi32(_mm256_castsi256_si128(halves),
                        _mm256_extracti128_si256(halves, 1));
                if (prepared) {
                    dots = _mm_sub_epi32(dots, _mm_cvtepi16_epi32(_mm_loadl_epi64(
                            (const __m128i *) (prepared[r].sums + k*nb + i))));
                }
                const __m128 scales = prepared ? _mm_loadu_ps(prepared[r].scales + k*nb + i) : _mm_setr_ps(
                        GGML_CPU_FP16_TO_FP32(y[kq*(i+0)+k].d),
                        GGML_CPU_FP16_TO_FP32(y[kq*(i+1)+k].d),
                        GGML_CPU_FP16_TO_FP32(y[kq*(i+2)+k].d),
                        GGML_CPU_FP16_TO_FP32(y[kq*(i+3)+k].d));
                sums[r] = _mm_add_ps(sums[r], _mm_mul_ps(scales, _mm_cvtepi32_ps(dots)));
            }
        }
        const float d[4] = { q2_0_block_d(x0), q2_0_block_d(x1), q2_0_block_d(x2), q2_0_block_d(x3) };
        for (int r = 0; r < nr; ++r) {
            float blocks[4];
            _mm_storeu_ps(blocks, sums[r]);
            for (int j = 0; j < 4; ++j) s[r] += d[j] * blocks[j];
        }
    }
    for (; i < nb; ++i) {
        const uint8_t * x = q2_0_block(vx, qk, i);
        float sums[4] = {0};
        for (int k = 0; k < kq; ++k) {
            const __m256i codes = q2_0_unpack32(q2_0_block_qs(x) + k*8);
            for (int r = 0; r < nr; ++r) {
                const block_q8_0 * y = vy[r];
                const int dot = prepared
                    ? hsum_i32_8(q2_0_codes_dot(codes, y[kq*i+k].qs)) - prepared[r].sums[k*nb+i]
                    : hsum_i32_8(q2_0_dot32_codes(codes, y[kq*i+k].qs));
                const float scale = prepared ? prepared[r].scales[k*nb+i] : GGML_CPU_FP16_TO_FP32(y[kq*i+k].d);
                sums[r] += scale * dot;
            }
        }
        for (int r = 0; r < nr; ++r) s[r] += q2_0_block_d(x) * sums[r];
    }
}

// Literal row counts and group sizes let the compiler keep each row's
// accumulators in registers.
static void q2_0_batch_dispatch(int n, float * s, const void * vx, const void * const * vy, int nr,
                                     const struct q2_0_prepared_act * prepared, int qk) {
#define Q2_0_BATCH_NR(QK) \
    switch (nr) { \
        case 1: q2_0_batch_impl(n, s, vx, vy, 1, prepared, QK); break; \
        case 2: q2_0_batch_impl(n, s, vx, vy, 2, prepared, QK); break; \
        case 3: q2_0_batch_impl(n, s, vx, vy, 3, prepared, QK); break; \
        case 4: q2_0_batch_impl(n, s, vx, vy, 4, prepared, QK); break; \
    }
    if (qk == QK2_0) {
        Q2_0_BATCH_NR(QK2_0)
    } else {
        Q2_0_BATCH_NR(QK2_0_G128)
    }
#undef Q2_0_BATCH_NR
}

static void q2_0_batch_rows(int n, float * const * dst, const void * vx, size_t stride,
                            const void * const * vy, int nr, int64_t rows, int qk) {
    assert((qk == QK2_0 || qk == QK2_0_G128) && n % qk == 0 && nr >= 1 && nr <= 4);
    // Amortize activation preparation over at least one output-column tile.
    if (n <= 16384 && rows >= 16) {
        // Reuse exact signed-byte sums and FP16-to-FP32 scale conversions across
        // output columns. Scratch is bounded at 12 KiB, with no weight repacking.
        struct q2_0_prepared_act prepared[4];
        const int kq = qk / QK8_0;
        const int nb = n / qk;
        for (int r = 0; r < nr; ++r) {
            const block_q8_0 * y = vy[r];
            // Adjacent entries feed adjacent weight blocks in one block-slice
            // pass, so four corrections need one load rather than a gather.
            for (int i = 0; i < nb; ++i) {
                for (int k = 0; k < kq; ++k) {
                    const block_q8_0 * b = &y[i*kq + k];
                    prepared[r].sums[k*nb + i] = (int16_t) hsum_i32_8(
                            q2_0_codes_dot(_mm256_set1_epi8(1), b->qs));
                    prepared[r].scales[k*nb + i] = GGML_CPU_FP16_TO_FP32(b->d);
                }
            }
        }
        for (int64_t row = 0; row < rows; ++row) {
            float sums[4];
            q2_0_batch_dispatch(n, sums, (const char *) vx + row * stride, vy, nr, prepared, qk);
            for (int r = 0; r < nr; ++r) dst[r][row] = sums[r];
        }
        return;
    }
    for (int64_t row = 0; row < rows; ++row) {
        const void * x = (const char *) vx + row * stride;
        if (nr == 1) {
            if (qk == QK2_0) {
                ggml_vec_dot_q2_0_q8_0(n, dst[0] + row, 0, x, 0, vy[0], 0, 1);
            } else {
                ggml_vec_dot_q2_0_g128_q8_0(n, dst[0] + row, 0, x, 0, vy[0], 0, 1);
            }
            continue;
        }
        float sums[4];
        q2_0_batch_dispatch(n, sums, x, vy, nr, NULL, qk);
        for (int r = 0; r < nr; ++r) dst[r][row] = sums[r];
    }
}

static void q2_0_rows(int n, float * const * dst, const void * vx, size_t stride,
                      const void * const * vy, int nr, int64_t rows) {
    q2_0_batch_rows(n, dst, vx, stride, vy, nr, rows, QK2_0);
}

static void q2_0_g128_rows(int n, float * const * dst, const void * vx, size_t stride,
                           const void * const * vy, int nr, int64_t rows) {
    q2_0_batch_rows(n, dst, vx, stride, vy, nr, rows, QK2_0_G128);
}
#endif

void ggml_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps( GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        qx = _mm256_sub_epi8( qx, off );

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_1), _mm_set1_epi8(8));
        const __m128i q4b_1_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_1, 4)), _mm_set1_epi8(8));
        const __m128i q4b_2_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_2), _mm_set1_epi8(8));
        const __m128i q4b_2_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_2, 4)), _mm_set1_epi8(8));

        const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
        const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
        const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
        const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
        const __m128i p_1 = _mm_add_epi16(p16_1_0, p16_1_1);
        const __m128i p_2 = _mm_add_epi16(p16_2_0, p16_2_1);
        const __m256 p =  sum_i16_pairs_float(p_2, p_1);

        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#elif defined(__SSSE3__)
    // set constants
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        _mm_prefetch(&x[ib] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = _mm_set1_ps( GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d) );

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[ib].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        _mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = _mm_set1_ps( GGML_CPU_FP16_TO_FP32(x[ib + 1].d) * GGML_CPU_FP16_TO_FP32(y[ib + 1].d) );

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i *)(y[ib + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        __m128 p0_d = _mm_mul_ps( d_0_1, p0 );
        __m128 p1_d = _mm_mul_ps( d_0_1, p1 );
        __m128 p2_d = _mm_mul_ps( d_2_3, p2 );
        __m128 p3_d = _mm_mul_ps( d_2_3, p3 );

        // Acummulate
        acc_0 = _mm_add_ps(p0_d, acc_0);
        acc_1 = _mm_add_ps(p1_d, acc_1);
        acc_2 = _mm_add_ps(p2_d, acc_2);
        acc_3 = _mm_add_ps(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);

#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

void ggml_vec_dot_q4_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

    int ib = 0;

#if defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    // Main loop
    for (; ib < nb; ++ib) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[ib].d);
        const float d1 = GGML_CPU_FP16_TO_FP32(y[ib].d);

        summs += GGML_CPU_FP16_TO_FP32(x[ib].m) * GGML_CPU_FP16_TO_FP32(y[ib].s);

        const __m256 d0v = _mm256_set1_ps( d0 );
        const __m256 d1v = _mm256_set1_ps( d1 );

        // Compute combined scales
        const __m256 d0d1 = _mm256_mul_ps( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        const __m256i qy = _mm256_loadu_si256( (const __m256i *)y[ib].qs );

        const __m256 xy = mul_sum_us8_pairs_float(qx, qy);

        // Accumulate d0*d1*x*y
#if defined(__AVX2__)
        acc = _mm256_fmadd_ps( d0d1, xy, acc );
#else
        acc = _mm256_add_ps( _mm256_mul_ps( d0d1, xy ), acc );
#endif
    }

    *s = hsum_float_8(acc) + summs;
#else
    UNUSED(nb);
    UNUSED(x);
    UNUSED(y);
    UNUSED(ib);
    ggml_vec_dot_q4_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_mxfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_MXFP4 == 0);
    static_assert(QK_MXFP4 == QK8_0, "QK_MXFP4 and QK8_0 must be the same");

    const block_mxfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_MXFP4;

    int ib = 0;
    float sumf = 0;

#if defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        const __m256 scale0 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 0].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 0].e));
        const __m256 scale1 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 1].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e));
        accum1 = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));

        const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        const __m256 deltas = quad_mx_delta_float(x[ib].e, y[ib].d, x[ib + 1].e, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);

#endif
    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e);
        int sumi1 = 0;
        int sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_fp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_fp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_nvfp4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_NVFP4 == 0);

    const block_nvfp4 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    const int nb = n / QK_NVFP4;
    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 16));

        const __m256i q8_01 = _mm256_loadu_si256((const __m256i *)y[2*ib + 0].qs);
        const __m256i q8_23 = _mm256_loadu_si256((const __m256i *)y[2*ib + 1].qs);

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        //reordering
        const __m256i q4_01 = MM256_SET_M128I(_mm_unpackhi_epi64(q4_01_lo,q4_01_hi), _mm_unpacklo_epi64(q4_01_lo,q4_01_hi));
        const __m256i q4_23 = MM256_SET_M128I(_mm_unpackhi_epi64(q4_23_lo,q4_23_hi),_mm_unpacklo_epi64(q4_23_lo,q4_23_hi));

        const __m256i p01 = mul_add_epi8(q4_01,q8_01);
        const __m256i p_1 = _mm256_madd_epi16(p01, mone);

        const __m256i p23 = mul_add_epi8(q4_23,q8_23);
        const __m256i p_2 = _mm256_madd_epi16(p23, mone);

        const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
        const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);

        const float s0 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[0]) * dy0;
        const float s1 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[1]) * dy0;
        const float s2 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[2]) * dy1;
        const float s3 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[3]) * dy1;

        const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = _mm256_fmadd_ps(scales01, _mm256_cvtepi32_ps(p_1), accum);
        accum = _mm256_fmadd_ps(scales23, _mm256_cvtepi32_ps(p_2), accum);
    }
    sumf = hsum_float_8(accum);

#elif defined(__AVX__)

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for(; ib < nb; ib++){

        const __m128i q4bits_01 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 0));
        const __m128i q4bits_23 = _mm_loadu_si128((const __m128i *)(x[ib].qs + 16));

        const __m128i q8_0 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 0));
        const __m128i q8_1 = _mm_loadu_si128((const __m128i *)(y[2*ib + 0].qs + 16));
        const __m128i q8_2 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 0));
        const __m128i q8_3 = _mm_loadu_si128((const __m128i *)(y[2*ib + 1].qs + 16));

        const __m128i q4_01_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_01, m4b));
        const __m128i q4_01_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_01, 4), m4b));
        const __m128i q4_23_lo = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_23, m4b));
        const __m128i q4_23_hi = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_23, 4), m4b));

        const __m128i q4_0 = _mm_unpacklo_epi64(q4_01_lo, q4_01_hi);
        const __m128i q4_1 = _mm_unpackhi_epi64(q4_01_lo, q4_01_hi);
        const __m128i q4_2 = _mm_unpacklo_epi64(q4_23_lo, q4_23_hi);
        const __m128i q4_3 = _mm_unpackhi_epi64(q4_23_lo, q4_23_hi);

        const __m128i p0_i32 = mul_sum_i8_pairs(q4_0, q8_0);
        const __m128i p1_i32 = mul_sum_i8_pairs(q4_1, q8_1);
        const __m128i p2_i32 = mul_sum_i8_pairs(q4_2, q8_2);
        const __m128i p3_i32 = mul_sum_i8_pairs(q4_3, q8_3);

        const __m128 p0 = _mm_cvtepi32_ps(p0_i32);
        const __m128 p1 = _mm_cvtepi32_ps(p1_i32);
        const __m128 p2 = _mm_cvtepi32_ps(p2_i32);
        const __m128 p3 = _mm_cvtepi32_ps(p3_i32);

        const __m256 p01 = _mm256_set_m128(p1, p0);
        const __m256 p23 = _mm256_set_m128(p3, p2);

        const float dy0 = GGML_CPU_FP16_TO_FP32(y[2*ib].d);
        const float dy1 = GGML_CPU_FP16_TO_FP32(y[2*ib+1].d);

        const float s0 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[0]) * dy0;
        const float s1 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[1]) * dy0;
        const float s2 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[2]) * dy1;
        const float s3 = GGML_CPU_UE4M3_TO_FP32(x[ib].d[3]) * dy1;

        const __m256 scales01 = _mm256_set_m128(_mm_set1_ps(s1), _mm_set1_ps(s0));
        const __m256 scales23 = _mm256_set_m128(_mm_set1_ps(s3), _mm_set1_ps(s2));

        accum = _mm256_add_ps(accum, _mm256_mul_ps(p01, scales01));
        accum = _mm256_add_ps(accum, _mm256_mul_ps(p23, scales23));
    }
    sumf = hsum_float_8(accum);

#endif

    for (;ib < nb; ++ib) {
        for (int s_idx = 0; s_idx < 4; ++s_idx) {
            const float d = GGML_CPU_UE4M3_TO_FP32(x[ib].d[s_idx]);
            const int q8_block = s_idx / 2;
            const int q8_off   = (s_idx % 2) * QK_NVFP4_SUB;
            const float dy = GGML_CPU_FP16_TO_FP32(y[2*ib + q8_block].d);

            int sumi_lo = 0, sumi_hi = 0;
            for (int j = 0; j < QK_NVFP4_SUB/2; ++j) {
                const uint8_t qv = x[ib].qs[s_idx*(QK_NVFP4_SUB/2) + j];
                sumi_lo += y[2*ib + q8_block].qs[q8_off + j +               0] * kvalues_fp4[qv & 0xf];
                sumi_hi += y[2*ib + q8_block].qs[q8_off + j + QK_NVFP4_SUB/2] * kvalues_fp4[qv >>  4];
            }

            sumf += dy * d * (sumi_lo + sumi_hi);
        }
    }
    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
        qx = _mm256_or_si256(qx, bxhi);

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = hsum_float_8(acc);
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8((char)0xF0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));

        __m256i bx_0 = bytes_from_nibbles_32(x[ib].qs);
        const __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_andnot_si128(bxhil, mask);
        bxhih = _mm_andnot_si128(bxhih, mask);
        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_SET_M128I(bxh, bxl);

        const __m256i by_0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx_0, by_0);

        /* Multiply q with scale and accumulate */
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
    }

    *s = hsum_float_8(acc);
#else
    UNUSED(nb);
    UNUSED(ib);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q5_0_q8_0_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q5_1_q8_1(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * GGML_RESTRICT x = vx;
    const block_q8_1 * GGML_RESTRICT y = vy;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0.0f;

    // Main loop
    for (; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d));

        summs += GGML_CPU_FP16_TO_FP32(x[ib].m) * GGML_CPU_FP16_TO_FP32(y[ib].s);

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);

        const __m256 dy = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib].d));
        const __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_us8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    *s = hsum_float_8(acc) + summs;
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8(0x10);

    float summs = 0.0f;

    // Main loop
    for (; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d));

        summs += GGML_CPU_FP16_TO_FP32(x[ib].m) * GGML_CPU_FP16_TO_FP32(y[ib].s);

        __m256i bx_0 = bytes_from_nibbles_32(x[ib].qs);
        const __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_and_si128(bxhil, mask);
        bxhih = _mm_and_si128(bxhih, mask);
        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_SET_M128I(bxh, bxl);

        const __m256 dy = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib].d));
        const __m256i by_0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_us8_pairs_float(bx_0, by_0);

        acc = _mm256_add_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, dy)), acc);
    }

    *s = hsum_float_8(acc) + summs;
#else
    UNUSED(nb);
    UNUSED(ib);
    UNUSED(x);
    UNUSED(y);
    ggml_vec_dot_q5_1_q8_1_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q8_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * GGML_RESTRICT x = vx;
    const block_q8_0 * GGML_RESTRICT y = vy;

    int ib = 0;
    float sumf = 0;

#if defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i qx_1_0 = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i qx_1_1 = _mm_loadu_si128((const __m128i *)x[ib].qs + 1);
        const __m128i qx_2_0 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i qx_2_1 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs + 1);
        const __m128i qy_1_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        const __m128i qy_1_1 = _mm_loadu_si128((const __m128i *)y[ib].qs + 1);
        const __m128i qy_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i qy_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m256 p = mul_sum_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_CPU_FP16_TO_FP32(x[ib].d)*GGML_CPU_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_tq1_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        // 16-bit sums
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        // first 32 bytes of 5 elements
        {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *) (x[i].qs));
            // 8-bit multiplies with shifts, masks and adds
            __m256i qx1 = _mm256_add_epi8(qx0, _mm256_add_epi8(qx0, qx0)); // 1 * 3
            __m256i qx2 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx0, 3), _mm256_set1_epi8(-8)), qx0); // 1 * 9
            __m256i qx3 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx1, 3), _mm256_set1_epi8(-8)), qx1); // 3 * 9
            __m256i qx4 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx2, 3), _mm256_set1_epi8(-8)), qx2); // 9 * 9

            // TODO: can _mm256_mulhi_epu16 be faster even if 16-bits?

            // Cancel the +1 from avg so that it behaves like a halving add
            qx0 = _mm256_subs_epu8(qx0, _mm256_set1_epi8(1));
            qx1 = _mm256_subs_epu8(qx1, _mm256_set1_epi8(1));
            qx2 = _mm256_subs_epu8(qx2, _mm256_set1_epi8(1));
            qx3 = _mm256_subs_epu8(qx3, _mm256_set1_epi8(1));
            qx4 = _mm256_subs_epu8(qx4, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx0 = _mm256_avg_epu8(qx0, _mm256_avg_epu8(qx0, _mm256_setzero_si256()));
            qx1 = _mm256_avg_epu8(qx1, _mm256_avg_epu8(qx1, _mm256_setzero_si256()));
            qx2 = _mm256_avg_epu8(qx2, _mm256_avg_epu8(qx2, _mm256_setzero_si256()));
            qx3 = _mm256_avg_epu8(qx3, _mm256_avg_epu8(qx3, _mm256_setzero_si256()));
            qx4 = _mm256_avg_epu8(qx4, _mm256_avg_epu8(qx4, _mm256_setzero_si256()));
            qx0 = _mm256_and_si256(_mm256_srli_epi16(qx0, 6), _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(_mm256_srli_epi16(qx1, 6), _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(_mm256_srli_epi16(qx2, 6), _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(_mm256_srli_epi16(qx3, 6), _mm256_set1_epi8(3));
            qx4 = _mm256_and_si256(_mm256_srli_epi16(qx4, 6), _mm256_set1_epi8(3));

            const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (y[i].qs +   0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (y[i].qs +  32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (y[i].qs +  64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (y[i].qs +  96));
            const __m256i qy4 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 128));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);
            qx4 = _mm256_maddubs_epi16(qx4, qy4);

            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
            sumi2 = _mm256_add_epi16(sumi2, qx4);
        }

        // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
        {
            __m128i qx0 = _mm_loadu_si128((const __m128i *) (x[i].qs + 32));
            uint32_t qh;
            memcpy(&qh, x[i].qh, sizeof(qh)); // potentially unaligned
            __m256i qx5_l = _mm256_cvtepu8_epi16(_mm_set1_epi32(qh));
            __m128i qx1 = _mm_add_epi8(qx0, _mm_add_epi8(qx0, qx0)); // 1 * 3
            __m128i qx2 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx0, 3), _mm_set1_epi8(-8)), qx0); // 1 * 9
            __m128i qx3 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx1, 3), _mm_set1_epi8(-8)), qx1); // 3 * 9
            __m128i qx4 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx2, 3), _mm_set1_epi8(-8)), qx2); // 9 * 9
            __m256i qx01 = MM256_SET_M128I(qx1, qx0);
            __m256i qx23 = MM256_SET_M128I(qx3, qx2);

            // avx2 does not have 8-bit multiplies, so 16-bit it is.
            qx5_l = _mm256_mullo_epi16(qx5_l, _mm256_set_epi16(27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1));
            qx5_l = _mm256_and_si256(qx5_l, _mm256_set1_epi16(0xFF));
            __m128i qx5 = _mm_packus_epi16(_mm256_castsi256_si128(qx5_l), _mm256_extracti128_si256(qx5_l, 1));

            __m256i qx45 = MM256_SET_M128I(qx5, qx4);

            // Cancel the +1 from avg so that it behaves like a halving add
            qx01 = _mm256_subs_epu8(qx01, _mm256_set1_epi8(1));
            qx23 = _mm256_subs_epu8(qx23, _mm256_set1_epi8(1));
            qx45 = _mm256_subs_epu8(qx45, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx01 = _mm256_avg_epu8(qx01, _mm256_avg_epu8(qx01, _mm256_setzero_si256()));
            qx23 = _mm256_avg_epu8(qx23, _mm256_avg_epu8(qx23, _mm256_setzero_si256()));
            qx45 = _mm256_avg_epu8(qx45, _mm256_avg_epu8(qx45, _mm256_setzero_si256()));
            qx01 = _mm256_and_si256(_mm256_srli_epi16(qx01, 6), _mm256_set1_epi8(3));
            qx23 = _mm256_and_si256(_mm256_srli_epi16(qx23, 6), _mm256_set1_epi8(3));
            qx45 = _mm256_and_si256(_mm256_srli_epi16(qx45, 6), _mm256_set1_epi8(3));

            const __m256i qy01 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 160));
            const __m256i qy23 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 192));
            const __m256i qy45 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 224));

            qx01 = _mm256_maddubs_epi16(qx01, qy01);
            qx23 = _mm256_maddubs_epi16(qx23, qy23);
            qx45 = _mm256_maddubs_epi16(qx45, qy45);

            sumi0 = _mm256_add_epi16(sumi0, qx01);
            sumi1 = _mm256_add_epi16(sumi1, qx23);
            sumi2 = _mm256_add_epi16(sumi2, qx45);
        }

        const __m256i ysum = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d));

        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(sumi1, sumi2));
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_tq1_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_tq2_0_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        // 16-bit sums, because 256*127 still fits
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *) (x[i].qs + j));
            __m256i qx1 = _mm256_srli_epi16(qx0, 2);
            __m256i qx2 = _mm256_srli_epi16(qx0, 4);
            __m256i qx3 = _mm256_srli_epi16(qx0, 6);

            // 0, 1, 2 (should not be 3)
            qx0 = _mm256_and_si256(qx0, _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(qx1, _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(qx2, _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(qx3, _mm256_set1_epi8(3));

            const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 +  0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 96));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);

            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
        }

        const __m256i ysum = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d));

        sumi0 = _mm256_add_epi16(sumi0, sumi1);
        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_tq2_0_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q2_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {

            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(0x3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float dall = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // load mins and scales from block_q2_K.scales[QK_K/16]
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
        const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

        // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
        const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i*)&y[i].bsums[0]));
        const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i*)&y[i].bsums[8]));

        // sumf += -dmin * summs in 32bits*8
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(MM256_SET_M128I(summs_1, summs_0))), acc);

        const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
        const __m128i scales[2] = { scales_0, scales_1 };

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K/128; ++j) {

            // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
            __m128i q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_0 = _mm_and_si128(q2bits, m3);
            const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_1 = _mm_and_si128(q2bits, m3);
            const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

            // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
            __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
            __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
            __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
            __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
            __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
            __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
            __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
            __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

            // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

            p0 = _mm_add_epi32(p0, p1);
            p2 = _mm_add_epi32(p2, p3);
            p4 = _mm_add_epi32(p4, p5);
            p6 = _mm_add_epi32(p6, p7);

            // isum in 32bits*4*2
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
        }

        // sumf += dall * isum - dmin * summs in 32bits
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q2_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q3_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        // high bit
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

        // integer accumulator
        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;
        int is  = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

            // prepare low and high bits
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            // load Q8 quants
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            // multiply with scales
            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

            // accumulate
            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi  = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));

        }

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i mone = _mm_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    const uint32_t *aux;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // Set up scales
        aux = (const uint32_t *)x[i].scales;
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m128i scales_0 = _mm_cvtepi8_epi16(scales128);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
        const __m128i scales[2] = { scales_0, scales_1 };

        // high bit *128*2 from block_q3_K.hmask[QK_K/8]
        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].hmask[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].hmask[16]);

        // integer accumulator
        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits *64*2 from block_q3_K.qs[QK_K/4]
            const __m128i q3bits_0 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;
            const __m128i q3bits_1 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;

            // prepare low and high bits
            const int bit = j << 2;

            const __m128i q3l_0 = _mm_and_si128(q3bits_0, m3);
            const __m128i q3l_1 = _mm_and_si128(q3bits_1, m3);
            const __m128i q3h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit)), bit), 2);
            const __m128i q3h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit)), bit), 2);

            const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
            const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
            const __m128i q3h_2 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+1)), bit+1), 2);
            const __m128i q3h_3 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+1)), bit+1), 2);

            const __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
            const __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
            const __m128i q3h_4 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+2)), bit+2), 2);
            const __m128i q3h_5 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+2)), bit+2), 2);

            const __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
            const __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
            const __m128i q3h_6 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+3)), bit+3), 2);
            const __m128i q3h_7 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+3)), bit+3), 2);

            // load Q8 quants from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

            __m128i p16_0 = _mm_maddubs_epi16(q3l_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q3l_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q3l_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q3l_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q3l_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q3l_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q3l_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q3l_7, q8_7);

            p16_0 = _mm_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm_sub_epi16(p16_3, q8s_3);
            p16_4 = _mm_sub_epi16(p16_4, q8s_4);
            p16_5 = _mm_sub_epi16(p16_5, q8s_5);
            p16_6 = _mm_sub_epi16(p16_6, q8s_6);
            p16_7 = _mm_sub_epi16(p16_7, q8s_7);

            // multiply with scales
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_7);

            // accumulate
            p16_0 = _mm_add_epi32(p16_0, p16_1);
            p16_2 = _mm_add_epi32(p16_2, p16_3);
            p16_4 = _mm_add_epi32(p16_4, p16_5);
            p16_6 = _mm_add_epi32(p16_6, p16_7);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));

        }

        // multiply with block scale and accumulate
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc);

#else
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q3_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q4_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q4_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q5_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#if defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc) + summs;

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m128i mone  = _mm_set1_epi8(1);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q5 = x[i].qs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].qh[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].qh[16]);
        __m128i hmask = mone;

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int bit = 0;

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            const __m128i q5bits_0 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;
            const __m128i q5bits_1 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;

            __m128i q5l_0 = _mm_and_si128(q5bits_0, m4);
            __m128i q5l_1 = _mm_and_si128(q5bits_1, m4);
            __m128i q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            __m128i q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            __m128i q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            __m128i q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_0 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm_madd_epi16(scale_0, p16_0);
            p16_1 = _mm_madd_epi16(scale_0, p16_1);

            q5l_0 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4);
            q5l_1 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4);
            q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_2 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_3 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_2 = _mm_madd_epi16(scale_1, p16_2);
            p16_3 = _mm_madd_epi16(scale_1, p16_3);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc) + summs;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(kmask1);
    UNUSED(kmask2);
    UNUSED(kmask3);
    UNUSED(utmp);
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        }

        sumi = _mm256_sub_epi32(sumi, q8sclsub);
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m15 = _mm_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        // handle the q6_k -32 offset separately using bsums
        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)y[i].bsums);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)y[i].bsums + 1);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh); qh += 16;

            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
            const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
            const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
            const __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
            const __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;

            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), q4h_7);

            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));

        }

        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
        const __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_q6_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

// only the non-VBMI IQ2_XXS / IQ3_XXS paths read it
#if (defined (__AVX__) || defined (__AVX2__)) && !(defined(__AVX512VBMI__) && defined(__AVX512BW__))
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)
// iq2xxs_grid as 2-bit level codes (8, 25, 43 -> 0, 1, 2), [0] for grid bytes 0-3 and [1] for bytes 4-7;
// generated from g = iq2xxs_grid[i] as code(byte b of g) << 2*(b%4)
static const uint8_t iq2xxs_grid_code[2][256] = {
    {
        0x00, 0x02, 0x05, 0x08, 0x0a, 0x11, 0x14, 0x20, 0x22, 0x28, 0x2a, 0x41, 0x44, 0x50, 0x58, 0x61,
        0x64, 0x80, 0x82, 0x8a, 0xa2, 0x01, 0x04, 0x10, 0x15, 0x40, 0x84, 0x98, 0x00, 0x02, 0x22, 0x82,
        0x01, 0x04, 0x10, 0x21, 0x24, 0x40, 0x42, 0x48, 0x60, 0x81, 0x84, 0x90, 0xa4, 0x00, 0x02, 0x08,
        0x20, 0x46, 0x69, 0x80, 0x91, 0x09, 0x10, 0x40, 0x84, 0xa4, 0x00, 0x05, 0x08, 0x14, 0x28, 0x41,
        0x44, 0x50, 0x52, 0x88, 0x04, 0x40, 0x02, 0x14, 0x01, 0x04, 0x10, 0x21, 0x40, 0x60, 0x84, 0x90,
        0x95, 0x00, 0x08, 0x20, 0x50, 0x5a, 0x80, 0x24, 0x45, 0x00, 0x08, 0x20, 0x25, 0x49, 0x80, 0x18,
        0x62, 0x00, 0x16, 0x01, 0x04, 0x10, 0x40, 0x81, 0x00, 0x05, 0xa0, 0x51, 0x00, 0x02, 0x0a, 0x44,
        0x61, 0x80, 0x82, 0x29, 0x48, 0x00, 0x02, 0x01, 0x04, 0x10, 0x40, 0x56, 0x00, 0x41, 0x64, 0x90,
        0x08, 0x20, 0x94, 0x44, 0x01, 0x04, 0x10, 0x18, 0x21, 0x24, 0x40, 0x48, 0x56, 0x60, 0x81, 0x84,
        0x90, 0x00, 0x20, 0x61, 0x80, 0x85, 0x01, 0x10, 0x48, 0x56, 0x68, 0x00, 0x08, 0x20, 0x80, 0x99,
        0x12, 0x24, 0x00, 0x01, 0x04, 0x10, 0x40, 0x45, 0x00, 0x58, 0x61, 0x82, 0x45, 0x90, 0x00, 0x08,
        0x11, 0x19, 0x20, 0x80, 0x88, 0x04, 0x42, 0xa4, 0x91, 0x90, 0x92, 0x0a, 0x01, 0x54, 0x00, 0x11,
        0x19, 0x64, 0x40, 0x08, 0x04, 0x10, 0x40, 0x68, 0x00, 0x55, 0x18, 0x60, 0x00, 0x05, 0x10, 0x12,
        0x84, 0x42, 0x00, 0x02, 0x0a, 0x41, 0x82, 0x04, 0x18, 0x40, 0x11, 0x01, 0x04, 0x10, 0x15, 0x40,
        0x60, 0x00, 0x46, 0x94, 0x09, 0x40, 0x60, 0x02, 0x04, 0x11, 0x04, 0x10, 0x24, 0x40, 0xa1, 0x16,
        0x80, 0x45, 0x00, 0x22, 0x44, 0x51, 0x81, 0x20, 0x02, 0x50, 0x85, 0x09, 0x00, 0x18, 0x50, 0x04,
    },
    {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05,
        0x05, 0x05, 0x05, 0x05, 0x05, 0x06, 0x06, 0x06, 0x06, 0x06, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
        0x08, 0x08, 0x08, 0x08, 0x09, 0x09, 0x0a, 0x0a, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
        0x10, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x12, 0x12, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x15,
        0x15, 0x16, 0x16, 0x18, 0x18, 0x18, 0x18, 0x18, 0x19, 0x19, 0x19, 0x1a, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x21, 0x21, 0x22, 0x22, 0x24, 0x24, 0x24, 0x24, 0x24, 0x25, 0x25, 0x25, 0x26,
        0x28, 0x28, 0x28, 0x2a, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
        0x40, 0x41, 0x41, 0x41, 0x41, 0x41, 0x42, 0x42, 0x42, 0x42, 0x42, 0x44, 0x44, 0x44, 0x44, 0x44,
        0x45, 0x45, 0x46, 0x48, 0x48, 0x48, 0x48, 0x48, 0x49, 0x49, 0x49, 0x49, 0x4a, 0x4a, 0x50, 0x50,
        0x50, 0x50, 0x50, 0x50, 0x50, 0x51, 0x51, 0x51, 0x52, 0x54, 0x54, 0x55, 0x56, 0x56, 0x58, 0x58,
        0x58, 0x58, 0x59, 0x5a, 0x60, 0x60, 0x60, 0x60, 0x61, 0x61, 0x62, 0x62, 0x64, 0x64, 0x65, 0x65,
        0x65, 0x68, 0x80, 0x80, 0x80, 0x80, 0x80, 0x81, 0x81, 0x81, 0x82, 0x84, 0x84, 0x84, 0x84, 0x84,
        0x84, 0x85, 0x85, 0x85, 0x86, 0x86, 0x86, 0x88, 0x89, 0x8a, 0x90, 0x90, 0x90, 0x90, 0x90, 0x91,
        0x91, 0x92, 0x94, 0x94, 0x94, 0x95, 0x98, 0x99, 0xa0, 0xa0, 0xa0, 0xa1, 0xa2, 0xa4, 0xa8, 0xa9,
    },
};

// look up 64 8-bit indices in one 256-entry byte table
static inline __m512i vbmi_lookup_256(const uint8_t * t, __m512i idx, __mmask64 b7) {
    const __m512i r0 = _mm512_permutex2var_epi8(_mm512_loadu_si512(t +   0), idx, _mm512_loadu_si512(t +  64));
    const __m512i r1 = _mm512_permutex2var_epi8(_mm512_loadu_si512(t + 128), idx, _mm512_loadu_si512(t + 192));
    return _mm512_mask_blend_epi8(b7, r0, r1);
}

// 64 u8 weight magnitudes q (signs in sg) times group j of nr q8 rows, int16 sub-block scales scj
static inline void xxs_vbmi_accum(__m512i q, __mmask64 sg, __m512i scj, __m512i * sumi,
                                  const void * const * vy, int i, int j, int nr) {
    for (int r = 0; r < nr; ++r) {
        const __m512i q8  = _mm512_loadu_si512(((const block_q8_K *) vy[r])[i].qs + 64*j);
        const __m512i q8s = _mm512_mask_sub_epi8(q8, sg, _mm512_setzero_si512(), q8);
#if defined(__AVX512VNNI__)
        sumi[r] = _mm512_dpwssd_epi32(sumi[r], _mm512_maddubs_epi16(q, q8s), scj);
#else
        sumi[r] = _mm512_add_epi32(sumi[r], _mm512_madd_epi16(_mm512_maddubs_epi16(q, q8s), scj));
#endif
    }
}

// Shared by vec_dot (nr = 1) and the batched rows so that a row's result does not depend on nr:
// each block is decoded once and dotted against nr <= 4 activation rows.
static BATCH_INLINE void iq2_xxs_vbmi_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    // qword m of a 64-weight group takes the [0]/[1] codes of grid index 8*(m/4) + m%4 (+16 per group)
    static const uint8_t k_entry[64] = {
         0, 64, 0, 0, 0, 0, 0, 0,   1, 65, 0, 0, 0, 0, 0, 0,   2, 66, 0, 0, 0, 0, 0, 0,   3, 67, 0, 0, 0, 0, 0, 0,
         8, 72, 0, 0, 0, 0, 0, 0,   9, 73, 0, 0, 0, 0, 0, 0,  10, 74, 0, 0, 0, 0, 0, 0,  11, 75, 0, 0, 0, 0, 0, 0,
    };
    // the 32 sign bytes (at 8*i + k, k < 4) made contiguous
    static const uint8_t k_signs[64] = {
         0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
        32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59,
    };

    const __m512i entry  = _mm512_loadu_si512(k_entry);
    const __m512i sperm  = _mm512_loadu_si512(k_signs);
    // levels 8, 25, 43 of the 2-bit codes, repeated so that only the low 2 bits of a permute index matter
    const __m512i levels = _mm512_set1_epi32(0x002B1908);
    const __m512i ksgn0  = _mm512_loadu_si512(ksigns_iq2xs);
    const __m512i ksgn1  = _mm512_loadu_si512(ksigns_iq2xs + 64);
    // byte b of each qword: bit 2b (code of weight b)
    const __m512i cshift = _mm512_set1_epi64(INT64_C(0x0E0C0A0806040200));
    // bytes 0-3 of each qword: bits 32 + 7k (the 4 sign indices)
    const __m512i sshift = _mm512_set1_epi64(INT64_C(0x352E2720));
    // scale m sits in the top nibble of int16 lane 4m+3: lanes 0-15 take scale 0, lanes 16-31 scale 1
    const __m512i sidx   = _mm512_mask_set1_epi16(_mm512_set1_epi16(3), 0xFFFF0000u, 7);

    __m512 accumf[4];
    for (int r = 0; r < nr; ++r) accumf[r] = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        // one qword per 32 weights: 4 grid indices, then 4 7-bit sign indices and a 4-bit scale
        const __m512i qs = _mm512_loadu_si512(x[i].qs);
        const __mmask64 b7 = _mm512_movepi8_mask(qs);
        const __m512i lo = vbmi_lookup_256(iq2xxs_grid_code[0], qs, b7);
        const __m512i hi = vbmi_lookup_256(iq2xxs_grid_code[1], qs, b7);
        const __m512i sg = _mm512_permutex2var_epi8(ksgn0, _mm512_multishift_epi64_epi8(sshift, qs), ksgn1);
        __mmask64 signs[4];
        _mm256_storeu_si256((__m256i *) signs, _mm512_castsi512_si256(_mm512_permutexvar_epi8(sperm, sg)));
        // sub-block scales 2*ls+1 as int16 (lanes 4m+3)
        const __m512i sc = _mm512_or_si512(_mm512_slli_epi16(_mm512_srli_epi16(qs, 12), 1), _mm512_set1_epi16(1));
        __m512i sumi[4];
        for (int r = 0; r < nr; ++r) sumi[r] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __m512i c   = _mm512_permutex2var_epi8(lo, _mm512_add_epi8(entry, _mm512_set1_epi8(16*j)), hi);
            const __m512i q2  = _mm512_permutexvar_epi8(_mm512_multishift_epi64_epi8(cshift, c), levels);
            const __m512i scj = _mm512_permutexvar_epi16(_mm512_add_epi16(sidx, _mm512_set1_epi16(8*j)), sc);
            xxs_vbmi_accum(q2, signs[j], scj, sumi, vy, i, j, nr);
        }
        for (int r = 0; r < nr; ++r) {
            const float d = xd * ((const block_q8_K *) vy[r])[i].d;
            accumf[r] = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi[r]), accumf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = 0.125f * _mm512_reduce_add_ps(accumf[r]);
}
#endif

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)

    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    const void * ys = vy;
    iq2_xxs_vbmi_impl(n, s, vx, &ys, 1);

#elif defined(__AVX2__)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq2_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX2__)

    const __m256i mone = _mm256_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*)bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_2);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*)k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            const __m256i q2_data = _mm256_loadu_si256((const __m256i*)q2);  q2 += 16;
            aux_gindex = _mm256_and_si256(q2_data, m511);

            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);

            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            const __m256i q2_1 = _mm256_set_epi64x(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                   iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                   iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            const __m256i q2_3 = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            const __m256i q2_4 = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = MM256_SET_M128I(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = MM256_SET_M128I(full_signs_h, full_signs_h);

            __m256i signs;
            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i dot3  = _mm256_maddubs_epi16(q2_3, q8s_3);
            const __m256i dot4  = _mm256_maddubs_epi16(q2_4, q8s_4);

            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const __m128i mone = _mm_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i bit_selector_mask_0 = _mm_loadu_si128((const __m128i*)bit_selector_mask_bytes);
    const __m128i bit_selector_mask_1 = _mm_loadu_si128((const __m128i*)bit_selector_mask_bytes + 1);
    const __m128i block_sign_shuffle_1_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1);
    const __m128i block_sign_shuffle_1_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1 + 1);
    const __m128i block_sign_shuffle_2_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2);
    const __m128i block_sign_shuffle_2_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2 + 1);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m128i bit_helper_0 = _mm_loadu_si128((const __m128i*)k_bit_helper);
    const __m128i bit_helper_1 = _mm_loadu_si128((const __m128i*)k_bit_helper + 1);
    const __m128i m511 = _mm_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        const int8_t   * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            const __m128i q2_data_0 = _mm_loadu_si128((const __m128i*)q2);
            const __m128i q2_data_1 = _mm_loadu_si128((const __m128i*)q2 + 1);  q2 += 16;
            aux_gindex = MM256_SET_M128I(_mm_and_si128(q2_data_1, m511), _mm_and_si128(q2_data_0, m511));

            const __m128i partial_sign_bits_0 = _mm_srli_epi16(q2_data_0, 9);
            const __m128i partial_sign_bits_1 = _mm_srli_epi16(q2_data_1, 9);
            const __m128i partial_sign_bits_upper_0 = _mm_srli_epi16(q2_data_0, 13);
            const __m128i partial_sign_bits_upper_1 = _mm_srli_epi16(q2_data_1, 13);
            const __m128i partial_sign_bits_for_counting_0 = _mm_xor_si128(partial_sign_bits_0, partial_sign_bits_upper_0);
            const __m128i partial_sign_bits_for_counting_1 = _mm_xor_si128(partial_sign_bits_1, partial_sign_bits_upper_1);

            const __m128i odd_bits_0 = _mm_shuffle_epi8(bit_helper_0, partial_sign_bits_for_counting_0);
            const __m128i odd_bits_1 = _mm_shuffle_epi8(bit_helper_1, partial_sign_bits_for_counting_1);
            const __m128i full_sign_bits_0 = _mm_or_si128(partial_sign_bits_0, odd_bits_0);
            const __m128i full_sign_bits_1 = _mm_or_si128(partial_sign_bits_1, odd_bits_1);

            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i q2_1_0 = _mm_set_epi64x(iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]]);
            const __m128i q2_3_0 = _mm_set_epi64x(iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
            const __m128i q2_3_1 = _mm_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]]);
            const __m128i q2_4_0 = _mm_set_epi64x(iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
            const __m128i q2_4_1 = _mm_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]]);

            // AVX2 full_signs_1 is full_sign_bits_0 here
            // AVX2 full_signs_2 is full_sign_bits_1 here
            __m128i signs_0, signs_1;
            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(signs_1, mone));

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const __m128i dot3_0  = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
            const __m128i dot3_1  = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
            const __m128i dot4_0  = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
            const __m128i dot4_1  = _mm_maddubs_epi16(q2_4_1, q8s_4_1);

            __m128i sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0));
            const __m128i sc1_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc1_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1));
            const __m128i sc2_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc2_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2));
            const __m128i sc3_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc3_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3));
            const __m128i sc4_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc4_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, sc1_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, sc1_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, sc2_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, sc2_1));
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot3_0, sc3_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot3_1, sc3_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot4_0, sc4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot4_1, sc4_1));
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq2_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX2__)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2_0 = _mm_loadu_si128((const __m128i*)k_mask2);
    const __m128i mask2_1 = _mm_loadu_si128((const __m128i*)k_mask2 + 1);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m128i scales16_0 = _mm_cvtepi8_epi16(scales8);
        const __m128i scales16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(scales8, 8));

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                  iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                  iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                  iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                  iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)]);
            qs += 8;

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);

            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 0)));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 1)));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 0)));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 1)));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq2_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)
// iq3xxs_grid split by byte: [b][i] = byte b of iq3xxs_grid[i]
static const uint8_t iq3xxs_grid_byte[4][256] = {
    {
        0x04, 0x14, 0x24, 0x0c, 0x1c, 0x3e, 0x04, 0x14, 0x0c, 0x14, 0x1c, 0x2c, 0x0c, 0x1c, 0x04, 0x14,
        0x0c, 0x2c, 0x04, 0x14, 0x0c, 0x24, 0x04, 0x04, 0x14, 0x24, 0x0c, 0x04, 0x14, 0x0c, 0x1c, 0x3e,
        0x0c, 0x3e, 0x2c, 0x0c, 0x3e, 0x04, 0x14, 0x2c, 0x04, 0x1c, 0x3e, 0x24, 0x3e, 0x1c, 0x2c, 0x0c,
        0x3e, 0x14, 0x14, 0x2c, 0x24, 0x04, 0x24, 0x34, 0x1c, 0x0c, 0x0c, 0x1c, 0x04, 0x14, 0x0c, 0x1c,
        0x04, 0x14, 0x24, 0x3e, 0x04, 0x04, 0x14, 0x0c, 0x04, 0x14, 0x0c, 0x1c, 0x04, 0x14, 0x0c, 0x04,
        0x14, 0x04, 0x14, 0x04, 0x0c, 0x34, 0x34, 0x0c, 0x2c, 0x04, 0x04, 0x24, 0x34, 0x0c, 0x2c, 0x14,
        0x04, 0x04, 0x14, 0x0c, 0x1c, 0x04, 0x14, 0x34, 0x0c, 0x14, 0x0c, 0x1c, 0x2c, 0x04, 0x14, 0x0c,
        0x04, 0x1c, 0x3e, 0x04, 0x04, 0x14, 0x0c, 0x3e, 0x04, 0x14, 0x3e, 0x04, 0x2c, 0x0c, 0x04, 0x24,
        0x04, 0x24, 0x2c, 0x1c, 0x1c, 0x3e, 0x0c, 0x24, 0x0c, 0x1c, 0x34, 0x2c, 0x0c, 0x04, 0x14, 0x0c,
        0x1c, 0x04, 0x2c, 0x14, 0x04, 0x14, 0x04, 0x0c, 0x24, 0x34, 0x0c, 0x1c, 0x04, 0x2c, 0x14, 0x14,
        0x0c, 0x1c, 0x04, 0x3e, 0x14, 0x04, 0x34, 0x14, 0x2c, 0x24, 0x34, 0x1c, 0x1c, 0x04, 0x24, 0x3e,
        0x2c, 0x3e, 0x1c, 0x3e, 0x24, 0x04, 0x3e, 0x04, 0x04, 0x34, 0x3e, 0x2c, 0x24, 0x0c, 0x24, 0x2c,
        0x1c, 0x04, 0x2c, 0x04, 0x14, 0x04, 0x14, 0x0c, 0x04, 0x04, 0x34, 0x34, 0x2c, 0x24, 0x14, 0x14,
        0x14, 0x1c, 0x04, 0x1c, 0x3e, 0x14, 0x14, 0x0c, 0x04, 0x24, 0x14, 0x24, 0x24, 0x34, 0x24, 0x0c,
        0x0c, 0x3e, 0x24, 0x04, 0x34, 0x24, 0x2c, 0x14, 0x1c, 0x1c, 0x0c, 0x1c, 0x2c, 0x3e, 0x04, 0x14,
        0x14, 0x34, 0x04, 0x14, 0x2c, 0x14, 0x04, 0x2c, 0x1c, 0x04, 0x0c, 0x0c, 0x04, 0x14, 0x24, 0x04,
    },
    {
        0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x14, 0x14, 0x1c, 0x24, 0x3e, 0x3e, 0x04, 0x04, 0x0c, 0x0c,
        0x14, 0x14, 0x1c, 0x1c, 0x24, 0x2c, 0x3e, 0x04, 0x04, 0x04, 0x0c, 0x14, 0x14, 0x1c, 0x1c, 0x1c,
        0x2c, 0x2c, 0x3e, 0x04, 0x04, 0x0c, 0x0c, 0x14, 0x3e, 0x0c, 0x1c, 0x24, 0x2c, 0x3e, 0x3e, 0x04,
        0x04, 0x1c, 0x2c, 0x1c, 0x34, 0x0c, 0x0c, 0x0c, 0x24, 0x34, 0x04, 0x04, 0x0c, 0x0c, 0x14, 0x14,
        0x1c, 0x1c, 0x1c, 0x24, 0x2c, 0x04, 0x04, 0x0c, 0x14, 0x14, 0x04, 0x04, 0x0c, 0x0c, 0x14, 0x1c,
        0x3e, 0x04, 0x04, 0x14, 0x1c, 0x24, 0x34, 0x04, 0x04, 0x2c, 0x14, 0x14, 0x24, 0x3e, 0x04, 0x14,
        0x24, 0x04, 0x04, 0x0c, 0x0c, 0x14, 0x14, 0x14, 0x1c, 0x24, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x14,
        0x1c, 0x34, 0x34, 0x3e, 0x04, 0x04, 0x0c, 0x0c, 0x14, 0x14, 0x1c, 0x24, 0x2c, 0x04, 0x0c, 0x0c,
        0x3e, 0x3e, 0x1c, 0x2c, 0x04, 0x14, 0x24, 0x3e, 0x04, 0x04, 0x0c, 0x24, 0x04, 0x0c, 0x0c, 0x14,
        0x14, 0x2c, 0x34, 0x3e, 0x04, 0x04, 0x14, 0x1c, 0x24, 0x24, 0x04, 0x04, 0x0c, 0x14, 0x2c, 0x3e,
        0x0c, 0x1c, 0x1c, 0x24, 0x3e, 0x04, 0x04, 0x14, 0x2c, 0x0c, 0x1c, 0x34, 0x1c, 0x34, 0x04, 0x0c,
        0x1c, 0x1c, 0x2c, 0x2c, 0x3e, 0x14, 0x1c, 0x24, 0x34, 0x34, 0x04, 0x24, 0x04, 0x2c, 0x34, 0x14,
        0x24, 0x3e, 0x04, 0x0c, 0x0c, 0x1c, 0x0c, 0x24, 0x3e, 0x04, 0x04, 0x14, 0x2c, 0x0c, 0x1c, 0x3e,
        0x04, 0x2c, 0x0c, 0x14, 0x14, 0x3e, 0x04, 0x1c, 0x2c, 0x14, 0x24, 0x14, 0x24, 0x24, 0x34, 0x14,
        0x34, 0x0c, 0x34, 0x1c, 0x1c, 0x24, 0x04, 0x2c, 0x1c, 0x04, 0x14, 0x04, 0x04, 0x04, 0x0c, 0x1c,
        0x2c, 0x14, 0x24, 0x0c, 0x24, 0x2c, 0x04, 0x0c, 0x1c, 0x34, 0x14, 0x24, 0x04, 0x04, 0x14, 0x1c,
    },
    {
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x2c,
        0x2c, 0x2c, 0x2c, 0x34, 0x34, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x24, 0x24, 0x24, 0x2c, 0x2c, 0x2c, 0x2c, 0x34, 0x3e,
        0x3e, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c,
        0x0c, 0x0c, 0x0c, 0x0c, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x1c, 0x1c, 0x1c,
        0x1c, 0x1c, 0x24, 0x24, 0x2c, 0x2c, 0x2c, 0x2c, 0x3e, 0x3e, 0x3e, 0x3e, 0x04, 0x04, 0x04, 0x04,
        0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x1c, 0x1c, 0x24, 0x24, 0x24, 0x2c, 0x2c, 0x2c, 0x2c, 0x34, 0x34, 0x34, 0x3e, 0x3e, 0x04, 0x04,
        0x04, 0x04, 0x04, 0x04, 0x0c, 0x14, 0x14, 0x14, 0x14, 0x14, 0x1c, 0x1c, 0x24, 0x24, 0x24, 0x2c,
        0x2c, 0x2c, 0x3e, 0x3e, 0x3e, 0x3e, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x14, 0x14, 0x14,
        0x1c, 0x1c, 0x24, 0x24, 0x24, 0x24, 0x2c, 0x2c, 0x34, 0x3e, 0x3e, 0x04, 0x04, 0x04, 0x04, 0x0c,
        0x0c, 0x14, 0x14, 0x1c, 0x1c, 0x24, 0x2c, 0x2c, 0x34, 0x3e, 0x3e, 0x04, 0x04, 0x04, 0x04, 0x04,
        0x04, 0x0c, 0x0c, 0x14, 0x14, 0x14, 0x1c, 0x1c, 0x1c, 0x1c, 0x24, 0x24, 0x2c, 0x2c, 0x2c, 0x34,
    },
    {
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c,
        0x0c, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14,
        0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x14, 0x1c, 0x1c, 0x1c, 0x1c,
        0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c,
        0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x1c, 0x24, 0x24,
        0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x24,
        0x24, 0x24, 0x24, 0x24, 0x24, 0x24, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c,
        0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x2c, 0x34, 0x34, 0x34, 0x34, 0x34,
        0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x34, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e,
        0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e, 0x3e,
    },
};

// shared by vec_dot and the batched rows, as iq2_xxs_vbmi_impl
static BATCH_INLINE void iq3_xxs_vbmi_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    // weight byte p of 64-weight group j is grid byte p%4 of index 16*j + p/4:
    // bytes 0/1 from the g0/g1 pair, bytes 2/3 from the g2/g3 pair (same selector)
    static const uint8_t k_wsel[64] = {
         0, 64,  0, 64,  1, 65,  1, 65,  2, 66,  2, 66,  3, 67,  3, 67,
         4, 68,  4, 68,  5, 69,  5, 69,  6, 70,  6, 70,  7, 71,  7, 71,
         8, 72,  8, 72,  9, 73,  9, 73, 10, 74, 10, 74, 11, 75, 11, 75,
        12, 76, 12, 76, 13, 77, 13, 77, 14, 78, 14, 78, 15, 79, 15, 79,
    };
    // the 32 sign bytes (at 8*i + k, k < 4) made contiguous
    static const uint8_t k_signs[64] = {
         0,  1,  2,  3,  8,  9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
        32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59,
    };

    const __m512i wsel   = _mm512_loadu_si512(k_wsel);
    const __m512i sperm  = _mm512_loadu_si512(k_signs);
    const __m512i ksgn0  = _mm512_loadu_si512(ksigns_iq2xs);
    const __m512i ksgn1  = _mm512_loadu_si512(ksigns_iq2xs + 64);
    // bytes 0-3 of each qword: bits 7k (the 4 sign indices)
    const __m512i sshift = _mm512_set1_epi64(INT64_C(0x150E0700));
    // scale m sits in the top nibble of int16 lane 4m+1 of aux: lanes 0-15 take scale 0, lanes 16-31 scale 1
    const __m512i sidx   = _mm512_mask_set1_epi16(_mm512_set1_epi16(1), 0xFFFF0000u, 5);
    const __mmask64 khi2 = 0xCCCCCCCCCCCCCCCCull;

    __m512 accumf[4];
    for (int r = 0; r < nr; ++r) accumf[r] = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const __m512i qs = _mm512_loadu_si512(x[i].qs);
        const __mmask64 b7 = _mm512_movepi8_mask(qs);
        const __m512i g0 = vbmi_lookup_256(iq3xxs_grid_byte[0], qs, b7);
        const __m512i g1 = vbmi_lookup_256(iq3xxs_grid_byte[1], qs, b7);
        const __m512i g2 = vbmi_lookup_256(iq3xxs_grid_byte[2], qs, b7);
        const __m512i g3 = vbmi_lookup_256(iq3xxs_grid_byte[3], qs, b7);
        // one qword per 32 weights: 4 7-bit sign indices and a 4-bit scale
        const __m512i aux = _mm512_cvtepu32_epi64(_mm256_loadu_si256((const __m256i *) (x[i].qs + QK_K/4)));
        const __m512i sg = _mm512_permutex2var_epi8(ksgn0, _mm512_multishift_epi64_epi8(sshift, aux), ksgn1);
        __mmask64 signs[4];
        _mm256_storeu_si256((__m256i *) signs, _mm512_castsi512_si256(_mm512_permutexvar_epi8(sperm, sg)));
        // sub-block scales 2*ls+1 as int16 (lanes 4m+1)
        const __m512i sc = _mm512_or_si512(_mm512_slli_epi16(_mm512_srli_epi16(aux, 12), 1), _mm512_set1_epi16(1));
        __m512i sumi[4];
        for (int r = 0; r < nr; ++r) sumi[r] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __m512i wj  = _mm512_add_epi8(wsel, _mm512_set1_epi8(16*j));
            const __m512i q3  = _mm512_mask_blend_epi8(khi2, _mm512_permutex2var_epi8(g0, wj, g1), _mm512_permutex2var_epi8(g2, wj, g3));
            const __m512i scj = _mm512_permutexvar_epi16(_mm512_add_epi16(sidx, _mm512_set1_epi16(8*j)), sc);
            xxs_vbmi_accum(q3, signs[j], scj, sumi, vy, i, j, nr);
        }
        for (int r = 0; r < nr; ++r) {
            const float d = xd * ((const block_q8_K *) vy[r])[i].d;
            accumf[r] = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi[r]), accumf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = 0.25f * _mm512_reduce_add_ps(accumf[r]);
}
#endif

void ggml_vec_dot_iq3_xxs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const block_q8_K    * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)

    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    const void * ys = vy;
    iq3_xxs_vbmi_impl(n, s, vx, &ys, 1);

#elif defined(__AVX2__)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            const __m128i q2_2_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_xxs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)
// iq3s_grid with each grid byte (odd, 1..15) packed into a nibble; generated from g = iq3s_grid[i] as
// lo[i] = (g & 0xf) | ((g >> 8) & 0xf) << 4, hi[i] = ((g >> 16) & 0xf) | ((g >> 24) & 0xf) << 4
static const uint8_t iq3s_grid_nib[2][512] = {
    {
        0x11, 0x13, 0x15, 0x1b, 0x1f, 0x31, 0x33, 0x35, 0x39, 0x3d, 0x51, 0x53, 0x5b, 0x77, 0x91, 0x95,
        0x9b, 0x9f, 0xb3, 0xb7, 0xd1, 0xd5, 0xf3, 0xf9, 0xff, 0x11, 0x13, 0x15, 0x19, 0x31, 0x33, 0x3b,
        0x51, 0x57, 0x5f, 0x73, 0x7b, 0x99, 0xd3, 0xdb, 0xf5, 0x11, 0x13, 0x1b, 0x1f, 0x31, 0x37, 0x3d,
        0x53, 0x5b, 0x71, 0x79, 0x95, 0x9b, 0x9f, 0xb3, 0xb7, 0xf1, 0xf7, 0x17, 0x33, 0x3b, 0x51, 0x55,
        0x73, 0x77, 0x7d, 0x99, 0xb1, 0xb5, 0xdf, 0xf3, 0xfb, 0x11, 0x37, 0x3f, 0x53, 0x59, 0x75, 0x91,
        0x97, 0xb3, 0xf1, 0x15, 0x19, 0x51, 0x55, 0x5d, 0x77, 0x93, 0x9b, 0x9f, 0xdd, 0xf7, 0x1d, 0x33,
        0x37, 0x73, 0xb5, 0xf3, 0x11, 0x15, 0x19, 0x51, 0x55, 0x5d, 0x77, 0xb1, 0xb9, 0x11, 0x13, 0x15,
        0x19, 0x31, 0x33, 0x37, 0x3b, 0x3f, 0x51, 0x55, 0x73, 0x79, 0x7d, 0xb9, 0xbd, 0xd3, 0xf5, 0x11,
        0x13, 0x17, 0x1d, 0x31, 0x39, 0x53, 0x71, 0x77, 0x93, 0xb1, 0xb5, 0xf1, 0xfd, 0x11, 0x35, 0x3b,
        0x3f, 0x51, 0x59, 0x75, 0x91, 0x97, 0xbb, 0xd1, 0xf5, 0x13, 0x19, 0x1f, 0x31, 0x37, 0x53, 0x5f,
        0x71, 0x79, 0x93, 0xd5, 0xf1, 0x17, 0x1b, 0x35, 0x39, 0x73, 0x77, 0x95, 0x9d, 0xb1, 0xb9, 0x13,
        0x31, 0x37, 0x53, 0x71, 0x75, 0xb3, 0x51, 0x59, 0x5f, 0x99, 0x9d, 0x13, 0x17, 0x31, 0x35, 0x53,
        0x7b, 0x93, 0xd5, 0xf1, 0x11, 0x13, 0x17, 0x1b, 0x1f, 0x31, 0x35, 0x39, 0x3d, 0x53, 0x57, 0x5f,
        0x71, 0x75, 0x93, 0x97, 0x9b, 0xb1, 0xb5, 0xdf, 0xf1, 0xf7, 0xfb, 0x11, 0x15, 0x31, 0x37, 0x3f,
        0x55, 0x5b, 0x73, 0x79, 0x95, 0xb3, 0x13, 0x19, 0x1f, 0x53, 0x57, 0x71, 0x7f, 0x93, 0xb7, 0xbf,
        0xf3, 0xf9, 0x11, 0x15, 0x1b, 0x33, 0x55, 0x59, 0x73, 0x77, 0x95, 0xb1, 0xdd, 0x13, 0x1f, 0x51,
        0x57, 0x75, 0x7b, 0x93, 0xf5, 0xfb, 0x19, 0x33, 0x55, 0x7f, 0x91, 0xb7, 0xf1, 0x11, 0x15, 0x1f,
        0x53, 0xbb, 0xd3, 0x1b, 0x33, 0x5d, 0x71, 0x97, 0xb1, 0x15, 0x33, 0x37, 0x3b, 0x3f, 0x55, 0x73,
        0x77, 0x7b, 0x95, 0x99, 0x9f, 0xb3, 0xd7, 0xf3, 0x13, 0x17, 0x1b, 0x39, 0x53, 0x57, 0x91, 0xd1,
        0xf5, 0xfd, 0x11, 0x35, 0x51, 0x75, 0x79, 0xb1, 0x13, 0x31, 0x39, 0x53, 0x57, 0x5f, 0x71, 0x93,
        0x97, 0x9f, 0xbb, 0xf7, 0x17, 0x33, 0x3d, 0x55, 0x73, 0xb5, 0xd1, 0xd9, 0x13, 0x31, 0x35, 0x5b,
        0x75, 0x99, 0xbd, 0xf7, 0x3d, 0x93, 0x13, 0x17, 0x51, 0x55, 0x7b, 0x11, 0x19, 0x35, 0x51, 0x59,
        0x5f, 0x75, 0x93, 0xb1, 0xf1, 0x15, 0x1f, 0x33, 0x37, 0x55, 0x71, 0x7b, 0x97, 0xb3, 0xbb, 0x13,
        0x17, 0x31, 0x3b, 0x53, 0x77, 0x91, 0xbf, 0xd5, 0xf1, 0x19, 0x33, 0x37, 0x51, 0x55, 0x73, 0x7b,
        0x11, 0x15, 0x59, 0x7f, 0x91, 0xf3, 0x1b, 0x1f, 0x53, 0xd5, 0x37, 0x79, 0xd1, 0x31, 0x3b, 0x71,
        0x97, 0xb3, 0x15, 0x31, 0x39, 0x55, 0x91, 0x99, 0x9f, 0xb5, 0xdd, 0xf9, 0x13, 0x17, 0x1b, 0x35,
        0x53, 0x75, 0xf5, 0x11, 0x33, 0x57, 0x71, 0x7d, 0xb7, 0x15, 0x1f, 0x31, 0x5f, 0x99, 0xb3, 0xdb,
        0xf7, 0x13, 0x19, 0x51, 0x75, 0x9d, 0x35, 0x5d, 0xb3, 0xb7, 0x95, 0x15, 0x19, 0x55, 0x33, 0x37,
        0x3b, 0x73, 0x77, 0xd1, 0x11, 0x51, 0x5f, 0xd9, 0x35, 0x79, 0x95, 0xbb, 0xd5, 0xf1, 0x11, 0x39,
        0x53, 0x91, 0x5b, 0x97, 0xd5, 0x11, 0x17, 0x79, 0xd1, 0x1b, 0x91, 0x33, 0x37, 0x11, 0x19, 0x1f,
        0x51, 0x55, 0x7d, 0x91, 0xb9, 0xd5, 0x15, 0x33, 0x59, 0x97, 0x9b, 0x13, 0x19, 0x31, 0x3d, 0x53,
        0x71, 0xb3, 0x15, 0x75, 0x7b, 0xb7, 0x13, 0x1b, 0x37, 0x51, 0xb1, 0x55, 0x95, 0x15, 0x73, 0x11,
    },
    {
        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13,
        0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x13, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15,
        0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x15, 0x17, 0x17, 0x17, 0x17, 0x17,
        0x17, 0x17, 0x17, 0x17, 0x17, 0x17, 0x17, 0x17, 0x17, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19, 0x19,
        0x19, 0x19, 0x19, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1b, 0x1d, 0x1d,
        0x1d, 0x1d, 0x1d, 0x1d, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x31, 0x31, 0x31,
        0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x31, 0x33,
        0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x35, 0x35, 0x35,
        0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x35, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37, 0x37,
        0x37, 0x37, 0x37, 0x37, 0x37, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x39, 0x3b,
        0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3b, 0x3d, 0x3d, 0x3d, 0x3d, 0x3d, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f,
        0x3f, 0x3f, 0x3f, 0x3f, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51,
        0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x51, 0x53, 0x53, 0x53, 0x53, 0x53,
        0x53, 0x53, 0x53, 0x53, 0x53, 0x53, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
        0x55, 0x55, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x57, 0x59, 0x59, 0x59,
        0x59, 0x59, 0x59, 0x59, 0x59, 0x59, 0x5b, 0x5b, 0x5b, 0x5b, 0x5b, 0x5b, 0x5b, 0x5d, 0x5d, 0x5d,
        0x5d, 0x5d, 0x5d, 0x5f, 0x5f, 0x5f, 0x5f, 0x5f, 0x5f, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71,
        0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x71, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73, 0x73,
        0x73, 0x73, 0x75, 0x75, 0x75, 0x75, 0x75, 0x75, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77, 0x77,
        0x77, 0x77, 0x77, 0x77, 0x79, 0x79, 0x79, 0x79, 0x79, 0x79, 0x79, 0x79, 0x7b, 0x7b, 0x7b, 0x7b,
        0x7b, 0x7b, 0x7b, 0x7b, 0x7d, 0x7d, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x91, 0x91, 0x91, 0x91, 0x91,
        0x91, 0x91, 0x91, 0x91, 0x91, 0x93, 0x93, 0x93, 0x93, 0x93, 0x93, 0x93, 0x93, 0x93, 0x93, 0x95,
        0x95, 0x95, 0x95, 0x95, 0x95, 0x95, 0x95, 0x95, 0x95, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97, 0x97,
        0x99, 0x99, 0x99, 0x99, 0x99, 0x99, 0x9b, 0x9b, 0x9b, 0x9b, 0x9d, 0x9d, 0x9d, 0x9f, 0x9f, 0x9f,
        0x9f, 0x9f, 0xb1, 0xb1, 0xb1, 0xb1, 0xb1, 0xb1, 0xb1, 0xb1, 0xb1, 0xb1, 0xb3, 0xb3, 0xb3, 0xb3,
        0xb3, 0xb3, 0xb3, 0xb5, 0xb5, 0xb5, 0xb5, 0xb5, 0xb5, 0xb7, 0xb7, 0xb7, 0xb7, 0xb7, 0xb7, 0xb7,
        0xb7, 0xb9, 0xb9, 0xb9, 0xb9, 0xb9, 0xbb, 0xbb, 0xbb, 0xbb, 0xbd, 0xbf, 0xbf, 0xbf, 0xd1, 0xd1,
        0xd1, 0xd1, 0xd1, 0xd1, 0xd3, 0xd3, 0xd3, 0xd3, 0xd5, 0xd5, 0xd5, 0xd5, 0xd5, 0xd5, 0xd7, 0xd7,
        0xd7, 0xd7, 0xd9, 0xd9, 0xd9, 0xdb, 0xdb, 0xdb, 0xdb, 0xdd, 0xdd, 0xdf, 0xdf, 0xf1, 0xf1, 0xf1,
        0xf1, 0xf1, 0xf1, 0xf1, 0xf1, 0xf1, 0xf3, 0xf3, 0xf3, 0xf3, 0xf3, 0xf5, 0xf5, 0xf5, 0xf5, 0xf5,
        0xf5, 0xf5, 0xf7, 0xf7, 0xf7, 0xf7, 0xf9, 0xf9, 0xf9, 0xf9, 0xf9, 0xfb, 0xfb, 0xfd, 0xfd, 0xff,
    },
};

// look up 64 9-bit indices (low 8 bits in qs, bit 8 in qh) in one 512-entry byte table
static inline __m512i iq3s_lookup_512(const uint8_t * t, __m512i qs, __mmask64 b7, __mmask64 qh) {
    return _mm512_mask_blend_epi8(qh, vbmi_lookup_256(t, qs, b7), vbmi_lookup_256(t + 256, qs, b7));
}
#endif

void ggml_vec_dot_iq3_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)

    // lane 4k+j of 64 weights takes nibble j of index k: bytes (lo, lo, hi, hi)[k], hi table at +64
    static const uint8_t k_expand[64] = {
         0,  0, 64, 64,  1,  1, 65, 65,  2,  2, 66, 66,  3,  3, 67, 67,
         4,  4, 68, 68,  5,  5, 69, 69,  6,  6, 70, 70,  7,  7, 71, 71,
         8,  8, 72, 72,  9,  9, 73, 73, 10, 10, 74, 74, 11, 11, 75, 75,
        12, 12, 76, 76, 13, 13, 77, 77, 14, 14, 78, 78, 15, 15, 79, 79,
    };

    const __m512i expand = _mm512_loadu_si512(k_expand);
    const __m512i m4     = _mm512_set1_epi8(0x0f);
    // int16 lanes 0-15 take scale 0, lanes 16-31 scale 1 (of the 8 broadcast per block)
    const __m512i sidx   = _mm512_maskz_set1_epi16(0xFFFF0000u, 1);

    __m512 accumf = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const __m512i qs = _mm512_loadu_si512(x[i].qs);
        __mmask64 qh, signs[4];
        memcpy(&qh, x[i].qh, sizeof(qh));
        memcpy(signs, x[i].signs, sizeof(signs));
        uint32_t scales;
        memcpy(&scales, x[i].scales, sizeof(scales));
        const __mmask64 b7 = _mm512_movepi8_mask(qs);
        const __m512i lo = iq3s_lookup_512(iq3s_grid_nib[0], qs, b7, qh);
        const __m512i hi = iq3s_lookup_512(iq3s_grid_nib[1], qs, b7, qh);
        // 8 sub-block scales 2*ls+1 as int16 (only indices 0-7 are permuted in)
        const __m128i sb  = _mm_cvtsi32_si128((int) scales);
        const __m128i s8  = _mm_unpacklo_epi8(_mm_and_si128(sb, _mm_set1_epi8(0x0f)), _mm_and_si128(_mm_srli_epi16(sb, 4), _mm_set1_epi8(0x0f)));
        const __m512i sc  = _mm512_castsi128_si512(_mm_or_si128(_mm_slli_epi16(_mm_cvtepu8_epi16(s8), 1), _mm_set1_epi16(1)));
        __m512i sumi = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            const __m512i g   = _mm512_permutex2var_epi8(lo, _mm512_add_epi8(expand, _mm512_set1_epi8(16*j)), hi);
            const __m512i q3  = _mm512_and_si512(_mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAAull, g, _mm512_srli_epi16(g, 4)), m4);
            const __m512i q8  = _mm512_loadu_si512(y[i].qs + 64*j);
            const __m512i q8s = _mm512_mask_sub_epi8(q8, signs[j], _mm512_setzero_si512(), q8);
            const __m512i scj = _mm512_permutexvar_epi16(_mm512_add_epi16(sidx, _mm512_set1_epi16(2*j)), sc);
#if defined(__AVX512VNNI__)
            sumi = _mm512_dpwssd_epi32(sumi, _mm512_maddubs_epi16(q3, q8s), scj);
#else
            sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(_mm512_maddubs_epi16(q3, q8s), scj));
#endif
        }
        accumf = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), accumf);
    }

    *s = _mm512_reduce_add_ps(accumf);

#elif defined(__AVX2__)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = hsum_float_8(accumf);

#elif defined(__AVX__)
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2_0 = _mm_loadu_si128((const __m128i*)k_mask2);
    const __m128i mask2_1 = _mm_loadu_si128((const __m128i*)k_mask2 + 1);

    const __m128i idx_mul_0 = _mm_set_epi32(32, 64, 128, 256);
    const __m128i idx_mul_1 = _mm_set_epi32(2, 4, 8, 16);
    const __m128i idx_mask  = _mm_set1_epi32(256);

    typedef union {
        __m128i  vec[4];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        const int8_t  * GGML_RESTRICT q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i qs_tmp = _mm_loadu_si128((const __m128i *)qs);
            const __m128i idx_l_0 = _mm_cvtepu8_epi16(qs_tmp);
            const __m128i idx_l_1 = _mm_cvtepu8_epi16(_mm_srli_si128(qs_tmp, 8)); qs += 16;
            idx.vec[0] = _mm_set1_epi32(qh[ib32+0]);
            idx.vec[1] = idx.vec[0];
            idx.vec[2] = _mm_set1_epi32(qh[ib32+1]);
            idx.vec[3] = idx.vec[2];

            idx.vec[0] = _mm_and_si128(_mm_mullo_epi32(idx.vec[0], idx_mul_0), idx_mask);
            idx.vec[1] = _mm_and_si128(_mm_mullo_epi32(idx.vec[1], idx_mul_1), idx_mask);
            idx.vec[2] = _mm_and_si128(_mm_mullo_epi32(idx.vec[2], idx_mul_0), idx_mask);
            idx.vec[3] = _mm_and_si128(_mm_mullo_epi32(idx.vec[3], idx_mul_1), idx_mask);

            idx.vec[0] = _mm_or_si128(idx.vec[0], _mm_cvtepi16_epi32(idx_l_0));
            idx.vec[1] = _mm_or_si128(idx.vec[1], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_0, 8)));
            idx.vec[2] = _mm_or_si128(idx.vec[2], _mm_cvtepi16_epi32(idx_l_1));
            idx.vec[3] = _mm_or_si128(idx.vec[3], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_1, 8)));

            const __m128i q2_1_0 = _mm_set_epi32(iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]]);
            const __m128i q2_2_0 = _mm_set_epi32(iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]]);

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | (signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | (signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = hsum_float_8(accumf);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq3_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__BMI2__)
// IQ1 weight times 8 is w = 8*g + delta (g in {-1,0,1}, delta = +-1); |w|*(2s+1) <= 135 fits u8, its sign goes onto q8.
// Index = (g & 3) | (delta < 0) << 2 | s << 3.
static const uint8_t iq1_wmag[64] = {
     1,   9, 0,   7,  1,   7, 0,   9,     3,  27, 0,  21,  3,  21, 0,  27,
     5,  45, 0,  35,  5,  35, 0,  45,     7,  63, 0,  49,  7,  49, 0,  63,
     9,  81, 0,  63,  9,  63, 0,  81,    11,  99, 0,  77, 11,  77, 0,  99,
    13, 117, 0,  91, 13,  91, 0, 117,    15, 135, 0, 105, 15, 105, 0, 135,
};
// sign of w depends only on index bits 0..2, so 16 bytes serve an in-lane shuffle
static const uint8_t iq1_wneg[16] = {
    0, 0, 0, 0x80, 0x80, 0, 0, 0x80,  0, 0, 0, 0x80, 0x80, 0, 0, 0x80,
};

// 64 weights (8 grid qwords) against nr rows of q8; hz carries each byte's delta/scale index bits
static inline void iq1_vnni_accum(const uint16_t * idx, __m512i hz, __m512i * sumi,
                                  const void * const * vy, int i, int k, int nr) {
    const __m512i g = _mm512_set_epi64(iq1s_grid[idx[7]], iq1s_grid[idx[6]], iq1s_grid[idx[5]], iq1s_grid[idx[4]],
                                       iq1s_grid[idx[3]], iq1s_grid[idx[2]], iq1s_grid[idx[1]], iq1s_grid[idx[0]]);
    const __m512i wi = _mm512_or_si512(_mm512_and_si512(g, _mm512_set1_epi8(3)), hz);
    const __m512i wmag = _mm512_permutexvar_epi8(wi, _mm512_loadu_si512(iq1_wmag));
    const __mmask64 wneg = _mm512_movepi8_mask(_mm512_shuffle_epi8(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *) iq1_wneg)), wi));
    for (int r = 0; r < nr; ++r) {
        const __m512i q8 = _mm512_loadu_si512(((const block_q8_K *) vy[r])[i].qs + 64*k);
        sumi[r] = _mm512_dpbusd_epi32(sumi[r], wmag, _mm512_mask_sub_epi8(q8, wneg, _mm512_setzero_si512(), q8));
    }
}

static BATCH_INLINE void iq1_s_vnni_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq1_s * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    // byte b of chunk k reads the index bits of group 2k + b/32, held in byte 2*group
    const __m512i hsel = _mm512_inserti64x4(_mm512_setzero_si512(), _mm256_set1_epi8(2), 1);

    __m512 accf[4];
    for (int r = 0; r < nr; ++r) accf[r] = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;
        // qh >> 12: scale in bits 0..2, delta sign in bit 3
        const __m128i t = _mm_srli_epi16(_mm_loadu_si128((const __m128i *) qh), 12);
        const __m512i hq = _mm512_castsi128_si512(_mm_or_si128(_mm_srli_epi16(_mm_and_si128(t, _mm_set1_epi16(8)), 1),
                                                               _mm_slli_epi16(_mm_and_si128(t, _mm_set1_epi16(7)), 3)));
        __m512i sumi[4];
        for (int r = 0; r < nr; ++r) sumi[r] = _mm512_setzero_si512();
        for (int k = 0; k < 4; ++k) {
            uint64_t packed[2];
            packed[0] = _pdep_u64(*(const uint32_t *)(qs + 8*k + 0), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[2*k + 0], 0x0700070007000700ULL);
            packed[1] = _pdep_u64(*(const uint32_t *)(qs + 8*k + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[2*k + 1], 0x0700070007000700ULL);
            const __m512i sel = _mm512_add_epi8(hsel, _mm512_set1_epi8(4*k));
            iq1_vnni_accum((const uint16_t *) packed, _mm512_permutexvar_epi8(sel, hq), sumi, vy, i, k, nr);
        }
        // the 1/8 of the u8 weights is folded in here (exact), so huge activations don't overflow a final rescale
        const float xd = 0.125f * GGML_CPU_FP16_TO_FP32(x[i].d);
        for (int r = 0; r < nr; ++r) {
            accf[r] = _mm512_fmadd_ps(_mm512_set1_ps(((const block_q8_K *) vy[r])[i].d * xd), _mm512_cvtepi32_ps(sumi[r]), accf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = _mm512_reduce_add_ps(accf[r]);
}

static BATCH_INLINE void iq1_m_vnni_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq1_m * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    // byte k of the multishift = 3-bit scale k: bits 16*(k/4) + 3*(k%4) of the scales word
    const __m128i sshift = _mm_setr_epi8(0, 3, 6, 9, 16, 19, 22, 25, 32, 35, 38, 41, 48, 51, 54, 57);
    const __m128i m4 = _mm_set1_epi8(4);
    // byte b of chunk k reads the index bits of qword 8k + b/8
    const __m512i qsel = _mm512_set_epi8(
        7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
        3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0);

    iq1m_scale_t scale;

    __m512 accf[4];
    for (int r = 0; r < nr; ++r) accf[r] = _mm512_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *) x[i].scales;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        // per-qword index bits: qh byte b holds qwords 2b (low nibble) and 2b+1, delta sign = nibble bit 3, scale b
        const __m128i s3 = _mm_slli_epi16(_mm_and_si128(_mm_multishift_epi64_epi8(sshift, _mm_set1_epi64x(*(const int64_t *) sc)),
                                                        _mm_set1_epi8(7)), 3);
        const __m128i qhv = _mm_loadu_si128((const __m128i *) qh);
        const __m128i hlo = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qhv, 1), m4), s3);
        const __m128i hhi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qhv, 5), m4), s3);
        const __m512i hq = _mm512_castsi256_si512(_mm256_set_m128i(_mm_unpackhi_epi8(hlo, hhi), _mm_unpacklo_epi8(hlo, hhi)));

        __m512i sumi[4];
        for (int r = 0; r < nr; ++r) sumi[r] = _mm512_setzero_si512();
        for (int k = 0; k < 4; ++k) {
            uint64_t packed[2];
            packed[0] = _pdep_u64(*(const uint32_t *)(qs + 8*k + 0), 0x00ff00ff00ff00ffULL)
                      | _pdep_u64(*(const uint16_t *)(qh + 4*k + 0) & 0x7777, 0x0f000f000f000f00ULL);
            packed[1] = _pdep_u64(*(const uint32_t *)(qs + 8*k + 4), 0x00ff00ff00ff00ffULL)
                      | _pdep_u64(*(const uint16_t *)(qh + 4*k + 2) & 0x7777, 0x0f000f000f000f00ULL);
            const __m512i sel = _mm512_add_epi8(qsel, _mm512_set1_epi8(8*k));
            iq1_vnni_accum((const uint16_t *) packed, _mm512_permutexvar_epi8(sel, hq), sumi, vy, i, k, nr);
        }
        const float xd = 0.125f * GGML_CPU_FP16_TO_FP32(scale.f16);
        for (int r = 0; r < nr; ++r) {
            accf[r] = _mm512_fmadd_ps(_mm512_set1_ps(((const block_q8_K *) vy[r])[i].d * xd), _mm512_cvtepi32_ps(sumi[r]), accf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = _mm512_reduce_add_ps(accf[r]);
}
#endif

#if defined(__AVX2__)
// Batched forms of the vec_dots below (nr <= 4 rows per decoded block). The AVX2 arms keep the
// single-row op order, so s[r] is bit-identical to vec_dot; routing vec_dot through them measured
// slower at nr = 1, hence the copies.
static BATCH_INLINE void iq1_s_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__BMI2__)
    iq1_s_vnni_impl(n, s, vx, vy, nr);
#else
    const block_iq1_s * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    __m256 accum[4];
    float accum1[4];
    for (int r = 0; r < nr; ++r) { accum[r] = _mm256_setzero_ps(); accum1[r] = 0; }
    for (int i = 0; i < nb; ++i) {
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;
        __m256i sumi[4];
        int sumi1[4];
        for (int r = 0; r < nr; ++r) { sumi[r] = _mm256_setzero_si256(); sumi1[r] = 0; }
        for (int ib = 0; ib < QK_K/32; ib += 2) {
#ifdef __BMI2__
            const uint64_t packed_idx1 = _pdep_u64(*(const uint32_t *)qs, 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib], 0x700070007000700ULL);
            const uint64_t packed_idx2 = _pdep_u64(*(const uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib + 1], 0x700070007000700ULL);
            const uint16_t *idx1 = (const uint16_t *)(&packed_idx1);
            const uint16_t *idx2 = (const uint16_t *)(&packed_idx2);
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);
#else
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                                    iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)],
                                                    iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
#endif
            qs += 8;
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const int s1 = (qh[ib+0] & 0x8000 ? -1 : 1) * ls1;
            const int s2 = (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
            for (int r = 0; r < nr; ++r) {
                const block_q8_K * GGML_RESTRICT y = (const block_q8_K *) vy[r] + i;
                const int8_t * GGML_RESTRICT q8 = y->qs + 32*ib;
                const __m256i dot1 = mul_add_epi8(q1b_1, _mm256_loadu_si256((const __m256i *)(q8 +  0)));
                const __m256i dot2 = mul_add_epi8(q1b_2, _mm256_loadu_si256((const __m256i *)(q8 + 32)));
                const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
                const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p1, p2));
                sumi1[r] += (y->bsums[2*ib+0] + y->bsums[2*ib+1]) * s1 + (y->bsums[2*ib+2] + y->bsums[2*ib+3]) * s2;
            }
        }
        for (int r = 0; r < nr; ++r) {
            const float d = ((const block_q8_K *) vy[r])[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
            accum[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), accum[r]);
            accum1[r] += d * sumi1[r];
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(accum[r]) + IQ1S_DELTA * accum1[r];
#endif
}

static BATCH_INLINE void iq1_m_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__BMI2__)
    iq1_m_vnni_impl(n, s, vx, vy, nr);
#else
    const block_iq1_m * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    iq1m_scale_t scale;

    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);
    const __m256i mone8 = _mm256_set1_epi8(1);
    const __m256i mtwo8 = _mm256_set1_epi8(2);
    // VPSHUFB cannot cross 128-bit lanes so odd shifts go to upper half.
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);

    __m256 accum1[4], accum2[4];
    for (int r = 0; r < nr; ++r) accum1[r] = accum2[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        // Extract 3-bit scales (16 values)
        __m256i scales = _mm256_set1_epi64x(*(const uint64_t*)sc);
        scales = _mm256_srlv_epi64(scales, scales_shift);
        scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);

        // Indices to repeat each scale 8 times.
        __m256i scales_idx1 = _mm256_set1_epi16(0x0100);
        __m256i scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8));

        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < nr; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
#ifdef __BMI2__
            const uint64_t packed_idx1 = _pdep_u64(*(const uint32_t *)qs, 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(const uint16_t*)(qh) & 0x7777, 0xf000f000f000f00ULL);
            const uint64_t packed_idx2 = _pdep_u64(*(const uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(const uint16_t*)(qh + 2) & 0x7777, 0xf000f000f000f00ULL);
            const uint16_t *idx1 = (const uint16_t *)(&packed_idx1);
            const uint16_t *idx2 = (const uint16_t *)(&packed_idx2);
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);

            // Convert signs to bytes 0x81 (negative) or 0x01 (positive)
            const uint64_t delta_sign = _pdep_u64(*(const uint32_t*)(qh) & 0x88888888, 0xf0f0f0f0f0f0f0f0ULL);
            const __m256i delta1 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign)));
            const __m256i delta2 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign >> 32)));
#else
            const __m256i q1b_1 = _mm256_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]
            );
            const __m256i q1b_2 = _mm256_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]
            );

            const __m256i delta1 = _mm256_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m256i delta2 = _mm256_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
#endif
            const __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
            const __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);
            scales_idx1 = _mm256_add_epi8(scales_idx1, mtwo8);
            scales_idx2 = _mm256_add_epi8(scales_idx2, mtwo8);

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 32*ib;
                const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
                const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
                const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
                const __m256i dot3 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_1, delta1));
                const __m256i dot4 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_2, delta2));
                const __m256i p1 = _mm256_madd_epi16(dot1, scale1);
                const __m256i p2 = _mm256_madd_epi16(dot2, scale2);
                const __m256i p3 = _mm256_madd_epi16(dot3, scale1);
                const __m256i p4 = _mm256_madd_epi16(dot4, scale2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_add_epi32(p1, p2));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_add_epi32(p3, p4));
            }

            qs += 8; qh += 4;
        }

        for (int r = 0; r < nr; ++r) {
            const __m256 d = _mm256_set1_ps(((const block_q8_K *) vy[r])[i].d * GGML_CPU_FP16_TO_FP32(scale.f16));
            accum1[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1[r]), accum1[r]);
            accum2[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2[r]), accum2[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(accum1[r]) + IQ1M_DELTA * hsum_float_8(accum2[r]);
#endif
}
#endif

void ggml_vec_dot_iq1_s_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__BMI2__)
    UNUSED(x); UNUSED(y); UNUSED(nb);
    const void * ys = vy;
    iq1_s_vnni_impl(n, s, vx, &ys, 1);
#elif defined __AVX2__

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
#ifdef __BMI2__
            const uint64_t packed_idx1 = _pdep_u64(*(const uint32_t *)qs, 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib], 0x700070007000700ULL);
            const uint64_t packed_idx2 = _pdep_u64(*(const uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL) | _pdep_u64(qh[ib + 1], 0x700070007000700ULL);
            const uint16_t *idx1 = (const uint16_t *)(&packed_idx1);
            const uint16_t *idx2 = (const uint16_t *)(&packed_idx2);
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);
#else
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                                    iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)],
                                                    iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
#endif
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#elif defined __AVX__
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q1b_1_0 = _mm_set_epi64x(iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m128i q1b_1_1 = _mm_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)]);
            const __m128i q1b_2_0 = _mm_set_epi64x(iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
            const __m128i q1b_2_1 = _mm_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)]);
            qs += 8;
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
            const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
            const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
            const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(ls1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(ls1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(ls2));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(ls2));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_CPU_FP16_TO_FP32(x[i].d);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq1_s_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq1_m_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * GGML_RESTRICT x = vx;
    const block_q8_K  * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

#if defined(__AVX512VBMI__) && defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__BMI2__)
    UNUSED(x); UNUSED(y); UNUSED(nb); UNUSED(scale);
    const void * ys = vy;
    iq1_m_vnni_impl(n, s, vx, &ys, 1);
#elif defined __AVX2__

    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);
    const __m256i mone8 = _mm256_set1_epi8(1);
    const __m256i mtwo8 = _mm256_set1_epi8(2);
    // VPSHUFB cannot cross 128-bit lanes so odd shifts go to upper half.
    const __m256i scales_shift = _mm256_set_epi64x(9, 3, 6, 0);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        // Extract 3-bit scales (16 values)
        __m256i scales = _mm256_set1_epi64x(*(const uint64_t*)sc);
        scales = _mm256_srlv_epi64(scales, scales_shift);
        scales = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scales, mask), 1), mone);

        // Indices to repeat each scale 8 times.
        __m256i scales_idx1 = _mm256_set1_epi16(0x0100);
        __m256i scales_idx2 = _mm256_add_epi8(scales_idx1, _mm256_set1_epi8(8));

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
#ifdef __BMI2__
            const uint64_t packed_idx1 = _pdep_u64(*(const uint32_t *)qs, 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(const uint16_t*)(qh) & 0x7777, 0xf000f000f000f00ULL);
            const uint64_t packed_idx2 = _pdep_u64(*(const uint32_t *)(qs + 4), 0x00ff00ff00ff00ffULL)
                                       | _pdep_u64(*(const uint16_t*)(qh + 2) & 0x7777, 0xf000f000f000f00ULL);
            const uint16_t *idx1 = (const uint16_t *)(&packed_idx1);
            const uint16_t *idx2 = (const uint16_t *)(&packed_idx2);
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[idx1[3]], iq1s_grid[idx1[2]], iq1s_grid[idx1[1]], iq1s_grid[idx1[0]]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[idx2[3]], iq1s_grid[idx2[2]], iq1s_grid[idx2[1]], iq1s_grid[idx2[0]]);

            // Convert signs to bytes 0x81 (negative) or 0x01 (positive)
            const uint64_t delta_sign = _pdep_u64(*(const uint32_t*)(qh) & 0x88888888, 0xf0f0f0f0f0f0f0f0ULL);
            const __m256i delta1 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign)));
            const __m256i delta2 = _mm256_or_si256(mone8, _mm256_cvtepi8_epi64(_mm_set1_epi32(delta_sign >> 32)));
#else
            const __m256i q1b_1 = _mm256_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]
            );
            const __m256i q1b_2 = _mm256_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]
            );

            const __m256i delta1 = _mm256_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m256i delta2 = _mm256_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
#endif
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const __m256i dot3 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_1, delta1));
            const __m256i dot4 = _mm256_maddubs_epi16(mone8, _mm256_sign_epi8(q8b_2, delta2));

            __m256i scale1 = _mm256_shuffle_epi8(scales, scales_idx1);
            __m256i scale2 = _mm256_shuffle_epi8(scales, scales_idx2);

            scales_idx1 = _mm256_add_epi8(scales_idx1, mtwo8);
            scales_idx2 = _mm256_add_epi8(scales_idx2, mtwo8);

            const __m256i p1 = _mm256_madd_epi16(dot1, scale1);
            const __m256i p2 = _mm256_madd_epi16(dot2, scale2);
            const __m256i p3 = _mm256_madd_epi16(dot3, scale1);
            const __m256i p4 = _mm256_madd_epi16(dot4, scale2);

            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p1, p2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p3, p4));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_TO_FP32(scale.f16));

        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#elif defined __AVX__
    const __m128i mask = _mm_set1_epi16(0x7);
    const __m128i mone = _mm_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q1b_1_0 = _mm_set_epi64x(
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]);
            const __m128i q1b_1_1 = _mm_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)]);
            const __m128i q1b_2_0 = _mm_set_epi64x(
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]);
            const __m128i q1b_2_1 = _mm_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)]);
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
            const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
            const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
            const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);

            const __m128i delta1_0 = _mm_set_epi64x(qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta1_1 = _mm_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta2_0 = _mm_set_epi64x(qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta2_1 = _mm_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);

            const __m128i dot3_0 = mul_add_epi8_sse(delta1_0, q8b_1_0);
            const __m128i dot3_1 = mul_add_epi8_sse(delta1_1, q8b_1_1);
            const __m128i dot4_0 = mul_add_epi8_sse(delta2_0, q8b_2_0);
            const __m128i dot4_1 = mul_add_epi8_sse(delta2_1, q8b_2_1);

            __m128i scale1_0 = _mm_set1_epi16(sc[ib/2] >> 0);
            __m128i scale1_1 = _mm_set1_epi16(sc[ib/2] >> 3);
            __m128i scale2_0 = _mm_set1_epi16(sc[ib/2] >> 6);
            __m128i scale2_1 = _mm_set1_epi16(sc[ib/2] >> 9);

            scale1_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_0, mask), 1), mone);
            scale1_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_1, mask), 1), mone);
            scale2_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_0, mask), 1), mone);
            scale2_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_1, mask), 1), mone);
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, scale1_0);
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, scale1_1);
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, scale2_0);
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, scale2_1);
            const __m128i p3_0 = _mm_madd_epi16(dot3_0, scale1_0);
            const __m128i p3_1 = _mm_madd_epi16(dot3_1, scale1_1);
            const __m128i p4_0 = _mm_madd_epi16(dot4_0, scale2_0);
            const __m128i p4_1 = _mm_madd_epi16(dot4_1, scale2_1);

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_add_epi32(p3_0, p4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_add_epi32(p3_1, p4_1));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * GGML_CPU_FP16_TO_FP32(scale.f16));

        accum1 = _mm256_add_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), accum1);
        accum2 = _mm256_add_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi2_1, sumi2_0))), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    UNUSED(scale);
    ggml_vec_dot_iq1_m_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

void ggml_vec_dot_iq4_nl_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * GGML_RESTRICT x = vx;
    const block_q8_0   * GGML_RESTRICT y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

#if defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 0].d)*GGML_CPU_FP16_TO_FP32(x[ib + 0].d)),
                _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 1].d)*GGML_CPU_FP16_TO_FP32(x[ib + 1].d)),
                _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));

        const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);

#endif
    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * GGML_RESTRICT x = vx;
    const block_q8_K   * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

#if defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m256i p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1));
            const __m256i p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2));
            sumi1 = _mm256_add_epi32(p_1, sumi1);
            sumi2 = _mm256_add_epi32(p_2, sumi2);
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
            const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
            const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
            const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
            const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
            const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
            const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
            const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, _mm_set1_epi16(ls1));
            const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, _mm_set1_epi16(ls1));
            const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, _mm_set1_epi16(ls2));
            const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, _mm_set1_epi16(ls2));
            sumi1_0 = _mm_add_epi32(p_1_0, sumi1_0);
            sumi1_1 = _mm_add_epi32(p_1_1, sumi1_1);
            sumi2_0 = _mm_add_epi32(p_2_0, sumi2_0);
            sumi2_1 = _mm_add_epi32(p_2_1, sumi2_1);
        }
        __m128i sumi12_0 = _mm_add_epi32(sumi1_0, sumi2_0);
        __m128i sumi12_1 = _mm_add_epi32(sumi1_1, sumi2_1);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(MM256_SET_M128I(sumi12_1, sumi12_0))), accum);
    }

    *s = hsum_float_8(accum);

#else
    UNUSED(x);
    UNUSED(y);
    UNUSED(nb);
    ggml_vec_dot_iq4_xs_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
#endif
}

#if defined(__AVX2__)
// Batched forms of the AVX2 vec_dots above: each weight sub-block is decoded
// once and dotted against nr <= 4 activation rows. Every row keeps its own
// accumulators in the single-row op order, so s[r] is bit-identical to vec_dot.

static BATCH_INLINE void iq2_xxs_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
#if defined(__AVX512VBMI__) && defined(__AVX512BW__)
    iq2_xxs_vbmi_impl(n, s, vx, vy, nr);
#else
    const block_iq2_xxs * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf[4];
    for (int r = 0; r < nr; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;
        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < nr; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i ls1 = _mm256_set1_epi16(2*(aux32[1] >> 28)+1);
            const __m256i ls2 = _mm256_set1_epi16(2*(aux32[3] >> 28)+1);
            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 32*ib32;
                const __m256i q8s_1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 +  0)), s2_1);
                const __m256i q8s_2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 + 32)), s2_2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), ls1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), ls2));
            }
        }
        for (int r = 0; r < nr; ++r) {
            const float d = xd * ((const block_q8_K *) vy[r])[i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = 0.125f * hsum_float_8(accumf[r]);
#endif
}

static BATCH_INLINE void iq2_xs_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq2_xs * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    const __m256i mone = _mm256_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i *) bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i *) block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i *) block_sign_shuffle_mask_2);
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i *) k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *) &aux_gindex;

    __m256 accumf[4];
    for (int r = 0; r < nr; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint16_t * GGML_RESTRICT q2 = x[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < nr; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {
            const __m256i q2_data = _mm256_loadu_si256((const __m256i *) q2); q2 += 16;
            aux_gindex = _mm256_and_si256(q2_data, m511);

            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            const __m256i q2_1 = _mm256_set_epi64x(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                   iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                   iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            const __m256i q2_3 = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            const __m256i q2_4 = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = MM256_SET_M128I(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = MM256_SET_M128I(full_signs_h, full_signs_h);

            __m256i signs[4];
            signs[0] = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs[1] = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs[2] = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs[3] = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            for (int k = 0; k < 4; ++k) {
                signs[k] = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(signs[k], bit_selector_mask), bit_selector_mask), mone);
            }

            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)));

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 32*ib32;
                const __m256i dot1 = _mm256_maddubs_epi16(q2_1, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 +  0)), signs[0]));
                const __m256i dot2 = _mm256_maddubs_epi16(q2_2, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 + 32)), signs[1]));
                const __m256i dot3 = _mm256_maddubs_epi16(q2_3, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 + 64)), signs[2]));
                const __m256i dot4 = _mm256_maddubs_epi16(q2_4, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 + 96)), signs[3]));
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot1, sc1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot2, sc2));
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot3, sc3));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot4, sc4));
            }
        }
        for (int r = 0; r < nr; ++r) {
            const float d = xd * ((const block_q8_K *) vy[r])[i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = 0.125f * hsum_float_8(accumf[r]);
}

static BATCH_INLINE void iq2_s_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq2_s * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);

    uint64_t aux64;

    __m256 accumf[4];
    for (int r = 0; r < nr; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)(x[i].qs + QK_K/8);

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8);

        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < nr; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            signs += 4;

            const __m256i sc1 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0));
            const __m256i sc2 = _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1));

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 32*ib32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), sc1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), sc2));
            }
        }
        for (int r = 0; r < nr; ++r) {
            const float d = xd * ((const block_q8_K *) vy[r])[i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = 0.125f * hsum_float_8(accumf[r]);
}

static BATCH_INLINE void iq4_nl_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq4_nl * GGML_RESTRICT x = vx;
    const int nb = n / QK4_NL;

    const __m128i values128 = _mm_loadu_si128((const __m128i *) kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    int ib = 0;
    __m256 accum1[4], accum2[4];
    for (int r = 0; r < nr; ++r) accum1[r] = accum2[r] = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *) x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *) x[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        // mul_add_epi8 split: |w| is shared, sign(w) is applied to each row
        const __m256i ax_1 = _mm256_sign_epi8(q4b_1, q4b_1);
        const __m256i ax_2 = _mm256_sign_epi8(q4b_2, q4b_2);
        const float xd_1 = GGML_CPU_FP16_TO_FP32(x[ib + 0].d);
        const float xd_2 = GGML_CPU_FP16_TO_FP32(x[ib + 1].d);
        for (int r = 0; r < nr; ++r) {
            const block_q8_0 * GGML_RESTRICT y = vy[r];
            const __m256i sy_1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) y[ib + 0].qs), q4b_1);
            const __m256i sy_2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *) y[ib + 1].qs), q4b_2);
            const __m256i p_1 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_1, sy_1), mone);
            const __m256i p_2 = _mm256_madd_epi16(_mm256_maddubs_epi16(ax_2, sy_2), mone);
            accum1[r] = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 0].d)*xd_1), _mm256_cvtepi32_ps(p_1), accum1[r]);
            accum2[r] = _mm256_fmadd_ps(_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 1].d)*xd_2), _mm256_cvtepi32_ps(p_2), accum2[r]);
        }
    }
    for (int r = 0; r < nr; ++r) {
        const block_q8_0 * GGML_RESTRICT y = vy[r];
        float sumf = hsum_float_8(_mm256_add_ps(accum1[r], accum2[r]));
        for (int j = ib; j < nb; ++j) {
            const float d = GGML_CPU_FP16_TO_FP32(y[j].d)*GGML_CPU_FP16_TO_FP32(x[j].d);
            int sumi1 = 0, sumi2 = 0;
            for (int k = 0; k < QK4_NL/2; ++k) {
                sumi1 += y[j].qs[k+       0] * kvalues_iq4nl[x[j].qs[k] & 0xf];
                sumi2 += y[j].qs[k+QK4_NL/2] * kvalues_iq4nl[x[j].qs[k] >>  4];
            }
            sumf += d * (sumi1 + sumi2);
        }
        s[r] = sumf;
    }
}

static BATCH_INLINE void iq3_xxs_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
#if defined(__AVX512VBMI__) && defined(__AVX512BW__)
    iq3_xxs_vbmi_impl(n, s, vx, vy, nr);
#else
    const block_iq3_xxs * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf[4];
    for (int r = 0; r < nr; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;
        const uint8_t * GGML_RESTRICT gas = x[i].qs + QK_K/4;
        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < nr; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i ls1 = _mm256_set1_epi16(2*(aux32[0] >> 28)+1);
            const __m256i ls2 = _mm256_set1_epi16(2*(aux32[1] >> 28)+1);
            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 32*ib32;
                const __m256i q8s_1 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 +  0)), s2_1);
                const __m256i q8s_2 = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(q8 + 32)), s2_2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), ls1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), ls2));
            }
        }
        for (int r = 0; r < nr; ++r) {
            const float d = xd * ((const block_q8_K *) vy[r])[i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = 0.25f * hsum_float_8(accumf[r]);
#endif
}

static BATCH_INLINE void iq3_s_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq3_s * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };
    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    const __m256i mask1 = _mm256_loadu_si256((const __m256i *) k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i *) k_mask2);
    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;
    index_t idx;

    __m256 accumf[4];
    for (int r = 0; r < nr; ++r) accumf[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT qs = x[i].qs;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const uint16_t * GGML_RESTRICT signs = (const uint16_t *)x[i].signs;
        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < nr; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256, mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            signs += 4;

            const __m256i ls1 = _mm256_set1_epi16(2*(x[i].scales[ib32/2] & 0xf)+1);
            const __m256i ls2 = _mm256_set1_epi16(2*(x[i].scales[ib32/2] >>  4)+1);
            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 32*ib32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);
                const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_1, q8s_1), ls1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(_mm256_maddubs_epi16(q2_2, q8s_2), ls2));
            }
        }
        for (int r = 0; r < nr; ++r) {
            const float d = xd * ((const block_q8_K *) vy[r])[i].d;
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(accumf[r]);
}

static BATCH_INLINE void q4_K_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_q4_K * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc[4];
    __m128 acc_m[4];
    for (int r = 0; r < nr; ++r) {
        acc[r] = _mm256_setzero_ps();
        acc_m[r] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float xdmin = GGML_CPU_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi[4];
        for (int r = 0; r < nr; ++r) {
            const block_q8_K * y = &((const block_q8_K *) vy[r])[i];
            const __m256i q8sums = _mm256_loadu_si256((const __m256i *) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[r] = _mm_fmadd_ps(_mm_set1_ps(-y->d * xdmin), _mm_cvtepi32_ps(prod), acc_m[r]);
            sumi[r] = _mm256_setzero_si256();
        }

        const uint8_t * GGML_RESTRICT q4 = x[i].qs;
        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i *) q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 64*j;
                const __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, _mm256_loadu_si256((const __m256i *)(q8 +  0))));
                const __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, _mm256_loadu_si256((const __m256i *)(q8 + 32))));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16l, p16h));
            }
        }

        for (int r = 0; r < nr; ++r) {
            const float d = ((const block_q8_K *) vy[r])[i].d * xd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for (int r = 0; r < nr; ++r) {
        __m128 m = _mm_add_ps(acc_m[r], _mm_movehl_ps(acc_m[r], acc_m[r]));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        s[r] = hsum_float_8(acc[r]) + _mm_cvtss_f32(m);
    }
}

static BATCH_INLINE void q3_K_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_q3_K * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    uint32_t aux[3];

    __m256 acc[4];
    for (int r = 0; r < nr; ++r) acc[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const uint8_t * GGML_RESTRICT q3 = x[i].qs;

        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

        __m256i sumi[4];
        for (int r = 0; r < nr; ++r) sumi[r] = _mm256_setzero_si256();

        int bit = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i sc_0 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0));
            const __m256i sc_1 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1));
            const __m256i sc_2 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2));
            const __m256i sc_3 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3));

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 128*j;
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 +  0));
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 + 64));
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 + 96));

                const __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
                const __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
                const __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
                const __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

                __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
                __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
                __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

                p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                p16_0 = _mm256_madd_epi16(sc_0, p16_0);
                p16_1 = _mm256_madd_epi16(sc_1, p16_1);
                p16_2 = _mm256_madd_epi16(sc_2, p16_2);
                p16_3 = _mm256_madd_epi16(sc_3, p16_3);

                p16_0 = _mm256_add_epi32(p16_0, p16_1);
                p16_2 = _mm256_add_epi32(p16_2, p16_3);
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_2));
            }
        }

        for (int r = 0; r < nr; ++r) {
            const float d = ((const block_q8_K *) vy[r])[i].d * xd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(acc[r]);
}

static BATCH_INLINE void q5_K_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_q5_K * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc[4];
    float summs[4];
    for (int r = 0; r < nr; ++r) {
        acc[r] = _mm256_setzero_ps();
        summs[r] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q5 = x[i].qs;

        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float xdmin = GGML_CPU_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);

        __m256i sumi[4];
        for (int r = 0; r < nr; ++r) {
            const block_q8_K * y = &((const block_q8_K *) vy[r])[i];
            const float dmin = -y->d * xdmin;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[r] += dmin * _mm_extract_epi32(hsum, 0);
            sumi[r] = _mm256_setzero_si256();
        }

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        int bit = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 64*j;
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 +  0));
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));

                __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

                p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                p16_1 = _mm256_madd_epi16(scale_1, p16_1);

                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_1));
            }
        }

        for (int r = 0; r < nr; ++r) {
            const float d = ((const block_q8_K *) vy[r])[i].d * xd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(acc[r]) + summs[r];
}

static BATCH_INLINE void q6_K_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_q6_K * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m15 = _mm256_set1_epi8(15);

    __m256 acc[4];
    for (int r = 0; r < nr; ++r) acc[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;

        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);

        __m256i q8sclsub[4], sumi[4];
        for (int r = 0; r < nr; ++r) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*)((const block_q8_K *) vy[r])[i].bsums);
            q8sclsub[r] = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            sumi[r] = _mm256_setzero_si256();
        }

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(12)), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, _mm256_set1_epi8(48));
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, _mm256_set1_epi8(-64)), 2);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            const __m256i sc_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 0)));
            const __m256i sc_1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 1)));
            const __m256i sc_2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 2)));
            const __m256i sc_3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 3)));
            is += 4;

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 128*j;
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 +  0));
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 + 64));
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 + 96));

                __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                p16_0 = _mm256_madd_epi16(sc_0, p16_0);
                p16_1 = _mm256_madd_epi16(sc_1, p16_1);
                p16_2 = _mm256_madd_epi16(sc_2, p16_2);
                p16_3 = _mm256_madd_epi16(sc_3, p16_3);

                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_1));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_2, p16_3));
            }
        }

        for (int r = 0; r < nr; ++r) {
            const float d = ((const block_q8_K *) vy[r])[i].d * xd;
            sumi[r] = _mm256_sub_epi32(sumi[r], q8sclsub[r]);
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(acc[r]);
}

static BATCH_INLINE void q2_K_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_q2_K * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc[4];
    for (int r = 0; r < nr; ++r) acc[r] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[i].d);
        const float xdmin = GGML_CPU_FP16_TO_FP32(x[i].dmin);

        const uint8_t * GGML_RESTRICT q2 = x[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);

        __m256i sumi[4];
        for (int r = 0; r < nr; ++r) {
            const block_q8_K * y = &((const block_q8_K *) vy[r])[i];
            const float dmin = -y->d * xdmin;
            const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y->bsums));
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(prod), acc[r]);
            sumi[r] = _mm256_setzero_si256();
        }

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        for (int j = 0; j < QK_K/128; ++j) {
            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            const __m256i sc_0 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0));
            const __m256i sc_1 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1));
            const __m256i sc_2 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2));
            const __m256i sc_3 = _mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3));

            for (int r = 0; r < nr; ++r) {
                const int8_t * GGML_RESTRICT q8 = ((const block_q8_K *) vy[r])[i].qs + 128*j;
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)(q8 +  0));
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)(q8 + 32));
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)(q8 + 64));
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)(q8 + 96));

                __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

                p0 = _mm256_madd_epi16(sc_0, p0);
                p1 = _mm256_madd_epi16(sc_1, p1);
                p2 = _mm256_madd_epi16(sc_2, p2);
                p3 = _mm256_madd_epi16(sc_3, p3);

                p0 = _mm256_add_epi32(p0, p1);
                p2 = _mm256_add_epi32(p2, p3);

                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p0, p2));
            }
        }

        for (int r = 0; r < nr; ++r) {
            const float d = ((const block_q8_K *) vy[r])[i].d * xd;
            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(acc[r]);
}

static BATCH_INLINE void iq4_xs_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_iq4_xs * GGML_RESTRICT x = vx;
    const int nb = n / QK_K;

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum[4];
    for (int r = 0; r < nr; ++r) accum[r] = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < nr; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            // mul_add_epi8 split: |x| once, sign(y, x) per row
            const __m256i ax_1 = _mm256_sign_epi8(q4b_1, q4b_1);
            const __m256i ax_2 = _mm256_sign_epi8(q4b_2, q4b_2);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m256i vls1 = _mm256_set1_epi16(ls1);
            const __m256i vls2 = _mm256_set1_epi16(ls2);
            for (int r = 0; r < nr; ++r) {
                const int8_t * q8 = ((const block_q8_K *) vy[r])[ibl].qs + 32*ib;
                const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)(q8 +  0));
                const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)(q8 + 32));
                const __m256i p16_1 = _mm256_maddubs_epi16(ax_1, _mm256_sign_epi8(q8b_1, q4b_1));
                const __m256i p16_2 = _mm256_maddubs_epi16(ax_2, _mm256_sign_epi8(q8b_2, q4b_2));
                const __m256i p_1 = _mm256_madd_epi16(p16_1, vls1);
                const __m256i p_2 = _mm256_madd_epi16(p16_2, vls2);
                sumi1[r] = _mm256_add_epi32(p_1, sumi1[r]);
                sumi2[r] = _mm256_add_epi32(p_2, sumi2[r]);
            }
        }
        const float xd = GGML_CPU_FP16_TO_FP32(x[ibl].d);
        for (int r = 0; r < nr; ++r) {
            accum[r] = _mm256_fmadd_ps(_mm256_set1_ps(xd*((const block_q8_K *) vy[r])[ibl].d),
                    _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accum[r]);
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(accum[r]);
}

static BATCH_INLINE void q4_0_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_q4_0 * GGML_RESTRICT x = vx;
    const int nb = n / QK8_0;

    const __m256i off = _mm256_set1_epi8( 8 );

    __m256 acc[4];
    for (int r = 0; r < nr; ++r) acc[r] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[ib].d);
        const __m256i qx = _mm256_sub_epi8(bytes_from_nibbles_32(x[ib].qs), off);
        for (int r = 0; r < nr; ++r) {
            const block_q8_0 * y = &((const block_q8_0 *) vy[r])[ib];
            const __m256 d = _mm256_set1_ps( xd * GGML_CPU_FP16_TO_FP32(y->d) );
            const __m256i qy = _mm256_loadu_si256((const __m256i *)y->qs);
            const __m256 q = mul_sum_i8_pairs_float(qx, qy);
            acc[r] = _mm256_fmadd_ps( d, q, acc[r] );
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(acc[r]);
}

static BATCH_INLINE void q8_0_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_q8_0 * GGML_RESTRICT x = vx;
    const int nb = n / QK8_0;

    __m256 acc[4];
    for (int r = 0; r < nr; ++r) acc[r] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const float xd = GGML_CPU_FP16_TO_FP32(x[ib].d);
        const __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        for (int r = 0; r < nr; ++r) {
            const block_q8_0 * y = &((const block_q8_0 *) vy[r])[ib];
            const __m256 d = _mm256_set1_ps( xd * GGML_CPU_FP16_TO_FP32(y->d) );
            const __m256i qy = _mm256_loadu_si256((const __m256i *)y->qs);
            const __m256 q = mul_sum_i8_pairs_float(qx, qy);
            acc[r] = _mm256_fmadd_ps( d, q, acc[r] );
        }
    }
    for (int r = 0; r < nr; ++r) s[r] = hsum_float_8(acc[r]);
}

static BATCH_INLINE void mxfp4_batch_impl(int n, float * s, const void * vx, const void * const * vy, int nr) {
    const block_mxfp4 * GGML_RESTRICT x = vx;
    const int nb = n / QK_MXFP4;

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1[4], accum2[4];
    for (int r = 0; r < nr; ++r) accum1[r] = accum2[r] = _mm256_setzero_ps();

    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i ax_1 = _mm256_sign_epi8(q4b_1, q4b_1);
        const __m256i ax_2 = _mm256_sign_epi8(q4b_2, q4b_2);
        const float xe0 = GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 0].e);
        const float xe1 = GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e);
        for (int r = 0; r < nr; ++r) {
            const block_q8_0 * y = &((const block_q8_0 *) vy[r])[ib];
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[0].qs);
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[1].qs);
            const __m256i p16_1 = _mm256_maddubs_epi16(ax_1, _mm256_sign_epi8(q8b_1, q4b_1));
            const __m256i p16_2 = _mm256_maddubs_epi16(ax_2, _mm256_sign_epi8(q8b_2, q4b_2));
            const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
            const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
            const __m256 scale0 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[0].d)*xe0);
            const __m256 scale1 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[1].d)*xe1);
            accum1[r] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1[r]);
            accum2[r] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(p_2), accum2[r]);
        }
    }

    for (int r = 0; r < nr; ++r) {
        float sumf = hsum_float_8(_mm256_add_ps(accum1[r], accum2[r]));
        for (int ibt = ib; ibt < nb; ++ibt) {
            const block_q8_0 * y = &((const block_q8_0 *) vy[r])[ibt];
            const float d = GGML_CPU_FP16_TO_FP32(y->d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ibt].e);
            int sumi1 = 0;
            int sumi2 = 0;
            for (int j = 0; j < QK_MXFP4/2; ++j) {
                sumi1 += y->qs[j +          0] * kvalues_fp4[x[ibt].qs[j] & 0xf];
                sumi2 += y->qs[j + QK_MXFP4/2] * kvalues_fp4[x[ibt].qs[j] >>  4];
            }
            sumf += d * (sumi1 + sumi2);
        }
        s[r] = sumf;
    }
}

// A one-token matvec streams cold weight rows; the hardware prefetcher stops at every 4 KiB page,
// so pull a row a few rows ahead into cache while the current one is computed.
// Runtime distance for tuning (GGML_PF_ROWS, 0 = off).
static int vec_dot_prefetch_rows_get(void) {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("GGML_PF_ROWS");
        v = e ? atoi(e) : 2;
    }
    return v;
}
static inline void vec_dot_prefetch_row(const char * base, int64_t row, size_t size) {
    const int ahead = vec_dot_prefetch_rows_get();
    if (ahead == 0) {
        return;
    }
    const char * p = base + (row + ahead) * size;
    for (size_t off = 0; off < size; off += 64) {
        _mm_prefetch(p + off, _MM_HINT_T0);
    }
}

// Literal row counts let the compiler keep each row's accumulators in registers;
// a single row goes straight to vec_dot.
#define VEC_DOT_BATCH_ROWS(NAME, NR) \
    for (int64_t row = 0; row < rows; ++row) { \
        float sums[4]; \
        vec_dot_prefetch_row((const char *) vx, row, stride); \
        NAME##_batch_impl(n, sums, (const char *) vx + row * stride, vy, NR); \
        for (int r = 0; r < NR; ++r) dst[r][row] = sums[r]; \
    }
#define VEC_DOT_BATCH_DEFINE(NAME, VEC_DOT_TYPE) \
    static void NAME##_rows(int n, float * const * dst, const void * vx, size_t stride, \
                            const void * const * vy, int nr, int64_t rows) { \
        switch (nr) { \
            case 1: \
                for (int64_t row = 0; row < rows; ++row) { \
                    vec_dot_prefetch_row((const char *) vx, row, stride); \
                    ggml_vec_dot_##NAME##_##VEC_DOT_TYPE(n, dst[0] + row, 0, (const char *) vx + row * stride, 0, vy[0], 0, 1); \
                } \
                break; \
            case 2: VEC_DOT_BATCH_ROWS(NAME, 2) break; \
            case 3: VEC_DOT_BATCH_ROWS(NAME, 3) break; \
            case 4: VEC_DOT_BATCH_ROWS(NAME, 4) break; \
            default: assert(false); \
        } \
    }
VEC_DOT_BATCH_DEFINE(q4_0,    q8_0)
VEC_DOT_BATCH_DEFINE(q8_0,    q8_0)
VEC_DOT_BATCH_DEFINE(q2_K,    q8_K)
VEC_DOT_BATCH_DEFINE(q3_K,    q8_K)
VEC_DOT_BATCH_DEFINE(q4_K,    q8_K)
VEC_DOT_BATCH_DEFINE(q5_K,    q8_K)
VEC_DOT_BATCH_DEFINE(q6_K,    q8_K)
VEC_DOT_BATCH_DEFINE(iq2_xxs, q8_K)
VEC_DOT_BATCH_DEFINE(iq2_xs,  q8_K)
VEC_DOT_BATCH_DEFINE(iq3_xxs, q8_K)
VEC_DOT_BATCH_DEFINE(iq1_s,   q8_K)
VEC_DOT_BATCH_DEFINE(iq1_m,   q8_K)
VEC_DOT_BATCH_DEFINE(iq4_nl,  q8_0)
VEC_DOT_BATCH_DEFINE(iq3_s,   q8_K)
VEC_DOT_BATCH_DEFINE(iq2_s,   q8_K)
VEC_DOT_BATCH_DEFINE(iq4_xs,  q8_K)
VEC_DOT_BATCH_DEFINE(mxfp4,   q8_0)
#undef VEC_DOT_BATCH_DEFINE
#undef VEC_DOT_BATCH_ROWS
#undef BATCH_INLINE

ggml_vec_dot_batch_rows_t ggml_get_vec_dot_batch_rows(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:      return q4_0_rows;
        case GGML_TYPE_Q8_0:      return q8_0_rows;
        case GGML_TYPE_Q2_K:      return q2_K_rows;
        case GGML_TYPE_Q3_K:      return q3_K_rows;
        case GGML_TYPE_Q4_K:      return q4_K_rows;
        case GGML_TYPE_Q5_K:      return q5_K_rows;
        case GGML_TYPE_Q6_K:      return q6_K_rows;
        case GGML_TYPE_IQ2_XXS:   return iq2_xxs_rows;
        case GGML_TYPE_IQ2_XS:    return iq2_xs_rows;
        case GGML_TYPE_IQ3_XXS:   return iq3_xxs_rows;
        case GGML_TYPE_IQ1_S:     return iq1_s_rows;
        case GGML_TYPE_IQ1_M:     return iq1_m_rows;
        case GGML_TYPE_IQ4_NL:    return iq4_nl_rows;
        case GGML_TYPE_IQ3_S:     return iq3_s_rows;
        case GGML_TYPE_IQ2_S:     return iq2_s_rows;
        case GGML_TYPE_IQ4_XS:    return iq4_xs_rows;
        case GGML_TYPE_MXFP4:     return mxfp4_rows;
        case GGML_TYPE_Q2_0:      return q2_0_rows;
        case GGML_TYPE_Q2_0_G128: return q2_0_g128_rows;
        default:                  return NULL;
    }
}
#endif
