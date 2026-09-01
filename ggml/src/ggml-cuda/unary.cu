#include "unary.cuh"

#if !defined(GGML_USE_HIP)
static __global__ void fp8_static_fake_quant_kernel(
        const float * src, float * dst, const float * scale, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float d = scale[0];
    const __nv_fp8_e4m3 q(src[i] / d);
    dst[i] = float(q) * d;
}

static __global__ void fp8_dynamic_fake_quant_kernel(
        const float * src, float * dst, const int32_t * marker, int64_t width) {
    const int64_t row = blockIdx.x;
    const float * row_src = src + row * width;
    float * row_dst = dst + row * width;
    float local_max = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_src[col]));
    }
    __shared__ float maxima[256];
    maxima[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float absmax = maxima[0];
    if (marker != nullptr) {
        const float upper_bound = __int_as_float(marker[0]);
        if (upper_bound > 0.0f) {
            absmax = fminf(absmax, upper_bound);
        }
    }
    const float d = absmax / 448.0f;
    const float inverse = absmax == 0.0f ? 0.0f : 1.0f / d;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const __nv_fp8_e4m3 q(row_src[col] * inverse);
        row_dst[col] = float(q) * d;
    }
}

static __global__ void int8_static_fake_quant_kernel(
        const float * src, float * dst, const float * scale, int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float d = scale[0];
    const float q = nearbyintf(fminf(127.0f, fmaxf(-128.0f, src[i] / d)));
    dst[i] = q * d;
}

static __global__ void int8_dynamic_fake_quant_kernel(
        const float * src, float * dst, int64_t width) {
    const int64_t row = blockIdx.x;
    const float * row_src = src + row * width;
    float * row_dst = dst + row * width;
    float local_max = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(row_src[col]));
    }
    __shared__ float maxima[256];
    maxima[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float d = maxima[0] / 127.0f;
    const float inverse = maxima[0] == 0.0f ? 0.0f : 1.0f / d;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float q = nearbyintf(fminf(127.0f, fmaxf(-128.0f, row_src[col] * inverse)));
        row_dst[col] = q * d;
    }
}

static __device__ __forceinline__ float bnb_block_scale(
        const ggml_bnb_scale_header * header, uint32_t block) {
    const uint8_t * bundle = reinterpret_cast<const uint8_t *>(header);
    if (header->nested_block_size == 0) {
        return reinterpret_cast<const float *>(bundle + header->absmax_offset)[block];
    }
    const uint8_t code = bundle[header->absmax_offset + block];
    const float nested_absmax =
        reinterpret_cast<const float *>(bundle + header->nested_absmax_offset)[block / header->nested_block_size];
    const float nested_value =
        reinterpret_cast<const float *>(bundle + header->nested_quant_map_offset)[code];
    return nested_value * nested_absmax + header->nested_offset;
}

static __global__ void bnb4_mul_mat_kernel(
        const uint8_t * weight,
        const ggml_bnb_scale_header * header,
        const float * input,
        float * dst,
        int64_t k,
        int64_t n) {
    const int64_t row = blockIdx.x;
    const int64_t token = blockIdx.y;
    const uint8_t * packed = weight + row * (k / 2);
    const float * x = input + token * k;
    const float * codebook = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(header) + header->quant_map_offset);
    float partial = 0.0f;
    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        const uint8_t byte = packed[col / 2];
        const uint8_t code = (col & 1) ? (byte & 0x0f) : (byte >> 4);
        const uint32_t scale_block = (uint64_t(row) * k + col) / header->block_size;
        // Match BitsAndBytes' declared BF16 dequantization dtype before the
        // FP32 accumulation performed by this correctness-first kernel.
        const nv_bfloat16 value = __float2bfloat16(
            codebook[code] * bnb_block_scale(header, scale_block));
        partial += __bfloat162float(value) * x[col];
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dst[token * n + row] = sums[0];
    }
}

static __device__ __forceinline__ float gptq_ao_scale(
        const ggml_gptq_ao_header * header, uint32_t group, uint32_t row) {
    const uint8_t * bundle = reinterpret_cast<const uint8_t *>(header);
    const uint16_t bits = reinterpret_cast<const uint16_t *>(
        bundle + header->scales_offset)[uint64_t(group) * header->rows + row];
    return header->scale_type == 1 ?
        __bfloat162float(*reinterpret_cast<const nv_bfloat16 *>(&bits)) :
        __half2float(*reinterpret_cast<const half *>(&bits));
}

static __global__ void gptq_ao_mul_mat_kernel(
        const uint8_t * weight,
        const ggml_gptq_ao_header * header,
        const float * input,
        float * dst,
        int64_t k,
        int64_t n) {
    const uint32_t row = blockIdx.x;
    const uint32_t token = blockIdx.y;
    const uint8_t * bundle = reinterpret_cast<const uint8_t *>(header);
    const uint8_t * zeros = bundle + header->zeros_offset;
    const uint16_t * g_idx = reinterpret_cast<const uint16_t *>(bundle + header->g_idx_offset);
    const float * x = input + uint64_t(token) * k;
    float partial = 0.0f;
    for (uint32_t col = threadIdx.x; col < uint32_t(k); col += blockDim.x) {
        const uint32_t word = reinterpret_cast<const uint32_t *>(weight)[uint64_t(col / 8) * n + row];
        const uint8_t code = (word >> (4 * (col % 8))) & 0x0f;
        const uint16_t group = g_idx[col];
        const uint8_t zero = zeros[uint64_t(group) * n + row];
        partial += (float(code) - zero) * gptq_ao_scale(header, group, row) * x[col];
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dst[uint64_t(token) * n + row] = sums[0];
    }
}

static __device__ __forceinline__ float nearest_e2m1(float value) {
    constexpr float magnitudes[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const float magnitude = fabsf(value);
    float best = magnitudes[0];
    float best_error = magnitude;
#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float error = fabsf(magnitude - magnitudes[i]);
        if (error < best_error) {
            best = magnitudes[i];
            best_error = error;
        }
    }
    return copysignf(best, value);
}

static __global__ void mxfp4_dynamic_fake_quant_kernel(
        const float * src, float * dst, int64_t n) {
    constexpr int group_size = 32;
    const int64_t group = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t index = group * group_size + lane;
    const float value = index < n ? src[index] : 0.0f;
    float amax = fabsf(value);
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    if (index < n) {
        const int exponent = amax > 0.0f ?
            max(0, min(254, __float2int_rn(log2f(amax)) - 2 + 127)) : 0;
        const float scale = ldexpf(1.0f, exponent - 127);
        dst[index] = nearest_e2m1(value / scale) * scale;
    }
}

static __global__ void mxfp8_dynamic_fake_quant_kernel(
        const float * src, float * dst, int64_t n) {
    constexpr int group_size = 32;
    const int64_t group = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t index = group * group_size + lane;
    const float value = index < n ? src[index] : 0.0f;
    float amax = fabsf(value);
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    if (index < n) {
        const int exponent = amax > 0.0f ?
            max(0, min(254, __float2int_ru(log2f(amax / 448.0f)) + 127)) : 0;
        const float scale = ldexpf(1.0f, exponent - 127);
        const __nv_fp8_e4m3 quantized(value / scale);
        dst[index] = static_cast<float>(quantized) * scale;
    }
}

static __global__ void fp8_group_dynamic_fake_quant_kernel(
        const float * src, float * dst, int64_t n) {
    constexpr int group_size = 32;
    const int64_t index = int64_t(blockIdx.x) * group_size + threadIdx.x;
    const float value = index < n ? src[index] : 0.0f;
    float amax = fabsf(value);
    for (int offset = group_size / 2; offset > 0; offset /= 2) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
    }
    if (index < n) {
        const nv_bfloat16 scale_bf16 = __float2bfloat16(amax / 448.0f);
        const float scale = __bfloat162float(scale_bf16);
        const __nv_fp8_e4m3 quantized(scale == 0.0f ? 0.0f : value / scale);
        dst[index] = static_cast<float>(quantized) * scale;
    }
}

static __global__ void mxfp8_mul_mat_kernel(
        const uint8_t * weight,
        const uint8_t * weight_scale,
        const float * input,
        float * dst,
        int64_t k,
        int64_t n) {
    const int64_t row = blockIdx.x;
    const int64_t token = blockIdx.y;
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        const float w = ggml_cuda_e4m3_to_fp32(weight[row * k + col]);
        const float scale = ggml_cuda_e8m0_to_fp32(weight_scale[row * (k / 32) + col / 32]);
        sum += w * scale * input[token * k + col];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < blockDim.x / 32 ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            dst[token * n + row] = sum;
        }
    }
}

static __global__ void fp8_group_mul_mat_kernel(
        const uint8_t * weight,
        const nv_bfloat16 * weight_scale,
        const float * input,
        float * dst,
        int64_t k,
        int64_t n) {
    const int64_t row = blockIdx.x;
    const int64_t token = blockIdx.y;
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        const float w = ggml_cuda_e4m3_to_fp32(weight[row * k + col]);
        const float scale = __bfloat162float(weight_scale[row * (k / 32) + col / 32]);
        sum += w * scale * input[token * k + col];
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) {
        warp_sums[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < blockDim.x / 32 ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            dst[token * n + row] = sum;
        }
    }
}
#endif

void ggml_cuda_fp8_static_fake_quant(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        const ggml_tensor * scale,
        float * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src);
    GGML_UNUSED(scale);
    GGML_UNUSED(dst);
    GGML_ABORT("static FP8 activation quantization is not implemented for HIP");
#else
    GGML_ASSERT(src->type == GGML_TYPE_F32 && ggml_is_contiguous(src));
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale));
    GGML_ASSERT(ggml_nelements(scale) == 1);
    const int64_t n = ggml_nelements(src);
    fp8_static_fake_quant_kernel<<<(n + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE,
                                    CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>(
        static_cast<const float *>(src->data), dst,
        static_cast<const float *>(scale->data), n);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void ggml_cuda_fp8_dynamic_fake_quant(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        const ggml_tensor * marker,
        float * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src);
    GGML_UNUSED(marker);
    GGML_UNUSED(dst);
    GGML_ABORT("dynamic FP8 activation quantization is not implemented for HIP");
#else
    GGML_ASSERT(src->type == GGML_TYPE_F32 && ggml_is_contiguous(src));
    GGML_ASSERT(marker == nullptr ||
        (marker->type == GGML_TYPE_I32 && ggml_is_contiguous(marker) && ggml_nelements(marker) == 1));
    const int64_t width = src->ne[0];
    const int64_t rows = ggml_nelements(src) / width;
    fp8_dynamic_fake_quant_kernel<<<rows, 256, 0, ctx.stream()>>>(
        static_cast<const float *>(src->data), dst,
        marker == nullptr ? nullptr : static_cast<const int32_t *>(marker->data), width);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void ggml_cuda_int8_static_fake_quant(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        const ggml_tensor * scale,
        float * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src);
    GGML_UNUSED(scale);
    GGML_UNUSED(dst);
    GGML_ABORT("static INT8 activation quantization is not implemented for HIP");
#else
    GGML_ASSERT(src->type == GGML_TYPE_F32 && ggml_is_contiguous(src));
    GGML_ASSERT(scale->type == GGML_TYPE_F32 && ggml_is_contiguous(scale));
    GGML_ASSERT(ggml_nelements(scale) == 1);
    const int64_t n = ggml_nelements(src);
    int8_static_fake_quant_kernel<<<(n + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE,
                                     CUDA_NEG_BLOCK_SIZE, 0, ctx.stream()>>>(
        static_cast<const float *>(src->data), dst,
        static_cast<const float *>(scale->data), n);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void ggml_cuda_int8_dynamic_fake_quant(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        float * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_ABORT("dynamic INT8 activation quantization is not implemented for HIP");
#else
    GGML_ASSERT(src->type == GGML_TYPE_F32 && ggml_is_contiguous(src));
    const int64_t width = src->ne[0];
    const int64_t rows = ggml_nelements(src) / width;
    int8_dynamic_fake_quant_kernel<<<rows, 256, 0, ctx.stream()>>>(
        static_cast<const float *>(src->data), dst, width);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void ggml_cuda_mxfp4_dynamic_fake_quant(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        float * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    GGML_ABORT("dynamic MXFP4 activation quantization is not implemented for HIP");
#else
    GGML_ASSERT(src->type == GGML_TYPE_F32 && ggml_is_contiguous(src));
    GGML_ASSERT(src->ne[0] % 32 == 0);
    const int64_t n = ggml_nelements(src);
    mxfp4_dynamic_fake_quant_kernel<<<n / 32, 32, 0, ctx.stream()>>>(
        static_cast<const float *>(src->data), dst, n);
    CUDA_CHECK(cudaGetLastError());
#endif
}

bool ggml_cuda_mul_mat_mxfp8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight,
        const ggml_tensor * input,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(weight);
    GGML_UNUSED(input);
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * scale = dst->src[2];
    const ggml_tensor * marker = dst->src[3];
    const bool mxfp8 = scale != nullptr && scale->type == GGML_TYPE_I8 &&
        marker != nullptr && marker->type == GGML_TYPE_I32;
    const bool grouped_fp8 = scale != nullptr && scale->type == GGML_TYPE_BF16 &&
        marker != nullptr && marker->type == GGML_TYPE_I16;
    if (weight->type != GGML_TYPE_F8_E4M3 || input->type != GGML_TYPE_F32 ||
            (!mxfp8 && !grouped_fp8) ||
            weight->ne[2] != 1 || weight->ne[3] != 1 ||
            input->ne[2] != 1 || input->ne[3] != 1 ||
            weight->ne[0] % 32 != 0 ||
            scale->ne[0] != weight->ne[0] / 32 || scale->ne[1] != weight->ne[1] ||
            !ggml_is_contiguous(weight) || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(scale) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t input_elements = ggml_nelements(input);
    ggml_cuda_pool_alloc<float> rounded(ctx.pool(), input_elements);
    if (mxfp8) {
        mxfp8_dynamic_fake_quant_kernel<<<input_elements / 32, 32, 0, ctx.stream()>>>(
            static_cast<const float *>(input->data), rounded.get(), input_elements);
    } else {
        fp8_group_dynamic_fake_quant_kernel<<<input_elements / 32, 32, 0, ctx.stream()>>>(
            static_cast<const float *>(input->data), rounded.get(), input_elements);
    }
    CUDA_CHECK(cudaGetLastError());
    const dim3 grid(weight->ne[1], input->ne[1]);
    if (mxfp8) {
        mxfp8_mul_mat_kernel<<<grid, 256, 0, ctx.stream()>>>(
            static_cast<const uint8_t *>(weight->data),
            static_cast<const uint8_t *>(scale->data), rounded.get(),
            static_cast<float *>(dst->data), weight->ne[0], weight->ne[1]);
    } else {
        fp8_group_mul_mat_kernel<<<grid, 256, 0, ctx.stream()>>>(
            static_cast<const uint8_t *>(weight->data),
            static_cast<const nv_bfloat16 *>(scale->data), rounded.get(),
            static_cast<float *>(dst->data), weight->ne[0], weight->ne[1]);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}

bool ggml_cuda_mul_mat_bnb4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight,
        const ggml_tensor * input,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(weight);
    GGML_UNUSED(input);
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * scale = dst->src[2];
    if ((weight->type != GGML_TYPE_BNB_NF4 && weight->type != GGML_TYPE_BNB_FP4) ||
            input->type != GGML_TYPE_F32 || scale == nullptr || scale->type != GGML_TYPE_I8 ||
            weight->ne[2] != 1 || weight->ne[3] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
            weight->ne[0] % 64 != 0 || input->ne[0] != weight->ne[0] ||
            !ggml_is_contiguous(weight) || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(scale) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const dim3 grid(weight->ne[1], input->ne[1]);
    bnb4_mul_mat_kernel<<<grid, 256, 0, ctx.stream()>>>(
        static_cast<const uint8_t *>(weight->data),
        static_cast<const ggml_bnb_scale_header *>(scale->data),
        static_cast<const float *>(input->data), static_cast<float *>(dst->data),
        weight->ne[0], weight->ne[1]);
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}

bool ggml_cuda_mul_mat_gptq_ao(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight,
        const ggml_tensor * input,
        ggml_tensor * dst) {
#if defined(GGML_USE_HIP)
    GGML_UNUSED(ctx);
    GGML_UNUSED(weight);
    GGML_UNUSED(input);
    GGML_UNUSED(dst);
    return false;
#else
    const ggml_tensor * auxiliary = dst->src[2];
    if (weight->type != GGML_TYPE_GPTQ_AO || input->type != GGML_TYPE_F32 ||
            auxiliary == nullptr || auxiliary->type != GGML_TYPE_I8 ||
            weight->ne[2] != 1 || weight->ne[3] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
            weight->ne[0] % 8 != 0 || input->ne[0] != weight->ne[0] ||
            !ggml_is_contiguous(weight) || !ggml_is_contiguous(input) ||
            !ggml_is_contiguous(auxiliary) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const dim3 grid(weight->ne[1], input->ne[1]);
    gptq_ao_mul_mat_kernel<<<grid, 256, 0, ctx.stream()>>>(
        static_cast<const uint8_t *>(weight->data),
        static_cast<const ggml_gptq_ao_header *>(auxiliary->data),
        static_cast<const float *>(input->data), static_cast<float *>(dst->data),
        weight->ne[0], weight->ne[1]);
    CUDA_CHECK(cudaGetLastError());
    return true;
#endif
}
#include "convert.cuh"

static __device__ __forceinline__ float op_abs(float x) {
    return fabsf(x);
}

static __device__ __forceinline__ float op_sgn(float x) {
    return (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
}

static __device__ __forceinline__ float op_neg(float x) {
    return -x;
}

static __device__ __forceinline__ float op_step(float x) {
    return x > 0.0f;
}

static __device__ __forceinline__ float op_gelu(float x) {
    return ggml_cuda_op_gelu_single(x);
}

static __device__ __forceinline__ float op_gelu_erf(float x) {
    const float SQRT_2_INV = 0.70710678118654752440084436210484f;

    return 0.5f*x*(1.0f + erff(x*SQRT_2_INV));
}

static __device__ __forceinline__ float op_gelu_quick(float x) {
    const float GELU_QUICK_COEF = -1.702f;

    return x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * x)));
}

static __device__ __forceinline__ float op_silu(float x) {
    return ggml_cuda_op_silu_single(x);
}

static __device__ __forceinline__ float op_tanh(float x) {
    return tanhf(x);
}

static __device__ __forceinline__ float op_relu(float x) {
    return fmaxf(x, 0);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float op_hardsigmoid(float x) {
    return fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_hardswish(float x) {
    return x * fminf(1.0f, fmaxf(0.0f, (x + 3.0f) / 6.0f));
}

static __device__ __forceinline__ float op_exp(float x) {
    return expf(x);
}

static __device__ __forceinline__ float op_sqr(float x) {
    return x * x;
}

static __device__ __forceinline__ float op_relu_sqr(float x) {
    const float r = fmaxf(x, 0.0f);
    return r * r;
}

static __device__ __forceinline__ float op_sqrt(float x) {
    return sqrtf(x);
}

static __device__ __forceinline__ float op_sin(float x) {
    return sinf(x);
}

static __device__ __forceinline__ float op_cos(float x) {
    return cosf(x);
}

static __device__ __forceinline__ float op_log(float x) {
    return logf(x);
}

static __device__ __forceinline__ float op_expm1(float x) {
    return expm1f(x);
}

static __device__ __forceinline__ float op_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float op_elu(float x) {
    return (x > 0.f) ? x : expm1f(x);
}

static __device__ __forceinline__ float op_floor(float x) {
    return floorf(x);
}

static __device__ __forceinline__ float op_ceil(float x) {
    return ceilf(x);
}

static __device__ __forceinline__ float op_round(float x) {
    return round(x);
}

static __device__ __forceinline__ float op_trunc(float x) {
    return trunc(x);
}

template <float (*op)(float), typename T>
static __global__ void unary_op_kernel(const T * x, T * dst, const int k) {
    ggml_cuda_pdl_lc();
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    ggml_cuda_pdl_sync();
    dst[i] = (T)op((float)x[i]);
}

template <float (*op)(float), typename T>
static __global__ void unary_strided_op_kernel(
        const T * x, T * dst, const int64_t k,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const int64_t sx1, const int64_t sx2, const int64_t sx3) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int64_t i0 = i % ne0;
    int64_t row = i / ne0;
    const int64_t i1 = row % ne1;
    row /= ne1;
    const int64_t i2 = row % ne2;
    const int64_t i3 = row / ne2;
    dst[i] = (T)op((float)x[i0 + i1*sx1 + i2*sx2 + i3*sx3]);
}

template <float (*op)(float), typename T>
static __global__ void unary_strided_256x24_op_kernel(
        const T * x, T * dst, const int k, const int64_t sx1, const int64_t sx2) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int i0 = i & 255;
    const int row = i >> 8;
    const int i2 = row / 24;
    const int i1 = row - i2*24;
    dst[i] = (T)op((float)x[i0 + int64_t(i1)*sx1 + int64_t(i2)*sx2]);
}

template <float (*op)(float), typename T>
static void unary_cuda(const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_NEG_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_op_kernel<op, T>, launch_params, x, dst, k);
}

template <float (*op)(float)>
void ggml_cuda_op_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_rows(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t k = ggml_nelements(src0);
    const int64_t blocks = (k + CUDA_NEG_BLOCK_SIZE - 1) / CUDA_NEG_BLOCK_SIZE;
    const bool qwen35_gate = !ggml_is_contiguous(src0) &&
        src0->ne[0] == 256 && src0->ne[1] == 24 && src0->ne[3] == 1;

#define GGML_CUDA_UNARY_STRIDED_ARGS(src) \
    (src)->nb[1] / ggml_element_size(src), \
    (src)->nb[2] / ggml_element_size(src), \
    (src)->nb[3] / ggml_element_size(src)

    if (src0->type == GGML_TYPE_F16) {
        if (ggml_is_contiguous(src0)) {
            unary_cuda<op>((const half *)src0_d, (half *)dst_d, k, stream);
        } else if (qwen35_gate) {
            unary_strided_256x24_op_kernel<op, half><<<blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(
                static_cast<const half *>(src0_d), static_cast<half *>(dst_d), int(k),
                src0->nb[1] / sizeof(half), src0->nb[2] / sizeof(half));
        } else {
            unary_strided_op_kernel<op, half><<<blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(
                static_cast<const half *>(src0_d), static_cast<half *>(dst_d), k,
                src0->ne[0], src0->ne[1], src0->ne[2], GGML_CUDA_UNARY_STRIDED_ARGS(src0));
        }
    } else {
        if (ggml_is_contiguous(src0)) {
            unary_cuda<op>((const float *)src0_d, (float *)dst_d, k, stream);
        } else if (qwen35_gate) {
            unary_strided_256x24_op_kernel<op, float><<<blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(
                static_cast<const float *>(src0_d), static_cast<float *>(dst_d), int(k),
                src0->nb[1] / sizeof(float), src0->nb[2] / sizeof(float));
        } else {
            unary_strided_op_kernel<op, float><<<blocks, CUDA_NEG_BLOCK_SIZE, 0, stream>>>(
                static_cast<const float *>(src0_d), static_cast<float *>(dst_d), k,
                src0->ne[0], src0->ne[1], src0->ne[2], GGML_CUDA_UNARY_STRIDED_ARGS(src0));
        }
    }

#undef GGML_CUDA_UNARY_STRIDED_ARGS
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_abs(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_abs>(ctx, dst);
}

void ggml_cuda_op_sgn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sgn>(ctx, dst);
}

void ggml_cuda_op_neg(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_neg>(ctx, dst);
}

void ggml_cuda_op_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_step>(ctx, dst);
}

void ggml_cuda_op_gelu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu>(ctx, dst);
}

void ggml_cuda_op_gelu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_gelu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_gelu_quick>(ctx, dst);
}

void ggml_cuda_op_silu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_silu>(ctx, dst);
}

void ggml_cuda_op_tanh(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_tanh>(ctx, dst);
}

void ggml_cuda_op_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_relu>(ctx, dst);
}

void ggml_cuda_op_sigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sigmoid>(ctx, dst);
}

void ggml_cuda_op_hardsigmoid(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardsigmoid>(ctx, dst);
}

void ggml_cuda_op_hardswish(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_hardswish>(ctx, dst);
}

void ggml_cuda_op_exp(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_exp>(ctx, dst);
}

void ggml_cuda_op_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqr>(ctx, dst);
}

void ggml_cuda_op_sqrt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sqrt>(ctx, dst);
}

void ggml_cuda_op_sin(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_sin>(ctx, dst);
}

void ggml_cuda_op_cos(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_cos>(ctx, dst);
}

void ggml_cuda_op_log(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_log>(ctx, dst);
}

void ggml_cuda_op_elu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_elu>(ctx, dst);
}

void ggml_cuda_op_floor(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_floor>(ctx, dst);
}

void ggml_cuda_op_ceil(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_ceil>(ctx, dst);
}

void ggml_cuda_op_round(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_round>(ctx, dst);
}

void ggml_cuda_op_trunc(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_trunc>(ctx, dst);
}

void ggml_cuda_op_expm1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_expm1>(ctx, dst);
}

void ggml_cuda_op_softplus(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary<op_softplus>(ctx, dst);
}
/* gated ops */

template <float (*op)(float), typename T>
static __global__ void unary_gated_op_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    ggml_cuda_pdl_sync();
    dst[i] = (T)(op((float)x[j0]) * (float)g[j1]);
}

template <float (*op)(float)>
static __global__ void unary_gated_op_bf16_kernel(
        const float * x, const float * g, nv_bfloat16 * dst,
        const int64_t k, const int64_t n, const int64_t o0, const int64_t o1) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
    dst[i] = op(x[j0]) * g[j1];
}

template <float (*op)(float), typename T>
static __global__ void unary_gated_strided_op_kernel(
        const T * x, const T * g, T * dst, const int64_t k,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const int64_t sx1, const int64_t sx2, const int64_t sx3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const int64_t i0 = i % ne0;
    int64_t row = i / ne0;
    const int64_t i1 = row % ne1;
    row /= ne1;
    const int64_t i2 = row % ne2;
    const int64_t i3 = row / ne2;
    const int64_t jx = i0 + i1*sx1 + i2*sx2 + i3*sx3;
    const int64_t jg = i0 + i1*sg1 + i2*sg2 + i3*sg3;
    dst[i] = (T)(op((float)x[jx]) * (float)g[jg]);
}

template <float (*op)(float)>
static __global__ void unary_gated_strided_op_bf16_kernel(
        const float * x, const float * g, nv_bfloat16 * dst, const int64_t k,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const int64_t sx1, const int64_t sx2, const int64_t sx3,
        const int64_t sg1, const int64_t sg2, const int64_t sg3) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const int64_t i0 = i % ne0;
    int64_t row = i / ne0;
    const int64_t i1 = row % ne1;
    row /= ne1;
    const int64_t i2 = row % ne2;
    const int64_t i3 = row / ne2;
    const int64_t jx = i0 + i1*sx1 + i2*sx2 + i3*sx3;
    const int64_t jg = i0 + i1*sg1 + i2*sg2 + i3*sg3;
    dst[i] = op(x[jx]) * g[jg];
}

template <float (*op)(float), typename T>
static __global__ void unary_gated_strided_256x24_op_kernel(
        const T * x, const T * g, T * dst, const int k,
        const int64_t sx1, const int64_t sx2,
        const int64_t sg1, const int64_t sg2) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const int i0 = i & 255;
    const int row = i >> 8;
    const int i2 = row / 24;
    const int i1 = row - i2*24;
    const int64_t jx = i0 + int64_t(i1)*sx1 + int64_t(i2)*sx2;
    const int64_t jg = i0 + int64_t(i1)*sg1 + int64_t(i2)*sg2;
    dst[i] = (T)(op((float)x[jx]) * (float)g[jg]);
}

template <float (*op)(float)>
static __global__ void unary_gated_strided_256x24_op_bf16_kernel(
        const float * x, const float * g, nv_bfloat16 * dst, const int k,
        const int64_t sx1, const int64_t sx2,
        const int64_t sg1, const int64_t sg2) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }

    const int i0 = i & 255;
    const int row = i >> 8;
    const int i2 = row / 24;
    const int i1 = row - i2*24;
    const int64_t jx = i0 + int64_t(i1)*sx1 + int64_t(i2)*sx2;
    const int64_t jg = i0 + int64_t(i1)*sg1 + int64_t(i2)*sg2;
    dst[i] = op(x[jx]) * g[jg];
}

template <float (*op)(float), typename T>
static void unary_gated_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(unary_gated_op_kernel<op, T>, launch_params, x, g, dst, k, n, o0, o1);
}

template <float (*op)(float)>
void ggml_cuda_op_unary_gated(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    const int32_t swapped = ((const int32_t *) dst->op_params)[1];

    if (src0->type == GGML_TYPE_F16) {
        half * src0_p = (half *) src0_d;
        half * src1_p = (half *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (half *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(half), src1_o / sizeof(half), stream);
    } else {
        float * src0_p = (float *) src0_d;
        float * src1_p = (float *) src1_d;

        if (!src1) {
            src0_p += swapped ? nc : 0;
            src1_p += swapped ? 0 : nc;
        }

        unary_gated_cuda<op>(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), stream);
    }
}

void ggml_cuda_op_reglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_relu>(ctx, dst);
}

void ggml_cuda_op_geglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu>(ctx, dst);
}

void ggml_cuda_op_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ctx.bf16_glu_outputs.erase(dst) != 0) {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];
        GGML_ASSERT(src1 != nullptr && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
                    dst->type == GGML_TYPE_F32 && ggml_is_contiguous_1(src0) &&
                    ggml_is_contiguous_1(src1) && ggml_is_contiguous(dst));
        const int64_t n = src0->ne[0];
        const int64_t k = ggml_nelements(dst);
        const int64_t blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
        unary_gated_op_bf16_kernel<op_silu><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, ctx.stream()>>>(
            static_cast<const float *>(src0->data), static_cast<const float *>(src1->data),
            static_cast<nv_bfloat16 *>(dst->data), k, n,
            src0->nb[1] / sizeof(float), src1->nb[1] / sizeof(float));
        ctx.humming_bf16_activations.insert(dst);
        return;
    }
    ggml_cuda_op_unary_gated<op_silu>(ctx, dst);
}

template <typename T>
static __global__ void up_clamp_swiglu_kernel(
        const T * gate, const T * up, T * dst,
        const int64_t k, const int64_t n,
        const int64_t gate_stride, const int64_t up_stride,
        const float up_min, const float up_max) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const int64_t gate_i = (i / n) * gate_stride + (i % n);
    const int64_t up_i   = (i / n) * up_stride   + (i % n);
    const float gate_v = (float) gate[gate_i];
    const float up_v   = fminf(fmaxf((float) up[up_i],     up_min),   up_max);

    ggml_cuda_pdl_sync();
    dst[i] = (T) (ggml_cuda_op_silu_single(gate_v) * up_v);
}

template <typename T>
static void up_clamp_swiglu_cuda(
        const T * gate, const T * up, T * dst,
        const int64_t k, const int64_t n,
        const int64_t gate_stride, const int64_t up_stride,
        const float up_min, const float up_max,
        cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = {
        (dim3) num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream
    };
    ggml_cuda_kernel_launch(
        up_clamp_swiglu_kernel<T>, launch_params,
        gate, up, dst, k, n, gate_stride, up_stride,
        up_min, up_max);
}

void ggml_cuda_op_up_clamp_swiglu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gate = dst->src[0];
    const ggml_tensor * up_clamp = dst->src[1];
    const ggml_tensor * up   = up_clamp->src[0];

    GGML_ASSERT(up_clamp->op   == GGML_OP_CLAMP);
    GGML_ASSERT(dst->op        == GGML_OP_GLU);
    GGML_ASSERT(dst->src[0] == gate);
    GGML_ASSERT(dst->src[1] == up_clamp);
    GGML_ASSERT(ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU);
    GGML_ASSERT(!ggml_get_op_params_i32(dst, 1));
    GGML_ASSERT(ggml_is_contiguous_1(gate));
    GGML_ASSERT(ggml_is_contiguous_1(up));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(gate, up));
    GGML_ASSERT(ggml_are_same_shape(gate, dst));
    GGML_ASSERT(gate->type == GGML_TYPE_F32 || gate->type == GGML_TYPE_F16);
    GGML_ASSERT(gate->type == up->type && gate->type == dst->type);

    float up_min;
    float up_max;
    memcpy(&up_min, up_clamp->op_params, sizeof(float));
    memcpy(&up_max, (const float *) up_clamp->op_params + 1, sizeof(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t n = gate->ne[0];
    cudaStream_t stream = ctx.stream();

    if (gate->type == GGML_TYPE_F16) {
        up_clamp_swiglu_cuda(
            (const half *) gate->data, (const half *) up->data, (half *) dst->data,
            k, n, gate->nb[1] / sizeof(half), up->nb[1] / sizeof(half),
            up_min, up_max, stream);
    } else {
        up_clamp_swiglu_cuda(
            (const float *) gate->data, (const float *) up->data, (float *) dst->data,
            k, n, gate->nb[1] / sizeof(float), up->nb[1] / sizeof(float),
            up_min, up_max, stream);
    }
}

void ggml_cuda_op_geglu_erf(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_erf>(ctx, dst);
}

void ggml_cuda_op_geglu_quick(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_unary_gated<op_gelu_quick>(ctx, dst);
}

// swiglu_oai

template <typename T>
static __global__ void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, float alpha, float limit) {
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    // perform base op and multiply with gate (either offset in same tensor or a separate one)
    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_cuda_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_cuda(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, const float alpha, const float limit, cudaStream_t stream) {
    const int64_t num_blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;
    swiglu_oai_kernel<<<num_blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(x, g, dst, k, n, o0, o1, alpha, limit);
}

void ggml_cuda_op_swiglu_oai(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_cuda(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

/* CUDA kernel + launcher for xIELU */

template <typename T>
static __global__ void xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    const float xi = ggml_cuda_cast<float>(x[i]);

    const float gate_pos = (xi > 0.0f);
    const float y_pos = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = fminf(xi, eps);
    const float y_neg = (expm1f(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;

    dst[i] = ggml_cuda_cast<T>(out);
}

template <typename T>
static void xielu_cuda(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_XIELU_BLOCK_SIZE) / CUDA_XIELU_BLOCK_SIZE;
    xielu_kernel<<<num_blocks, CUDA_XIELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, alpha_n, alpha_p, beta, eps);
}

void ggml_cuda_op_xielu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);

    if (src0->type == GGML_TYPE_F16) {
        xielu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    } else {
        xielu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), alpha_n, alpha_p, beta, eps, stream);
    }
}



/* silu_back */

static __device__ __forceinline__ float op_silu_back(float grad, float x) {
    const float s = 1.0f / (1.0f + expf(-x));
    return grad * s * (1.0f + x * (1.0f - s));
}

template <class T>
static __global__ void silu_back_kernel(const T * grad, const T * xf, T * dst, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_silu_back((float)grad[i], (float)xf[i]);
}

template <class T>
static void silu_back_cuda(const T * grad, const T * x, T * dst, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SILU_BACK_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE;
    silu_back_kernel<<<num_blocks, CUDA_SILU_BACK_BLOCK_SIZE, 0, stream>>>(grad, x, dst, k);
}

void ggml_cuda_op_silu_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // input from forward pass
    const ggml_tensor * src1 = dst->src[1]; // grads of forward pass output

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    if (src0->type == GGML_TYPE_F16) {
        silu_back_cuda((const half *)src0_d, (const half *)src1_d, (half *)dst_d, ggml_nelements(src0), stream);
    } else {
        silu_back_cuda((const float*)src0_d, (const float*)src1_d, (float *)dst_d, ggml_nelements(src0), stream);
    }
}

/* leaky relu */

static __device__ __forceinline__ float op_leaky_relu(float x, const float negative_slope) {
    return fmaxf(x, 0) + fminf(x, 0.0f) * negative_slope;
}

template <class T>
static __global__ void leaky_relu_kernel(const T * x, T * dst, const int k, const float negative_slope) {
    const int i  = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = (T)op_leaky_relu((float)x[i], negative_slope);
}

template <class T>
static void leaky_relu_cuda(const T * x, T * dst, const int k, const float negative_slope, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE;
    leaky_relu_kernel<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(x, dst, k, negative_slope);
}

void ggml_cuda_op_leaky_relu(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        leaky_relu_cuda((const half *)src0_d, (half *)dst_d, ggml_nelements(src0), negative_slope, stream);
    } else {
        leaky_relu_cuda((const float *)src0_d, (float *)dst_d, ggml_nelements(src0), negative_slope, stream);
    }
}

/* fused unary + mul */

template <float (*op)(float)>
static void ggml_cuda_op_unary_mul_impl(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node,
                                        ggml_tensor * mul_node, const ggml_tensor * bf16_activation) {
    // unary_node: UNARY op applied to unary_node->src[0]
    // mul_node:   MUL(a, b) where one of a/b is unary_node
    // Output goes to mul_node->data

    const ggml_tensor * unary_src = unary_node->src[0];  // input to the unary op
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(unary_src->nb[0] == ggml_element_size(unary_src));
    GGML_ASSERT(other_src->nb[0] == ggml_element_size(other_src));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src));

    GGML_ASSERT(unary_src->type == GGML_TYPE_F32 || unary_src->type == GGML_TYPE_F16);
    GGML_ASSERT(unary_src->type == other_src->type);
    GGML_ASSERT(unary_src->type == mul_node->type);

    cudaStream_t stream = ctx.stream();

    const int64_t k  = ggml_nelements(mul_node);
    const int64_t nc = unary_src->ne[0];
    const int64_t unary_stride = unary_src->nb[1];
    const int64_t other_stride = other_src->nb[1];

    const bool simple_rows = ggml_is_contiguous_1(unary_src) && ggml_is_contiguous_1(other_src);
    const bool qwen35_gate = !simple_rows && unary_src->ne[0] == 256 && unary_src->ne[1] == 24 &&
        unary_src->ne[3] == 1 && other_src->ne[0] == 256 && other_src->ne[1] == 24 && other_src->ne[3] == 1;
    const int64_t blocks = (k + CUDA_GLU_BLOCK_SIZE - 1) / CUDA_GLU_BLOCK_SIZE;

#define GGML_CUDA_UNARY_MUL_STRIDED_ARGS(src) \
    (src)->nb[1] / ggml_element_size(src), \
    (src)->nb[2] / ggml_element_size(src), \
    (src)->nb[3] / ggml_element_size(src)

    if (unary_src->type == GGML_TYPE_F16) {
        if (simple_rows) {
            unary_gated_cuda<op>((const half *) unary_src->data, (const half *) other_src->data,
                                 (half *) mul_node->data, k, nc,
                                 unary_stride / sizeof(half), other_stride / sizeof(half), stream);
        } else if (qwen35_gate) {
            unary_gated_strided_256x24_op_kernel<op, half><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(
                static_cast<const half *>(unary_src->data), static_cast<const half *>(other_src->data),
                static_cast<half *>(mul_node->data), int(k),
                unary_src->nb[1] / sizeof(half), unary_src->nb[2] / sizeof(half),
                other_src->nb[1] / sizeof(half), other_src->nb[2] / sizeof(half));
        } else {
            unary_gated_strided_op_kernel<op, half><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(
                static_cast<const half *>(unary_src->data), static_cast<const half *>(other_src->data),
                static_cast<half *>(mul_node->data), k,
                unary_src->ne[0], unary_src->ne[1], unary_src->ne[2],
                GGML_CUDA_UNARY_MUL_STRIDED_ARGS(unary_src),
                GGML_CUDA_UNARY_MUL_STRIDED_ARGS(other_src));
        }
    } else {
        if (bf16_activation != nullptr) {
            if (simple_rows) {
                unary_gated_op_bf16_kernel<op><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(
                    static_cast<const float *>(unary_src->data), static_cast<const float *>(other_src->data),
                    static_cast<nv_bfloat16 *>(mul_node->data), k, nc,
                    unary_stride / sizeof(float), other_stride / sizeof(float));
            } else if (qwen35_gate) {
                unary_gated_strided_256x24_op_bf16_kernel<op><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(
                    static_cast<const float *>(unary_src->data), static_cast<const float *>(other_src->data),
                    static_cast<nv_bfloat16 *>(mul_node->data), int(k),
                    unary_src->nb[1] / sizeof(float), unary_src->nb[2] / sizeof(float),
                    other_src->nb[1] / sizeof(float), other_src->nb[2] / sizeof(float));
            } else {
                unary_gated_strided_op_bf16_kernel<op><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(
                    static_cast<const float *>(unary_src->data), static_cast<const float *>(other_src->data),
                    static_cast<nv_bfloat16 *>(mul_node->data), k,
                    unary_src->ne[0], unary_src->ne[1], unary_src->ne[2],
                    GGML_CUDA_UNARY_MUL_STRIDED_ARGS(unary_src),
                    GGML_CUDA_UNARY_MUL_STRIDED_ARGS(other_src));
            }
            ctx.humming_bf16_activations.insert(bf16_activation);
        } else {
            if (simple_rows) {
                unary_gated_cuda<op>((const float *) unary_src->data, (const float *) other_src->data,
                                     (float *) mul_node->data, k, nc,
                                     unary_stride / sizeof(float), other_stride / sizeof(float), stream);
            } else if (qwen35_gate) {
                unary_gated_strided_256x24_op_kernel<op, float><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(
                    static_cast<const float *>(unary_src->data), static_cast<const float *>(other_src->data),
                    static_cast<float *>(mul_node->data), int(k),
                    unary_src->nb[1] / sizeof(float), unary_src->nb[2] / sizeof(float),
                    other_src->nb[1] / sizeof(float), other_src->nb[2] / sizeof(float));
            } else {
                unary_gated_strided_op_kernel<op, float><<<blocks, CUDA_GLU_BLOCK_SIZE, 0, stream>>>(
                    static_cast<const float *>(unary_src->data), static_cast<const float *>(other_src->data),
                    static_cast<float *>(mul_node->data), k,
                    unary_src->ne[0], unary_src->ne[1], unary_src->ne[2],
                    GGML_CUDA_UNARY_MUL_STRIDED_ARGS(unary_src),
                    GGML_CUDA_UNARY_MUL_STRIDED_ARGS(other_src));
            }
        }
    }

#undef GGML_CUDA_UNARY_MUL_STRIDED_ARGS
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_unary_mul(ggml_backend_cuda_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node,
                            const ggml_tensor * bf16_activation) {
    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:
            ggml_cuda_op_unary_mul_impl<op_silu>(ctx, unary_node, mul_node, bf16_activation);
            break;
        case GGML_UNARY_OP_SIGMOID:
            ggml_cuda_op_unary_mul_impl<op_sigmoid>(ctx, unary_node, mul_node, bf16_activation);
            break;
        case GGML_UNARY_OP_SOFTPLUS:
            ggml_cuda_op_unary_mul_impl<op_softplus>(ctx, unary_node, mul_node, bf16_activation);
            break;
        default:
            GGML_ABORT("Unsupported unary op for fused unary+mul");
    }
}

/* fused relu + sqr */

void ggml_cuda_op_relu_sqr(ggml_backend_cuda_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node) {
    const ggml_tensor * src = relu_node->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->type == sqr_node->type);

    const int k = ggml_nelements(src);
    if (src->type == GGML_TYPE_F16) {
        unary_cuda<op_relu_sqr>((const half *)src->data, (half *)sqr_node->data, k, stream);
    } else {
        unary_cuda<op_relu_sqr>((const float *)src->data, (float *)sqr_node->data, k, stream);
    }
}
