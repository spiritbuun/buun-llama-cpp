#include "fp8-channel-bf16.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
#include "unary.cuh"
#include <cublasLt.h>

namespace {

template<int Capacity = 0, bool FuseSwiGLU = false>
__global__ void pack_bf16(const float * src, const float * up, nv_bfloat16 * dst,
        const int32_t * marker, int64_t k, int64_t m) {
    const int64_t row = blockIdx.x;
    const auto load = [&](int64_t col) {
        if (row >= m) return 0.0f;
        const int64_t index = row*k + col;
        if constexpr (FuseSwiGLU) {
            return ggml_cuda_op_silu_single(src[index]) * up[index];
        } else {
            return src[index];
        }
    };
    float retained[Capacity > 0 ? Capacity : 1];
    float maximum = 0.0f;
    if constexpr (Capacity > 0) {
#pragma unroll
        for (int i = 0; i < Capacity; ++i) {
            retained[i] = load(threadIdx.x + i*256);
            maximum = fmaxf(maximum, fabsf(retained[i]));
        }
    } else if (row < m) {
        for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
            maximum = fmaxf(maximum, fabsf(load(col)));
        }
    }
    __shared__ float maxima[256 / WARP_SIZE];
    maximum = block_reduce<block_reduce_method::MAX, 256>(maximum, maxima);
    const float upper_bound = __int_as_float(marker[0]);
    if (upper_bound > 0.0f) maximum = fminf(maximum, upper_bound);
    const float scale = maximum / 448.0f;
    const float inverse = maximum == 0.0f ? 0.0f : 1.0f / scale;
    const auto emit = [&](int64_t col, float value) {
        const __nv_fp8_e4m3 q(value * inverse);
        // Keep this rounding before GEMM; applying scale after native FP8 MMA
        // is not numerically equivalent to the ordinary BF16 execution path.
        dst[row*k + col] = __float2bfloat16_rn(float(q) * scale);
    };
    if constexpr (Capacity > 0) {
#pragma unroll
        for (int i = 0; i < Capacity; ++i) emit(threadIdx.x + i*256, retained[i]);
    } else {
        for (int64_t col = threadIdx.x; col < k; col += blockDim.x) emit(col, load(col));
    }
}

__device__ uint16_t bf16_bits(uint8_t value) {
    const unsigned magnitude = value & 127;
    if (magnitude >= 8 && magnitude != 127) {
        return uint16_t((unsigned(value & 128) << 8) | ((magnitude << 4) + 0x3c00));
    }
    // Preserve CUDA's signed-zero, subnormal and NaN conversions.
    __nv_fp8_e4m3 x;
    x.__x = value;
    return __bfloat16_as_ushort(__float2bfloat16_rn(float(x)));
}

__global__ void unpack_bf16(const uint32_t * src, uint2 * dst, int64_t groups) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= groups) return;
    const uint32_t x = src[i];
    dst[i] = make_uint2(uint32_t(bf16_bits(x)) | (uint32_t(bf16_bits(x >> 8)) << 16),
                       uint32_t(bf16_bits(x >> 16)) | (uint32_t(bf16_bits(x >> 24)) << 16));
}

__global__ void scale_output(const float * src, float * dst,
        const nv_bfloat16 * scale, int64_t n, int64_t count) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[i] * float(scale[i % n]);
}

struct matmul_plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    ~matmul_plan() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

bool algo_attribute(const cublasLtMatmulAlgo_t & algo,
        cublasLtMatmulAlgoConfigAttributes_t attribute, int expected) {
    int value = -1;
    size_t written = 0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algo, attribute, &value, sizeof(value), &written) ==
        CUBLAS_STATUS_SUCCESS && written == sizeof(value) && value == expected;
}

} // namespace

bool ggml_cuda_mul_mat_fp8_channel_bf16(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool fuse_swiglu) {
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const auto * scale = dst->src[2];
    const auto * marker = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc < 1200 || cc >= 1300 || w->type != GGML_TYPE_F8_E4M3 ||
            x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !scale || scale->type != GGML_TYPE_BF16 || !marker || marker->type != GGML_TYPE_I32 ||
            ggml_nelements(marker) != 1 || !ggml_is_contiguous(marker) ||
            !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(scale) || !ggml_is_contiguous(dst) ||
            w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 ||
            x->ne[0] != w->ne[0] || dst->ne[0] != w->ne[1] || dst->ne[1] != x->ne[1] ||
            scale->ne[0] != w->ne[1] || ggml_nelements(scale) != w->ne[1] ||
            w->ne[0] % 16 || w->ne[1] % 16 || x->ne[1] < 32 || uintptr_t(w->data) % 16 ||
            ggml_cuda_humming_fp8_is_repacked(w) || ctx.humming_bf16_activations.count(x) ||
            ctx.humming_bf16_activation_uses.count(x)) return false;
    const int64_t k = w->ne[0], n = w->ne[1], m = x->ne[1], padded_m = (m + 15)/16*16;
    if (fuse_swiglu) {
        if (m < 384 || (k != 5120 && k != 17408) || x->op != GGML_OP_GLU ||
                ggml_get_glu_op(x) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(x, 1) ||
                !x->src[0] || !x->src[1]) return false;
        for (const auto * input : { x->src[0], x->src[1] }) {
            if (input->type != GGML_TYPE_F32 || !ggml_is_contiguous(input) ||
                    !ggml_are_same_shape(input, x) || ctx.humming_bf16_activations.count(input) ||
                    ctx.humming_bf16_activation_uses.count(input)) return false;
        }
    }
    matmul_plan plan;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transpose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16BF, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16BF, k, padded_m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_32F, n, padded_m, n));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&plan.preference));
    const size_t max_workspace = 8*1024*1024;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(plan.preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));
    const auto handle = reinterpret_cast<cublasLtHandle_t>(ctx.cublas_handle());
    int version = 0;
    CUBLAS_CHECK(cublasGetVersion(ctx.cublas_handle(), &version));
    // Only these runtime/shapes/configurations have exact-output and PP gates.
    // Never select a heuristic by ordinal: other releases can put split-K there.
    const bool tune = cc == 1200 && version == 120804 && k == 5120 && m == 2048 &&
        (n == 6144 || n == 14336 || n == 17408);
    cublasLtMatmulHeuristicResult_t choices[3]{};
    int count = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, plan.op, plan.a, plan.b,
        plan.c, plan.c, plan.preference, tune ? 3 : 1, choices, &count);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) return false;
    CUBLAS_CHECK(status);
    if (count == 0) return false;
    auto selected = choices[0];
    if (tune) {
        for (int i = 0; i < count; ++i) {
            const auto & candidate = choices[i];
            if (candidate.state == CUBLAS_STATUS_SUCCESS && candidate.workspaceSize <= max_workspace &&
                    algo_attribute(candidate.algo, CUBLASLT_ALGO_CONFIG_ID, 21) &&
                    algo_attribute(candidate.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, n == 6144 ? 19 : 21) &&
                    algo_attribute(candidate.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, 10) &&
                    algo_attribute(candidate.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1)) {
                selected = candidate;
                break;
            }
        }
    }
    CUBLAS_CHECK(selected.state);
    ggml_cuda_pool_alloc<nv_bfloat16> weights(ctx.pool(), n*k);
    ggml_cuda_pool_alloc<nv_bfloat16> input(ctx.pool(), padded_m*k);
    ggml_cuda_pool_alloc<float> output(ctx.pool(), n*padded_m);
    ggml_cuda_pool_alloc<char> workspace(ctx.pool(), std::max(size_t(1), selected.workspaceSize));
    const auto * src = static_cast<const float *>(fuse_swiglu ? x->src[0]->data : x->data);
    const auto * up = fuse_swiglu ? static_cast<const float *>(x->src[1]->data) : nullptr;
    const auto * bound = static_cast<const int32_t *>(marker->data);
    if (fuse_swiglu) {
        if (k == 5120) pack_bf16<20, true><<<padded_m, 256, 0, ctx.stream()>>>(src, up, input.get(), bound, k, m);
        else           pack_bf16<68, true><<<padded_m, 256, 0, ctx.stream()>>>(src, up, input.get(), bound, k, m);
    } else if (k == 5120) {
        pack_bf16<20><<<padded_m, 256, 0, ctx.stream()>>>(src, up, input.get(), bound, k, m);
    } else if (k == 17408) {
        pack_bf16<68><<<padded_m, 256, 0, ctx.stream()>>>(src, up, input.get(), bound, k, m);
    } else {
        pack_bf16<<<padded_m, 256, 0, ctx.stream()>>>(src, up, input.get(), bound, k, m);
    }
    unpack_bf16<<<(n*k/4 + 255)/256, 256, 0, ctx.stream()>>>(
        static_cast<const uint32_t *>(w->data), reinterpret_cast<uint2 *>(weights.get()), n*k/4);
    CUDA_CHECK(cudaGetLastError());
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(handle, plan.op, &alpha, weights.get(), plan.a, input.get(), plan.b,
        &beta, output.get(), plan.c, output.get(), plan.c, &selected.algo,
        workspace.get(), selected.workspaceSize, ctx.stream()));
    scale_output<<<(n*m + 255)/256, 256, 0, ctx.stream()>>>(output.get(), static_cast<float *>(dst->data),
        static_cast<const nv_bfloat16 *>(scale->data), n, n*m);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else
bool ggml_cuda_mul_mat_fp8_channel_bf16(ggml_backend_cuda_context &, ggml_tensor *, bool) {
    return false;
}
#endif
