#include "fp8-channel.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
#include <cublasLt.h>
#include "unary.cuh"
#include "fp8-cutlass/fp8-channel-dual-pack.cuh"
#include "fp8-cutlass.cuh"
using fp8_cutlass_fn = decltype(&buun_fp8_cutlass);
static fp8_cutlass_fn fp8_cutlass_provider() {
#if defined(GGML_CUDA_FP8_CUTLASS)
    return buun_fp8_cutlass;
#else
    return nullptr;
#endif
}

// Use the same row scale and E4M3 rounding as fp8_dynamic_fake_quant_kernel,
// but keep quantized activations packed until the GEMM epilogue.
template<int Capacity = 0, bool FuseSwiGLU = false>
static __global__ void fp8_channel_pack(
        const float * src, __nv_fp8_e4m3 * dst, float * scales,
        const int32_t * marker, int64_t k, int64_t m,
        const float * up = nullptr) {
    const int64_t row = blockIdx.x;
    float maximum = 0.0f;
    const auto load = [&](int64_t col) {
        if (row >= m) return 0.0f;
        const int64_t index = row*k + col;
        if constexpr (FuseSwiGLU) {
            return ggml_cuda_op_silu_single(src[index]) * up[index];
        } else {
            return src[index];
        }
    };
    // Nonzero Capacity is dispatched only for k == 256*Capacity.
    float retained[Capacity > 0 ? Capacity : 1];
    if constexpr (Capacity > 0) {
#pragma unroll
        for (int i = 0; i < Capacity; ++i) {
            const int64_t col = threadIdx.x + i*256;
            retained[i] = load(col);
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
    if (upper_bound > 0.0f) {
        maximum = fminf(maximum, upper_bound);
    }
    const float scale = maximum / 448.0f;
    const float inverse = maximum == 0.0f ? 0.0f : 1.0f / scale;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const auto emit = [&](int64_t col, float value_in) {
        dst[row*k + col] = __nv_fp8_e4m3(value_in * inverse);
    };
    if constexpr (Capacity > 0) {
#pragma unroll
        for (int i = 0; i < Capacity; ++i) {
            emit(threadIdx.x + i*256, retained[i]);
        }
    } else {
        for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
            emit(col, load(col));
        }
    }
}

static void fp8_channel_pack_launch(const float * src, __nv_fp8_e4m3 * dst, float * scales,
        const int32_t * marker, int64_t k, int64_t m, int64_t padded_m,
        cudaStream_t stream, const float * up = nullptr) {
    if (up) {
        if (k == 5120) {
            fp8_channel_pack<20, true><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m, up);
        } else {
            GGML_ASSERT(k == 17408);
            fp8_channel_pack<68, true><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m, up);
        }
        return;
    }
    // Retain a row across the reduction for these measured widths, avoiding
    // its second global read. Other widths keep the generic streaming loop.
    if (k == 5120) {
        fp8_channel_pack<20><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m);
    } else if (k == 17408) {
        fp8_channel_pack<68><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m);
    } else {
        fp8_channel_pack<><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m);
    }
}

struct fp8_channel_lt_descriptors {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    ~fp8_channel_lt_descriptors() {
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

// Combine the split reduction and channel scaling without
// materializing the reduced matrix. Round the partial sum to F32 before scaling.
static __global__ void fp8_channel_reduce_finish(
        const float * src, float * dst, const float * weight_scale,
        const float * input_scale, int64_t n, int64_t stride) {
    const int64_t col = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (col >= n) return;
    const int64_t i = row*n + col;
    // Two F32 partials need only one rounded F32 addition. Preserve the
    // initial +0 and explicit rounding (including fast-math FTZ behavior).
    const float value = __fadd_rn(__fadd_rn(0.0f, src[i]), src[stride+i]);
    dst[i] = (value * input_scale[row]) * weight_scale[col];
}

#endif

bool ggml_cuda_fp8_channel_pair(ggml_backend_cuda_context & ctx, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * dst) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080 && defined(GGML_CUDA_FP8_CUTLASS)
    const auto provider = buun_fp8_cutlass_ffn;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_BLACKWELL || !gate || !up || gate->op != GGML_OP_MUL_MAT || up->op != GGML_OP_MUL_MAT ||
            dst->op != GGML_OP_GLU || ggml_get_glu_op(dst) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(dst, 1) ||
            dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
            ctx.humming_bf16_activations.count(dst) || ctx.humming_bf16_activation_uses.count(dst) ||
            ctx.bf16_glu_outputs.count(dst) ||
            !ggml_are_same_shape(gate, up) || !ggml_are_same_shape(gate, dst) ||
            gate->src[1] != up->src[1]) return false;
    const auto * x = gate->src[1];
    if (!x || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || x->ne[2] != 1 || x->ne[3] != 1 ||
            x->ne[0] != 5120 || x->ne[1] < 1009 || x->ne[1] > 65535 ||
            ctx.humming_bf16_activations.count(x) || ctx.humming_bf16_activation_uses.count(x)) return false;
    const int64_t k = x->ne[0], m = x->ne[1], n = dst->ne[0], padded_m = (m + 15)/16*16;
    if (n % 64 || dst->ne[1] != m) return false;
    for (auto * mm : {gate, up}) {
        const auto * w = mm->src[0]; const auto * s = mm->src[2]; const auto * marker = mm->src[3];
        if (!w || !s || !marker || w->type != GGML_TYPE_F8_E4M3 || s->type != GGML_TYPE_F32 ||
                marker->type != GGML_TYPE_I32 || ggml_nelements(marker) != 1 || !ggml_is_contiguous(marker) ||
                !ggml_is_contiguous(w) || !ggml_is_contiguous(s) || !ggml_is_contiguous(mm) ||
                mm->type != GGML_TYPE_F32 || w->ne[0] != k || w->ne[1] != n || w->ne[2] != 1 || w->ne[3] != 1 ||
                s->ne[0] != n || ggml_nelements(s) != n || uintptr_t(w->data)%16 ||
                ggml_cuda_humming_fp8_is_repacked(w) ||
                ctx.humming_bf16_activations.count(mm) || ctx.humming_bf16_activation_uses.count(mm)) return false;
    }
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> input(ctx.pool(), k*padded_m);
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> input_up(ctx.pool(), k*padded_m);
    ggml_cuda_pool_alloc<float> scales(ctx.pool(), padded_m);
    ggml_cuda_pool_alloc<float> scales_up(ctx.pool(), padded_m);
    size_t required = 0;
    const auto invoke = [&](void * workspace, size_t capacity) {
        return provider(gate->src[0]->data, up->src[0]->data, input.get(), input_up.get(),
            static_cast<const float *>(gate->src[2]->data), static_cast<const float *>(up->src[2]->data),
            scales.get(), scales_up.get(), static_cast<float *>(dst->data),
            static_cast<const int32_t *>(gate->src[3]->data), static_cast<const int32_t *>(up->src[3]->data),
            int(m), int(n), int(k), 2,
            ctx.device, ggml_cuda_info().devices[ctx.device].nsm, workspace, capacity, &required, ctx.stream());
    };
    if (invoke(nullptr, 0) != 0) return false;
    ggml_cuda_pool_alloc<char> workspace(ctx.pool(), std::max(size_t(1), required));
    ffn_dual_pack_kernel<<<padded_m, 256, 0, ctx.stream()>>>(
        static_cast<const float *>(x->data), input.get(), input_up.get(), scales.get(), scales_up.get(),
        static_cast<const int32_t *>(gate->src[3]->data), static_cast<const int32_t *>(up->src[3]->data), int(m));
    CUDA_CHECK(cudaGetLastError());
    const int status = invoke(workspace.get(), std::max(size_t(1), required));
    if (status != 0) GGML_ABORT("FP8 paired FFN: launch status %d", status);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(gate); GGML_UNUSED(up); GGML_UNUSED(dst);
    return false;
#endif
}

bool ggml_cuda_mul_mat_fp8_channel_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool fuse_swiglu) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const auto * scale = dst->src[2];
    const auto * marker = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (cc != GGML_CUDA_CC_BLACKWELL || w->type != GGML_TYPE_F8_E4M3 ||
            x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !scale || scale->type != GGML_TYPE_F32 ||
            !marker || marker->type != GGML_TYPE_I32 || ggml_nelements(marker) != 1 ||
            !ggml_is_contiguous(marker) || !ggml_is_contiguous(w) || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(scale) || !ggml_is_contiguous(dst) ||
            w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 ||
            x->ne[0] != w->ne[0] || dst->ne[0] != w->ne[1] || dst->ne[1] != x->ne[1] ||
            scale->ne[0] != w->ne[1] || ggml_nelements(scale) != w->ne[1] ||
            w->ne[0] % 16 || w->ne[1] % 16 || x->ne[1] < 32 ||
            uintptr_t(w->data) % 16 ||
            ggml_cuda_humming_fp8_is_repacked(w) ||
            ctx.humming_bf16_activations.count(x) || ctx.humming_bf16_activation_uses.count(x)) {
        return false;
    }
    const int64_t k = w->ne[0], n = w->ne[1], m = x->ne[1];
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
    // Native F32 channel scales use the qualified two-part FP8 accumulation.
    // BF16-scale operands are owned by the separate precision-preserving executor.
    const int64_t padded_m = (m + 15) / 16 * 16;
    constexpr int split_k = 2;
    if (m > 65535) return false;
    const auto * pack_src = static_cast<const float *>(fuse_swiglu ? x->src[0]->data : x->data);
    const auto * pack_up = fuse_swiglu ? static_cast<const float *>(x->src[1]->data) : nullptr;
    if (k % (16*split_k)) return false;
    // Use the fused F32 epilogue for large prefill batches; smaller batches keep Lt.
    if (padded_m >= 1024) {
        if (auto provider = fp8_cutlass_provider()) {
            ggml_cuda_pool_alloc<__nv_fp8_e4m3> input(ctx.pool(), k*padded_m);
            ggml_cuda_pool_alloc<float> scales(ctx.pool(), padded_m);
            size_t required = 0;
            const auto invoke = [&](void * workspace, size_t capacity) {
                return provider(w->data, input.get(), static_cast<const float *>(scale->data),
                    scales.get(), static_cast<float *>(dst->data), int(m), int(n), int(k), split_k,
                    ctx.device, ggml_cuda_info().devices[ctx.device].nsm,
                    workspace, capacity, &required, ctx.stream());
            };
            if (invoke(nullptr, 0) == 0) {
                ggml_cuda_pool_alloc<char> workspace(ctx.pool(), std::max(size_t(1), required));
                fp8_channel_pack_launch(
                    pack_src, input.get(), scales.get(),
                    static_cast<const int32_t *>(marker->data), k, m, padded_m, ctx.stream(), pack_up);
                CUDA_CHECK(cudaGetLastError());
                const int status = invoke(workspace.get(), std::max(size_t(1), required));
                if (status != 0) GGML_ABORT("FP8 CUTLASS: launch status %d", status);
                CUDA_CHECK(cudaGetLastError());
                return true;
            }
        }
    }
    const int64_t part_k = k/split_k;
    const cudaDataType_t gemm_type = CUDA_R_8F_E4M3;
    fp8_channel_lt_descriptors plan;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transpose = CUBLAS_OP_T;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA,
                                               &transpose, sizeof(transpose)));
    // FAST_ACCUM stays disabled: retain periodic FP32 accumulation.
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.a, gemm_type, part_k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.b, gemm_type, part_k, padded_m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&plan.c, CUDA_R_32F, n, padded_m, n));
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&plan.preference));
    const size_t max_workspace = 8 * 1024 * 1024;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(plan.preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));
    // NVIDIA documents that a cuBLAS handle encapsulates an Lt handle. Reuse
    // this context's handle and stream; do not add process-global GPU storage.
    const auto handle = reinterpret_cast<cublasLtHandle_t>(ctx.cublas_handle());
    cublasLtMatmulHeuristicResult_t heuristic{};
    int count = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(handle, plan.op, plan.a, plan.b,
        plan.c, plan.c, plan.preference, 1, &heuristic, &count);
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) return false;
    CUBLAS_CHECK(status);
    if (count == 0) return false;
    CUBLAS_CHECK(heuristic.state);
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> packed(ctx.pool(), k * padded_m);
    ggml_cuda_pool_alloc<float> input_scales(ctx.pool(), padded_m);
    ggml_cuda_pool_alloc<float> unscaled(ctx.pool(), n * padded_m * split_k);
    ggml_cuda_pool_alloc<char> workspace(ctx.pool(), std::max(size_t(1), heuristic.workspaceSize));
    fp8_channel_pack_launch(pack_src, packed.get(), input_scales.get(),
        static_cast<const int32_t *>(marker->data), k, m, padded_m, ctx.stream(), pack_up);
    CUDA_CHECK(cudaGetLastError());
    const void * gemm_w = w->data;
    const void * gemm_x = packed.get();
    const float alpha = 1.0f, beta = 0.0f;
    const auto multiply = [&](const void * input, float * output) {
        for (int part = 0; part < split_k; ++part) {
            const size_t offset = part*part_k;
            float * partial = output + part*n*padded_m;
            CUBLAS_CHECK(cublasLtMatmul(handle, plan.op, &alpha,
                static_cast<const char *>(gemm_w)+offset, plan.a,
                static_cast<const char *>(input)+offset, plan.b,
                &beta, partial, plan.c, partial, plan.c, &heuristic.algo,
                workspace.get(), heuristic.workspaceSize, ctx.stream()));
        }
    };
    multiply(gemm_x, unscaled.get());
    const dim3 grid((n + 255)/256, m);
    fp8_channel_reduce_finish<<<grid, 256, 0, ctx.stream()>>>(
        unscaled.get(), static_cast<float *>(dst->data), static_cast<const float *>(scale->data),
        input_scales.get(), n, n*padded_m);
    CUDA_CHECK(cudaGetLastError());
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(fuse_swiglu);
    return false;
#endif
}
