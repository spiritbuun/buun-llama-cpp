#include "fp8-channel.cuh"
#include <type_traits>
#include <fstream>

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
#include <cublasLt.h>
#include "unary.cuh"
#if defined(__linux__)
#include <dlfcn.h>

using fp8_cutlass_fn = int (*)(const void *, const void *, const float *, const float *, float *,
    int, int, int, int, int, int, void *, size_t, size_t *, cudaStream_t);

static fp8_cutlass_fn fp8_cutlass_provider() {
    static const auto fn = []() -> fp8_cutlass_fn {
        const char * path = getenv("GGML_CUDA_FP8_CUTLASS_LIBRARY");
        if (!path) return nullptr;
        void * library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!library) GGML_ABORT("FP8 CUTLASS experiment: %s", dlerror());
        auto result = reinterpret_cast<fp8_cutlass_fn>(dlsym(library, "buun_fp8_cutlass"));
        if (!result) GGML_ABORT("FP8 CUTLASS experiment: missing entry point");
        // Keep code loaded while CUDA graphs can refer to its kernels.
        return result;
    }();
    return fn;
}
#endif

// Use the same row scale and E4M3 rounding as fp8_dynamic_fake_quant_kernel,
// but keep quantized activations packed until the GEMM epilogue.
template<typename T, int Capacity = 0, bool FuseSwiGLU = false>
static __global__ void fp8_channel_pack(
        const float * src, T * dst, float * scales,
        const int32_t * marker, int64_t k, int64_t m,
        __nv_fp8_e4m3 * round_residual = nullptr, const float * up = nullptr) {
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
        const __nv_fp8_e4m3 q(value_in * inverse);
        if constexpr (std::is_same<T, nv_bfloat16>::value) {
            dst[row*k + col] = __float2bfloat16_rn(float(q) * scale);
        } else {
            dst[row*k + col] = q;
        }
        if (round_residual) {
            // BF16 rounding error is <= |q|/256 in normalized coordinates.
            // Multiplying by 128 fits E4M3 (<=224); zero rows stay zero.
            const float value = float(q) * scale;
            const float rounded = __bfloat162float(__float2bfloat16_rn(value));
            round_residual[row*k + col] = __nv_fp8_e4m3((rounded-value) * inverse * 128.0f);
        }
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

template<typename T>
static void fp8_channel_pack_launch(const float * src, T * dst, float * scales,
        const int32_t * marker, int64_t k, int64_t m, int64_t padded_m,
        cudaStream_t stream, __nv_fp8_e4m3 * round_residual = nullptr, const float * up = nullptr) {
    if constexpr (std::is_same<T, __nv_fp8_e4m3>::value) {
        if (up) {
            if (k == 5120) {
                fp8_channel_pack<T, 20, true><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m, nullptr, up);
            } else {
                GGML_ASSERT(k == 17408);
                fp8_channel_pack<T, 68, true><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m, nullptr, up);
            }
            return;
        }
    }
    // Retain a row across the reduction for these measured widths, avoiding
    // its second global read. Other widths keep the generic streaming loop.
    if (k == 5120) {
        fp8_channel_pack<T, 20><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m, round_residual);
    } else if (k == 17408) {
        fp8_channel_pack<T, 68><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m, round_residual);
    } else {
        fp8_channel_pack<T><<<padded_m, 256, 0, stream>>>(src, dst, scales, marker, k, m, round_residual);
    }
}

template<typename S>
static __global__ void fp8_channel_finish(
        const float * src, float * dst, const S * weight_scale,
        const float * input_scale, int64_t n, int64_t count, bool scaled_input,
        bool wide_scale, const float * round_correction) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        if (wide_scale || round_correction) {
            const double value = double(src[i]) + (round_correction ? double(round_correction[i])/128.0 : 0.0);
            dst[i] = float(value * (scaled_input ? 1.0 : double(input_scale[i/n])) * double(float(weight_scale[i%n])));
        } else {
            dst[i] = (scaled_input ? src[i] : src[i] * input_scale[i/n]) * float(weight_scale[i%n]);
        }
    }
}

// Diagnostic controls: separate GEMM precision from the location of the
// baseline's BF16 activation rounding. Not a proposed public option.
static __global__ void fp8_channel_unpack_reference(
        const __nv_fp8_e4m3 * src, nv_bfloat16 * dst, int64_t count,
        const float * scales, int64_t k) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float value = float(src[i]);
        dst[i] = __float2bfloat16_rn(scales ? value * scales[i/k] : value);
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

static __global__ void fp8_channel_sum_partials(float * src, int64_t stride, int64_t count, int parts) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < count) {
        double sum = 0.0;
        for (int p = 0; p < parts; ++p) sum += double(src[p*stride+i]);
        src[i] = float(sum);
    }
}

// Research variant: combine the split reduction and channel scaling without
// materializing the reduced matrix. Round the partial sum to F32 before scaling.
template<typename S>
static __global__ void fp8_channel_reduce_finish(
        const float * src, float * dst, const S * weight_scale,
        const float * input_scale, int64_t n, int64_t stride, int parts) {
    const int64_t col = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t row = blockIdx.y;
    if (col >= n) return;
    const int64_t i = row*n + col;
    float value = src[i];
    if (parts == 2) {
        // Two F32 partials need only one rounded F32 addition. Preserve the
        // initial +0 and explicit rounding (including fast-math FTZ behavior).
        value = __fadd_rn(__fadd_rn(0.0f, src[i]), src[stride+i]);
    } else if (parts > 1) {
        double sum = 0.0;
        for (int p = 0; p < parts; ++p) sum += double(src[p*stride+i]);
        value = float(sum);
    }
    dst[i] = (value * input_scale[row]) * float(weight_scale[col]);
}

static __global__ void fp8_channel_check_reference(
        const __nv_fp8_e4m3 * w, const __nv_fp8_e4m3 * x, const float * input_scales,
        const void * weight_scales, bool scale_f32, bool rounded, const float * actual,
        double * report, int64_t k, int64_t n, int64_t m) {
    const int sample = threadIdx.x;
    if (sample >= 16) return;
    const int64_t row = (sample % 4) * (n-1) / 3;
    const int64_t token = (sample / 4) * (m-1) / 3;
    const float d = input_scales[token];
    double sum = 0.0;
    for (int64_t col = 0; col < k; ++col) {
        const float q = float(x[token*k + col]);
        const double value = rounded ? float(__float2bfloat16_rn(q*d)) : q;
        sum += double(float(w[row*k + col])) * value;
    }
    const float scale = scale_f32 ? static_cast<const float *>(weight_scales)[row] :
        float(static_cast<const nv_bfloat16 *>(weight_scales)[row]);
    report[2*sample] = sum * (rounded ? 1.0 : double(d)) * double(scale);
    report[2*sample+1] = actual[token*n + row];
}

static __global__ void fp8_channel_compare_all(
        const float * actual, const float * reference, double * report, int64_t count) {
    double error = 0.0, norm = 0.0, maximum_error = 0.0, maximum_value = 0.0;
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
            i < count; i += int64_t(gridDim.x)*blockDim.x) {
        const double a = actual[i], b = reference[i], d = a-b;
        error += d*d;
        norm += b*b;
        maximum_error = fmax(maximum_error, fabs(d));
        maximum_value = fmax(maximum_value, fabs(b));
    }
    __shared__ double values[4][256];
    values[0][threadIdx.x] = error;
    values[1][threadIdx.x] = norm;
    values[2][threadIdx.x] = maximum_error;
    values[3][threadIdx.x] = maximum_value;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            values[0][threadIdx.x] += values[0][threadIdx.x + stride];
            values[1][threadIdx.x] += values[1][threadIdx.x + stride];
            values[2][threadIdx.x] = fmax(values[2][threadIdx.x], values[2][threadIdx.x + stride]);
            values[3][threadIdx.x] = fmax(values[3][threadIdx.x], values[3][threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        for (int j = 0; j < 4; ++j) report[4*blockIdx.x+j] = values[j][0];
    }
}
#endif

bool ggml_cuda_mul_mat_fp8_channel_lt(ggml_backend_cuda_context & ctx, ggml_tensor * dst, bool fuse_swiglu) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12080
    // Experimental gate while the native FP8 numerical path is qualified.
    static const bool enabled = getenv("GGML_CUDA_FP8_LT") != nullptr;
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const auto * scale = dst->src[2];
    const auto * marker = dst->src[3];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!enabled || cc < 1200 || cc >= 1300 || w->type != GGML_TYPE_F8_E4M3 ||
            x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            !scale || (scale->type != GGML_TYPE_F32 && scale->type != GGML_TYPE_BF16) ||
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
    if (getenv("GGML_CUDA_FP8_LT_KEEP_SKINNY") && n <= 128) {
        return false;
    }
    const int64_t padded_m = (m + 15) / 16 * 16;
    const char * reference = getenv("GGML_CUDA_FP8_LT_REFERENCE");
    const bool bf16_reference = reference != nullptr;
    const bool fused_bf16 = reference && std::string(reference) == "fused-bf16";
    const bool rounded_reference = fused_bf16 || (reference && std::string(reference) == "rounded-bf16");
    const bool round_correction = !bf16_reference && getenv("GGML_CUDA_FP8_LT_ROUND_CORRECTION");
    const bool wide_scale = getenv("GGML_CUDA_FP8_LT_WIDE_SCALE") != nullptr;
    if (fuse_swiglu && (bf16_reference || round_correction || wide_scale ||
            getenv("GGML_CUDA_FP8_LT_CHECK_ALL") || getenv("GGML_CUDA_FP8_LT_CHECK") ||
            getenv("GGML_CUDA_FP8_LT_DUMP"))) return false;
    const auto * pack_src = static_cast<const float *>(fuse_swiglu ? x->src[0]->data : x->data);
    const auto * pack_up = fuse_swiglu ? static_cast<const float *>(x->src[1]->data) : nullptr;
    const char * split_env = getenv("GGML_CUDA_FP8_LT_SPLIT_K");
    const int split_k = !bf16_reference && split_env ? atoi(split_env) : 1;
    GGML_ASSERT(split_k == 1 || split_k == 2 || split_k == 4);
    const bool fused_finish = getenv("GGML_CUDA_FP8_LT_FUSED_FINISH") &&
        !bf16_reference && !round_correction && !wide_scale && m <= 65535 &&
        !getenv("GGML_CUDA_FP8_LT_CHECK_ALL");
    if (k % (16*split_k)) return false;
#if defined(__linux__)
    // Use the fused F32 epilogue for large prefill batches; smaller batches keep Lt.
    if (padded_m >= 1024 && !bf16_reference && !round_correction && !wide_scale && split_k <= 2 &&
            scale->type == GGML_TYPE_F32 && !getenv("GGML_CUDA_FP8_LT_CHECK_ALL") &&
            !getenv("GGML_CUDA_FP8_LT_CHECK") && !getenv("GGML_CUDA_FP8_LT_DUMP")) {
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
                    static_cast<const int32_t *>(marker->data), k, m, padded_m, ctx.stream(), nullptr, pack_up);
                CUDA_CHECK(cudaGetLastError());
                const int status = invoke(workspace.get(), std::max(size_t(1), required));
                if (status != 0) GGML_ABORT("FP8 CUTLASS experiment: launch status %d", status);
                CUDA_CHECK(cudaGetLastError());
                return true;
            }
        }
    }
#endif
    const int64_t part_k = k/split_k;
    const cudaDataType_t gemm_type = bf16_reference ? CUDA_R_16BF : CUDA_R_8F_E4M3;
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
    if (status == CUBLAS_STATUS_NOT_SUPPORTED) {
        return false;
    }
    CUBLAS_CHECK(status);
    if (count == 0) {
        return false;
    }
    CUBLAS_CHECK(heuristic.state);
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> packed(ctx.pool(), k * padded_m);
    ggml_cuda_pool_alloc<float> input_scales(ctx.pool(), padded_m);
    ggml_cuda_pool_alloc<float> unscaled(ctx.pool(), n * padded_m * split_k);
    ggml_cuda_pool_alloc<char> workspace(ctx.pool(), std::max(size_t(1), heuristic.workspaceSize));
    ggml_cuda_pool_alloc<__nv_fp8_e4m3> residual_x(ctx.pool());
    ggml_cuda_pool_alloc<float> residual_y(ctx.pool());
    if (round_correction) {
        residual_x.alloc(k*padded_m);
        residual_y.alloc(n*padded_m*split_k);
    }
    ggml_cuda_pool_alloc<nv_bfloat16> reference_w(ctx.pool());
    ggml_cuda_pool_alloc<nv_bfloat16> reference_x(ctx.pool());
    if (bf16_reference) {
        reference_w.alloc(n*k);
        reference_x.alloc(padded_m*k);
    }
    if (fused_bf16) {
        fp8_channel_pack_launch(
            static_cast<const float *>(x->data), reference_x.get(), input_scales.get(),
            static_cast<const int32_t *>(marker->data), k, m, padded_m, ctx.stream());
    } else {
        fp8_channel_pack_launch(
            pack_src, packed.get(), input_scales.get(),
            static_cast<const int32_t *>(marker->data), k, m, padded_m, ctx.stream(), residual_x.get(), pack_up);
    }
    CUDA_CHECK(cudaGetLastError());
    const void * gemm_w = w->data;
    const void * gemm_x = packed.get();
    if (bf16_reference) {
        fp8_channel_unpack_reference<<<(n*k + 255)/256, 256, 0, ctx.stream()>>>(
            static_cast<const __nv_fp8_e4m3 *>(w->data), reference_w.get(), n*k, nullptr, k);
        if (!fused_bf16) {
            fp8_channel_unpack_reference<<<(padded_m*k + 255)/256, 256, 0, ctx.stream()>>>(
                packed.get(), reference_x.get(), padded_m*k, rounded_reference ? input_scales.get() : nullptr, k);
        }
        CUDA_CHECK(cudaGetLastError());
        gemm_w = reference_w.get();
        gemm_x = reference_x.get();
    }
    const float alpha = 1.0f, beta = 0.0f;
    const auto multiply = [&](const void * input, float * output) {
        for (int part = 0; part < split_k; ++part) {
            const size_t offset = part*part_k*(bf16_reference ? sizeof(nv_bfloat16) : 1);
            float * partial = output + part*n*padded_m;
            CUBLAS_CHECK(cublasLtMatmul(handle, plan.op, &alpha,
                static_cast<const char *>(gemm_w)+offset, plan.a,
                static_cast<const char *>(input)+offset, plan.b,
                &beta, partial, plan.c, partial, plan.c, &heuristic.algo,
                workspace.get(), heuristic.workspaceSize, ctx.stream()));
        }
        if (split_k > 1 && !fused_finish) {
            fp8_channel_sum_partials<<<(n*m+255)/256, 256, 0, ctx.stream()>>>(
                output, n*padded_m, n*m, split_k);
            CUDA_CHECK(cudaGetLastError());
        }
    };
    multiply(gemm_x, unscaled.get());
    if (round_correction) {
        multiply(residual_x.get(), residual_y.get());
    }
    const int64_t elements = n*m;
    if (fused_finish) {
        const dim3 grid((n + 255)/256, m);
        if (scale->type == GGML_TYPE_F32) {
            fp8_channel_reduce_finish<<<grid, 256, 0, ctx.stream()>>>(
                unscaled.get(), static_cast<float *>(dst->data), static_cast<const float *>(scale->data),
                input_scales.get(), n, n*padded_m, split_k);
        } else {
            fp8_channel_reduce_finish<<<grid, 256, 0, ctx.stream()>>>(
                unscaled.get(), static_cast<float *>(dst->data), static_cast<const nv_bfloat16 *>(scale->data),
                input_scales.get(), n, n*padded_m, split_k);
        }
    } else if (scale->type == GGML_TYPE_F32) {
        fp8_channel_finish<<<(elements + 255)/256, 256, 0, ctx.stream()>>>(
            unscaled.get(), static_cast<float *>(dst->data), static_cast<const float *>(scale->data),
            input_scales.get(), n, elements, rounded_reference, wide_scale, residual_y.get());
    } else {
        fp8_channel_finish<<<(elements + 255)/256, 256, 0, ctx.stream()>>>(
            unscaled.get(), static_cast<float *>(dst->data), static_cast<const nv_bfloat16 *>(scale->data),
            input_scales.get(), n, elements, rounded_reference, wide_scale, residual_y.get());
    }
    CUDA_CHECK(cudaGetLastError());
    if (!bf16_reference && getenv("GGML_CUDA_FP8_LT_CHECK_ALL")) {
        // Observe a second GEMM on exactly the same packed input, but promote
        // both FP8 operands losslessly to BF16. Do not feed this result onward.
        reference_w.alloc(n*k);
        reference_x.alloc(padded_m*k);
        ggml_cuda_pool_alloc<float> expected(ctx.pool(), n*padded_m);
        ggml_cuda_pool_alloc<double> report(ctx.pool(), 4*128);
        fp8_channel_unpack_reference<<<(n*k + 255)/256, 256, 0, ctx.stream()>>>(
            static_cast<const __nv_fp8_e4m3 *>(w->data), reference_w.get(), n*k, nullptr, k);
        fp8_channel_unpack_reference<<<(padded_m*k + 255)/256, 256, 0, ctx.stream()>>>(
            packed.get(), reference_x.get(), padded_m*k, nullptr, k);
        CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
            n, padded_m, k, &alpha, reference_w.get(), CUDA_R_16BF, k,
            reference_x.get(), CUDA_R_16BF, k, &beta, expected.get(), CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        fp8_channel_compare_all<<<128, 256, 0, ctx.stream()>>>(
            unscaled.get(), expected.get(), report.get(), n*m);
        double host[4*128];
        CUDA_CHECK(cudaMemcpyAsync(host, report.get(), sizeof(host), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        double error = 0.0, norm = 0.0, max_error = 0.0, max_value = 0.0;
        for (int i = 0; i < 128; ++i) {
            error += host[4*i]; norm += host[4*i+1];
            max_error = std::max(max_error, host[4*i+2]);
            max_value = std::max(max_value, host[4*i+3]);
        }
        fprintf(stderr, "FP8_LT_PAIR %s k=%lld n=%lld m=%lld relative_rms=%.12g max_abs=%.12g reference_max=%.12g\n",
            w->name, (long long) k, (long long) n, (long long) m,
            sqrt(error/std::max(norm, 1e-300)), max_error, max_value);
    }
    static int checked = 0;
    if (const char * dump_dir = getenv("GGML_CUDA_FP8_LT_DUMP")) {
        if (!fused_bf16 && std::string(w->name).find("blk.0.") == 0) {
            const std::string prefix = std::string(dump_dir) + "/" + w->name;
            std::ofstream shape(prefix + ".shape");
            shape << k << " " << n << " " << m << "\n";
            const auto dump = [&](const char * suffix, const void * ptr, size_t size) {
                std::vector<char> bytes(size);
                CUDA_CHECK(cudaMemcpyAsync(bytes.data(), ptr, size, cudaMemcpyDeviceToHost, ctx.stream()));
                CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
                std::ofstream file(prefix + suffix, std::ios::binary);
                file.write(bytes.data(), bytes.size());
                GGML_ASSERT(file.good());
            };
            dump(".input", x->data, k*m*sizeof(float));
            dump(".packed", packed.get(), k*m);
            dump(".scales", input_scales.get(), m*sizeof(float));
            dump(".output", dst->data, n*m*sizeof(float));
        }
    }
    if (!fused_bf16 && getenv("GGML_CUDA_FP8_LT_CHECK") && checked++ < 4) {
        ggml_cuda_pool_alloc<double> report(ctx.pool(), 32);
        fp8_channel_check_reference<<<1, 32, 0, ctx.stream()>>>(
            static_cast<const __nv_fp8_e4m3 *>(w->data), packed.get(), input_scales.get(),
            scale->data, scale->type == GGML_TYPE_F32, rounded_reference || round_correction,
            static_cast<const float *>(dst->data), report.get(), k, n, m);
        double host[32];
        CUDA_CHECK(cudaMemcpyAsync(host, report.get(), sizeof(host), cudaMemcpyDeviceToHost, ctx.stream()));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        for (int i = 0; i < 16; ++i) {
            fprintf(stderr, "FP8_LT_CHECK %s k=%lld n=%lld m=%lld sample=%d ideal=%.12g actual=%.12g\n",
                w->name, (long long) k, (long long) n, (long long) m, i, host[2*i], host[2*i+1]);
        }
    }
    return true;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_UNUSED(fuse_swiglu);
    return false;
#endif
}
