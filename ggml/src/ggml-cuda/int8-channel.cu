#include "int8-channel.cuh"
#include "humming-fp8.cuh"
#include "mmvq.cuh"
#include "unary.cuh"

#if !defined(GGML_USE_HIP)

#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"

namespace ggml_cuda_int8_detail {

using namespace cute;

__device__ __forceinline__ int8_t float_to_i8_rn(float value) {
    uint32_t result;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
    return reinterpret_cast<const int8_t &>(result);
}

template <typename input_t>
__global__ void quantize_rows_i8(
        const input_t * input,
        int8_t * quantized,
        float * scales,
        int64_t width);

__global__ void quantize_rows_i8_static(
        const float * input,
        int8_t * quantized,
        float * scales,
        const float * scale_ptr,
        int64_t width) {
    const int64_t row = blockIdx.x;
    const float scale = *scale_ptr;
    const float inverse = 1.0f / scale;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float4 * row_input = reinterpret_cast<const float4 *>(input + row * width);
    char4 * row_output = reinterpret_cast<char4 *>(quantized + row * width);
    for (int64_t col = threadIdx.x; col < width / 4; col += blockDim.x) {
        const float4 value = row_input[col];
        row_output[col] = {
            float_to_i8_rn(value.x * inverse),
            float_to_i8_rn(value.y * inverse),
            float_to_i8_rn(value.z * inverse),
            float_to_i8_rn(value.w * inverse),
        };
    }
}

__global__ void quantize_rows_i8_static_asym(
        const float * input,
        int8_t * quantized,
        float * scales,
        const int64_t * params,
        int64_t width) {
    const int64_t row = blockIdx.x;
    const uint64_t raw = static_cast<uint64_t>(*params);
    const float scale = __uint_as_float(static_cast<uint32_t>(raw));
    const int8_t zero = static_cast<int8_t>(raw >> 32);
    const float inverse = 1.0f / scale;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float4 * row_input = reinterpret_cast<const float4 *>(input + row * width);
    char4 * row_output = reinterpret_cast<char4 *>(quantized + row * width);
    for (int64_t col = threadIdx.x; col < width / 4; col += blockDim.x) {
        const float4 value = row_input[col];
        row_output[col] = {
            float_to_i8_rn(value.x * inverse + zero),
            float_to_i8_rn(value.y * inverse + zero),
            float_to_i8_rn(value.z * inverse + zero),
            float_to_i8_rn(value.w * inverse + zero),
        };
    }
}

__global__ void sum_i8_rows(const int8_t * weight, int32_t * sums, int64_t width) {
    const int64_t row = blockIdx.x;
    int sum = 0;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        sum += weight[row * width + col];
    }
    __shared__ int partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sums[row] = partial[0];
    }
}

template<typename scale_t>
__global__ void correct_i8_asym_output(
        float * output,
        const int32_t * weight_sums,
        const int64_t * params,
        const scale_t * weight_scales,
        int64_t rows,
        int64_t count) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const uint64_t raw = static_cast<uint64_t>(*params);
    const float input_scale = __uint_as_float(static_cast<uint32_t>(raw));
    const int8_t zero = static_cast<int8_t>(raw >> 32);
    const int64_t row = index % rows;
    output[index] -= float(zero) * input_scale * float(weight_scales[row]) * float(weight_sums[row]);
}

template<typename scale_t>
__global__ void add_i8_outlier_output(
        const float * input,
        const int8_t * weight,
        const uint8_t * outlier_columns,
        const scale_t * weight_scales,
        float * output,
        int64_t width,
        int64_t rows) {
    const int64_t row = blockIdx.x;
    const int64_t token = blockIdx.y;
    const float * x = input + token * width;
    const int8_t * w = weight + row * width;
    float sum = 0.0f;
    for (int64_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float value = x[col];
        if (outlier_columns[col]) {
            sum += float(w[col]) * value;
        }
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[token * rows + row] += partial[0] * float(weight_scales[row]);
    }
}

__global__ void mark_i8_outlier_columns(
        const float * input,
        uint8_t * outlier_columns,
        const int32_t * threshold_bits,
        int64_t width,
        int64_t rows) {
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= width) {
        return;
    }
    const float threshold = __int_as_float(*threshold_bits);
    bool outlier = false;
    if (threshold > 0.0f) {
        for (int64_t row = 0; row < rows; ++row) {
            outlier |= fabsf(input[row * width + col]) > threshold;
        }
    }
    outlier_columns[col] = outlier;
}

template <typename output_t>
__global__ void swiglu_stacked_f16(
        const half * gate_up,
        output_t * output,
        int64_t width,
        int64_t count) {
    const int64_t pair = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t index = pair * 2;
    if (index + 1 < count) {
        const int64_t row = index / width;
        const int64_t col = index - row * width;
        const int64_t base = row * 2 * width + col;
        const float2 gate = __half22float2(*reinterpret_cast<const half2 *>(gate_up + base));
        const float2 up = __half22float2(*reinterpret_cast<const half2 *>(gate_up + base + width));
        const float out0 = ggml_cuda_op_silu_single(gate.x) * up.x;
        const float out1 = ggml_cuda_op_silu_single(gate.y) * up.y;
        if constexpr (std::is_same_v<output_t, nv_bfloat16>) {
            *reinterpret_cast<nv_bfloat162 *>(output + index) = __floats2bfloat162_rn(out0, out1);
        } else {
            *reinterpret_cast<float2 *>(output + index) = {out0, out1};
        }
    } else if (index < count) {
        const int64_t row = index / width;
        const int64_t col = index - row * width;
        const int64_t base = row * 2 * width + col;
        output[index] = output_t(
            ggml_cuda_op_silu_single(__half2float(gate_up[base])) *
            __half2float(gate_up[base + width]));
    }
}

template <>
__global__ void quantize_rows_i8<float>(
        const float * input,
        int8_t * quantized,
        float * scales,
        int64_t width) {
    const int64_t row = blockIdx.x;
    const float * row_input = input + row * width;
    int8_t * row_output = quantized + row * width;

    const int64_t vectors = width / 4;
    const float4 * row_input4 = reinterpret_cast<const float4 *>(row_input);
    float local_max = 0.0f;
    for (int64_t col = threadIdx.x; col < vectors; col += blockDim.x) {
        const float4 value = row_input4[col];
        local_max = fmaxf(local_max, fabsf(value.x));
        local_max = fmaxf(local_max, fabsf(value.y));
        local_max = fmaxf(local_max, fabsf(value.z));
        local_max = fmaxf(local_max, fabsf(value.w));
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

    const float scale = maxima[0] / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float inverse = maxima[0] == 0.0f ? 0.0f : 127.0f / maxima[0];
    char4 * row_output4 = reinterpret_cast<char4 *>(row_output);
    for (int64_t col = threadIdx.x; col < vectors; col += blockDim.x) {
        const float4 value = row_input4[col];
        row_output4[col] = {
            float_to_i8_rn(value.x * inverse),
            float_to_i8_rn(value.y * inverse),
            float_to_i8_rn(value.z * inverse),
            float_to_i8_rn(value.w * inverse),
        };
    }
}

__global__ void quantize_rows_i8_outliers(
        const float * input,
        int8_t * quantized,
        float * scales,
        const uint8_t * outlier_columns,
        int64_t width) {
    const int64_t row = blockIdx.x;
    const float * row_input = input + row * width;
    int8_t * row_output = quantized + row * width;

    const int64_t vectors = width / 4;
    const float4 * row_input4 = reinterpret_cast<const float4 *>(row_input);
    float local_max = 0.0f;
    for (int64_t col = threadIdx.x; col < vectors; col += blockDim.x) {
        const float4 value = row_input4[col];
        const float values[4] = { value.x, value.y, value.z, value.w };
#pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            const float magnitude = fabsf(values[lane]);
            if (!outlier_columns[4 * col + lane]) {
                local_max = fmaxf(local_max, magnitude);
            }
        }
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

    const float scale = maxima[0] / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float inverse = maxima[0] == 0.0f ? 0.0f : 127.0f / maxima[0];
    char4 * row_output4 = reinterpret_cast<char4 *>(row_output);
    for (int64_t col = threadIdx.x; col < vectors; col += blockDim.x) {
        const float4 value = row_input4[col];
        row_output4[col] = {
            outlier_columns[4 * col + 0] ? int8_t(0) : float_to_i8_rn(value.x * inverse),
            outlier_columns[4 * col + 1] ? int8_t(0) : float_to_i8_rn(value.y * inverse),
            outlier_columns[4 * col + 2] ? int8_t(0) : float_to_i8_rn(value.z * inverse),
            outlier_columns[4 * col + 3] ? int8_t(0) : float_to_i8_rn(value.w * inverse),
        };
    }
}

template <>
__global__ void quantize_rows_i8<nv_bfloat16>(
        const nv_bfloat16 * input,
        int8_t * quantized,
        float * scales,
        int64_t width) {
    const int64_t row = blockIdx.x;
    const nv_bfloat16 * row_input = input + row * width;
    int8_t * row_output = quantized + row * width;

    const int64_t vectors = width / 8;
    const nv_bfloat162 * row_input2 = reinterpret_cast<const nv_bfloat162 *>(row_input);
    float local_max = 0.0f;
    for (int64_t col = threadIdx.x; col < vectors; col += blockDim.x) {
        const int64_t pair = col * 4;
        const float2 v0 = __bfloat1622float2(row_input2[pair + 0]);
        const float2 v1 = __bfloat1622float2(row_input2[pair + 1]);
        const float2 v2 = __bfloat1622float2(row_input2[pair + 2]);
        const float2 v3 = __bfloat1622float2(row_input2[pair + 3]);
        local_max = fmaxf(local_max, fabsf(v0.x));
        local_max = fmaxf(local_max, fabsf(v0.y));
        local_max = fmaxf(local_max, fabsf(v1.x));
        local_max = fmaxf(local_max, fabsf(v1.y));
        local_max = fmaxf(local_max, fabsf(v2.x));
        local_max = fmaxf(local_max, fabsf(v2.y));
        local_max = fmaxf(local_max, fabsf(v3.x));
        local_max = fmaxf(local_max, fabsf(v3.y));
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

    const float scale = maxima[0] / 127.0f;
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float inverse = maxima[0] == 0.0f ? 0.0f : 127.0f / maxima[0];
    int64_t * row_output8 = reinterpret_cast<int64_t *>(row_output);
    for (int64_t col = threadIdx.x; col < vectors; col += blockDim.x) {
        const int64_t pair = col * 4;
        const float2 v0 = __bfloat1622float2(row_input2[pair + 0]);
        const float2 v1 = __bfloat1622float2(row_input2[pair + 1]);
        const float2 v2 = __bfloat1622float2(row_input2[pair + 2]);
        const float2 v3 = __bfloat1622float2(row_input2[pair + 3]);
        const char4 q0 = {
            float_to_i8_rn(v0.x * inverse),
            float_to_i8_rn(v0.y * inverse),
            float_to_i8_rn(v1.x * inverse),
            float_to_i8_rn(v1.y * inverse),
        };
        const char4 q1 = {
            float_to_i8_rn(v2.x * inverse),
            float_to_i8_rn(v2.y * inverse),
            float_to_i8_rn(v3.x * inverse),
            float_to_i8_rn(v3.y * inverse),
        };
        int64_t packed;
        memcpy(&packed, &q0, sizeof(q0));
        memcpy(reinterpret_cast<char *>(&packed) + sizeof(q0), &q1, sizeof(q1));
        row_output8[col] = packed;
    }
}

template<typename scale_t>
__global__ void scale_i32_output(
        const int32_t * input,
        float * output,
        const float * activation_scales,
        const scale_t * weight_scales,
        int64_t rows,
        int64_t count) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int64_t row = index % rows;
    const int64_t token = index / rows;
    output[index] = float(input[index]) * activation_scales[token] * float(weight_scales[row]);
}

template<typename scale_t>
__global__ void gemv_i8_channel(
        const int8_t * __restrict__ weight,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scales,
        const scale_t * __restrict__ weight_scales,
        float * __restrict__ output,
        int64_t width) {
    const int64_t row = blockIdx.x;
    const int4 * weight16 = reinterpret_cast<const int4 *>(weight + row * width);
    const int4 * activation16 = reinterpret_cast<const int4 *>(activation);

    int sum = 0;
    for (int64_t col = threadIdx.x; col < width / 16; col += blockDim.x) {
        const int4 w = weight16[col];
        const int4 x = activation16[col];
        sum = __dp4a(w.x, x.x, sum);
        sum = __dp4a(w.y, x.y, sum);
        sum = __dp4a(w.z, x.z, sum);
        sum = __dp4a(w.w, x.w, sum);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ int warp_sums[32];
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        sum = lane < blockDim.x / warpSize ? warp_sums[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            output[row] = float(sum) * activation_scales[0] * float(weight_scales[row]);
        }
    }
}

template<typename scale_t>
__global__ void gemv_i8_channel_swiglu(
        const int8_t * __restrict__ up_weight,
        const int8_t * __restrict__ gate_weight,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scales,
        const scale_t * __restrict__ up_scales,
        const scale_t * __restrict__ gate_scales,
        float * __restrict__ output,
        int64_t width) {
    const int64_t row = blockIdx.x;
    const int4 * up16 = reinterpret_cast<const int4 *>(up_weight + row * width);
    const int4 * gate16 = reinterpret_cast<const int4 *>(gate_weight + row * width);
    const int4 * activation16 = reinterpret_cast<const int4 *>(activation);

    int up_sum = 0;
    int gate_sum = 0;
    for (int64_t col = threadIdx.x; col < width / 16; col += blockDim.x) {
        const int4 x = activation16[col];
        const int4 up = up16[col];
        const int4 gate = gate16[col];
        up_sum = __dp4a(up.x, x.x, up_sum);
        up_sum = __dp4a(up.y, x.y, up_sum);
        up_sum = __dp4a(up.z, x.z, up_sum);
        up_sum = __dp4a(up.w, x.w, up_sum);
        gate_sum = __dp4a(gate.x, x.x, gate_sum);
        gate_sum = __dp4a(gate.y, x.y, gate_sum);
        gate_sum = __dp4a(gate.z, x.z, gate_sum);
        gate_sum = __dp4a(gate.w, x.w, gate_sum);
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
        gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
    }

    __shared__ int up_warp_sums[32];
    __shared__ int gate_warp_sums[32];
    const int lane = threadIdx.x % warpSize;
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) {
        up_warp_sums[warp] = up_sum;
        gate_warp_sums[warp] = gate_sum;
    }
    __syncthreads();

    if (warp == 0) {
        up_sum = lane < blockDim.x / warpSize ? up_warp_sums[lane] : 0;
        gate_sum = lane < blockDim.x / warpSize ? gate_warp_sums[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            up_sum += __shfl_down_sync(0xffffffff, up_sum, offset);
            gate_sum += __shfl_down_sync(0xffffffff, gate_sum, offset);
        }
        if (lane == 0) {
            const float activation_scale = activation_scales[0];
            const float up = float(up_sum) * activation_scale * float(up_scales[row]);
            const float gate = float(gate_sum) * activation_scale * float(gate_scales[row]);
            output[row] = ggml_cuda_op_silu_single(gate) * up;
        }
    }
}

template<typename weight_scale_t, typename output_t>
bool cutlass_scaled_i8_sm80(
        ggml_backend_cuda_context & ctx,
        const int8_t * activation,
        const int8_t * weight,
        const float * activation_scales,
        const weight_scale_t * weight_scales,
        output_t * output,
        int m,
        int n,
        int k) {
    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        TileShape, WarpShape, float, 4, 1>;
    using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
    using ScaleM = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap, float, Stride<Int<1>, Int<0>, Int<0>>>;
    using ScaleN = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, weight_scale_t, Stride<Int<0>, Int<1>, Int<0>>>;
    using MulN = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTMulN = cutlass::epilogue::threadblock::Sm80EVT<MulN, ScaleN, Accum>;
    using MulM = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, output_t, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTCompute = cutlass::epilogue::threadblock::Sm80EVT<MulM, ScaleM, EVTMulN>;
    using D = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap, output_t, cutlass::FloatRoundStyle::round_to_nearest,
        Stride<int64_t, Int<1>, Int<0>>>;
    using EVTD = cutlass::epilogue::threadblock::Sm80EVT<D, EVTCompute>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
        float, cutlass::layout::RowMajor, 4,
        int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        TileShape, WarpShape, InstructionShape, EVTD,
        cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, 5,
        cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    typename D::Arguments d_args{output, {int64_t(n), Int<1>{}, Int<0>{}}};
    typename ScaleM::Arguments scale_m_args{activation_scales};
    typename ScaleN::Arguments scale_n_args{weight_scales};
    typename EVTMulN::Arguments scale_n_evt{scale_n_args, {}, {}};
    typename EVTCompute::Arguments compute_evt{scale_m_args, scale_n_evt, {}};
    typename EVTD::Arguments epilogue_args{compute_evt, d_args};

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemmSplitKParallel,
        {m, n, k}, 1, epilogue_args,
        activation, weight, nullptr, nullptr,
        0, 0, 0, 0,
        k, k, n, n};
    Gemm gemm;
    const size_t workspace_size = gemm.get_workspace_size(args);
    ggml_cuda_pool_alloc<uint8_t> workspace(ctx.pool());
    if (workspace_size != 0) {
        workspace.alloc(workspace_size);
    }
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        return false;
    }
    return gemm(args, workspace.get(), ctx.stream()) == cutlass::Status::kSuccess;
}

static void quantize_activation(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * input,
        int8_t * quantized,
        float * scales,
        int64_t k,
        int64_t m) {
    const nv_bfloat16 * cached_bf16 = ggml_cuda_get_cached_bf16_input(
        ctx, input, size_t(k) * m);
    if (cached_bf16 != nullptr) {
        GGML_ASSERT(k % 8 == 0);
        quantize_rows_i8<nv_bfloat16><<<m, 256, 0, ctx.stream()>>>(
            cached_bf16, quantized, scales, k);
    } else {
        quantize_rows_i8<float><<<m, 256, 0, ctx.stream()>>>(
            static_cast<const float *>(input->data), quantized, scales, k);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ggml_cuda_int8_detail

bool ggml_cuda_int8_channel_supports(
        const ggml_tensor * weight,
        const ggml_tensor * input,
        const ggml_tensor * scale,
        const ggml_tensor * output,
        int cc) {
    return weight != nullptr && input != nullptr && scale != nullptr && output != nullptr &&
        weight->type == GGML_TYPE_I8 && input->type == GGML_TYPE_F32 &&
        output->type == GGML_TYPE_F32 &&
        (scale->type == GGML_TYPE_F32 || scale->type == GGML_TYPE_F16 || scale->type == GGML_TYPE_BF16) &&
        cc >= GGML_CUDA_CC_TURING &&
        ggml_is_contiguous(weight) && ggml_is_contiguous(input) &&
        ggml_is_contiguous(output) && ggml_is_contiguous(scale) &&
        weight->ne[2] == 1 && weight->ne[3] == 1 &&
        input->ne[2] == 1 && input->ne[3] == 1 &&
        weight->ne[0] == input->ne[0] && output->ne[0] == weight->ne[1] &&
        output->ne[1] == input->ne[1] && scale->ne[0] == weight->ne[1] &&
        scale->ne[1] == 1 && scale->ne[2] == 1 && scale->ne[3] == 1 &&
        weight->ne[0] % 4 == 0 && weight->ne[1] % 4 == 0 && input->ne[1] > 0;
}

bool ggml_cuda_mul_mat_int8_channel(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight,
        const ggml_tensor * input,
        ggml_tensor * output) {
    using namespace ggml_cuda_int8_detail;
    const ggml_tensor * scale = output->src[2];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_int8_channel_supports(weight, input, scale, output, cc)) {
        return false;
    }

    const int64_t k = weight->ne[0];
    const int64_t n = weight->ne[1];
    const int64_t m = input->ne[1];
    if (k % 4 != 0 || n % 4 != 0 || m <= 0) {
        return false;
    }

    ggml_cuda_pool_alloc<int8_t> quantized_storage(ctx.pool());
    ggml_cuda_pool_alloc<float> activation_scale_storage(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> weight_sum_storage(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> outlier_column_storage(ctx.pool());
    int8_t * quantized = nullptr;
    float * activation_scales = nullptr;
    uint8_t * outlier_columns = nullptr;
    const ggml_tensor * static_input_scale = output->src[3];
    const bool static_asym = static_input_scale != nullptr && static_input_scale->type == GGML_TYPE_I64;
    const bool thresholded = static_input_scale != nullptr && static_input_scale->type == GGML_TYPE_I32;
    if (static_input_scale != nullptr) {
        if ((static_input_scale->type != GGML_TYPE_F32 && !static_asym && !thresholded) ||
            !ggml_is_contiguous(static_input_scale) ||
            ggml_nelements(static_input_scale) != 1 || input->type != GGML_TYPE_F32) {
            return false;
        }
        quantized = quantized_storage.alloc(size_t(k) * m);
        activation_scales = activation_scale_storage.alloc(m);
        if (static_asym) {
            quantize_rows_i8_static_asym<<<m, 256, 0, ctx.stream()>>>(
                static_cast<const float *>(input->data), quantized, activation_scales,
                static_cast<const int64_t *>(static_input_scale->data), k);
        } else if (thresholded) {
            outlier_columns = outlier_column_storage.alloc(k);
            mark_i8_outlier_columns<<<(k + 255) / 256, 256, 0, ctx.stream()>>>(
                static_cast<const float *>(input->data), outlier_columns,
                static_cast<const int32_t *>(static_input_scale->data), k, m);
            quantize_rows_i8_outliers<<<m, 256, 0, ctx.stream()>>>(
                static_cast<const float *>(input->data), quantized, activation_scales,
                outlier_columns, k);
        } else {
            quantize_rows_i8_static<<<m, 256, 0, ctx.stream()>>>(
                static_cast<const float *>(input->data), quantized, activation_scales,
                static_cast<const float *>(static_input_scale->data), k);
        }
        CUDA_CHECK(cudaGetLastError());
    } else if (ctx.consume_int8_channel_activation(input)) {
        quantized = static_cast<int8_t *>(input->data);
        activation_scales = reinterpret_cast<float *>(quantized + size_t(k) * m);
    } else {
        quantized = quantized_storage.alloc(size_t(k) * m);
        activation_scales = activation_scale_storage.alloc(m);
        quantize_activation(ctx, input, quantized, activation_scales, k, m);
    }
    cudaStream_t stream = ctx.stream();
    int32_t * weight_sums = nullptr;
    if (static_asym) {
        weight_sums = weight_sum_storage.alloc(n);
        sum_i8_rows<<<n, 256, 0, stream>>>(
            static_cast<const int8_t *>(weight->data), weight_sums, k);
        CUDA_CHECK(cudaGetLastError());
    }

    const auto correct_asym = [&](auto * weight_scales) {
        if (!static_asym) {
            return;
        }
        const int64_t count = n * m;
        correct_i8_asym_output<<<(count + 255) / 256, 256, 0, stream>>>(
            static_cast<float *>(output->data), weight_sums,
            static_cast<const int64_t *>(static_input_scale->data), weight_scales, n, count);
        CUDA_CHECK(cudaGetLastError());
    };
    const auto correct_outliers = [&](auto * weight_scales) {
        if (!thresholded) {
            return;
        }
        add_i8_outlier_output<<<dim3(n, m), 256, 0, stream>>>(
            static_cast<const float *>(input->data), static_cast<const int8_t *>(weight->data),
            outlier_columns, weight_scales,
            static_cast<float *>(output->data), k, n);
        CUDA_CHECK(cudaGetLastError());
    };

    if (cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_ADA_LOVELACE && m == 1 && k % 16 == 0) {
        constexpr int threads = 64;
        if (scale->type == GGML_TYPE_F32) {
            gemv_i8_channel<<<n, threads, 0, stream>>>(
                static_cast<const int8_t *>(weight->data), quantized, activation_scales,
                static_cast<const float *>(scale->data), static_cast<float *>(output->data), k);
            correct_asym(static_cast<const float *>(scale->data));
            correct_outliers(static_cast<const float *>(scale->data));
        } else if (scale->type == GGML_TYPE_F16) {
            gemv_i8_channel<<<n, threads, 0, stream>>>(
                static_cast<const int8_t *>(weight->data), quantized, activation_scales,
                static_cast<const half *>(scale->data), static_cast<float *>(output->data), k);
            correct_asym(static_cast<const half *>(scale->data));
            correct_outliers(static_cast<const half *>(scale->data));
        } else {
            gemv_i8_channel<<<n, threads, 0, stream>>>(
                static_cast<const int8_t *>(weight->data), quantized, activation_scales,
                static_cast<const nv_bfloat16 *>(scale->data), static_cast<float *>(output->data), k);
            correct_asym(static_cast<const nv_bfloat16 *>(scale->data));
            correct_outliers(static_cast<const nv_bfloat16 *>(scale->data));
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    if (cc >= GGML_CUDA_CC_AMPERE && cc < GGML_CUDA_CC_ADA_LOVELACE && m > 128) {
        auto * dst = static_cast<float *>(output->data);
        const bool launched = scale->type == GGML_TYPE_F32 ?
            cutlass_scaled_i8_sm80(ctx, quantized, static_cast<const int8_t *>(weight->data),
                activation_scales, static_cast<const float *>(scale->data), dst, m, n, k) :
            scale->type == GGML_TYPE_F16 ?
            cutlass_scaled_i8_sm80(ctx, quantized, static_cast<const int8_t *>(weight->data),
                activation_scales, static_cast<const cutlass::half_t *>(scale->data), dst, m, n, k) :
            cutlass_scaled_i8_sm80(ctx, quantized, static_cast<const int8_t *>(weight->data),
                activation_scales, static_cast<const cutlass::bfloat16_t *>(scale->data), dst, m, n, k);
        if (launched) {
            if (scale->type == GGML_TYPE_F32) {
                correct_asym(static_cast<const float *>(scale->data));
                correct_outliers(static_cast<const float *>(scale->data));
            } else if (scale->type == GGML_TYPE_F16) {
                correct_asym(static_cast<const half *>(scale->data));
                correct_outliers(static_cast<const half *>(scale->data));
            } else {
                correct_asym(static_cast<const nv_bfloat16 *>(scale->data));
                correct_outliers(static_cast<const nv_bfloat16 *>(scale->data));
            }
            return true;
        }
    }

    ggml_cuda_pool_alloc<int32_t> accumulator(ctx.pool(), size_t(n) * m);
    const int32_t alpha = 1;
    const int32_t beta = 0;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), stream));
    CUBLAS_CHECK(cublasGemmEx(
        ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        weight->data, CUDA_R_8I, k,
        quantized, CUDA_R_8I, k,
        &beta,
        accumulator.get(), CUDA_R_32I, n,
        CUBLAS_COMPUTE_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    const int64_t count = n * m;
    const int blocks = int((count + 255) / 256);
    if (scale->type == GGML_TYPE_F32) {
        scale_i32_output<<<blocks, 256, 0, stream>>>(
            accumulator.get(), static_cast<float *>(output->data), activation_scales,
            static_cast<const float *>(scale->data), n, count);
        correct_asym(static_cast<const float *>(scale->data));
        correct_outliers(static_cast<const float *>(scale->data));
    } else if (scale->type == GGML_TYPE_F16) {
        scale_i32_output<<<blocks, 256, 0, stream>>>(
            accumulator.get(), static_cast<float *>(output->data), activation_scales,
            static_cast<const half *>(scale->data), n, count);
        correct_asym(static_cast<const half *>(scale->data));
        correct_outliers(static_cast<const half *>(scale->data));
    } else {
        scale_i32_output<<<blocks, 256, 0, stream>>>(
            accumulator.get(), static_cast<float *>(output->data), activation_scales,
            static_cast<const nv_bfloat16 *>(scale->data), n, count);
        correct_asym(static_cast<const nv_bfloat16 *>(scale->data));
        correct_outliers(static_cast<const nv_bfloat16 *>(scale->data));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_mul_mat_int8_channel_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * weight,
        const ggml_tensor * input,
        const ggml_tensor * scale,
        ggml_tensor * output,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    using namespace ggml_cuda_int8_detail;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (fusion == nullptr || fusion->residual == nullptr || fusion->rms_weight == nullptr ||
            fusion->x_bias != nullptr || fusion->gate != nullptr ||
            fusion->gate_bias != nullptr || fusion->x_scale != nullptr ||
            fusion->gate_scale != nullptr ||
            output->src[3] != nullptr ||
            !ggml_cuda_int8_channel_supports(weight, input, scale, output, cc)) {
        return false;
    }

    const int64_t k = weight->ne[0];
    const int64_t n = weight->ne[1];
    const int64_t m = input->ne[1];
    if (cc < GGML_CUDA_CC_AMPERE || cc >= GGML_CUDA_CC_ADA_LOVELACE ||
            k % 8 != 0 || m <= 128) {
        return false;
    }

    ggml_cuda_pool_alloc<int8_t> quantized(ctx.pool(), size_t(k) * m);
    ggml_cuda_pool_alloc<float> activation_scales(ctx.pool(), m);
    quantize_activation(ctx, input, quantized.get(), activation_scales.get(), k, m);

    const size_t output_count = size_t(n) * m;
    auto & scratch = ctx.humming_outputs[ctx.curr_stream_no];
    if (scratch.count < output_count) {
        if (scratch.ptr != nullptr) {
            scratch.retired.push_back(scratch.ptr);
        }
        CUDA_CHECK(cudaMalloc(&scratch.ptr, output_count * sizeof(nv_bfloat16)));
        scratch.count = output_count;
    }

    const bool launched = scale->type == GGML_TYPE_F32 ?
        cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(weight->data),
            activation_scales.get(), static_cast<const float *>(scale->data),
            reinterpret_cast<cutlass::bfloat16_t *>(scratch.ptr), m, n, k) : scale->type == GGML_TYPE_F16 ?
        cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(weight->data),
            activation_scales.get(), static_cast<const cutlass::half_t *>(scale->data),
            reinterpret_cast<cutlass::bfloat16_t *>(scratch.ptr), m, n, k) :
        cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(weight->data),
            activation_scales.get(), static_cast<const cutlass::bfloat16_t *>(scale->data),
            reinterpret_cast<cutlass::bfloat16_t *>(scratch.ptr), m, n, k);
    return launched && ggml_cuda_humming_finish_residual_rms(
        ctx, fusion, scratch.ptr, output, n, m, ctx.stream());
}

bool ggml_cuda_mul_mat_int8_channel_swiglu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * input,
        ggml_tensor * output,
        bool retain_bf16_output) {
    using namespace ggml_cuda_int8_detail;
    const ggml_tensor * up_weight = up != nullptr ? up->src[0] : nullptr;
    const ggml_tensor * up_scale = up != nullptr ? up->src[2] : nullptr;
    const ggml_tensor * gate_weight = gate != nullptr ? gate->src[0] : nullptr;
    const ggml_tensor * gate_scale = gate != nullptr ? gate->src[2] : nullptr;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (up == nullptr || gate == nullptr || input == nullptr || output == nullptr ||
            up->src[1] != input || gate->src[1] != input ||
            up->src[3] != nullptr || gate->src[3] != nullptr ||
            !ggml_are_same_shape(up, gate) || !ggml_are_same_shape(up, output) ||
            !ggml_cuda_int8_channel_supports(up_weight, input, up_scale, up, cc) ||
            !ggml_cuda_int8_channel_supports(gate_weight, input, gate_scale, gate, cc)) {
        return false;
    }

    const int64_t k = up_weight->ne[0];
    const int64_t n = up_weight->ne[1];
    const int64_t m = input->ne[1];
    if (cc < GGML_CUDA_CC_AMPERE || cc >= GGML_CUDA_CC_ADA_LOVELACE || k % 8 != 0) {
        return false;
    }
    if (m <= 128 && m != 1) {
        return false;
    }

    ggml_cuda_pool_alloc<int8_t> quantized(ctx.pool(), size_t(k) * m);
    ggml_cuda_pool_alloc<float> activation_scales(ctx.pool(), m);
    quantize_activation(ctx, input, quantized.get(), activation_scales.get(), k, m);

    if (m == 1 && up_scale->type == gate_scale->type) {
        constexpr int threads = 64;
        if (up_scale->type == GGML_TYPE_F32) {
            gemv_i8_channel_swiglu<<<n, threads, 0, ctx.stream()>>>(
                static_cast<const int8_t *>(up_weight->data),
                static_cast<const int8_t *>(gate_weight->data), quantized.get(), activation_scales.get(),
                static_cast<const float *>(up_scale->data), static_cast<const float *>(gate_scale->data),
                static_cast<float *>(output->data), k);
        } else if (up_scale->type == GGML_TYPE_F16) {
            gemv_i8_channel_swiglu<<<n, threads, 0, ctx.stream()>>>(
                static_cast<const int8_t *>(up_weight->data),
                static_cast<const int8_t *>(gate_weight->data), quantized.get(), activation_scales.get(),
                static_cast<const half *>(up_scale->data), static_cast<const half *>(gate_scale->data),
                static_cast<float *>(output->data), k);
        } else {
            gemv_i8_channel_swiglu<<<n, threads, 0, ctx.stream()>>>(
                static_cast<const int8_t *>(up_weight->data),
                static_cast<const int8_t *>(gate_weight->data), quantized.get(), activation_scales.get(),
                static_cast<const nv_bfloat16 *>(up_scale->data),
                static_cast<const nv_bfloat16 *>(gate_scale->data),
                static_cast<float *>(output->data), k);
        }
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    const bool adjacent_weights = gate_weight->buffer == up_weight->buffer &&
        static_cast<const char *>(gate_weight->data) + ggml_nbytes(gate_weight) == up_weight->data;
    const bool adjacent_scales = gate_scale->buffer == up_scale->buffer &&
        gate_scale->type == up_scale->type &&
        static_cast<const char *>(gate_scale->data) + ggml_nbytes(gate_scale) == up_scale->data;
    if (n % 128 == 0 && adjacent_weights && adjacent_scales) {
        ggml_cuda_pool_alloc<half> gate_up_output(ctx.pool(), size_t(2 * n) * m);
        const bool launched = gate_scale->type == GGML_TYPE_F32 ?
            cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(gate_weight->data),
                activation_scales.get(), static_cast<const float *>(gate_scale->data),
                reinterpret_cast<cutlass::half_t *>(gate_up_output.get()), m, 2 * n, k) :
            gate_scale->type == GGML_TYPE_F16 ?
            cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(gate_weight->data),
                activation_scales.get(), static_cast<const cutlass::half_t *>(gate_scale->data),
                reinterpret_cast<cutlass::half_t *>(gate_up_output.get()), m, 2 * n, k) :
            cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(gate_weight->data),
                activation_scales.get(), static_cast<const cutlass::bfloat16_t *>(gate_scale->data),
                reinterpret_cast<cutlass::half_t *>(gate_up_output.get()), m, 2 * n, k);
        if (launched) {
            const size_t count = size_t(n) * m;
            constexpr int threads = 256;
            const int blocks = ((count + 1) / 2 + threads - 1) / threads;
            if (retain_bf16_output) {
                swiglu_stacked_f16<<<blocks, threads, 0, ctx.stream()>>>(
                    gate_up_output.get(), static_cast<nv_bfloat16 *>(output->data), n, count);
                ctx.humming_bf16_activations.insert(output);
            } else {
                swiglu_stacked_f16<<<blocks, threads, 0, ctx.stream()>>>(
                    gate_up_output.get(), static_cast<float *>(output->data), n, count);
            }
            CUDA_CHECK(cudaGetLastError());
            return true;
        }
    }

    ggml_cuda_pool_alloc<nv_bfloat16> up_output(ctx.pool(), size_t(n) * m);
    ggml_cuda_pool_alloc<nv_bfloat16> gate_output(ctx.pool(), size_t(n) * m);

    const auto launch = [&](const ggml_tensor * weight, const ggml_tensor * scale,
                            nv_bfloat16 * dst) {
        return scale->type == GGML_TYPE_F32 ?
            cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(weight->data),
                activation_scales.get(), static_cast<const float *>(scale->data),
                reinterpret_cast<cutlass::bfloat16_t *>(dst), m, n, k) : scale->type == GGML_TYPE_F16 ?
            cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(weight->data),
                activation_scales.get(), static_cast<const cutlass::half_t *>(scale->data),
                reinterpret_cast<cutlass::bfloat16_t *>(dst), m, n, k) :
            cutlass_scaled_i8_sm80(ctx, quantized.get(), static_cast<const int8_t *>(weight->data),
                activation_scales.get(), static_cast<const cutlass::bfloat16_t *>(scale->data),
                reinterpret_cast<cutlass::bfloat16_t *>(dst), m, n, k);
    };
    if (!launch(up_weight, up_scale, up_output.get()) ||
            !launch(gate_weight, gate_scale, gate_output.get())) {
        return false;
    }

    const size_t count = size_t(n) * m;
    if (retain_bf16_output) {
        ggml_cuda_humming_fp8_swiglu_bf16(
            up_output.get(), gate_output.get(),
            static_cast<nv_bfloat16 *>(output->data), count, ctx.stream());
        ctx.humming_bf16_activations.insert(output);
    } else {
        ggml_cuda_humming_fp8_output_bf16_to_f32(
            up_output.get(), gate_output.get(),
            static_cast<float *>(output->data), count, ctx.stream());
    }
    return true;
}

#else

bool ggml_cuda_int8_channel_supports(
        const ggml_tensor *,
        const ggml_tensor *,
        const ggml_tensor *,
        const ggml_tensor *,
        int) {
    return false;
}

bool ggml_cuda_mul_mat_int8_channel(
        ggml_backend_cuda_context &,
        const ggml_tensor *,
        const ggml_tensor *,
        ggml_tensor *) {
    return false;
}

bool ggml_cuda_mul_mat_int8_channel_fused(
        ggml_backend_cuda_context &,
        const ggml_tensor *,
        const ggml_tensor *,
        const ggml_tensor *,
        ggml_tensor *,
        const ggml_cuda_mm_fusion_args_host *) {
    return false;
}

bool ggml_cuda_mul_mat_int8_channel_swiglu(
        ggml_backend_cuda_context &,
        const ggml_tensor *,
        const ggml_tensor *,
        const ggml_tensor *,
        ggml_tensor *,
        bool) {
    return false;
}

#endif
