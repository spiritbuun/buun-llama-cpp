#include "humming-fp8.cuh"
#include "unary.cuh"

#if !defined(GGML_USE_HIP)

// Humming uses preprocessor gates to remove storage and control paths that a
// generated kernel configuration does not need. Every instantiation in this
// translation unit has the same dense/channel-scaled policy; N/K and tile
// shapes remain ordinary template parameters below.
#define HUMMING_INPUT_SCALE_GROUP_SIZE 0
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE 0
#define HUMMING_HAS_ZERO_POINT 0
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE 1
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE_2 0
#define HUMMING_HAS_BIAS 0
#define HUMMING_HAS_INPUT_SCALE 0
#define HUMMING_IS_INDEXED_GEMM 0
#define HUMMING_IS_GROUPED_GEMM 0
#define HUMMING_IS_GROUPED_CONTIGUOUS_GEMM 0
#define HUMMING_USE_MBARRIER 0
#define HUMMING_USE_WARP_SPEC 0
#define HUMMING_REDUCE_OVERLAP_LAST_STAGE_ONLY 0

#include <humming/kernel/humming.cuh>
#include <humming/kernel/process.cuh>

#include <atomic>
#include <thread>
#include <vector>

namespace {

// Humming commit 636ba8564 (Apache-2.0) is vendored under humming-vendor/.
// These policy classes are the dense BF16 x channel-scaled E4M3 subset of its
// generated kernel contract. Keeping the subset here makes the production
// dispatch independent of Python, PyTorch, NVRTC, and a writable JIT cache.
class fp8_mma_op {
public:
    static constexpr MmaType kMmaType = MmaType::MMA;
    using MmaShape = Shape<16, 8, 16>;
    using ValTypeC = float;
    using ValTypeD = float;

    static constexpr uint32_t kATypeBits = 16;
    static constexpr uint32_t kBTypeBits = 16;
    static constexpr uint32_t kCTypeBits = 32;
    static constexpr uint32_t kDTypeBits = 32;
    static constexpr bool kNativeMixed = false;

    using ARegisters = uint32_t[4];
    using BRegisters = uint32_t[2];
    using CRegisters = float[4];
    using DRegisters = float[4];

    __device__ __forceinline__ static void fma(uint32_t * a, uint32_t * b, float * c, float * d) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(b[0]), "r"(b[1]),
              "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    }
};

template <uint32_t N, uint32_t K>
class fp8_layer_config {
public:
    static constexpr auto kShapeN = N;
    static constexpr auto kShapeK = K;
    static constexpr auto kPadShapeN = 0u;
    static constexpr auto kPadShapeK = 0u;
    static constexpr auto kNumExperts = 0u;
    static constexpr auto kInputScaleGroupSize = 0u;
    static constexpr auto kWeightScaleGroupSize = 0u;
    static constexpr auto kWeightScaleGroupSizeN = 0u;
    static constexpr auto kWeightScaleType = WeightScaleType::CHANNEL;
    static constexpr auto kWeightScale2Type = WeightScale2Type::NONE;
    static constexpr auto kUseIntWeightScale = false;
    static constexpr auto kUseFusedE8m0Scale = false;
    static constexpr auto kHasZeroPoint = false;
    static constexpr auto kIsFpZeroPoint = false;
    static constexpr auto kHasBias = false;
    static constexpr auto kMmaType = MmaType::MMA;
    static constexpr auto kUsePackedKLayout = false;
    static constexpr auto kMmaTypeId = 0u;
    static constexpr auto kIsChannelWeightScale = true;
    static constexpr auto kIsBlockWeightScale = false;
    static constexpr auto kIsGroupWeightScale = false;
    static constexpr auto kIsTensorWeightScale = false;
    static constexpr auto kIsChannelWeightScale2 = false;
    static constexpr auto kIsTensorWeightScale2 = false;
    static constexpr auto kHasChannelWeightScale = true;
    static constexpr auto kHasTensorWeightScale = false;
    static constexpr auto kHasInputScale = false;
};

class fp8_compute_config {
public:
    static constexpr auto kUseF16Accum = false;
    static constexpr auto kUseBatchInvariant = false;
    static constexpr auto kUseMMajorInputScale = false;
    static constexpr auto kGemmType = GemmType::DENSE;
    static constexpr auto kGemmTypeId = 0u;
    static constexpr auto kIsIndexedGemm = false;
    static constexpr auto kIsGroupedGemm = false;
    static constexpr auto kIsGroupedContiguousGemm = false;
    static constexpr auto kIsGroupedMaskedGemm = false;
};

template <uint32_t Threads, uint32_t Raster, bool UseStreamK>
class fp8_tuning_config {
public:
    static constexpr auto kUseStreamK = UseStreamK;
    static constexpr auto kNumStages = 3u;
    static constexpr auto kNumCtasPerSm = 1u;
    static constexpr auto kUseWarpSpec = false;
    static constexpr auto kUseMBarrier = false;
    static constexpr auto kUseCpAsync = true;
    static constexpr auto kUseTma = false;
    static constexpr auto kUseTmaA = false;
    static constexpr auto kUseTmaAS = false;
    static constexpr auto kUseTmaB = false;
    static constexpr auto kUseTmaC = false;
    static constexpr auto kUseTmaBS = false;
    static constexpr auto kUseTmaBS2 = false;
    static constexpr auto kUseTmaBZP = false;
    static constexpr auto kUseTmaBias = false;
    static constexpr auto kReduceOverlapLastStageOnly = false;
    static constexpr auto kNumWriteSplits = 1u;
    static constexpr auto kMultiCastSizeA = 1u;
    static constexpr auto kMultiCastSizeB = 1u;
    static constexpr auto kRasterGroupM = Raster;
    static constexpr auto kNumThreads = Threads;
    static constexpr auto kNumMathThreads = Threads;
    static constexpr auto kNumLoadThreads = Threads;
};

using fp8_a = FloatingPointType<16, 8, 7>;
using fp8_b = FloatingPointType<8, 4, 3>;
using fp8_c = BFloat16;
using fp8_s = FloatingPointType<16, 8, 7>;

template <uint32_t N, uint32_t K, class BlockShape, class WarpShape, uint32_t Threads, uint32_t Raster,
          bool UseStreamK>
void launch_fp8(
        const nv_bfloat16 * input,
        const void * weight,
        const nv_bfloat16 * scale,
        void * output,
        int32_t * locks,
        uint32_t m,
        int sms,
        cudaStream_t stream) {
    using layer = fp8_layer_config<N, K>;
    using tuning = fp8_tuning_config<Threads, Raster, UseStreamK>;
    using storage = SharedStorage<fp8_mma_op, BlockShape, WarpShape, fp8_a, fp8_b, fp8_s,
                                  layer, fp8_compute_config, tuning>;
    constexpr auto kernel = humming<
        fp8_mma_op,
        Shape<0, N, K>, BlockShape, WarpShape, Shape<0, 0, 0>,
        fp8_a, fp8_b, fp8_c, fp8_s,
        layer, fp8_compute_config, tuning>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(storage)));
    kernel<<<sms, Threads, sizeof(storage), stream>>>(
        const_cast<nv_bfloat16 *>(input),
        const_cast<void *>(weight),
        output,
        nullptr,
        const_cast<nv_bfloat16 *>(scale),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        locks,
        m,
        1,
        false);
}

__global__ void reorder_channel_scale(
        const nv_bfloat16 * src, nv_bfloat16 * dst, uint32_t n) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    // This channel scale is applied on C, so Humming consumes each 32-row
    // tile in accumulator order:
    // [0,1,8,9,16,17,24,25, 2,3,10,11,..., 6,7,14,15,...].
    const uint32_t tile = index & ~31u;
    const uint32_t lane = index & 31u;
    const uint32_t source_lane = ((lane & 7u) >> 1) * 8u + (lane & 1u) + (lane >> 3) * 2u;
    dst[index] = src[tile + source_lane];
}

__global__ void input_f32_to_bf16(
        const float * src, nv_bfloat16 * dst, uint32_t n) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        dst[index] = __float2bfloat16(src[index]);
    }
}

__global__ void output_bf16_to_f32(
        const nv_bfloat16 * src,
        const nv_bfloat16 * gate,
        float * dst,
        uint32_t n) {
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t index = pair * 2;
    if (index + 1 < n) {
        float2 value = __bfloat1622float2(reinterpret_cast<const nv_bfloat162 *>(src)[pair]);
        if (gate != nullptr) {
            const float2 gate_value = __bfloat1622float2(reinterpret_cast<const nv_bfloat162 *>(gate)[pair]);
            value.x *= ggml_cuda_op_silu_single(gate_value.x);
            value.y *= ggml_cuda_op_silu_single(gate_value.y);
        }
        reinterpret_cast<float2 *>(dst)[pair] = value;
    } else if (index < n) {
        float value = __bfloat162float(src[index]);
        if (gate != nullptr) {
            value *= ggml_cuda_op_silu_single(__bfloat162float(gate[index]));
        }
        dst[index] = value;
    }
}

__global__ void output_bf16_swiglu_bf16(
        const nv_bfloat16 * src,
        const nv_bfloat16 * gate,
        nv_bfloat16 * dst,
        uint32_t n) {
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t index = pair * 2;
    if (index + 1 < n) {
        const float2 value = __bfloat1622float2(reinterpret_cast<const nv_bfloat162 *>(src)[pair]);
        const float2 gate_value = __bfloat1622float2(reinterpret_cast<const nv_bfloat162 *>(gate)[pair]);
        reinterpret_cast<nv_bfloat162 *>(dst)[pair] = __floats2bfloat162_rn(
            value.x * ggml_cuda_op_silu_single(gate_value.x),
            value.y * ggml_cuda_op_silu_single(gate_value.y));
    } else if (index < n) {
        const float value = __bfloat162float(src[index]) *
                            ggml_cuda_op_silu_single(__bfloat162float(gate[index]));
        dst[index] = __float2bfloat16(value);
    }
}

__global__ void output_bf16_residual_add(
        const nv_bfloat16 * src,
        const float * residual,
        float * dst,
        uint32_t n) {
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t index = pair * 2;
    if (index + 1 < n) {
        float2 value = __bfloat1622float2(reinterpret_cast<const nv_bfloat162 *>(src)[pair]);
        const float2 residual_value = reinterpret_cast<const float2 *>(residual)[pair];
        value.x += residual_value.x;
        value.y += residual_value.y;
        reinterpret_cast<float2 *>(dst)[pair] = value;
    } else if (index < n) {
        dst[index] = __bfloat162float(src[index]) + residual[index];
    }
}

template <int block_size>
__global__ void residual_rms_prepare(
        const nv_bfloat16 * src,
        const float * residual,
        const float * norm_weight,
        float * residual_out,
        float * norm_out,
        nv_bfloat16 * norm_bf16,
        int ncols,
        float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t offset = int64_t(row) * ncols;

    float sum = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const int64_t i = offset + col;
        const float value = __bfloat162float(src[i]) + residual[i];
        residual_out[i] = value;
        sum += value * value;
    }

    extern __shared__ float s_sum[];
    sum = block_reduce<block_reduce_method::SUM, block_size>(sum, s_sum);
    const float scale = rsqrtf(sum / ncols + eps);

    for (int col = tid; col < ncols; col += block_size) {
        const int64_t i = offset + col;
        const float value = residual_out[i] * scale * norm_weight[col];
        if (norm_out != nullptr) {
            norm_out[i] = value;
        }
        norm_bf16[i] = __float2bfloat16(value);
    }
}

template <int block_size, int max_iters>
__global__ void residual_rms_prepare_cached(
        const nv_bfloat16 * src,
        const float * residual,
        const float * norm_weight,
        float * residual_out,
        float * norm_out,
        nv_bfloat16 * norm_bf16,
        int ncols,
        float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t offset = int64_t(row) * ncols;

    float values[max_iters];
    float sum = 0.0f;
#pragma unroll
    for (int j = 0; j < max_iters; ++j) {
        const int col = tid + j * block_size;
        float value = 0.0f;
        if (col < ncols) {
            const int64_t i = offset + col;
            value = __bfloat162float(src[i]) + residual[i];
            residual_out[i] = value;
            sum += value * value;
        }
        values[j] = value;
    }

    extern __shared__ float s_sum[];
    sum = block_reduce<block_reduce_method::SUM, block_size>(sum, s_sum);
    const float scale = rsqrtf(sum / ncols + eps);

#pragma unroll
    for (int j = 0; j < max_iters; ++j) {
        const int col = tid + j * block_size;
        if (col < ncols) {
            const int64_t i = offset + col;
            const float value = values[j] * scale * norm_weight[col];
            if (norm_out != nullptr) {
                norm_out[i] = value;
            }
            norm_bf16[i] = __float2bfloat16(value);
        }
    }
}

} // namespace

bool ggml_cuda_humming_fp8_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_HUMMING_FP8");
        return value == nullptr || std::atoi(value) != 0;
    }();
    return enabled;
}

bool ggml_cuda_humming_fp8_supports_shape(int64_t n, int64_t k, int64_t m, int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    const bool retained = (n == 17408 && k == 5120) ||
                          (n == 5120 && k == 17408) ||
                          (n == 12288 && k == 5120) ||
                          (n == 10240 && k == 5120) ||
                          (n == 6144 && k == 5120) ||
                          (n == 5120 && k == 6144) ||
                          (n == 1024 && k == 5120);
    return m >= 1 && retained;
}

void ggml_cuda_humming_fp8_repack_host(
        const void * src_void, void * dst_void, int64_t n, int64_t k) {
    GGML_ASSERT(n % 64 == 0 && k % 64 == 0);
    const auto * src = static_cast<const uint8_t *>(src_void);
    auto * dst = static_cast<uint32_t *>(dst_void);
    const uint32_t n_tiles = n / 64;
    const uint32_t k_tiles = k / 64;
    const uint64_t out_stride = 4 * uint64_t(n); // uint32_t elements
    std::atomic<uint32_t> next_n_tile{0};

    const unsigned int n_threads = std::max(1u, std::min(std::thread::hardware_concurrency(), n_tiles));
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned int worker = 0; worker < n_threads; ++worker) {
        workers.emplace_back([&] {
            for (;;) {
                const uint32_t bn = next_n_tile.fetch_add(1, std::memory_order_relaxed);
                if (bn >= n_tiles) {
                    return;
                }
                for (uint32_t bk = 0; bk < k_tiles; ++bk) {
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint64_t out_row = uint64_t(bk) * 4 + i;
                        const uint64_t out_col_base = uint64_t(bn) * 256;
                        for (uint32_t lane = 0; lane < 32; ++lane) {
                            for (uint32_t j = 0; j < 8; ++j) {
                                uint32_t packed = 0;
                                constexpr uint32_t interleave[4] = {0, 2, 1, 3};
                                for (uint32_t byte = 0; byte < 4; ++byte) {
                                    uint32_t q = i * 32 + j * 4 + interleave[byte];
                                    const uint32_t i6 = q % 2; q /= 2;
                                    const uint32_t i5 = q % 2; q /= 2;
                                    const uint32_t i4 = q % 2; q /= 2;
                                    const uint32_t i3 = q % 4; q /= 4;
                                    const uint32_t i1 = q;
                                    const uint32_t src_i = i3 * 2 + i4;
                                    const uint32_t src_j = i1 * 2 + i5;
                                    const uint64_t row = uint64_t(bn) * 64 + src_i * 8 + lane / 4;
                                    const uint64_t col = uint64_t(bk) * 64 + src_j * 8 + (lane % 4) * 2 + i6;
                                    packed |= uint32_t(src[row * k + col]) << (8 * byte);
                                }
                                dst[out_row * out_stride + out_col_base + lane * 8 + j] = packed;
                            }
                        }
                    }
                }
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
}

void ggml_cuda_humming_fp8_unrepack_host(
        const void * src_void, void * dst_void, int64_t n, int64_t k) {
    GGML_ASSERT(n % 64 == 0 && k % 64 == 0);
    const auto * src = static_cast<const uint32_t *>(src_void);
    auto * dst = static_cast<uint8_t *>(dst_void);
    const uint32_t n_tiles = n / 64;
    const uint32_t k_tiles = k / 64;
    const uint64_t in_stride = 4 * uint64_t(n); // uint32_t elements
    std::atomic<uint32_t> next_n_tile{0};

    const unsigned int n_threads = std::max(1u, std::min(std::thread::hardware_concurrency(), n_tiles));
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (unsigned int worker = 0; worker < n_threads; ++worker) {
        workers.emplace_back([&] {
            for (;;) {
                const uint32_t bn = next_n_tile.fetch_add(1, std::memory_order_relaxed);
                if (bn >= n_tiles) {
                    return;
                }
                for (uint32_t bk = 0; bk < k_tiles; ++bk) {
                    for (uint32_t i = 0; i < 4; ++i) {
                        const uint64_t in_row = uint64_t(bk) * 4 + i;
                        const uint64_t in_col_base = uint64_t(bn) * 256;
                        for (uint32_t lane = 0; lane < 32; ++lane) {
                            for (uint32_t j = 0; j < 8; ++j) {
                                const uint32_t packed = src[in_row * in_stride + in_col_base + lane * 8 + j];
                                constexpr uint32_t interleave[4] = {0, 2, 1, 3};
                                for (uint32_t byte = 0; byte < 4; ++byte) {
                                    uint32_t q = i * 32 + j * 4 + interleave[byte];
                                    const uint32_t i6 = q % 2; q /= 2;
                                    const uint32_t i5 = q % 2; q /= 2;
                                    const uint32_t i4 = q % 2; q /= 2;
                                    const uint32_t i3 = q % 4; q /= 4;
                                    const uint32_t i1 = q;
                                    const uint32_t dst_i = i3 * 2 + i4;
                                    const uint32_t dst_j = i1 * 2 + i5;
                                    const uint64_t row = uint64_t(bn) * 64 + dst_i * 8 + lane / 4;
                                    const uint64_t col = uint64_t(bk) * 64 + dst_j * 8 + (lane % 4) * 2 + i6;
                                    dst[row * k + col] = uint8_t(packed >> (8 * byte));
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (std::thread & worker : workers) {
        worker.join();
    }
}

void ggml_cuda_humming_fp8_repack(
        const void * src, void * dst, int64_t n, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(n % 64 == 0 && k % 64 == 0);
    const dim3 grid(n / 64, k / 64, 1);
    weight_repack_nk<8, 16, true, false, false, false, 0, false>
        <<<grid, 32, 0, stream>>>(
            static_cast<const uint32_t *>(src),
            static_cast<uint32_t *>(dst),
            nullptr,
            n, k, n, k, 3);
}

void ggml_cuda_humming_fp8_reorder_scale(
        const nv_bfloat16 * src, nv_bfloat16 * dst, int64_t n, cudaStream_t stream) {
    GGML_ASSERT(n % 64 == 0);
    reorder_channel_scale<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

void ggml_cuda_humming_fp8_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const nv_bfloat16 * scale,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int sms,
        cudaStream_t stream) {
#define GGML_HUMMING_LAUNCH(N, K, BM, BN, BK, WM, WN, WK, THREADS, RASTER, STREAMK) \
    launch_fp8<N, K, Shape<BM, BN, BK>, Shape<WM, WN, WK>, THREADS, RASTER, STREAMK>( \
        input, weight, scale, output, locks, m, sms, stream)

    if (m <= 16) {
        if (n == 17408 && k == 5120) { GGML_HUMMING_LAUNCH(17408, 5120, 16, 256, 32, 16, 64, 32, 128, 28, false); return; }
        if (n == 5120 && k == 17408) { GGML_HUMMING_LAUNCH(5120, 17408, 16, 64, 256, 16, 64, 32, 256, 8, false); return; }
        if (n == 12288 && k == 5120) { GGML_HUMMING_LAUNCH(12288, 5120, 16, 128, 128, 16, 64, 32, 256, 28, true); return; }
        if (n == 10240 && k == 5120) { GGML_HUMMING_LAUNCH(10240, 5120, 16, 128, 128, 16, 64, 32, 256, 28, false); return; }
        if (n == 6144 && k == 5120)  { GGML_HUMMING_LAUNCH(6144, 5120, 16, 128, 128, 16, 64, 32, 256, 28, false); return; }
        if (n == 5120 && k == 6144)  { GGML_HUMMING_LAUNCH(5120, 6144, 16, 64, 256, 16, 64, 32, 256, 23, false); return; }
        if (n == 1024 && k == 5120)  { GGML_HUMMING_LAUNCH(1024, 5120, 16, 64, 256, 16, 64, 32, 256, 28, true); return; }
    }

    if (m > 16) {
        if (n == 17408 && k == 5120) { GGML_HUMMING_LAUNCH(17408, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 5120 && k == 17408) { GGML_HUMMING_LAUNCH(5120, 17408, 128, 256, 64, 64, 64, 64, 256, 1, true); return; }
        if (n == 12288 && k == 5120) { GGML_HUMMING_LAUNCH(12288, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 10240 && k == 5120) { GGML_HUMMING_LAUNCH(10240, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 6144 && k == 5120)  { GGML_HUMMING_LAUNCH(6144, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 5120 && k == 6144)  { GGML_HUMMING_LAUNCH(5120, 6144, 128, 256, 64, 64, 64, 64, 256, 2, true); return; }
        if (n == 1024 && k == 5120)  { GGML_HUMMING_LAUNCH(1024, 5120, 128, 64, 64, 32, 64, 32, 256, 3, true); return; }
    }

#undef GGML_HUMMING_LAUNCH
    GGML_ABORT("unsupported Humming E4M3 shape");
}

void ggml_cuda_humming_fp8_input_f32_to_bf16(
        const float * src, nv_bfloat16 * dst, int64_t n, cudaStream_t stream) {
    input_f32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(src, dst, n);
}

void ggml_cuda_humming_fp8_output_bf16_to_f32(
        const nv_bfloat16 * src,
        const nv_bfloat16 * gate,
        float * dst,
        int64_t n,
        cudaStream_t stream) {
    output_bf16_to_f32<<<((n + 1) / 2 + 255) / 256, 256, 0, stream>>>(src, gate, dst, n);
}

void ggml_cuda_humming_fp8_swiglu_bf16(
        const nv_bfloat16 * src,
        const nv_bfloat16 * gate,
        nv_bfloat16 * dst,
        int64_t n,
        cudaStream_t stream) {
    output_bf16_swiglu_bf16<<<((n + 1) / 2 + 255) / 256, 256, 0, stream>>>(src, gate, dst, n);
}

void ggml_cuda_humming_fp8_residual_add(
        const nv_bfloat16 * src,
        const float * residual,
        float * dst,
        int64_t n,
        cudaStream_t stream) {
    output_bf16_residual_add<<<((n + 1) / 2 + 255) / 256, 256, 0, stream>>>(src, residual, dst, n);
}

void ggml_cuda_humming_residual_rms_prepare(
        const nv_bfloat16 * src,
        const float * residual,
        const float * norm_weight,
        float * residual_out,
        float * norm_out,
        nv_bfloat16 * norm_bf16,
        int64_t ncols,
        int64_t nrows,
        float eps,
        cudaStream_t stream) {
    static const int requested_block = [] {
        const char * value = std::getenv("GGML_CUDA_HUMMING_RMS_BLOCK");
        return value ? std::atoi(value) : 1024;
    }();
    if (requested_block == 1024 && ncols == 5120) {
        residual_rms_prepare_cached<1024, 5><<<nrows, 1024, 32 * sizeof(float), stream>>>(
            src, residual, norm_weight, residual_out, norm_out, norm_bf16, ncols, eps);
    } else if (requested_block == 1024 && ncols <= 8192) {
        residual_rms_prepare_cached<1024, 8><<<nrows, 1024, 32 * sizeof(float), stream>>>(
            src, residual, norm_weight, residual_out, norm_out, norm_bf16, ncols, eps);
    } else if (requested_block == 256) {
        residual_rms_prepare<256><<<nrows, 256, 32 * sizeof(float), stream>>>(
            src, residual, norm_weight, residual_out, norm_out, norm_bf16, ncols, eps);
    } else if (requested_block == 1024) {
        residual_rms_prepare<1024><<<nrows, 1024, 32 * sizeof(float), stream>>>(
            src, residual, norm_weight, residual_out, norm_out, norm_bf16, ncols, eps);
    } else {
        residual_rms_prepare<512><<<nrows, 512, 32 * sizeof(float), stream>>>(
            src, residual, norm_weight, residual_out, norm_out, norm_bf16, ncols, eps);
    }
}

#endif
