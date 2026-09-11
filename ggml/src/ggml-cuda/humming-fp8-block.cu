#include "humming-fp8-block.cuh"

#if !defined(GGML_USE_HIP)

#define HUMMING_INPUT_SCALE_GROUP_SIZE 0
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE 128
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE_N 128
#define HUMMING_HAS_ZERO_POINT 0
#define HUMMING_IS_FP_ZERO_POINT 0
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE 0
#define HUMMING_IS_BLOCK_WEIGHT_SCALE 1
#define HUMMING_IS_GROUP_WEIGHT_SCALE 0
#define HUMMING_IS_TENSOR_WEIGHT_SCALE 0
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE_2 0
#define HUMMING_IS_TENSOR_WEIGHT_SCALE_2 0
#define HUMMING_HAS_CHANNEL_WEIGHT_SCALE 0
#define HUMMING_HAS_TENSOR_WEIGHT_SCALE 0
#define HUMMING_HAS_BIAS 0
#define HUMMING_HAS_INPUT_SCALE 0
#define HUMMING_IS_INDEXED_GEMM 0
#define HUMMING_IS_GROUPED_GEMM 0
#define HUMMING_IS_GROUPED_CONTIGUOUS_GEMM 0
#define HUMMING_USE_MBARRIER 0
#define HUMMING_USE_WARP_SPEC 0
#define HUMMING_REDUCE_OVERLAP_LAST_STAGE_ONLY 0

#include <humming/kernel/humming.cuh>

namespace {

class fp8_block_mma_op {
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
class fp8_block_layer_config {
public:
    static constexpr auto kShapeN = N;
    static constexpr auto kShapeK = K;
    static constexpr auto kPadShapeN = 0u;
    static constexpr auto kPadShapeK = 0u;
    static constexpr auto kNumExperts = 0u;
    static constexpr auto kInputScaleGroupSize = 0u;
    static constexpr auto kWeightScaleGroupSize = 128u;
    static constexpr auto kWeightScaleGroupSizeN = 128u;
    static constexpr auto kWeightScaleType = WeightScaleType::BLOCK;
    static constexpr auto kWeightScale2Type = WeightScale2Type::NONE;
    static constexpr auto kUseIntWeightScale = false;
    static constexpr auto kUseFusedE8m0Scale = false;
    static constexpr auto kHasZeroPoint = false;
    static constexpr auto kIsFpZeroPoint = false;
    static constexpr auto kHasBias = false;
    static constexpr auto kMmaType = MmaType::MMA;
    static constexpr auto kUsePackedKLayout = false;
    static constexpr auto kMmaTypeId = 0u;
    static constexpr auto kIsChannelWeightScale = false;
    static constexpr auto kIsBlockWeightScale = true;
    static constexpr auto kIsGroupWeightScale = false;
    static constexpr auto kIsTensorWeightScale = false;
    static constexpr auto kIsChannelWeightScale2 = false;
    static constexpr auto kIsTensorWeightScale2 = false;
    static constexpr auto kHasChannelWeightScale = false;
    static constexpr auto kHasTensorWeightScale = false;
    static constexpr auto kHasInputScale = false;
};

class fp8_block_compute_config {
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
class fp8_block_tuning_config {
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

using fp8_block_a = FloatingPointType<16, 8, 7>;
using fp8_block_b = FloatingPointType<8, 4, 3>;
using fp8_block_c = BFloat16;
using fp8_block_s = FloatingPointType<32, 8, 23>;

template <uint32_t N, uint32_t K, class BlockShape, class WarpShape, uint32_t Threads,
          uint32_t Raster, bool UseStreamK>
void launch_fp8_block(
        const nv_bfloat16 * input,
        const void * weight,
        const float * scale,
        void * output,
        int32_t * locks,
        uint32_t m,
        int sms,
        cudaStream_t stream) {
    using layer = fp8_block_layer_config<N, K>;
    using tuning = fp8_block_tuning_config<Threads, Raster, UseStreamK>;
    using storage = SharedStorage<fp8_block_mma_op, BlockShape, WarpShape,
                                  fp8_block_a, fp8_block_b, fp8_block_s,
                                  layer, fp8_block_compute_config, tuning>;
    constexpr auto kernel = humming<
        fp8_block_mma_op,
        Shape<0, N, K>, BlockShape, WarpShape, Shape<0, 0, 0>,
        fp8_block_a, fp8_block_b, fp8_block_c, fp8_block_s,
        layer, fp8_block_compute_config, tuning>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(storage)));
    kernel<<<sms, Threads, sizeof(storage), stream>>>(
        const_cast<nv_bfloat16 *>(input),
        const_cast<void *>(weight),
        output,
        nullptr,
        const_cast<float *>(scale),
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

} // namespace

bool ggml_cuda_humming_fp8_block_supports_shape(int64_t n, int64_t k, int64_t m, int cc) {
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

void ggml_cuda_humming_fp8_block_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const float * scale,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
        int64_t m,
        int sms,
        cudaStream_t stream) {
#define GGML_HUMMING_FP8_BLOCK_LAUNCH(N, K, BM, BN, BK, WM, WN, WK, THREADS, RASTER, STREAMK) \
    launch_fp8_block<N, K, Shape<BM, BN, BK>, Shape<WM, WN, WK>, THREADS, RASTER, STREAMK>( \
        input, weight, scale, output, locks, m, sms, stream)

    if (m <= 16) {
        if (n == 17408 && k == 5120) { GGML_HUMMING_FP8_BLOCK_LAUNCH(17408, 5120, 16, 256, 32, 16, 64, 32, 128, 28, true); return; }
        if (n == 5120 && k == 17408) { GGML_HUMMING_FP8_BLOCK_LAUNCH(5120, 17408, 16, 64, 256, 16, 64, 32, 256, 8, true); return; }
        if (n == 12288 && k == 5120) { GGML_HUMMING_FP8_BLOCK_LAUNCH(12288, 5120, 16, 128, 128, 16, 64, 32, 256, 28, true); return; }
        if (n == 10240 && k == 5120) { GGML_HUMMING_FP8_BLOCK_LAUNCH(10240, 5120, 16, 128, 128, 16, 64, 32, 256, 28, true); return; }
        if (n == 6144 && k == 5120)  { GGML_HUMMING_FP8_BLOCK_LAUNCH(6144, 5120, 16, 128, 128, 16, 64, 32, 256, 28, true); return; }
        if (n == 5120 && k == 6144)  { GGML_HUMMING_FP8_BLOCK_LAUNCH(5120, 6144, 16, 64, 256, 16, 64, 32, 256, 23, true); return; }
        if (n == 1024 && k == 5120)  { GGML_HUMMING_FP8_BLOCK_LAUNCH(1024, 5120, 16, 64, 128, 16, 64, 32, 128, 28, true); return; }
    }

    if (m > 16) {
        if (n == 17408 && k == 5120) { GGML_HUMMING_FP8_BLOCK_LAUNCH(17408, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 5120 && k == 17408) { GGML_HUMMING_FP8_BLOCK_LAUNCH(5120, 17408, 128, 256, 64, 64, 64, 64, 256, 1, true); return; }
        if (n == 12288 && k == 5120) { GGML_HUMMING_FP8_BLOCK_LAUNCH(12288, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 10240 && k == 5120) { GGML_HUMMING_FP8_BLOCK_LAUNCH(10240, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 6144 && k == 5120)  { GGML_HUMMING_FP8_BLOCK_LAUNCH(6144, 5120, 128, 256, 64, 64, 64, 64, 256, 3, true); return; }
        if (n == 5120 && k == 6144)  { GGML_HUMMING_FP8_BLOCK_LAUNCH(5120, 6144, 128, 256, 64, 64, 64, 64, 256, 2, true); return; }
        if (n == 1024 && k == 5120)  { GGML_HUMMING_FP8_BLOCK_LAUNCH(1024, 5120, 128, 128, 64, 64, 64, 64, 128, 3, true); return; }
    }

#undef GGML_HUMMING_FP8_BLOCK_LAUNCH
    GGML_ABORT("unsupported Humming block-E4M3 shape");
}

#endif
