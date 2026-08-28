#include "humming-nvfp4.cuh"

#if !defined(GGML_USE_HIP)

#define HUMMING_INPUT_SCALE_GROUP_SIZE 0
#define HUMMING_WEIGHT_SCALE_GROUP_SIZE 16
#define HUMMING_HAS_ZERO_POINT 0
#define HUMMING_IS_CHANNEL_WEIGHT_SCALE 0
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

namespace {

class nvfp4_mma_op {
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
class nvfp4_layer_config {
public:
    static constexpr auto kShapeN = N;
    static constexpr auto kShapeK = K;
    static constexpr auto kPadShapeN = 0u;
    static constexpr auto kPadShapeK = 0u;
    static constexpr auto kNumExperts = 0u;
    static constexpr auto kInputScaleGroupSize = 0u;
    static constexpr auto kWeightScaleGroupSize = 16u;
    static constexpr auto kWeightScaleGroupSizeN = 0u;
    static constexpr auto kWeightScaleType = WeightScaleType::GROUP;
    static constexpr auto kWeightScale2Type = WeightScale2Type::TENSOR;
    static constexpr auto kUseIntWeightScale = false;
    static constexpr auto kUseFusedE8m0Scale = false;
    static constexpr auto kHasZeroPoint = false;
    static constexpr auto kIsFpZeroPoint = false;
    static constexpr auto kHasBias = false;
    static constexpr auto kMmaType = MmaType::MMA;
    static constexpr auto kUsePackedKLayout = false;
    static constexpr auto kMmaTypeId = 0u;
    static constexpr auto kIsChannelWeightScale = false;
    static constexpr auto kIsBlockWeightScale = false;
    static constexpr auto kIsGroupWeightScale = true;
    static constexpr auto kIsTensorWeightScale = false;
    static constexpr auto kIsChannelWeightScale2 = false;
    static constexpr auto kIsTensorWeightScale2 = true;
    static constexpr auto kHasChannelWeightScale = false;
    static constexpr auto kHasTensorWeightScale = true;
    static constexpr auto kHasInputScale = false;
};

class nvfp4_compute_config {
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

template <uint32_t Threads, uint32_t Raster, bool StreamK = true, uint32_t CtasPerSm = 1>
class nvfp4_tuning_config {
public:
    static constexpr auto kUseStreamK = StreamK;
    static constexpr auto kNumStages = 3u;
    static constexpr auto kNumCtasPerSm = CtasPerSm;
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

using nvfp4_a = FloatingPointType<16, 8, 7>;
using nvfp4_b = FloatingPointType<4, 2, 1>;
using nvfp4_c = BFloat16;
using nvfp4_s = FloatingPointType<8, 4, 3>;

template <uint32_t N, uint32_t K, class BlockShape, class WarpShape, uint32_t Threads, uint32_t Raster,
          bool StreamK = true, uint32_t CtasPerSm = 1>
void launch_nvfp4(
        const nv_bfloat16 * input,
        const void * weight,
        const void * scale,
        const float * tensor_scale,
        nv_bfloat16 * output,
        int32_t * locks,
        uint32_t m,
        int sms,
        cudaStream_t stream) {
    using layer = nvfp4_layer_config<N, K>;
    using tuning = nvfp4_tuning_config<Threads, Raster, StreamK, CtasPerSm>;
    using storage = SharedStorage<nvfp4_mma_op, BlockShape, WarpShape, nvfp4_a, nvfp4_b, nvfp4_s,
                                  layer, nvfp4_compute_config, tuning>;
    constexpr auto kernel = humming<
        nvfp4_mma_op,
        Shape<0, N, K>, BlockShape, WarpShape, Shape<0, 0, 0>,
        nvfp4_a, nvfp4_b, nvfp4_c, nvfp4_s,
        layer, nvfp4_compute_config, tuning>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(storage)));
    kernel<<<sms * tuning::kNumCtasPerSm, Threads, sizeof(storage), stream>>>(
        const_cast<nv_bfloat16 *>(input),
        const_cast<void *>(weight),
        output,
        nullptr,
        const_cast<void *>(scale),
        nullptr,
        nullptr,
        const_cast<float *>(tensor_scale),
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

__device__ __forceinline__ uint8_t canonical_nibble(const block_nvfp4 & block, uint32_t index) {
    const uint32_t sub = index / QK_NVFP4_SUB;
    const uint32_t lane = index % QK_NVFP4_SUB;
    const uint8_t packed = block.qs[sub * (QK_NVFP4_SUB / 2) + (lane & 7u)];
    return lane < 8 ? packed & 0x0f : packed >> 4;
}

__global__ void extract_canonical_nvfp4(
        const block_nvfp4 * canonical,
        uint8_t * packed_weight,
        uint8_t * reordered_scale,
        uint32_t n,
        uint32_t k) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t weight_bytes = uint64_t(n) * k / 2;
    if (index < weight_bytes) {
        const uint32_t row = index / (k / 2);
        const uint32_t byte_col = index % (k / 2);
        const uint32_t element = 2 * byte_col;
        const block_nvfp4 & block = canonical[uint64_t(row) * (k / QK_NVFP4) + element / QK_NVFP4];
        const uint32_t in_block = element % QK_NVFP4;
        packed_weight[index] = canonical_nibble(block, in_block) |
                               (canonical_nibble(block, in_block + 1) << 4);
    }

    const uint64_t scale_bytes = uint64_t(n) * k / QK_NVFP4_SUB;
    if (index < scale_bytes) {
        const uint32_t group = index / n;
        const uint32_t dst_row = index % n;
        const uint32_t tile = dst_row & ~63u;
        const uint32_t lane = dst_row & 63u;
        const uint32_t src_row = tile + (lane & 7u) * 8u + (lane >> 3);
        const block_nvfp4 & block = canonical[uint64_t(src_row) * (k / QK_NVFP4) + group / 4];
        reordered_scale[index] = block.d[group & 3u];
    }
}

__global__ void restore_canonical_nvfp4(
        const uint32_t * packed_weight,
        const uint8_t * reordered_scale,
        block_nvfp4 * canonical,
        uint32_t n,
        uint32_t k) {
    const uint64_t block_index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t blocks_per_row = k / QK_NVFP4;
    if (block_index >= uint64_t(n) * blocks_per_row) {
        return;
    }

    const uint32_t row = block_index / blocks_per_row;
    const uint32_t block_col = block_index % blocks_per_row;
    const uint32_t row_tile = row / 64;
    const uint32_t row_lane = row % 64;

    block_nvfp4 block = {};
    const uint32_t scale_row = (row & ~63u) + (row_lane & 7u) * 8u + (row_lane >> 3);
    for (uint32_t sub = 0; sub < QK_NVFP4 / QK_NVFP4_SUB; ++sub) {
        block.d[sub] = reordered_scale[(block_col * 4u + sub) * n + scale_row];
    }

    // Inverse of weight_repack_nk<4, 16, packed>. Each 64x64 canonical
    // tile is written as four K-major rows with 2*N uint32 words per row.
    constexpr uint8_t inverse_interleave[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    for (uint32_t element = 0; element < QK_NVFP4; ++element) {
        const uint32_t c = element;
        const uint32_t pack_thread = 4u * (row_lane & 7u) + ((c & 7u) >> 1);
        const uint32_t out_row = block_col * 4u + c / 16u;
        const uint32_t out_col = row_tile * 128u + pack_thread * 4u + row_lane / 16u;
        const uint32_t word = packed_weight[uint64_t(out_row) * (2u * n) + out_col];
        const uint32_t tuple = ((row_lane / 8u) & 1u) * 4u + ((c / 8u) & 1u) * 2u + (c & 1u);
        const uint8_t value = (word >> (4u * inverse_interleave[tuple])) & 0x0fu;
        const uint32_t byte = (element / 16u) * 8u + (element & 7u);
        if ((element & 15u) < 8u) {
            block.qs[byte] = value;
        } else {
            block.qs[byte] |= value << 4;
        }
    }
    canonical[block_index] = block;
}

} // namespace

bool ggml_cuda_humming_nvfp4_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_CUDA_HUMMING_NVFP4");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

bool ggml_cuda_humming_nvfp4_supports_shape(int64_t n, int64_t k, int64_t m, int cc) {
    if (!GGML_CUDA_CC_IS_NVIDIA(cc) || cc < GGML_CUDA_CC_AMPERE || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
        return false;
    }
    return m > 0 && ((n == 17408 && k == 5120) || (n == 5120 && k == 17408));
}

void ggml_cuda_humming_nvfp4_repack_upload(
        const void * canonical_host,
        void * repacked_device,
        int64_t n,
        int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(n % 64 == 0 && k % 64 == 0);
    const size_t canonical_size = size_t(n) * size_t(k) / QK_NVFP4 * sizeof(block_nvfp4);
    const size_t weight_size = size_t(n) * size_t(k) / 2;
    const size_t scale_size = size_t(n) * size_t(k) / QK_NVFP4_SUB;
    GGML_ASSERT(canonical_size == weight_size + scale_size);

    void * canonical_device = nullptr;
    void * row_major_weight = nullptr;
    CUDA_CHECK(cudaMalloc(&canonical_device, canonical_size));
    CUDA_CHECK(cudaMalloc(&row_major_weight, weight_size));
    CUDA_CHECK(cudaMemcpyAsync(
        canonical_device, canonical_host, canonical_size, cudaMemcpyHostToDevice, stream));
    ggml_cuda_humming_nvfp4_prepare(
        canonical_device,
        row_major_weight,
        repacked_device,
        static_cast<uint8_t *>(repacked_device) + weight_size,
        n, k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(row_major_weight));
    CUDA_CHECK(cudaFree(canonical_device));
}

void ggml_cuda_humming_nvfp4_unrepack(
        const void * repacked_device,
        void * canonical_device,
        int64_t n,
        int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(n % 64 == 0 && k % 64 == 0);
    const size_t weight_size = size_t(n) * size_t(k) / 2;
    const uint64_t blocks = uint64_t(n) * uint64_t(k) / QK_NVFP4;
    restore_canonical_nvfp4<<<(blocks + 255) / 256, 256, 0, stream>>>(
        static_cast<const uint32_t *>(repacked_device),
        static_cast<const uint8_t *>(repacked_device) + weight_size,
        static_cast<block_nvfp4 *>(canonical_device),
        n, k);
}

void ggml_cuda_humming_nvfp4_prepare(
        const void * canonical,
        void * original_weight,
        void * repacked_weight,
        void * repacked_scale,
        int64_t n,
        int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(n % 64 == 0 && k % 64 == 0);
    const uint64_t work = uint64_t(n) * k / 2;
    extract_canonical_nvfp4<<<(work + 255) / 256, 256, 0, stream>>>(
        static_cast<const block_nvfp4 *>(canonical),
        static_cast<uint8_t *>(original_weight),
        static_cast<uint8_t *>(repacked_scale),
        n, k);
    const dim3 grid(n / 64, k / 64, 1);
    weight_repack_nk<4, 16, true, false, false, false, 0, false>
        <<<grid, 32, 0, stream>>>(
            static_cast<const uint32_t *>(original_weight),
            static_cast<uint32_t *>(repacked_weight),
            nullptr,
            n, k, n, k, 3);
}

void ggml_cuda_humming_nvfp4_launch(
        const nv_bfloat16 * input,
        const void * weight,
        const void * scale,
        const float * tensor_scale,
        nv_bfloat16 * output,
        int32_t * locks,
        int64_t n,
        int64_t k,
    int64_t m,
    int sms,
    cudaStream_t stream) {
#define GGML_HUMMING_NVFP4_LAUNCH(N, K, BM, BN, BK, WM, WN, WK, THREADS, RASTER) \
    launch_nvfp4<N, K, Shape<BM, BN, BK>, Shape<WM, WN, WK>, THREADS, RASTER>( \
        input, weight, scale, tensor_scale, output, locks, m, sms, stream)
#define GGML_HUMMING_NVFP4_LAUNCH_TUNED(N, K, BM, BN, BK, WM, WN, WK, THREADS, RASTER, STREAMK, CTAS) \
    launch_nvfp4<N, K, Shape<BM, BN, BK>, Shape<WM, WN, WK>, THREADS, RASTER, STREAMK, CTAS>( \
        input, weight, scale, tensor_scale, output, locks, m, sms, stream)

    static const int raster_up = [] {
        const char * value = std::getenv("GGML_CUDA_HUMMING_NVFP4_RASTER_UP");
        return value ? std::atoi(value) : 3;
    }();
    static const int raster_down = [] {
        const char * value = std::getenv("GGML_CUDA_HUMMING_NVFP4_RASTER_DOWN");
        return value ? std::atoi(value) : 1;
    }();
    if (n == 17408 && k == 5120) {
        if (m <= 16) {
            GGML_HUMMING_NVFP4_LAUNCH(17408, 5120, 16, 256, 64, 16, 64, 64, 128, 1);
        } else if (raster_up == 1) {
            GGML_HUMMING_NVFP4_LAUNCH(17408, 5120, 128, 256, 64, 64, 64, 64, 256, 1);
        } else if (raster_up == 5) {
            GGML_HUMMING_NVFP4_LAUNCH(17408, 5120, 128, 256, 64, 64, 64, 64, 256, 5);
        } else {
            GGML_HUMMING_NVFP4_LAUNCH(17408, 5120, 128, 256, 64, 64, 64, 64, 256, 3);
        }
        return;
    }
    if (n == 5120 && k == 17408) {
        if (m <= 16) {
            GGML_HUMMING_NVFP4_LAUNCH(5120, 17408, 16, 128, 128, 16, 32, 64, 128, 1);
        } else if (m <= 128) {
            GGML_HUMMING_NVFP4_LAUNCH(5120, 17408, 128, 64, 64, 32, 64, 32, 256, 1);
        } else if (raster_down == 3) {
            GGML_HUMMING_NVFP4_LAUNCH(5120, 17408, 128, 256, 64, 64, 64, 64, 256, 3);
        } else if (raster_down == 5) {
            GGML_HUMMING_NVFP4_LAUNCH(5120, 17408, 128, 256, 64, 64, 64, 64, 256, 5);
        } else {
            GGML_HUMMING_NVFP4_LAUNCH(5120, 17408, 128, 256, 64, 64, 64, 64, 256, 1);
        }
        return;
    }

#undef GGML_HUMMING_NVFP4_LAUNCH
#undef GGML_HUMMING_NVFP4_LAUNCH_TUNED
    GGML_ABORT("unsupported Humming NVFP4 shape");
}

#endif
