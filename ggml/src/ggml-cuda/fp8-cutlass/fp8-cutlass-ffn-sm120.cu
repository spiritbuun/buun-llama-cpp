// SPDX-License-Identifier: Apache-2.0
// Paired FFN epilogue; the retained provider defines the base
// split2 GEMM. Build against unmodified CUTLASS 4.3.4 headers.
#include "fp8-cutlass-sm120.cu"
using Retained                 = Plan<true, false, true>;
static constexpr int PairGroup = 64;

struct PairedEpilogue {
    using Base               = Retained::Epi;
    using ElementC           = float;
    using ElementD           = float;
    using ThreadEpilogueOp   = typename Base::ThreadEpilogueOp;
    using StrideC            = typename Base::StrideC;
    using StrideD            = typename Base::StrideD;
    using LoadPipeline       = typename Base::LoadPipeline;
    using LoadPipelineState  = typename Base::LoadPipelineState;
    using StorePipeline      = typename Base::StorePipeline;
    using StorePipelineState = typename Base::StorePipelineState;
    using PipelineStorage    = typename Base::PipelineStorage;
    using EpilogueTile       = Shape<_128, _128>;

    struct TensorStorage {};

    struct SharedStorage {
        TensorStorage   tensors;
        PipelineStorage pipeline;
    };

    static constexpr bool RequiresTransactionBytes = false;

    struct Arguments {
        const float * xs;
        const float * ws;
        float *       output;
        const float * ws_up = nullptr;
        const float * xs_up = nullptr;
    };

    using Params = Arguments;
    Params params;

    CUTLASS_HOST_DEVICE PairedEpilogue(const Params & p, TensorStorage &) : params(p) {}

    template <class Shape> static Params to_underlying_arguments(const Shape &, const Arguments & args, void *) {
        return args;
    }

    template <class Shape> static size_t get_workspace_size(const Shape &, const Arguments &) { return 0; }

    template <class Shape>
    static cutlass::Status initialize_workspace(const Shape &,
                                                const Arguments &,
                                                void *,
                                                cudaStream_t,
                                                cutlass::CudaHostAdapter * = nullptr) {
        return cutlass::Status::kSuccess;
    }

    template <class Shape> static bool can_implement(const Shape & shape, const Arguments & args) {
        return get<0>(shape) > 0 && get<1>(shape) % 128 == 0 && args.xs && args.ws && args.output &&
               uintptr_t(args.output) % alignof(float2) == 0;
    }

    template <class Shape> CUTLASS_HOST_DEVICE static constexpr int get_store_pipe_increment(Shape) { return 1; }

    CUTLASS_DEVICE static void prefetch_tma_descriptors(const Params &) {}

    CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }

    template <class... Args> CUTLASS_DEVICE auto load(LoadPipeline, LoadPipelineState state, Args &&...) {
        return state;
    }

    CUTLASS_DEVICE void load_tail(LoadPipeline, LoadPipelineState) {}

    CUTLASS_DEVICE auto store_tail(LoadPipeline, LoadPipelineState load, StorePipeline, StorePipelineState store) {
        return cute::make_tuple(load, store);
    }

    template <class PS, class TS, class TC, class Engine, class Layout, class Mma>
    CUTLASS_DEVICE auto store(LoadPipeline,
                              LoadPipelineState load,
                              StorePipeline,
                              StorePipelineState store,
                              PS                 problem,
                              TS,
                              TC                           tile,
                              cute::Tensor<Engine, Layout> accum,
                              Mma                          mma,
                              int                          thread,
                              TensorStorage &,
                              int = -1) {
        auto          coordinates = mma.get_slice(thread).partition_C(make_identity_tensor(Shape<_128, _128>{}));
        constexpr int pair_stride = PairGroup == 1 ? 1 : (PairGroup / 16) * size<0>(Layout{}) * size<1>(Layout{});
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); i += 2) {
            const int row = get<0>(tile) * 128 + get<0>(coordinates(i));
            const int col = get<1>(tile) * 128 + get<1>(coordinates(i));
            if ((col & PairGroup) == 0 && row < get<0>(problem) && col < get<1>(problem)) {
                const float xs = params.xs[row];
                const int dst_col = (col / (2 * PairGroup)) * PairGroup + col % PairGroup;
                float values[2];
                CUTLASS_PRAGMA_UNROLL
                for (int j = 0; j < 2; ++j) {
                    const float gate = (accum(i + j) * xs) * params.ws[params.ws_up ? dst_col + j : col + j];
                    const float up = (accum((i + j) ^ pair_stride) * (params.xs_up ? params.xs_up[row] : xs)) *
                        (params.ws_up ? params.ws_up[dst_col + j] : params.ws[col + j + PairGroup]);
                    values[j] = (gate / (1.0f + expf(-gate))) * up;
                }
                *reinterpret_cast<float2 *>(params.output + int64_t(row) * (get<1>(problem) / 2) + dst_col) =
                    make_float2(values[0], values[1]);
            }
        }
        return cute::make_tuple(load, store);
    }
};

// DualMain extends SeparateMain; preserve their definition order.
// clang-format off
#include "fp8-cutlass-ffn-load.cuh"
#include "fp8-cutlass-ffn-dual.cuh"
// clang-format on
