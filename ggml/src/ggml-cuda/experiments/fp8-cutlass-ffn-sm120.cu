// SPDX-License-Identifier: Apache-2.0
// Experimental paired FFN epilogue; the retained provider defines the base
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
        return get<0>(shape) > 0 && get<1>(shape) % 128 == 0 && args.xs && args.ws && args.output;
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
        for (int i = 0; i < size(accum); ++i) {
            const int row = get<0>(tile) * 128 + get<0>(coordinates(i));
            const int col = get<1>(tile) * 128 + get<1>(coordinates(i));
            if ((col & PairGroup) == 0 && row < get<0>(problem) && col < get<1>(problem)) {
                const float xs      = params.xs[row];
                const int   dst_col = (col / (2 * PairGroup)) * PairGroup + col % PairGroup;
                const float gate    = (accum(i) * xs) * params.ws[params.ws_up ? dst_col : col];
                const float up      = (accum(i ^ pair_stride) * (params.xs_up ? params.xs_up[row] : xs)) *
                                 (params.ws_up ? params.ws_up[dst_col] : params.ws[col + PairGroup]);
                params.output[int64_t(row) * (get<1>(problem) / 2) + dst_col] = (gate / (1.0f + expf(-gate))) * up;
            }
        }
        return cute::make_tuple(load, store);
    }
};

extern "C" int ffn_epilogue_layout_check() {
    typename Retained::BaseMain::TiledMma mma;
    for (int thread = 0; thread < size(mma); ++thread) {
        auto      coords = mma.get_slice(thread).partition_C(make_identity_tensor(Shape<_128, _128>{}));
        const int step   = PairGroup == 1 ? 1 : (PairGroup / 16) * size<0>(coords) * size<1>(coords);
        for (int i = 0; i < size(coords); ++i) {
            const auto a = coords(i), b = coords(i ^ step);
            if (get<0>(a) != get<0>(b) || (get<1>(a) ^ PairGroup) != get<1>(b)) {
                printf("thread=%d step=%d size=%d,%d,%d\n", thread, step, int(size<0>(coords)), int(size<1>(coords)),
                       int(size<2>(coords)));
                for (int j = 0; j < size(coords); ++j) {
                    printf("i=%d m=%d n=%d\n", j, int(get<0>(coords(j))), int(get<1>(coords(j))));
                }
                return -1;
            }
        }
    }
    return 0;
}

#include "fp8-cutlass-ffn-dual.cuh"
#include "fp8-cutlass-ffn-load.cuh"

// Standalone reference entry point for exact-output qualification.
__global__ void ffn_reference(const float * gate, const float * up, float * output, int64_t count) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
    }
}

extern "C" int ffn_epilogue_reference(const float * gate,
                                      const float * up,
                                      float *       output,
                                      int64_t       count,
                                      cudaStream_t  stream) {
    ffn_reference<<<(count + 255) / 256, 256, 0, stream>>>(gate, up, output, count);
    return int(cudaGetLastError());
}
