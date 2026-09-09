// SPDX-License-Identifier: Apache-2.0
// Experimental integration using NVIDIA CUTLASS 4.3.4 (BSD-3-Clause).
// Configuration informed by vLLM v0.28.0 (Apache-2.0), commit2cf0a691:
// csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm{,_sm120_fp8_dispatch}.cuh
// Unlike vLLM's wrapper, this has no Torch dependency, writes F32, scales in
// the native order, and can reduce two separately accumulated K partitions.
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <type_traits>

using namespace cute;
namespace fusion = cutlass::epilogue::fusion;

// Each half uses CUTLASS's original MMA loop, including its accumulator reset
// and pipeline releases. Keep the first result in registers instead of passing
// a partial matrix between two launches. The caller requires whole K tiles in
// both halves; the enclosing kernel advances its own state by the full count.
template <class Base> struct RegisterSplitMain : Base {
    template <class Fragment>
    CUTLASS_DEVICE void mma(typename Base::MainloopPipeline pipeline,
                           typename Base::PipelineState state,
                           Fragment & accum, int tiles, int thread,
                           typename Base::TensorStorage & storage,
                           typename Base::Params const & params) {
        auto first = make_fragment_like(accum);
        Base::mma(pipeline, state, first, tiles / 2, thread, storage, params);
        state.advance(tiles / 2);
        Base::mma(pipeline, state, accum, tiles / 2, thread, storage, params);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
            accum(i) = first(i) + accum(i);
        }
    }
};

template <bool Scale, bool Add, bool RegisterSplit = false> struct Plan {
    using Tile    = Shape<_128, _128, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using XScale  = fusion::Sm90ColBroadcast<0, Tile, float, float, Stride<_1, _0, _0>, 4, false>;
    using WScale  = fusion::Sm90RowBroadcast<0, Tile, float, float, Stride<_0, _1, _0>, 4, false>;
    using Mul     = fusion::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Sum =
        fusion::Sm90EVT<fusion::Sm90Compute<cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
                        fusion::Sm90SrcFetch<float>,
                        fusion::Sm90AccFetch>;
    using Acc      = std::conditional_t<Add, Sum, fusion::Sm90AccFetch>;
    using Inner    = fusion::Sm90EVT<Mul, Acc, XScale>;
    using Scaled   = fusion::Sm90EVT<Mul, Inner, WScale>;
    using Identity = fusion::Sm90EVT<
        fusion::
            Sm90Compute<cutlass::epilogue::thread::Identity, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
        fusion::Sm90AccFetch>;
    using EVT = std::conditional_t<Scale, Scaled, Identity>;
    using Epi =
        typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm120,
                                                                  cutlass::arch::OpClassTensorOp,
                                                                  Tile,
                                                                  Cluster,
                                                                  cutlass::epilogue::collective::EpilogueTileAuto,
                                                                  float,
                                                                  float,
                                                                  float,
                                                                  cutlass::layout::RowMajor,
                                                                  4,
                                                                  float,
                                                                  cutlass::layout::RowMajor,
                                                                  4,
                                                                  cutlass::epilogue::collective::EpilogueScheduleAuto,
                                                                  EVT>::CollectiveOp;
    using BaseMain = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassTensorOp,
        cutlass::float_e4m3_t,
        cutlass::layout::RowMajor,
        16,
        cutlass::float_e4m3_t,
        cutlass::layout::ColumnMajor,
        16,
        float,
        Tile,
        Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using Main = std::conditional_t<RegisterSplit, RegisterSplitMain<BaseMain>, BaseMain>;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
    using Op     = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static typename Op::Arguments args(const void *  w,
                                       const void *  x,
                                       const float * ws,
                                       const float * xs,
                                       const float * prev,
                                       float *       out,
                                       int           m,
                                       int           n,
                                       int           k,
                                       int           ld,
                                       int           device,
                                       int           sms) {
        typename Kernel::StrideA sa{ ld, _1{}, int64_t(0) };
        typename Kernel::StrideB sb{ ld, _1{}, int64_t(0) };
        typename Kernel::StrideC sc{ n, _1{}, int64_t(0) };
        typename Kernel::StrideD sd{ n, _1{}, int64_t(0) };
        typename EVT::Arguments  evt{};
        if constexpr (Scale) {
            typename Inner::Arguments inner{
                {},
                { xs, 0.0f, {} },
                {}
            };
            evt = {
                inner, { ws, 0.0f, {} },
                 {}
            };
        }
        cutlass::KernelHardwareInfo hw;
        hw.device_id = device;
        hw.sm_count  = sms;
        return {
            cutlass::gemm::GemmUniversalMode::kGemm,
            { m, n, k, 1 },
            { static_cast<const cutlass::float_e4m3_t *>(x), sa, static_cast<const cutlass::float_e4m3_t *>(w), sb },
            { evt, prev, sc, out, sd },
            hw
        };
    }
};

// Probe first (workspace=nullptr), then run with the caller's context-owned
// workspace. No static device storage, allocations, or synchronization here.
extern "C" int buun_fp8_cutlass(const void *  w,
                                const void *  x,
                                const float * ws,
                                const float * xs,
                                float *       out,
                                int           m,
                                int           n,
                                int           k,
                                int           parts,
                                int           device,
                                int           sms,
                                void *        workspace,
                                size_t        capacity,
                                size_t *      required,
                                cudaStream_t  stream) {
    if (m < 1 || n < 1 || k < 1 || (parts != 1 && parts != 2) || n % 16 || k % (16 * parts)) {
        return -1;
    }
    if (parts == 2 && k % 256 == 0) {
        using Register = Plan<true, false, true>;
        auto args = Register::args(w, x, ws, xs, nullptr, out, m, n, k, k, device, sms);
        if (Register::Op::can_implement(args) == cutlass::Status::kSuccess) {
            *required = Register::Op::get_workspace_size(args);
            if (!workspace) {
                return 0;
            }
            if (capacity < *required) {
                return -2;
            }
            typename Register::Op op;
            return int(op.run(args, workspace, stream));
        }
    }
    using Single               = Plan<true, false>;
    using First                = Plan<false, false>;
    using Last                 = Plan<true, true>;
    const size_t partial_bytes = parts == 2 ? (size_t(m) * n * sizeof(float) + 255) / 256 * 256 : 0;
    auto *       partial       = static_cast<float *>(workspace);
    void *       scratch       = workspace ? static_cast<char *>(workspace) + partial_bytes : nullptr;
    auto         a             = Single::args(w, x, ws, xs, nullptr, out, m, n, k, k, device, sms);
    auto         f             = First::args(w, x, ws, xs, nullptr, partial, m, n, k / parts, k, device, sms);
    auto l = Last::args(static_cast<const char *>(w) + k / parts, static_cast<const char *>(x) + k / parts, ws, xs,
                        partial, out, m, n, k / parts, k, device, sms);
    const auto supported =
        parts == 1 ? Single::Op::can_implement(a) :
                     (First::Op::can_implement(f) == cutlass::Status::kSuccess ? Last::Op::can_implement(l) :
                                                                                 cutlass::Status::kErrorNotSupported);
    if (supported != cutlass::Status::kSuccess) {
        return int(supported);
    }
    *required =
        partial_bytes + (parts == 1 ? Single::Op::get_workspace_size(a) :
                                      std::max(First::Op::get_workspace_size(f), Last::Op::get_workspace_size(l)));
    if (!workspace) {
        return 0;
    }
    if (capacity < *required) {
        return -2;
    }
    if (parts == 1) {
        typename Single::Op op;
        return int(op.run(a, scratch, stream));
    }
    typename First::Op first;
    auto               status = first.run(f, scratch, stream);
    if (status != cutlass::Status::kSuccess) {
        return int(status);
    }
    typename Last::Op last;
    return int(last.run(l, scratch, stream));
}
