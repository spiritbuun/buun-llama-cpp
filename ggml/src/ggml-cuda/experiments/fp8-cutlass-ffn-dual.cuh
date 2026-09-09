/*
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// Private dual-activation experiment. The shared-memory copy/MMA pipeline is
// adapted from NVIDIA CUTLASS sm120_mma_tma.hpp (BSD-3-Clause); no vendor file
// is changed. Preserve the retained two separate K accumulations.
struct DualMain : SeparateMain {
    using Parent = SeparateMain;
    using Plain  = Retained::BaseMain;

    struct Arguments : Parent::Arguments {
        const cutlass::float_e4m3_t * ptr_a_up;
        const int32_t *               gate_marker;
        const int32_t *               up_marker;
    };

    struct TransferBytes {
        const int32_t * gate;
        const int32_t * up;

        CUTLASS_DEVICE bool same() const { return *gate == *up; }

        // The CUTLASS kernel reads this field only while initializing its
        // device pipeline. Load issues exactly this many bytes per stage.
        CUTLASS_DEVICE operator uint32_t() const {
            return Plain::TmaTransactionBytes + (same() ? 0 : Plain::TmaTransactionBytesMK);
        }
    };

    struct Params : Parent::Params {
        typename Plain::Params::TMA_A up_a;
        TransferBytes                 tma_transaction_bytes;
    };

    using OriginalStorage = typename Plain::TensorStorage;

    struct DualStorage : OriginalStorage {
        alignas(1024)
            cute::array_aligned<typename Plain::SmemAllocTypeA, cute::cosize_v<typename Plain::SmemLayoutA>> smem_up_a;
    };

    using TensorStorage = DualStorage;

    struct SharedStorage {
        TensorStorage                   tensors;
        typename Plain::PipelineStorage pipeline_storage;
    };

    template <class PS> static Params to_underlying_arguments(PS const & shape, const Arguments & a, void * workspace) {
        auto p            = Parent::to_underlying_arguments(shape, a, workspace);
        auto [m, n, k, l] = shape;
        auto t            = make_tensor(recast_ptr<typename Plain::TmaInternalElementA>(a.ptr_a_up),
                                        make_layout(make_shape(m, k, l), a.dA));
        auto tma = make_tma_copy(typename Plain::GmemTiledCopyA{}, t, typename Plain::SmemLayoutA{}(_, _, _0{}),
                                 Shape<_128, _128>{}, _1{});
        return {
            p, tma, { a.gate_marker, a.up_marker }
        };
    }

    template <class PS> static bool can_implement(PS const & shape, const Arguments & a) {
        return a.ptr_a_up && a.gate_marker && a.up_marker && uintptr_t(a.ptr_a_up) % 16 == 0 &&
               Parent::can_implement(shape, a);
    }

    CUTLASS_DEVICE static void prefetch_tma_descriptors(const Params & p) {
        Parent::prefetch_tma_descriptors(p);
        cute::prefetch_tma_descriptor(p.up_a.get_tma_descriptor());
    }

    template <class PS> CUTLASS_DEVICE auto load_init(PS const & shape, const Params & p) const {
        auto first        = Parent::load_init(shape, p);
        auto [m, n, k, l] = shape;
        auto a            = p.up_a.get_tma_tensor(make_shape(m, k, l));
        auto ga           = local_tile(a, typename Plain::TileShape{}, make_coord(_, _, _), Step<_1, Underscore, _1>{});
        return cute::append(first, ga);
    }

    template <class Inputs, class Coord, class Iterator>
    CUTLASS_DEVICE void load(const Params &                   p,
                             typename Plain::MainloopPipeline pipeline,
                             typename Plain::PipelineState    state,
                             const Inputs &                   inputs,
                             const Coord &                    coord,
                             Iterator                         iter,
                             int                              count,
                             int,
                             uint32_t,
                             TensorStorage & smem) {
        if (cute::elect_one_sync()) {
            auto          sa     = make_tensor(make_smem_ptr(smem.smem_A.data()), typename Plain::SmemLayoutA{});
            auto          sa2    = make_tensor(make_smem_ptr(smem.smem_up_a.data()), typename Plain::SmemLayoutA{});
            auto          sg     = make_tensor(make_smem_ptr(smem.smem_B.data()), typename Parent::HalfLayout{});
            constexpr int offset = typename Plain::SmemLayoutB{}(make_coord(64, 0, 0));
            auto          su  = make_tensor(make_smem_ptr(smem.smem_B.data() + offset), typename Parent::HalfLayout{});
            auto [m, n, k, l] = coord;
            auto ga           = get<0>(inputs)(_, _, m, _, l);
            auto ga2          = get<4>(inputs)(_, _, m, _, l);
            auto gg           = get<2>(inputs)(_, _, n, _, l);
            auto gu           = get<3>(inputs)(_, _, n, _, l);
            auto ta           = p.tma_load_a.get_slice(0);
            auto ta2          = p.up_a.get_slice(0);
            auto tg           = p.gate.get_slice(0);
            auto tu           = p.up.get_slice(0);
            auto tga          = ta.partition_S(ga);
            auto tsa          = ta.partition_D(sa);
            auto tga2         = ta2.partition_S(ga2);
            auto tsa2         = ta2.partition_D(sa2);
            auto tgg          = tg.partition_S(gg);
            auto tsg          = tg.partition_D(sg);
            auto tgu          = tu.partition_S(gu);
            auto tsu          = tu.partition_D(su);
            CUTLASS_PRAGMA_NO_UNROLL
            for (; count > 0; --count) {
                pipeline.producer_acquire(state);
                auto *    barrier = pipeline.producer_get_barrier(state);
                const int stage   = state.index();
                copy(p.tma_load_a.with(*barrier, uint16_t(0)), tga(_, _, _, *iter), tsa(_, _, _, stage));
                if (!p.tma_transaction_bytes.same()) {
                    copy(p.up_a.with(*barrier, uint16_t(0)), tga2(_, _, _, *iter), tsa2(_, _, _, stage));
                }
                copy(p.gate.with(*barrier, uint16_t(0)), tgg(_, _, _, *iter), tsg(_, _, _, stage));
                copy(p.up.with(*barrier, uint16_t(0)), tgu(_, _, _, *iter), tsu(_, _, _, stage));
                ++iter;
                ++state;
            }
        }
    }

    template <class Fragment>
    CUTLASS_DEVICE void mma_half(typename Plain::MainloopPipeline pipeline,
                                 typename Plain::PipelineState    state,
                                 Fragment &                       accum,
                                 int                              count,
                                 int                              thread,
                                 TensorStorage &                  storage) {
        clear(accum);
        auto sa  = make_tensor(make_smem_ptr(storage.smem_A.data()), typename Plain::SmemLayoutA{});
        auto sa2 = make_tensor(make_smem_ptr(storage.smem_up_a.data()), typename Plain::SmemLayoutA{});
        auto sb  = make_tensor(make_smem_ptr(storage.smem_B.data()), typename Plain::SmemLayoutB{});
        typename Plain::TiledMma mma;
        auto                     tm         = mma.get_thread_slice(thread);
        auto                     ra         = tm.partition_fragment_A(sa(_, _, _0{}));
        auto                     ra2        = tm.partition_fragment_A(sa2(_, _, _0{}));
        auto                     rb         = tm.partition_fragment_B(sb(_, _, _0{}));
        auto                     ca         = make_tiled_copy_A(typename Plain::SmemCopyAtomA{}, mma);
        auto                     cb         = make_tiled_copy_B(typename Plain::SmemCopyAtomB{}, mma);
        auto                     ta         = ca.get_thread_slice(thread);
        auto                     tb         = cb.get_thread_slice(thread);
        auto                     tsa        = ta.partition_S(as_position_independent_swizzle_tensor(sa));
        auto                     tsa2       = ta.partition_S(as_position_independent_swizzle_tensor(sa2));
        auto                     tsb        = tb.partition_S(as_position_independent_swizzle_tensor(sb));
        auto                     tra        = ta.retile_D(ra);
        auto                     tra2       = ta.retile_D(ra2);
        auto                     trb        = tb.retile_D(rb);
        constexpr int            blocks     = size<2>(decltype(ra){});
        int                      stage      = state.index();
        auto                     copy_block = [&](auto kb) {
            copy(ca, tsa(_, _, kb, stage), tra(_, _, kb));
            copy(ca, tsa2(_, _, kb, stage), tra2(_, _, kb));
            copy(cb, tsb(_, _, kb, stage), trb(_, _, kb));
        };
        auto gemm_block = [&](auto kb) {
            // Same independent MMA atoms/K order; each N half uses its own A.
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<1>(accum); ++mi) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<2>(accum); ++ni) {
                    const int ns = (mi & 1) ? size<2>(accum) - 1 - ni : ni;
                    if (ns < size<2>(accum) / 2) {
                        cute::gemm(mma, ra(_, mi, kb), rb(_, ns, kb), accum(_, mi, ns));
                    } else {
                        cute::gemm(mma, ra2(_, mi, kb), rb(_, ns, kb), accum(_, mi, ns));
                    }
                }
            }
        };
        pipeline.consumer_wait(state);
        copy_block(_0{});
        CUTLASS_PRAGMA_NO_UNROLL
        for (; count > 1; --count) {
            for_each(make_int_sequence<blocks>{}, [&](auto kb) {
                auto next = ((kb + 1) == blocks) ? 0 : kb + 1;
                if (kb == blocks - 1) {
                    cutlass::arch::NamedBarrier::sync(thr_size(mma),
                                                      cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                    pipeline.consumer_release(state);
                    ++state;
                    stage = state.index();
                    pipeline.consumer_wait(state);
                }
                copy_block(next);
                gemm_block(kb);
            });
        }
        for_each(make_int_sequence<blocks>{}, [&](auto kb) {
            auto next = ((kb + 1) == blocks) ? 0 : kb + 1;
            if (kb == blocks - 1) {
                cutlass::arch::NamedBarrier::sync(thr_size(mma),
                                                  cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
                pipeline.consumer_release(state);
                ++state;
            }
            if (next > 0) {
                copy_block(next);
            }
            gemm_block(kb);
        });
    }

    template <class Fragment>
    CUTLASS_DEVICE void mma(typename Plain::MainloopPipeline pipeline,
                            typename Plain::PipelineState    state,
                            Fragment &                       accum,
                            int                              count,
                            int                              thread,
                            TensorStorage &                  storage,
                            const Params &                   p) {
        if (p.tma_transaction_bytes.same()) {
            Parent::mma(pipeline, state, accum, count, thread, static_cast<OriginalStorage &>(storage), p);
            return;
        }
        auto first = make_fragment_like(accum);
        mma_half(pipeline, state, first, count / 2, thread, storage);
        state.advance(count / 2);
        mma_half(pipeline, state, accum, count / 2, thread, storage);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(accum); ++i) {
            accum(i) = first(i) + accum(i);
        }
    }
};

using DualKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, DualMain, PairedEpilogue, void>;
using DualOp     = cutlass::gemm::device::GemmUniversalAdapter<DualKernel>;

// x/xu and xs/xsu must be quantizations of the same original F32 activation
// matrix using the respective marker bounds. Equal marker bits permit the
// consumer to use x for both projections; xu need not be populated in that case.
// Marker values must remain stable for the pack-and-GEMM sequence on its stream.
extern "C" int buun_fp8_cutlass_ffn(const void *    gate,
                                    const void *    up,
                                    const void *    x,
                                    const void *    xu,
                                    const float *   gs,
                                    const float *   us,
                                    const float *   xs,
                                    const float *   xsu,
                                    float *         out,
                                    const int32_t * gate_marker,
                                    const int32_t * up_marker,
                                    int             m,
                                    int             n,
                                    int             k,
                                    int             parts,
                                    int             device,
                                    int             sms,
                                    void *          workspace,
                                    size_t          capacity,
                                    size_t *        required,
                                    cudaStream_t    stream) {
    if (m < 1 || n < 1 || n % 64 || k < 1 || k % 256 || parts != 2) {
        return -1;
    }
    cutlass::KernelHardwareInfo hw;
    hw.device_id = device;
    hw.sm_count  = sms;
    typename DualKernel::StrideA sa{ k, _1{}, int64_t(0) };
    typename DualKernel::StrideB sb{ k, _1{}, int64_t(0) };
    DualMain::Arguments          main{
                 { { static_cast<const cutlass::float_e4m3_t *>(x), sa, static_cast<const cutlass::float_e4m3_t *>(gate), sb },
                  static_cast<const cutlass::float_e4m3_t *>(up) },
        static_cast<const cutlass::float_e4m3_t *>(xu),
        gate_marker,
        up_marker
    };
    DualOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, { m, 2 * n, k, 1 },
         main, { xs, gs, out, us, xsu },
         hw
    };
    auto status = DualOp::can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return int(status);
    }
    *required = DualOp::get_workspace_size(args);
    if (!workspace) {
        return 0;
    }
    if (capacity < *required) {
        return -2;
    }
    DualOp op;
    return int(op.run(args, workspace, stream));
}
