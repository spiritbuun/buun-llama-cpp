/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/atom/copy_traits.hpp"
#include "cute/numeric/int.hpp"
#include "cute/pointer.hpp"
#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

namespace cute {

// On Volta, ld/st.global.cg.v4.u32 disassembles to LDG/STG.E.128.STRONG.GPU in SASS.
struct SM70_LDG_GLOBAL_CG_128b : UniversalCopy<uint_bit_t<128>> {
    using SRegisters = uint_bit_t<128>[1];
    using DRegisters = uint_bit_t<128>[1];

    CUTE_HOST_DEVICE static void
    copy(uint_bit_t<128> const& gmem_src, uint_bit_t<128>& smem_dst) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        uint32_t r0, r1, r2, r3;
        auto gmem_ptr = reinterpret_cast<uint32_t const*>(&gmem_src);
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                     : "l"(gmem_ptr));
        uint4 tmp = make_uint4(r0, r1, r2, r3);
        smem_dst = reinterpret_cast<uint_bit_t<128> const&>(tmp);
#else
        smem_dst = gmem_src;
#endif
    }
};

template <>
struct Copy_Traits<SM70_LDG_GLOBAL_CG_128b> {
    using ThrID = Layout<_1>;
    using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<uint_bit_t<128>>::value>>>;
    using DstLayout = Layout<Shape<_1, Int<sizeof_bits<uint_bit_t<128>>::value>>>;
    using RefLayout = SrcLayout;
};

struct SM70_STG_GLOBAL_CG_128b : UniversalCopy<uint_bit_t<128>> {
    using SRegisters = uint_bit_t<128>[1];
    using DRegisters = uint_bit_t<128>[1];

    CUTE_HOST_DEVICE static void
    copy(uint_bit_t<128> const& src, uint_bit_t<128>& gmem_dst) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        const uint4 tmp = reinterpret_cast<uint4 const&>(src);
        auto gmem_ptr = reinterpret_cast<uint32_t*>(&gmem_dst);
        asm volatile("st.global.cg.v4.u32 [%0], {%1, %2, %3, %4};\n"
                     :
                     : "l"(gmem_ptr), "r"(tmp.x), "r"(tmp.y), "r"(tmp.z), "r"(tmp.w));
#else
        gmem_dst = src;
#endif
    }
};

template <>
struct Copy_Traits<SM70_STG_GLOBAL_CG_128b> {
    using ThrID = Layout<_1>;
    using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<uint_bit_t<128>>::value>>>;
    using DstLayout = Layout<Shape<_1, Int<sizeof_bits<uint_bit_t<128>>::value>>>;
    using RefLayout = SrcLayout;

    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void
    copy_unpack(Copy_Traits        const&,
                Tensor<TS, SLayout> const& src,
                Tensor<TD, DLayout>      & dst)
    {
        static_assert(is_rmem<TS>::value, "Expected register source for st.global.cg.");
        static_assert(is_gmem<TD>::value, "Expected global destination for st.global.cg.");

        Tensor rS = recast<uint_bit_t<128> const>(src);
        Tensor rD = recast<uint_bit_t<128>>(dst);

        CUTE_STATIC_ASSERT_V(size(rS) == Int<1>{},
            "st.global.cg src layout doesn't vectorize into a single 128-bit register.");
        CUTE_STATIC_ASSERT_V(size(rD) == Int<1>{},
            "st.global.cg dst layout doesn't vectorize into a single 128-bit global store.");

        SM70_STG_GLOBAL_CG_128b::copy(rS[0], rD[0]);
    }
};

}  // namespace cute

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_fwd_kernel_traits  {
    using Element = cutlass::half_t;
    using ElementAccum = float;
    using index_t = int64_t;
    using MMA_Atom_Arch = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
    using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, Element>;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    static_assert(kBlockM % kNWarps == 0, "warp-stationary requires blockM divisible by CTA warps");
    static constexpr int kWarpRows = kBlockM / kNWarps;
    static_assert(kWarpRows == 8 || kWarpRows == 16 || kWarpRows == 32 || kWarpRows == 64,
                  "SM70 forward P-fragment conversion only supports kWarpRows == 8, 16, 32, or 64");

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_1, _4, _1>>,
        Tile<Int<kWarpRows>, Int<kBlockN / 4>, _4>
    >;
    static constexpr int kMmaThreads = decltype(size(TiledMma{}))::value;
    static_assert(kMmaThreads % 32 == 0, "SM70 TiledMma must use a whole number of warps");
    static_assert(kNThreads % kMmaThreads == 0, "threadblock threads must be divisible by TiledMma threads");
    static constexpr bool Share_Q_K_smem = false;
    static constexpr bool Is_Q_in_regs = false;

    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // Keep P in a simple row-major layout so the register-only C->A conversion can
    // follow the logical coordinates directly without reasoning about a physical swizzle.
    using SmemLayoutAtomP = Layout<Shape<_8, Int<kBlockN>>,
                                   Stride<Int<kBlockN>, _1>>;
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<kBlockM>, Int<kBlockN>>{}));

    // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemPSize = size(SmemLayoutP{}) * sizeof(Element);
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;
    static_assert(kSmemSize <= 96 * 1024, "kSmemSize must fit within the 96KB shared memory limit on SM70");

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = SM70_LDG_GLOBAL_CG_128b;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read

    // from how many rows does each thread have to fetch
    static constexpr int kGmemRowsPerThread = kBlockN / (kNThreads / kGmemThreadsPerRow);
    // Here we assign a contiguous tile to each thread, rather than a 1x8 row every 
    // (kNThreads / kGmemThreadsPerRow) rows, ensuring that the elements assigned to each thread
    // do not cross a page boundary. This way, each thread need only fetch 1 page index per
    // mainloop iteration. R>udimentary testing shows no slowdown.
    using GmemTiledCopyQKVPaged = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));
    using SmemTiledCopyOToReg = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per load
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<SM70_STG_GLOBAL_CG_128b, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    static constexpr int kOaccumElemsPerAccess = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static_assert(kBlockKSmem % kOaccumElemsPerAccess == 0, "kBlockKSmem must be divisible by the Oaccum vector width");
    static constexpr int kOaccumThreadsPerRow = kBlockKSmem / kOaccumElemsPerAccess;
    static_assert(kNThreads % kOaccumThreadsPerRow == 0, "kNThreads must be divisible by the Oaccum threads-per-row");
    static constexpr int kOaccumRows = kNThreads / kOaccumThreadsPerRow;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kOaccumRows>, Int<kOaccumThreadsPerRow>>,
                                        Stride<Int<kOaccumThreadsPerRow>, _1>>;
    using SmemTiledCopyOaccumToReg = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<SM70_STG_GLOBAL_CG_128b, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
    using GmemTiledCopyRotcossinPaged = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _4>, Stride<_4, _1>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinContPaged = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>{}));  // Val layout, 8 vals per load
};

// Is_V_in_regs is an option to reduce smem usage, but will increase register pressue.
// No_double_buffer is another option to reduce smem usage, but will slow things down.
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         int AtomLayoutMSdP_=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=2,
         bool Is_V_in_regs_=false, bool No_double_buffer_=false, typename elem_type=cutlass::half_t>
struct Flash_bwd_kernel_traits {

};

////////////////////////////////////////////////////////////////////////////////////////////////////
