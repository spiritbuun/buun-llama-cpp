// Private SM120 producer extension: two 64-row weight loads populate the
// retained 128-row B tile. The consumer/MMA and split2 accumulation are unchanged.
struct SeparateMain : Retained::Main {
    using Base       = Retained::Main;
    using HalfLayout = decltype(composition(
        typename Base::SmemLayoutB{},
        make_layout(Shape<_64, _128, Int<Base::DispatchPolicy::Stages>>{}, Stride<_1, _128, Int<128 * 128>>{})));
    using HalfTma    = decltype(make_tma_copy(typename Base::GmemTiledCopyB{},
                                              make_tensor(recast_ptr<typename Base::TmaInternalElementB>(nullptr),
                                                          repeat_like(typename Base::StrideB{}, int32_t(0)),
                                                          typename Base::StrideB{}),
                                              HalfLayout{}(_, _, _0{}),
                                              Shape<_64, _128>{},
                                              _1{}));

    struct Arguments : Base::Arguments {
        const cutlass::float_e4m3_t * ptr_up;
    };

    struct Params : Base::Params {
        HalfTma gate;
        HalfTma up;
    };

    template <class PS>
    static Params to_underlying_arguments(PS const & shape, const Arguments & args, void * workspace) {
        auto [m, n, k, l]       = shape;
        const auto native_shape = make_shape(m, n / 2, k, l);
        // Build only descriptors for valid original allocations. The inherited
        // full-width B descriptor is unused by our load path, never dereferenced.
        auto       base         = Base::to_underlying_arguments(native_shape, args, workspace);
        auto       gate_tensor  = make_tensor(recast_ptr<typename Base::TmaInternalElementB>(args.ptr_B),
                                              make_layout(make_shape(n / 2, k, l), args.dB));
        auto       up_tensor    = make_tensor(recast_ptr<typename Base::TmaInternalElementB>(args.ptr_up),
                                              make_layout(make_shape(n / 2, k, l), args.dB));
        HalfTma    gate         = make_tma_copy(typename Base::GmemTiledCopyB{}, gate_tensor, HalfLayout{}(_, _, _0{}),
                                                Shape<_64, _128>{}, _1{});
        HalfTma    up           = make_tma_copy(typename Base::GmemTiledCopyB{}, up_tensor, HalfLayout{}(_, _, _0{}),
                                                Shape<_64, _128>{}, _1{});
        return { base, gate, up };
    }

    template <class PS> static bool can_implement(PS const & shape, const Arguments & args) {
        auto [m, n, k, l] = shape;
        return n % 128 == 0 && args.ptr_up && Base::can_implement(make_shape(m, n / 2, k, l), args);
    }

    CUTLASS_DEVICE static void prefetch_tma_descriptors(const Params & p) {
        cute::prefetch_tma_descriptor(p.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(p.gate.get_tma_descriptor());
        cute::prefetch_tma_descriptor(p.up.get_tma_descriptor());
    }

    template <class PS> CUTLASS_DEVICE auto load_init(PS const & shape, const Params & p) const {
        auto [m, n, k, l] = shape;
        using X           = Underscore;
        auto a            = p.tma_load_a.get_tma_tensor(make_shape(m, k, l));
        auto gate         = p.gate.get_tma_tensor(make_shape(n / 2, k, l));
        auto up           = p.up.get_tma_tensor(make_shape(n / 2, k, l));
        // The second returned tensor is scheduling coordinates only. Actual
        // global loads use the independently bounded gate/up TMA tensors.
        auto virtual_b    = make_identity_tensor(make_shape(n, k, l));
        auto ga           = local_tile(a, typename Base::TileShape{}, make_coord(_, _, _), Step<_1, X, _1>{});
        auto gb           = local_tile(virtual_b, typename Base::TileShape{}, make_coord(_, _, _), Step<X, _1, _1>{});
        auto gg           = local_tile(gate, Shape<_128, _64, _128>{}, make_coord(_, _, _), Step<X, _1, _1>{});
        auto gu           = local_tile(up, Shape<_128, _64, _128>{}, make_coord(_, _, _), Step<X, _1, _1>{});
        return cute::make_tuple(ga, gb, gg, gu);
    }

    template <class Inputs, class Coord, class Iterator>
    CUTLASS_DEVICE void load(const Params &                  p,
                             typename Base::MainloopPipeline pipeline,
                             typename Base::PipelineState    state,
                             const Inputs &                  inputs,
                             const Coord &                   coord,
                             Iterator                        iter,
                             int                             count,
                             int,
                             uint32_t,
                             typename Base::TensorStorage & smem) {
        if (cute::elect_one_sync()) {
            auto          sa     = make_tensor(make_smem_ptr(smem.smem_A.data()), typename Base::SmemLayoutA{});
            auto          sg     = make_tensor(make_smem_ptr(smem.smem_B.data()), HalfLayout{});
            constexpr int offset = typename Base::SmemLayoutB{}(make_coord(64, 0, 0));
            auto          su     = make_tensor(make_smem_ptr(smem.smem_B.data() + offset), HalfLayout{});
            auto [m, n, k, l]    = coord;
            auto ga              = get<0>(inputs)(_, _, m, _, l);
            auto gg              = get<2>(inputs)(_, _, n, _, l);
            auto gu              = get<3>(inputs)(_, _, n, _, l);
            auto ta              = p.tma_load_a.get_slice(0);
            auto tg              = p.gate.get_slice(0);
            auto tu              = p.up.get_slice(0);
            auto tga             = ta.partition_S(ga);
            auto tsa             = ta.partition_D(sa);
            auto tgg             = tg.partition_S(gg);
            auto tsg             = tg.partition_D(sg);
            auto tgu             = tu.partition_S(gu);
            auto tsu             = tu.partition_D(su);
            CUTLASS_PRAGMA_NO_UNROLL
            for (; count > 0; --count) {
                pipeline.producer_acquire(state);
                auto *    barrier = pipeline.producer_get_barrier(state);
                const int stage   = state.index();
                copy(p.tma_load_a.with(*barrier, uint16_t(0)), tga(_, _, _, *iter), tsa(_, _, _, stage));
                copy(p.gate.with(*barrier, uint16_t(0)), tgg(_, _, _, *iter), tsg(_, _, _, stage));
                copy(p.up.with(*barrier, uint16_t(0)), tgu(_, _, _, *iter), tsu(_, _, _, stage));
                ++iter;
                ++state;
            }
        }
    }
};

extern "C" int ffn_separate_layout_check() {
    SeparateMain::HalfLayout        half;
    SeparateMain::Base::SmemLayoutB full;
    const int                       offset = full(make_coord(64, 0, 0));
    for (int s = 0; s < SeparateMain::DispatchPolicy::Stages; ++s) {
        for (int k = 0; k < 128; ++k) {
            for (int n = 0; n < 64; ++n) {
                if (half(make_coord(n, k, s)) != full(make_coord(n, k, s))) {
                    return -1;
                }
                if (half(make_coord(n, k, s)) + offset != full(make_coord(n + 64, k, s))) {
                    return -2;
                }
            }
        }
    }
    return ffn_epilogue_layout_check();
}
