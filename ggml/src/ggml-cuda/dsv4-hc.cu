#include "common.cuh"
#include "dsv4-hc.cuh"
#include "unary.cuh"


static constexpr int DSV4_HC = 4;
static constexpr int DSV4_HC_POST_TILE_EMBD = 64;
static constexpr int DSV4_HC_MIX_MAX_TOKENS = 8;

template <bool use_repeated_block>
static __global__ void hc_combine_f32(
        const float * residual, const float * block, const float * inject,
        float * dst, int64_t n_embd, int64_t hc, float inv_hc) {
    const int64_t c = blockIdx.y;
    const int64_t t = blockIdx.z;

    // The scatter weight is constant across the embedding tile. Computing it
    // once also preserves the exact SCALE -> SIGMOID -> SCALE boundaries.
    __shared__ float weight_shared;
    if (threadIdx.x == 0) {
        const float scaled = __fmaf_rn(inv_hc, inject[c + hc*t], 0.0f);
        const float sigmoid = 1.0f / (1.0f + expf(-scaled));
        weight_shared = __fmaf_rn(2.0f, sigmoid, 0.0f);
    }
    __syncthreads();

    const int64_t e = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_embd) {
        return;
    }

    const int64_t i = e + n_embd*(c + hc*t);
    const int64_t ib = use_repeated_block ? i : e + n_embd*t;
    const float product = __fmul_rn(block[ib], weight_shared);
    dst[i] = __fadd_rn(residual[i], product);
}


// 1/x without the IEEE division slow-path branch; bit-identical to 1.0f/x for x in [1e-30, 1e30]
// (checked exhaustively). Every sinkhorn sum here is >= eps > 0.
static __device__ __forceinline__ float dsv4_hc_rcp(float x) {
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return 1.0f/x;
#else
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return __fmaf_rn(r, __fmaf_rn(-x, r, 1.0f), r);
#endif
}

// the sinkhorn helpers are fully unrolled so comb stays in registers; element idst + DSV4_HC*isrc
static __device__ __forceinline__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
#pragma unroll
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
#pragma unroll
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = dsv4_hc_rcp(sum);
#pragma unroll
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ __forceinline__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
#pragma unroll
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
#pragma unroll
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = dsv4_hc_rcp(sum);
#pragma unroll
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

// row softmax + eps of the comb mixes of one token, then the sinkhorn: one column pass even when
// n_iter is 0
static __device__ __forceinline__ void dsv4_hc_comb_sinkhorn(
        float * comb, const float * mixes, float scale_comb, const float * base, int64_t sm0, int64_t sb0,
        float eps, int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

#pragma unroll
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
#pragma unroll
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
#pragma unroll
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = dsv4_hc_rcp(sum);
#pragma unroll
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }
}

// pre/post of one token, lanes 0..2*DSV4_HC-1: mixes and dst already point at that token; returns the
// lane's value (0 on the other lanes)
static __device__ float dsv4_hc_params_pre_post(
        const float * mixes, const float * scale, const float * base, float * dst,
        int64_t sm0, int64_t ss0, int64_t sb0, int64_t sd0, float eps) {
    const int lane = threadIdx.x % WARP_SIZE;

    float v = 0.0f;
    if (lane < 2*DSV4_HC) {
        // The unfused graph stores between MUL and ADD. Explicit round-to-nearest
        // intrinsics preserve that two-kernel arithmetic instead of contracting an FMA.
        const float affine = __fadd_rn(__fmul_rn(mixes[lane*sm0], scale[(lane / DSV4_HC)*ss0]), base[lane*sb0]);
        const float sig = ggml_cuda_op_sigmoid_single(affine);
        v = lane < DSV4_HC ? __fadd_rn(sig, eps) : __fmul_rn(sig, 2.0f);
        dst[lane*sd0] = v;
    }
    return v;
}

// comb of one token, computed by a full warp
static __device__ void dsv4_hc_params_comb(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t sm0,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        float eps,
        int32_t n_iter) {
    // every lane runs the whole sinkhorn in registers, so no shuffle sits on the dependency chain;
    // lane l < 16 stores element l
    const int lane = threadIdx.x % WARP_SIZE;

    float comb[DSV4_HC*DSV4_HC];
    dsv4_hc_comb_sinkhorn(comb, mixes, scale[2*ss0], base, sm0, sb0, eps, n_iter);

    if (lane < DSV4_HC*DSV4_HC) {
        float v = 0.0f;
#pragma unroll
        for (int i = 0; i < DSV4_HC*DSV4_HC; ++i) {
            v = lane == i ? comb[i] : v;
        }
        dst[(2*DSV4_HC + lane)*sd0] = v;
    }
}

static __global__ void dsv4_hc_params_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        float eps,
        int32_t n_iter) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    dsv4_hc_params_pre_post(mixes + it*sm1, scale, base, dst + it*sd1, sm0, ss0, sb0, sd0, eps);
    dsv4_hc_params_comb(mixes + it*sm1, scale, base, dst + it*sd1, sm0, ss0, sb0, sd0, eps, n_iter);
}

// RMS_NORM -> MUL_MAT(f32 hc_fn) -> DSV4_HC_PARAMS [-> DSV4_HC_PRE [-> RMS_NORM -> MUL]] in one launch.
// rms_norm only scales each token by a scalar, so every block dots its columns of the raw streams against
// the 24 hc_fn rows and accumulates sum(x^2) alongside; the last block to finish sums the slices and
// applies rsqrt(mean + eps) before the params epilogue. With apply, the other blocks wait for the last
// block to publish pre, then mix the streams into the DSV4_HC_PRE output while it runs the sinkhorn.
// With norm, the blocks also accumulate the Gram matrix G of the streams: the mixed row y = scale*x*pre
// has sum(y^2) = scale^2 pre^T G pre, so the last block publishes its rms scale with pre and the mixing
// blocks apply the norm without another grid barrier.
static constexpr int DSV4_HC_PRE_FUSED_MAX_TOKENS = 4;

// a warp per token for the sinkhorn. The grid never exceeds the block size, so the last block reduces one
// partial row per thread; 64-thread blocks would cap the grid below one pass over n_embd 5120
static constexpr int DSV4_HC_PRE_FUSED_THREADS = 128;
static constexpr int DSV4_HC_MIX = (2 + DSV4_HC)*DSV4_HC;
static constexpr int DSV4_HC_GRAM = DSV4_HC*(DSV4_HC + 1)/2;

// ungated DSV4_HC_PRE: dst[i0, it] = scale * sum_ih x[i0, ih, it]*pre[ih, it]
struct dsv4_hc_pre_apply_args {
    const float * x;
    float * dst;
    int64_t n_embd;
    int64_t sx0, sx1, sx2;
    int64_t sd0, sd1;
    float scale;
};

// DSV4_HC_POST with comb, producing the stream x of the fused hc_pre: dst[i0, ih, it] =
// x[i0, it]*post[ih, it] + sum_isrc residual[i0, isrc, it]*comb[ih, isrc, it]
// dst is contiguous and laid out like the stream, so it shares the stream's indexing
struct dsv4_hc_post_fold_args {
    const float * x;
    const float * residual;
    const float * post;
    const float * comb;
    float * dst;
    int64_t sx0, sx1;
    int64_t sr0, sr1, sr2;
    int64_t sp0, sp1;
    int64_t sc0, sc1, sc2;
};

// RMS_NORM -> MUL(w) of the DSV4_HC_PRE output into dst (contiguous rows)
struct dsv4_hc_pre_norm_args {
    const float * w;
    float * dst;
    float * rs; // [n_tokens] rms scales, published with pre
    int64_t sd1;
    float eps;
};

// ticket: the block ticket counter; launch_epoch: bumped once pre is published
template <int n_tokens, bool apply, bool fold, bool norm>
static __global__ void dsv4_hc_pre_fused_f32(
        const float * x, const float * w, const float * scale, const float * base,
        float * partials, float * dst, unsigned int * ticket_counter, unsigned int * launch_epoch, int64_t k, int64_t sx1, int64_t sw1, int64_t ss0,
        int64_t sb0, int64_t sd1, float rms_eps, float eps, int32_t n_iter, const dsv4_hc_pre_apply_args ap,
        const dsv4_hc_post_fold_args hp, const dsv4_hc_pre_norm_args np) {
    static_assert(apply || !norm, "norm needs apply");
    constexpr int nwarps = DSV4_HC_PRE_FUSED_THREADS/WARP_SIZE;
    // per token: the mixes, sum(x^2), and with norm the upper triangle of the streams' Gram matrix
    constexpr int nmix = DSV4_HC_MIX + 1 + (norm ? DSV4_HC_GRAM : 0);
    constexpr int nacc = n_tokens*nmix;
    // w may alias the fold's dst as far as the compiler knows, so stores ahead of the dot hold back its loads;
    // with more tokens, keeping the folded values live until the end costs more in spills
    constexpr bool fold_store_last = fold && n_tokens == 1;

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    // read before the ticket, so the last block cannot have bumped it for this launch yet
    unsigned int epoch = 0;
    if (apply && threadIdx.x == 0) {
        epoch = __ldcg(launch_epoch);
    }

    const int64_t n_embd = k/DSV4_HC;
    float acc[nacc] = {};
    // each thread owns stream columns i0, all DSV4_HC streams of them, so the Gram products stay in registers
    for (int64_t i0 = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; i0 < n_embd; i0 += (int64_t) gridDim.x*blockDim.x) {
        float xs[n_tokens][DSV4_HC];
#pragma unroll
        for (int t = 0; t < n_tokens; ++t) {
            if constexpr (fold) {
                // same product and order as dsv4_hc_post_f32<true>; the apply below reads it back after the ticket
                float r[DSV4_HC];
#pragma unroll
                for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                    r[isrc] = hp.residual[i0*hp.sr0 + isrc*hp.sr1 + t*hp.sr2];
                }
                const float xb = hp.x[i0*hp.sx0 + t*hp.sx1];
#pragma unroll
                for (int ih = 0; ih < DSV4_HC; ++ih) {
                    float xv = __fmul_rn(xb, hp.post[ih*hp.sp0 + t*hp.sp1]);
#pragma unroll
                    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                        xv += r[isrc] * hp.comb[ih*hp.sc0 + isrc*hp.sc1 + t*hp.sc2];
                    }
                    xs[t][ih] = xv;
                    if constexpr (!fold_store_last) {
                        hp.dst[t*sx1 + ih*n_embd + i0] = xv;
                    }
                }
            } else {
#pragma unroll
                for (int ih = 0; ih < DSV4_HC; ++ih) {
                    xs[t][ih] = x[t*sx1 + ih*n_embd + i0];
                }
            }
        }
#pragma unroll
        for (int ih = 0; ih < DSV4_HC; ++ih) {
            float wv[DSV4_HC_MIX];
#pragma unroll
            for (int j = 0; j < DSV4_HC_MIX; ++j) {
                wv[j] = w[j*sw1 + ih*n_embd + i0];
            }
#pragma unroll
            for (int t = 0; t < n_tokens; ++t) {
#pragma unroll
                for (int j = 0; j < DSV4_HC_MIX; ++j) {
                    acc[t*nmix + j] += wv[j]*xs[t][ih];
                }
                acc[t*nmix + DSV4_HC_MIX] += xs[t][ih]*xs[t][ih];
            }
        }
        if constexpr (norm) {
#pragma unroll
            for (int t = 0; t < n_tokens; ++t) {
                int g = DSV4_HC_MIX + 1;
#pragma unroll
                for (int ia = 0; ia < DSV4_HC; ++ia) {
#pragma unroll
                    for (int ib = ia; ib < DSV4_HC; ++ib) {
                        acc[t*nmix + g++] += xs[t][ia]*xs[t][ib];
                    }
                }
            }
        }
        if constexpr (fold_store_last) {
#pragma unroll
            for (int t = 0; t < n_tokens; ++t) {
#pragma unroll
                for (int ih = 0; ih < DSV4_HC; ++ih) {
                    hp.dst[t*sx1 + ih*n_embd + i0] = xs[t][ih];
                }
            }
        }
    }

    __shared__ float red[nwarps][nacc];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    // all shuffles before the stores, so the independent reductions overlap
#pragma unroll
    for (int a = 0; a < nacc; ++a) {
        acc[a] = warp_reduce_sum(acc[a]);
    }
    if (lane == 0) {
#pragma unroll
        for (int a = 0; a < nacc; ++a) {
            red[warp][a] = acc[a];
        }
    }
    __syncthreads();

    for (int a = threadIdx.x; a < nacc; a += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int iw = 0; iw < nwarps; ++iw) {
            v += red[iw][a];
        }
        partials[blockIdx.x*nacc + a] = v;
    }

    const unsigned int ticket = ggml_cuda_block_ticket(ticket_counter);
    if (ticket != gridDim.x - 1) {
        if constexpr (apply) {
            // the host caps the grid at the resident block count, so the last ticket always gets scheduled
            __shared__ float pre[n_tokens*DSV4_HC];
            __shared__ float rs[norm ? n_tokens : 1];
            if (threadIdx.x == 0) {
                while (*(volatile unsigned int *) launch_epoch == epoch) {}
                __threadfence();
            }
            __syncthreads();
            if (threadIdx.x < n_tokens*DSV4_HC) {
                pre[threadIdx.x] = __ldcg(dst + (threadIdx.x / DSV4_HC)*sd1 + threadIdx.x % DSV4_HC);
            }
            if constexpr (norm) {
                if (threadIdx.x < n_tokens) {
                    rs[threadIdx.x] = __ldcg(np.rs + threadIdx.x);
                }
            }
            __syncthreads();

            const int64_t nr = ap.n_embd*n_tokens;
            for (int64_t ir = (int64_t) ticket*blockDim.x + threadIdx.x; ir < nr; ir += (int64_t) (gridDim.x - 1)*blockDim.x) {
                const int64_t i0 = ir % ap.n_embd;
                const int64_t it = ir / ap.n_embd;
                // same product and order as dsv4_hc_pre_f32<false>
                float sum = 0.0f;
#pragma unroll
                for (int ih = 0; ih < DSV4_HC; ++ih) {
                    // with fold, other blocks wrote x in this launch: bypass L1
                    const float xv = fold ? __ldcg(ap.x + i0*ap.sx0 + ih*ap.sx1 + it*ap.sx2) : ap.x[i0*ap.sx0 + ih*ap.sx1 + it*ap.sx2];
                    const float wv = pre[it*DSV4_HC + ih];
                    if (ih == 0) {
                        sum = xv * wv;
                    } else {
                        sum += xv * wv;
                    }
                }
                const float v = ap.scale * sum;
                ap.dst[i0*ap.sd0 + it*ap.sd1] = v;
                if constexpr (norm) {
                    // same order as rms_norm_f32 with a fused MUL
                    np.dst[i0 + it*np.sd1] = rs[it] * v * np.w[i0];
                }
            }
        }
        return;
    }
    __threadfence();

    // the params scale and base, loaded now so their latency hides under the partials round trip
    constexpr int nscale = 3; // pre, post, comb
    constexpr int nsb = nscale + DSV4_HC_MIX;
    static_assert(nsb <= nwarps*WARP_SIZE, "a thread per params weight");
    float sbv = 0.0f;
    if (threadIdx.x < nscale) {
        sbv = scale[threadIdx.x*ss0];
    } else if (threadIdx.x < nsb) {
        sbv = base[(threadIdx.x - nscale)*sb0];
    }

    // warp w sums partial rows w, w + nwarps, ...; lane a owns accumulator a, so the loads coalesce
    // and no shuffles are needed
    // a single token takes two accumulators per lane per pass, so its partials load in one round; more
    // tokens take one, to stay clear of spills
    constexpr int na = n_tokens == 1 ? 2 : 1;
    for (int a0 = lane; a0 < nacc; a0 += na*WARP_SIZE) {
        float pv[na][WARP_SIZE];
#pragma unroll
        for (int ia = 0; ia < na; ++ia) {
            const int a = a0 + ia*WARP_SIZE;
#pragma unroll
            for (int ib = 0; ib < WARP_SIZE; ++ib) {
                const unsigned int b = warp + ib*nwarps;
                pv[ia][ib] = a < nacc && b < gridDim.x ? __ldcg(partials + b*nacc + a) : 0.0f;
            }
        }
#pragma unroll
        for (int ia = 0; ia < na; ++ia) {
            const int a = a0 + ia*WARP_SIZE;
            float v = 0.0f;
#pragma unroll
            for (int ib = 0; ib < WARP_SIZE; ++ib) {
                v += pv[ia][ib];
            }
            if (a < nacc) {
                red[warp][a] = v;
            }
        }
    }
    __syncthreads();

    __shared__ float mixes[nacc];
    for (int a = threadIdx.x; a < nacc; a += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int iw = 0; iw < nwarps; ++iw) {
            v += red[iw][a];
        }
        mixes[a] = v;
    }
    __shared__ float scale_base[nsb];
    if (threadIdx.x < nsb) {
        scale_base[threadIdx.x] = sbv;
    }
    __syncthreads();

    // one warp per token: pre/post first so the waiting blocks can start, then the sinkhorn. A single token
    // runs its sinkhorn on the second warp instead, alongside pre/post; each warp scales the mixes it reads.
    static_assert(n_tokens <= nwarps, "one warp per token");
    constexpr bool comb_apart = n_tokens == 1;
    static_assert(!comb_apart || nwarps >= 2, "the sinkhorn needs its own warp");
    const int t = comb_apart ? 0 : warp;
    const bool pre_warp  = warp < n_tokens;
    const bool comb_warp = comb_apart ? warp == 1 : pre_warp;
    float * m = mixes + t*nmix;
    if (pre_warp || comb_warp) {
        const float rs = rsqrtf(m[DSV4_HC_MIX]/k + rms_eps);
        const int lo = pre_warp  ? 0           : 2*DSV4_HC;
        const int hi = comb_warp ? DSV4_HC_MIX : 2*DSV4_HC;
        if (lane >= lo && lane < hi) {
            m[lane] *= rs;
        }
        __syncwarp();
    }
    if (pre_warp) {
        const float p = dsv4_hc_params_pre_post(m, scale_base, scale_base + nscale, dst + t*sd1, 1, 1, 1, 1, eps);
        if constexpr (norm) {
            // sum(y^2) = scale^2 pre^T G pre over the stored upper triangle of G
            float pre[DSV4_HC];
#pragma unroll
            for (int ih = 0; ih < DSV4_HC; ++ih) {
                pre[ih] = __shfl_sync(0xffffffff, p, ih, WARP_SIZE);
            }
            float yy = 0.0f;
            int g = DSV4_HC_MIX + 1;
#pragma unroll
            for (int ia = 0; ia < DSV4_HC; ++ia) {
#pragma unroll
                for (int ib = ia; ib < DSV4_HC; ++ib) {
                    yy += (ia == ib ? 1.0f : 2.0f)*pre[ia]*pre[ib]*m[g++];
                }
            }
            if (lane == 0) {
                // the quadratic form can round below zero when the streams cancel
                np.rs[t] = rsqrtf(ap.scale*ap.scale*fmaxf(yy, 0.0f)/ap.n_embd + np.eps);
            }
        }
    }
    if constexpr (apply) {
        // with comb_apart only warp 0 wrote what is published, so the sinkhorn warp starts without waiting on it
        if constexpr (comb_apart) {
            __syncwarp();
        } else {
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            __threadfence();
            atomicExch(launch_epoch, epoch + 1);
        }
    }
    if (comb_warp) {
        dsv4_hc_params_comb(m, scale_base, scale_base + nscale, dst + t*sd1, 1, 1, 1, 1, eps, n_iter);
    }
}

bool ggml_cuda_dsv4_hc_pre_fused_supported(const ggml_tensor * rms, const ggml_tensor * mm, const ggml_tensor * params) {
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * w = mm->src[0];
    return x->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 && mm->src[1] == rms && params->src[0] == mm &&
        x->ne[1] >= 1 && x->ne[1] <= DSV4_HC_PRE_FUSED_MAX_TOKENS && x->ne[2] == 1 && x->ne[3] == 1 &&
        x->nb[0] == sizeof(float) && w->nb[0] == sizeof(float) &&
        w->ne[0] == x->ne[0] && w->ne[1] == DSV4_HC_MIX && w->ne[2] == 1 && w->ne[3] == 1 &&
        mm->ne[0] == DSV4_HC_MIX && mm->ne[1] == x->ne[1] && mm->ne[2] == 1 && mm->ne[3] == 1 &&
        params->src[1]->type == GGML_TYPE_F32 && params->src[2]->type == GGML_TYPE_F32 &&
        params->type == GGML_TYPE_F32 && params->nb[0] == sizeof(float) &&
        x->ne[0] % DSV4_HC == 0;
}

bool ggml_cuda_dsv4_hc_post_fold_supported(const ggml_tensor * hc_post, const ggml_tensor * rms) {
    const ggml_tensor * x = rms->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * comb = hc_post->src[3];
    return hc_post->op == GGML_OP_DSV4_HC_POST && comb != nullptr && ggml_get_op_params_i32(hc_post, 0) == 0 &&
        hc_post->type == GGML_TYPE_F32 && hc_post->src[0]->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 &&
        hc_post->src[2]->type == GGML_TYPE_F32 && comb->type == GGML_TYPE_F32 &&
        residual->ne[1] == DSV4_HC && hc_post->ne[1] == DSV4_HC && ggml_is_contiguous(hc_post) &&
        ggml_cuda_is_whole_reshape(x, hc_post) && x->ne[1] == hc_post->ne[2];
}

bool ggml_cuda_dsv4_hc_pre_norm_supported(const ggml_tensor * x, const ggml_tensor * hc_pre, const ggml_tensor * rms,
        const ggml_tensor * mul) {
    if (rms->op != GGML_OP_RMS_NORM || rms->src[0] != hc_pre || rms->type != GGML_TYPE_F32 ||
            mul->op != GGML_OP_MUL || (mul->src[0] != rms && mul->src[1] != rms)) {
        return false;
    }
    // the norm comes from the Gram matrix of the streams the hc_pre dot reads, so they must be what it mixes
    if (!ggml_is_contiguous(hc_pre->src[0]) || !ggml_cuda_is_whole_reshape(x, hc_pre->src[0])) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == rms ? mul->src[1] : mul->src[0];
    return w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && w->ne[0] == hc_pre->ne[0] && ggml_nrows(w) == 1 &&
        mul->type == GGML_TYPE_F32 && ggml_are_same_shape(mul, hc_pre) && mul->nb[0] == sizeof(float);
}

bool ggml_cuda_dsv4_hc_pre_apply_supported(const ggml_tensor * params, const ggml_tensor * hc_pre) {
    const ggml_tensor * x   = hc_pre->src[0];
    const ggml_tensor * pre = hc_pre->src[1];
    return hc_pre->op == GGML_OP_DSV4_HC_PRE && ggml_get_op_params_i32(hc_pre, 1) == 0 &&
        hc_pre->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 &&
        pre->view_src == params && pre->view_offs == 0 && pre->nb[0] == params->nb[0] && pre->nb[1] == params->nb[1] &&
        pre->ne[0] == DSV4_HC && pre->ne[1] == params->ne[1] &&
        x->ne[1] == DSV4_HC && x->ne[2] == params->ne[1] && x->ne[3] == 1 &&
        hc_pre->ne[0] == x->ne[0] && hc_pre->ne[1] == x->ne[2];
}

using dsv4_hc_pre_fused_kernel_t = decltype(&dsv4_hc_pre_fused_f32<1, false, false, false>);

// indexed [apply][fold][norm]; norm needs apply
template <int NT>
static dsv4_hc_pre_fused_kernel_t dsv4_hc_pre_fused_kernel(bool apply, bool fold, bool norm) {
    static const dsv4_hc_pre_fused_kernel_t kernels[2][2][2] = {
        { { dsv4_hc_pre_fused_f32<NT, false, false, false>, nullptr },
          { dsv4_hc_pre_fused_f32<NT, false, true,  false>, nullptr } },
        { { dsv4_hc_pre_fused_f32<NT, true,  false, false>, dsv4_hc_pre_fused_f32<NT, true, false, true> },
          { dsv4_hc_pre_fused_f32<NT, true,  true,  false>, dsv4_hc_pre_fused_f32<NT, true, true,  true> } },
    };
    return kernels[apply][fold][norm];
}

static dsv4_hc_pre_fused_kernel_t dsv4_hc_pre_fused_kernel(int n_tokens, bool apply, bool fold, bool norm) {
    switch (n_tokens) {
        case 1: return dsv4_hc_pre_fused_kernel<1>(apply, fold, norm);
        case 2: return dsv4_hc_pre_fused_kernel<2>(apply, fold, norm);
        case 3: return dsv4_hc_pre_fused_kernel<3>(apply, fold, norm);
        case 4: return dsv4_hc_pre_fused_kernel<4>(apply, fold, norm);
        default: GGML_ABORT("unsupported token count");
    }
}

// co-resident blocks of an apply kernel, whose mixing blocks spin until the last block publishes
static int dsv4_hc_pre_fused_max_resident(int device, dsv4_hc_pre_fused_kernel_t kernel, int n_tokens, bool fold, bool norm) {
    static int occ[GGML_CUDA_MAX_DEVICES][DSV4_HC_PRE_FUSED_MAX_TOKENS][2][2] = {};
    int & o = occ[device][n_tokens - 1][fold][norm];
    if (o == 0) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, (const void *) kernel, DSV4_HC_PRE_FUSED_THREADS, 0));
    }
    return o * ggml_cuda_info().devices[device].nsm;
}

void ggml_cuda_op_dsv4_hc_pre_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * rms, const ggml_tensor * mm, ggml_tensor * params, ggml_tensor * hc_pre,
        ggml_tensor * hc_post, ggml_tensor * norm) {
    GGML_ASSERT(hc_pre || !norm);
    const ggml_tensor * x     = rms->src[0];
    const ggml_tensor * w     = mm->src[0];
    const ggml_tensor * scale = params->src[1];
    const ggml_tensor * base  = params->src[2];

    const int64_t k  = x->ne[0];
    const int     nt = (int) x->ne[1];
    const dsv4_hc_pre_fused_kernel_t kernel = dsv4_hc_pre_fused_kernel(nt, hc_pre, hc_post, norm);
    // a stream column per thread, at least two blocks: with apply, the last block computes params while the others mix
    const int nthreads = DSV4_HC_PRE_FUSED_THREADS;
    int nblocks = (int) std::max<int64_t>(2, std::min<int64_t>(nthreads, k/DSV4_HC/nthreads));
    if (hc_pre) {
        nblocks = std::min(nblocks, dsv4_hc_pre_fused_max_resident(ctx.device, kernel, nt, hc_post, norm));
    }

    // the norm's rms scales follow the block partials
    const size_t nacc = size_t(nt)*(DSV4_HC_MIX + 1 + (norm ? DSV4_HC_GRAM : 0));
    ggml_cuda_pool_alloc<float> partials(ctx.pool(), nblocks*nacc + (norm ? nt : 0));

    const float rms_eps = ggml_get_op_params_f32(rms, 0);
    const float eps     = ggml_get_op_params_f32(params, 0);
    const int32_t n_iter = ggml_get_op_params_i32(params, 1);

    dsv4_hc_pre_apply_args ap = {};
    if (hc_pre) {
        const ggml_tensor * hx = hc_pre->src[0];
        ap.x      = (const float *) hx->data;
        ap.dst    = (float *) hc_pre->data;
        ap.n_embd = hx->ne[0];
        ap.sx0    = hx->nb[0]/sizeof(float);
        ap.sx1    = hx->nb[1]/sizeof(float);
        ap.sx2    = hx->nb[2]/sizeof(float);
        ap.sd0    = hc_pre->nb[0]/sizeof(float);
        ap.sd1    = hc_pre->nb[1]/sizeof(float);
        ap.scale  = ggml_get_op_params_f32(hc_pre, 0);
    }

    dsv4_hc_post_fold_args hp = {};
    if (hc_post) {
        const ggml_tensor * px = hc_post->src[0];
        const ggml_tensor * pr = hc_post->src[1];
        const ggml_tensor * pp = hc_post->src[2];
        const ggml_tensor * pc = hc_post->src[3];
        hp.x        = (const float *) px->data;
        hp.residual = (const float *) pr->data;
        hp.post     = (const float *) pp->data;
        hp.comb     = (const float *) pc->data;
        hp.dst      = (float *) hc_post->data;
        hp.sx0 = px->nb[0]/sizeof(float); hp.sx1 = px->nb[1]/sizeof(float);
        hp.sr0 = pr->nb[0]/sizeof(float); hp.sr1 = pr->nb[1]/sizeof(float); hp.sr2 = pr->nb[2]/sizeof(float);
        hp.sp0 = pp->nb[0]/sizeof(float); hp.sp1 = pp->nb[1]/sizeof(float);
        hp.sc0 = pc->nb[0]/sizeof(float); hp.sc1 = pc->nb[1]/sizeof(float); hp.sc2 = pc->nb[2]/sizeof(float);
    }

    dsv4_hc_pre_norm_args np = {};
    if (norm) {
        const bool rms_first = norm->src[0]->op == GGML_OP_RMS_NORM && norm->src[0]->src[0] == hc_pre;
        np.w        = (const float *) norm->src[rms_first ? 1 : 0]->data;
        np.dst      = (float *) norm->data;
        np.rs       = partials.get() + nblocks*nacc;
        np.sd1      = norm->nb[1]/sizeof(float);
        np.eps      = ggml_get_op_params_f32(norm->src[rms_first ? 0 : 1], 0);
    }

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            dim3(nblocks, 1, 1), dim3(nthreads, 1, 1), 0, ctx.stream());

    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) w->data, (const float *) scale->data,
            (const float *) base->data, partials.get(), (float *) params->data,
            ggml_cuda_ticket_counter(ctx, GGML_CUDA_TICKET_DSV4_HC_PRE),
            ggml_cuda_ticket_counter(ctx, GGML_CUDA_TICKET_DSV4_HC_PRE_EPOCH), k,
            int64_t(x->nb[1]/sizeof(float)), int64_t(w->nb[1]/sizeof(float)),
            int64_t(scale->nb[0]/sizeof(float)), int64_t(base->nb[0]/sizeof(float)),
            int64_t(params->nb[1]/sizeof(float)), rms_eps, eps, n_iter, ap, hp, np);
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    float comb[DSV4_HC*DSV4_HC];
    dsv4_hc_comb_sinkhorn(comb, mixes + it*sm1, scale[2*ss0], base, sm0, sb0, eps, n_iter);

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
            const float product = __fmul_rn(xv, wv);
            sum = ih == 0 ? product : __fadd_rn(sum, product);
        } else {
            wv = weights[ih*sw0 + it*sw1];
            if (ih == 0) {
                sum = xv * wv;
            } else {
                sum += xv * wv;
            }
        }
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

template <bool has_comb>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t tiles_per_token,
        int64_t hc,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        bool    from_inject,
        float   inject_scale) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x / tiles_per_token;
    const int64_t tile = (int64_t) blockIdx.x - it*tiles_per_token;
    const int64_t i0 = tile*DSV4_HC_POST_TILE_EMBD + threadIdx.x;
    const int64_t idst = blockIdx.y * blockDim.y + threadIdx.y;

    ggml_cuda_pdl_sync();

    if (i0 >= n_embd || idst >= hc) {
        return;
    }

    float pv = post[idst*sp0 + it*sp1];
    if (from_inject) {
        // same SCALE -> SIGMOID -> SCALE boundaries as the unfused graph
        const float scaled = __fmaf_rn(inject_scale, pv, 0.0f);
        pv = __fmaf_rn(2.0f, 1.0f / (1.0f + expf(-scaled)), 0.0f);
    }
    float sum = __fmul_rn(x[i0*sx0 + it*sx1], pv);
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum = __fadd_rn(sum, residual[i0*sr0 + idst*sr1 + it*sr2]);
    }
    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_params(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mixes->ne[0] == 24);
    GGML_ASSERT(dst->ne[0] == 24);
    GGML_ASSERT(dst->ne[1] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == 24);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int tokens_per_block = 8;
    const dim3 block_dims(WARP_SIZE, tokens_per_block, 1);
    const dim3 grid_dims((n_tokens + tokens_per_block - 1) / tokens_per_block, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
            grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_params_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data,
            (float *) dst->data, n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float), nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];
    GGML_ASSERT(hc > 0);

    const int64_t tiles_per_token = (n_embd + DSV4_HC_POST_TILE_EMBD - 1) / DSV4_HC_POST_TILE_EMBD;
    const dim3 block_dims(DSV4_HC_POST_TILE_EMBD, DSV4_HC, 1);
    const dim3 grid_dims(tiles_per_token*n_tokens, (hc + DSV4_HC - 1)/DSV4_HC, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? dsv4_hc_post_f32<true> : dsv4_hc_post_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, tiles_per_token, hc,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            ggml_get_op_params_i32(dst, 0) != 0, ggml_get_op_params_f32(dst, 1));
}

void ggml_cuda_op_hc_combine_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * residual, const ggml_tensor * block,
        const ggml_tensor * repeated, const ggml_tensor * inject,
        ggml_tensor * dst, bool use_repeated_block) {
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(block->type    == GGML_TYPE_F32);
    GGML_ASSERT(repeated->type == GGML_TYPE_F32);
    GGML_ASSERT(inject->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type      == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual));
    GGML_ASSERT(ggml_is_contiguous(block));
    GGML_ASSERT(ggml_is_contiguous(repeated));
    GGML_ASSERT(ggml_is_contiguous(inject));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ggml_cuda_set_device(ctx.device);
    const int64_t n_embd = residual->ne[0];
    const int64_t hc = residual->ne[1];
    const float inv_hc = 1.0f / (float) hc;
    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_embd + block_size - 1) / block_size, hc, residual->ne[2]);

    // This kernel has no ggml_cuda_pdl_sync(), so it must use ordinary stream
    // ordering rather than the PDL-capable ggml_cuda_kernel_launch() helper.
    if (use_repeated_block) {
        hc_combine_f32<true><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) residual->data, (const float *) repeated->data,
            (const float *) inject->data, (float *) dst->data,
            n_embd, hc, inv_hc);
    } else {
        hc_combine_f32<false><<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) residual->data, (const float *) block->data,
            (const float *) inject->data, (float *) dst->data,
            n_embd, hc, inv_hc);
    }
    CUDA_CHECK(cudaGetLastError());
}

// hc_mix: the gated pre-mix of one hyper-connection for a few tokens (decode and MTP verify).
// Two launches. The down + inject matvec folds in the per-stream RMSNorm: the warps of a block
// split hc_dim, so the block sees every stream whole and scales its partial dot products by the
// stream's norm. The up matvec applies the sigmoid gate and the stream mean. The bf16 weights
// are read once for all tokens.

static constexpr int DSV4_HC_MIX_DOWN_WARPS = 8; // split hc_dim, DSV4_HC_MIX_DOWN_WARPS/DSV4_HC per stream
static constexpr int DSV4_HC_MIX_DOWN_ROWS  = 8; // rows per block, sharing each activation load
static constexpr int DSV4_HC_MIX_UP_WARPS   = 8;
static constexpr int DSV4_HC_MIX_UP_COLS    = 2; // output columns per warp; hc == 4 rows each

static __device__ __forceinline__ void dsv4_hc_bf16x8(const uint4 v, float * f) {
    const uint32_t u[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        f[2*k + 0] = __uint_as_float(u[k] << 16);
        f[2*k + 1] = __uint_as_float(u[k] & 0xffff0000u);
    }
}

// rs gets the [hc, nt] RMSNorm scales for the up matvec's gate product
template <int nt>
static __global__ void dsv4_hc_mix_down_bf16(
        const float * __restrict__ x, const float * __restrict__ gamma,
        const uint4 * __restrict__ w_down, const uint4 * __restrict__ w_inject,
        float * __restrict__ lo, float * __restrict__ inject, float * __restrict__ rs,
        const int n_embd, const int r, const int n_rows,
        const int64_t sw_down, const int64_t sw_inject, const float eps, const float scale,
        const char * __restrict__ pf, const int64_t pf_bytes) {
    constexpr int W     = DSV4_HC_MIX_DOWN_WARPS;
    constexpr int R     = DSV4_HC_MIX_DOWN_ROWS;
    constexpr int parts = W/DSV4_HC;
    // the down matvec does not saturate DRAM, so it also pulls the first half of the up weights into L2
    // (the whole of them would not fit)
    ggml_cuda_prefetch_l2(pf, pf_bytes);

    const int warp   = threadIdx.x / WARP_SIZE;
    const int lane   = threadIdx.x % WARP_SIZE;
    const int row0   = blockIdx.x*R;
    const int hc_dim = DSV4_HC*n_embd;
    const int n8w    = hc_dim/8/W;

    const uint4 * w[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
        const int row = min(row0 + k, n_rows - 1); // a short last block repeats its last row
        w[k] = row < r ? w_down + row*sw_down : w_inject + (row - r)*sw_inject;
    }

    const float4 * x4 = (const float4 *) x;
    const float4 * g4 = (const float4 *) gamma;
    float acc[R][nt] = {};
    float ss[nt] = {};
#pragma unroll 2
    for (int j = warp*n8w + lane; j < (warp + 1)*n8w; j += WARP_SIZE) {
        float wf[R][8];
#pragma unroll
        for (int k = 0; k < R; ++k) {
            dsv4_hc_bf16x8(w[k][j], wf[k]);
        }
        const float4 ga = g4[2*j + 0];
        const float4 gb = g4[2*j + 1];
#pragma unroll
        for (int t = 0; t < nt; ++t) {
            const float4 a = x4[t*(hc_dim/4) + 2*j + 0];
            const float4 b = x4[t*(hc_dim/4) + 2*j + 1];
            ss[t] += a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w + b.x*b.x + b.y*b.y + b.z*b.z + b.w*b.w;
            const float xg[8] = { a.x*ga.x, a.y*ga.y, a.z*ga.z, a.w*ga.w, b.x*gb.x, b.y*gb.y, b.z*gb.z, b.w*gb.w };
#pragma unroll
            for (int k = 0; k < R; ++k) {
                acc[k][t] += wf[k][0]*xg[0] + wf[k][1]*xg[1] + wf[k][2]*xg[2] + wf[k][3]*xg[3]
                           + wf[k][4]*xg[4] + wf[k][5]*xg[5] + wf[k][6]*xg[6] + wf[k][7]*xg[7];
            }
        }
    }

    __shared__ float s_acc[W][R][nt];
    __shared__ float s_ss[W][nt];
#pragma unroll
    for (int t = 0; t < nt; ++t) {
#pragma unroll
        for (int k = 0; k < R; ++k) {
            acc[k][t] = warp_reduce_sum(acc[k][t]);
        }
        ss[t] = warp_reduce_sum(ss[t]);
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < nt; ++t) {
#pragma unroll
            for (int k = 0; k < R; ++k) {
                s_acc[warp][k][t] = acc[k][t];
            }
            s_ss[warp][t] = ss[t];
        }
    }
    __syncthreads();

    if (threadIdx.x >= R*nt) {
        return;
    }
    const int k = threadIdx.x / nt;
    const int t = threadIdx.x % nt;
    float sum = 0.0f;
#pragma unroll
    for (int h = 0; h < DSV4_HC; ++h) {
        float sq = 0.0f;
        float d  = 0.0f;
#pragma unroll
        for (int p = h*parts; p < (h + 1)*parts; ++p) {
            sq += s_ss[p][t];
            d  += s_acc[p][k][t];
        }
        const float s = rsqrtf(sq/n_embd + eps);
        sum += s*d;
        if (blockIdx.x == 0 && k == 0) {
            rs[t*DSV4_HC + h] = s;
        }
    }

    const int row = row0 + k;
    if (row < r) {
        const float v = scale*sum;
        lo[t*r + row] = v/(1.0f + expf(-v));
    } else if (row < n_rows) {
        inject[t*DSV4_HC + row - r] = sum;
    }
}

template <int nt>
static __global__ void dsv4_hc_mix_up_bf16(
        const float * __restrict__ x, const float * __restrict__ gamma, const float * __restrict__ rs,
        const float * __restrict__ lo, const uint4 * __restrict__ w_up,
        float * __restrict__ mixed, const int n_embd, const int r, const int64_t sw_up, const float scale) {
    constexpr int COLS = DSV4_HC_MIX_UP_COLS;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int h    = lane / 8; // lanes 8h..8h+7 reduce the gate row of stream h
    const int sub  = lane % 8;
    const int hc_dim = DSV4_HC*n_embd;
    const int i0 = (blockIdx.x*DSV4_HC_MIX_UP_WARPS + warp)*COLS;

    // the gate product's inputs do not depend on the down matvec: load them before waiting on lo
    float xv[COLS][nt];
    float gv[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int i = min(i0 + c, n_embd - 1) + n_embd*h;
        gv[c] = gamma[i];
#pragma unroll
        for (int t = 0; t < nt; ++t) {
            xv[c][t] = x[t*hc_dim + i];
        }
    }

    extern __shared__ float4 lo_s4[]; // [nt][r/4]
    const float4 * lo4 = (const float4 *) lo;
    for (int k = threadIdx.x; k < nt*r/4; k += blockDim.x) {
        lo_s4[k] = lo4[k];
    }
    __syncthreads();

    const uint4 * w[COLS];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        w[c] = w_up + (int64_t) (min(i0 + c, n_embd - 1) + n_embd*h)*sw_up;
    }

    // all columns' loads of a step are issued before their math to keep enough bytes in flight
    float acc[COLS][nt] = {};
#pragma unroll 5
    for (int j = sub; j < r/8; j += 8) {
        float wf[COLS][8];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            dsv4_hc_bf16x8(w[c][j], wf[c]);
        }
#pragma unroll
        for (int t = 0; t < nt; ++t) {
            const float4 a = lo_s4[t*(r/4) + 2*j + 0];
            const float4 b = lo_s4[t*(r/4) + 2*j + 1];
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                acc[c][t] += wf[c][0]*a.x + wf[c][1]*a.y + wf[c][2]*a.z + wf[c][3]*a.w
                           + wf[c][4]*b.x + wf[c][5]*b.y + wf[c][6]*b.z + wf[c][7]*b.w;
            }
        }
    }

#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int i = i0 + c;
        if (i >= n_embd) {
            break;
        }
#pragma unroll
        for (int t = 0; t < nt; ++t) {
            float g = acc[c][t];
            g += __shfl_xor_sync(0xffffffff, g, 4, 8);
            g += __shfl_xor_sync(0xffffffff, g, 2, 8);
            g += __shfl_xor_sync(0xffffffff, g, 1, 8);
            // same product and stream order as dsv4_hc_pre_f32<true>
            const float xn = rs[t*DSV4_HC + h]*xv[c][t]*gv[c];
            const float p  = __fmul_rn(xn, 1.0f/(1.0f + expf(-g)));
            const float p1 = __shfl_sync(0xffffffff, p,  8, WARP_SIZE);
            const float p2 = __shfl_sync(0xffffffff, p, 16, WARP_SIZE);
            const float p3 = __shfl_sync(0xffffffff, p, 24, WARP_SIZE);
            if (lane == 0) {
                mixed[t*n_embd + i] = scale*__fadd_rn(__fadd_rn(__fadd_rn(p, p1), p2), p3);
            }
        }
    }
}

template <int nt>
static void dsv4_hc_mix_launch(
        cudaStream_t stream, const float * x, const float * gamma, const uint4 * w_down, const uint4 * w_inject,
        const uint4 * w_up, float * lo, float * rs, float * mixed, float * inject, int n_embd, int r, int n_inject,
        int64_t sw_down, int64_t sw_inject, int64_t sw_up, float eps, float scale) {
    const int n_rows = r + n_inject;
    const int64_t up_bytes = (int64_t) DSV4_HC*n_embd*sw_up*16;
    dsv4_hc_mix_down_bf16<nt><<<(n_rows + DSV4_HC_MIX_DOWN_ROWS - 1)/DSV4_HC_MIX_DOWN_ROWS, DSV4_HC_MIX_DOWN_WARPS*WARP_SIZE, 0, stream>>>(
            x, gamma, w_down, w_inject, lo, inject, rs, n_embd, r, n_rows, sw_down, sw_inject, eps, scale,
            (const char *) w_up, up_bytes/2);

    const int cols_per_block = DSV4_HC_MIX_UP_WARPS*DSV4_HC_MIX_UP_COLS;
    dsv4_hc_mix_up_bf16<nt><<<(n_embd + cols_per_block - 1)/cols_per_block, DSV4_HC_MIX_UP_WARPS*WARP_SIZE, nt*r*sizeof(float), stream>>>(
            x, gamma, rs, lo, w_up, mixed, n_embd, r, sw_up, scale);
}

bool ggml_cuda_dsv4_hc_mix_supported(const ggml_tensor * op) {
    const ggml_tensor * x = op->src[0];
    const int64_t n_embd = x->ne[0];
    const int64_t r      = op->src[2]->ne[1];
    if (x->ne[1] != DSV4_HC || x->ne[2] > DSV4_HC_MIX_MAX_TOKENS || n_embd % 16 != 0 || r % 8 != 0) {
        return false;
    }
    for (int i = 2; i <= 4; ++i) {
        const ggml_tensor * w = op->src[i];
        if (w && (w->type != GGML_TYPE_BF16 || w->nb[1] % 16 != 0)) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * w_norm   = dst->src[1];
    const ggml_tensor * w_down   = dst->src[2];
    const ggml_tensor * w_up     = dst->src[3];
    const ggml_tensor * w_inject = dst->src[4];

    const int n_embd = x->ne[0];
    const int hc     = x->ne[1];
    const int nt     = x->ne[2];
    const int r      = w_down->ne[1];

    const float eps   = ggml_get_op_params_f32(dst, 0);
    const float scale = ggml_get_op_params_f32(dst, 1);

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> lo(ctx.pool(), (size_t) r*nt);
    ggml_cuda_pool_alloc<float> rs(ctx.pool(), (size_t) hc*nt);

    float * mixed  = (float *) dst->data;
    float * inject = mixed + (int64_t) n_embd*nt;
    const uint4 * wi = w_inject ? (const uint4 *) w_inject->data : nullptr;
    const int64_t swi = w_inject ? w_inject->nb[1]/16 : 0;

#define DSV4_HC_MIX_CASE(N) case N: dsv4_hc_mix_launch<N>(stream, (const float *) x->data, (const float *) w_norm->data, \
        (const uint4 *) w_down->data, wi, (const uint4 *) w_up->data, lo.get(), rs.get(), mixed, inject, n_embd, r, \
        w_inject ? hc : 0, w_down->nb[1]/16, swi, w_up->nb[1]/16, eps, scale); break;
    switch (nt) {
        DSV4_HC_MIX_CASE(1)
        DSV4_HC_MIX_CASE(2)
        DSV4_HC_MIX_CASE(3)
        DSV4_HC_MIX_CASE(4)
        DSV4_HC_MIX_CASE(5)
        DSV4_HC_MIX_CASE(6)
        DSV4_HC_MIX_CASE(7)
        DSV4_HC_MIX_CASE(8)
        default: GGML_ABORT("hc_mix: unsupported token count %d", nt);
    }
#undef DSV4_HC_MIX_CASE
    CUDA_CHECK(cudaGetLastError());
}
