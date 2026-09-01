#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;
static constexpr int DSV4_HC_POST_TILE_EMBD = 64;

template <bool use_repeated_block>
static __global__ void hc_combine_f32(
        const float * residual, const float * block, const float * inject,
        float * dst, int64_t n_embd, int64_t hc, float inv_hc) {
    const int64_t e = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_embd) {
        return;
    }

    const int64_t c = blockIdx.y;
    const int64_t t = blockIdx.z;
    const int64_t i = e + n_embd*(c + hc*t);
    // Preserve the rounding boundaries of SCALE -> SIGMOID -> SCALE -> MUL -> ADD.
    // SCALE is fma(scale, value, +0). Keep both SCALE store boundaries exact,
    // including their signed-zero behavior.
    const float scaled = __fmaf_rn(inv_hc, inject[c + hc*t], 0.0f);
    const float sigmoid = 1.0f / (1.0f + expf(-scaled));
    const float weight = __fmaf_rn(2.0f, sigmoid, 0.0f);
    const int64_t ib = use_repeated_block ? i : e + n_embd*t;
    const float product = __fmul_rn(block[ib], weight);
    dst[i] = __fadd_rn(residual[i], product);
}


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
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
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_pre  = scale[0*ss0];
    const float scale_post = scale[1*ss0];
    const float scale_comb = scale[2*ss0];

    for (int i = 0; i < DSV4_HC; ++i) {
        // The unfused graph stores between MUL and ADD. Explicit round-to-nearest
        // intrinsics preserve that two-kernel arithmetic instead of contracting an FMA.
        const float pre_mul = __fmul_rn(mixes[i*sm0 + it*sm1], scale_pre);
        const float pre_affine = __fadd_rn(pre_mul, base[i*sb0]);
        const float pre = __fadd_rn(1.0f/(1.0f + expf(-pre_affine)), eps);
        dst[i*sd0 + it*sd1] = pre;

        const int post_idx = DSV4_HC + i;
        const float post_mul = __fmul_rn(mixes[post_idx*sm0 + it*sm1], scale_post);
        const float post_affine = __fadd_rn(post_mul, base[post_idx*sb0]);
        const float post = __fmul_rn(1.0f/(1.0f + expf(-post_affine)), 2.0f);
        dst[post_idx*sd0 + it*sd1] = post;
    }

    float comb[DSV4_HC*DSV4_HC];
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb +
                base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f/sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx]*inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int idx = 0; idx < DSV4_HC*DSV4_HC; ++idx) {
        dst[(comb_offset + idx)*sd0 + it*sd1] = comb[idx];
    }
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
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
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

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

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
        int64_t sd0,
        int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0*sx0 + it*sx2] * weights[it*sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float wv = weights[ih*sw0 + it*sw1];
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = sum;
}

static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t tiles_per_token,
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
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x / tiles_per_token;
    const int64_t tile = (int64_t) blockIdx.x - it*tiles_per_token;
    const int64_t i0 = tile*DSV4_HC_POST_TILE_EMBD + threadIdx.x;
    const int64_t idst = threadIdx.y;

    ggml_cuda_pdl_sync();

    if (i0 >= n_embd) {
        return;
    }

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    for (int64_t isrc = 0; isrc < DSV4_HC; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
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

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
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

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];
    GGML_ASSERT(hc == DSV4_HC);

    const int64_t tiles_per_token = (n_embd + DSV4_HC_POST_TILE_EMBD - 1) / DSV4_HC_POST_TILE_EMBD;
    const dim3 block_dims(DSV4_HC_POST_TILE_EMBD, DSV4_HC, 1);
    const dim3 grid_dims(tiles_per_token*n_tokens, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_f32, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, tiles_per_token,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
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
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    if (use_repeated_block) {
        ggml_cuda_kernel_launch(hc_combine_f32<true>, launch_params,
            (const float *) residual->data, (const float *) repeated->data,
            (const float *) inject->data, (float *) dst->data,
            n_embd, hc, inv_hc);
    } else {
        ggml_cuda_kernel_launch(hc_combine_f32<false>, launch_params,
            (const float *) residual->data, (const float *) block->data,
            (const float *) inject->data, (float *) dst->data,
            n_embd, hc, inv_hc);
    }
}
