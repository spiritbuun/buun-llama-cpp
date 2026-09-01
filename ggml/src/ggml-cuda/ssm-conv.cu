#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

// Long-prefill recurrent convolution directly over the graph's natural split
// inputs: a short channel-major saved prefix and a token-major projection view.
// This avoids transposing the entire projection merely to read it back during
// convolution. Each CTA keeps the overlapping window in registers, so eight
// outputs require only d_conv - 1 + 8 input loads per channel.
template <typename body_t>
static __device__ __forceinline__ float ssm_conv_split_load(const body_t * src, int64_t i) {
    return float(src[i]);
}

template <>
__device__ __forceinline__ float ssm_conv_split_load<nv_bfloat16>(const nv_bfloat16 * src, int64_t i) {
    return __bfloat162float(src[i]);
}

template <typename out_t>
static __device__ __forceinline__ void ssm_conv_split_store(out_t * dst, int64_t i, float value) {
    dst[i] = value;
}

template <>
__device__ __forceinline__ void ssm_conv_split_store<nv_bfloat16>(nv_bfloat16 * dst, int64_t i, float value) {
    dst[i] = __float2bfloat16_rn(value);
}

template <typename body_t, typename out_t, int d_conv, int split_n_t>
static __global__ void ssm_conv_split_input(
        const float * __restrict__ prefix,
        const body_t * __restrict__ body,
        const float * __restrict__ weight,
        out_t * __restrict__ dst,
        int64_t channels,
        int64_t n_t,
        int64_t prefix_seq_stride,
        int64_t body_seq_stride,
        int64_t dst_seq_stride) {
    const int64_t channel = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
    const int64_t token0  = int64_t(blockIdx.z) * split_n_t;
    const int64_t seq     = blockIdx.x;
    if (channel >= channels || token0 >= n_t) {
        return;
    }

    const float * prefix_seq = prefix + seq * prefix_seq_stride;
    const body_t * body_seq  = body   + seq * body_seq_stride;
    out_t *       dst_seq    = dst    + seq * dst_seq_stride;

    float w[d_conv];
#pragma unroll
    for (int j = 0; j < d_conv; ++j) {
        w[j] = weight[channel * d_conv + j];
    }

    float x[d_conv - 1 + split_n_t];
#pragma unroll
    for (int j = 0; j < d_conv - 1 + split_n_t; ++j) {
        const int64_t col = token0 + j;
        x[j] = col < d_conv - 1
            ? prefix_seq[channel * (d_conv - 1) + col]
            : ssm_conv_split_load(body_seq, (col - (d_conv - 1)) * channels + channel);
    }

#pragma unroll
    for (int t = 0; t < split_n_t; ++t) {
        if (token0 + t >= n_t) {
            break;
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < d_conv; ++j) {
            sum += x[t + j] * w[j];
        }
        ssm_conv_split_store(dst_seq, (token0 + t) * channels + channel,
                             ggml_cuda_op_silu_single(sum));
    }
}

// Preserve only the suffix observed by recurrent-state VIEW/CPY nodes. The
// full concat has no other consumer when this optimization is selected.
template <typename body_t>
static __global__ void concat_split_input_tail(
        const float * __restrict__ prefix,
        const body_t * __restrict__ body,
        float * __restrict__ dst,
        int64_t prefix_cols,
        int64_t body_cols,
        int64_t channels,
        int64_t tail_start,
        int64_t prefix_seq_stride,
        int64_t body_seq_stride,
        int64_t dst_seq_stride) {
    const int64_t tail_cols = prefix_cols + body_cols - tail_start;
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n = tail_cols * channels;
    if (i >= n) {
        return;
    }
    const int64_t seq = blockIdx.y;
    const int64_t channel = i / tail_cols;
    const int64_t col = tail_start + i % tail_cols;
    const float * prefix_seq = prefix + seq * prefix_seq_stride;
    const body_t * body_seq  = body   + seq * body_seq_stride;
    float *       dst_seq    = dst    + seq * dst_seq_stride;
    dst_seq[channel * (prefix_cols + body_cols) + col] = col < prefix_cols
        ? prefix_seq[channel * prefix_cols + col]
        : ssm_conv_split_load(body_seq, (col - prefix_cols) * channels + channel);
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    constexpr int short_threads = 128;
    GGML_ASSERT(nr % short_threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + short_threads - 1) / short_threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, short_threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, short_threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            // Prefill is latency-bound by the long serial token loop at 32
            // tokens per CTA. Match the successful channel-last scheduling
            // shape used by contemporary GDN implementations: 256 channels x
            // 8 tokens. The arithmetic for every output element is unchanged.
            constexpr int long_threads = 256;
            constexpr int64_t split_n_t = 8;
            GGML_ASSERT(nr % long_threads == 0);
            dim3          blocks(n_s, (nr + long_threads - 1) / long_threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = long_threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, long_threads, kNC, split_n_t><<<blocks, long_threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

// ============================================================================
// Tree-mode SSM Conv: follows parent pointers for convolution window
// ============================================================================

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_tree_f32(const float * __restrict__ src0,
                                         const float * __restrict__ src1,
                                         const int32_t * __restrict__ parent_ids,
                                         const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                         const int src1_nb1,
                                         float * __restrict__ dst, const int dst_nb0, const int dst_nb1,
                                         const int dst_nb2,
                                         const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    // Load weights
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    for (int64_t i = 0; i < n_t; i++) {
        // Walk parent chain to find conv window ancestors
        // ancestors[d_conv-1] = current token i
        // ancestors[k] = parent of ancestors[k+1], or negative for old state region
        int ancestors[d_conv];
        ancestors[d_conv - 1] = (int)i;
        for (int k = (int)d_conv - 2; k >= 0; k--) {
            int prev = ancestors[k + 1];
            if (prev >= 0) {
                ancestors[k] = parent_ids[prev]; // -1 means initial state
            } else {
                ancestors[k] = prev - 1; // keep going into old state region
            }
        }

        // Compute convolution using ancestor slots
        // Slot mapping: token index p (>=0) maps to column (d_conv-1+p) in conv_input
        //               negative values -1,-2,... map to columns (d_conv-2),(d_conv-3),... (old state)
        float sumf = 0.0f;
#pragma unroll
        for (size_t k = 0; k < d_conv; k++) {
            int slot = (int)(d_conv - 1) + ancestors[k];
            sumf += x_block[tid * stride_x + slot] * w[k];
        }
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_tree_f32_cuda(const float * src0, const float * src1, const int32_t * parent_ids,
                                   const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                   const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                                   const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                                   const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);
    const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        ssm_conv_tree_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(
            src0, src1, parent_ids, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
            dst, dst_nb0, dst_nb1, dst_nb2, n_t);
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 9 right now.");
    }
}

void ggml_cuda_op_ssm_conv_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_input
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const struct ggml_tensor * src2 = dst->src[2];  // parent_ids

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = dst->ne[1];                 // tokens per sequence
    const int64_t n_s = dst->ne[2];                 // number of sequences

    GGML_ASSERT(dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    GGML_ASSERT(src2->type == GGML_TYPE_I32);

    const float *   src0_d = (const float *)   src0->data;
    const float *   src1_d = (const float *)   src1->data;
    const int32_t * pids_d = (const int32_t *) src2->data;
    float *         dst_d  = (float *) dst->data;
    cudaStream_t    stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Tree conv always fuses silu (same as normal decode path)
    ssm_conv_tree_f32_cuda<true>(src0_d, src1_d, pids_d,
        src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
        dst_d, dst->nb[0], dst->nb[1], dst->nb[2], nc, nr, n_t, n_s, stream);
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

void ggml_cuda_op_ssm_conv_split_input(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * prefix,
        const ggml_tensor * body_transposed,
        ggml_tensor * concat_dst,
        const ggml_tensor * weight,
        ggml_tensor * silu_dst,
        int64_t tail_start,
        bool body_bf16,
        bool output_bf16) {
    GGML_ASSERT(prefix->type == GGML_TYPE_F32 && body_transposed->type == GGML_TYPE_F32);
    GGML_ASSERT(concat_dst->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32);
    GGML_ASSERT(silu_dst->type == GGML_TYPE_F32);

    const int64_t d_conv  = weight->ne[0];
    const int64_t channels = prefix->ne[1];
    const int64_t n_t     = body_transposed->ne[0];
    const int64_t n_s     = prefix->ne[2];
    GGML_ASSERT(d_conv == 4);
    GGML_ASSERT(prefix->ne[0] == d_conv - 1);
    GGML_ASSERT(body_transposed->ne[1] == channels && body_transposed->ne[2] == n_s);
    GGML_ASSERT(concat_dst->ne[0] == d_conv - 1 + n_t && concat_dst->ne[1] == channels && concat_dst->ne[2] == n_s);
    GGML_ASSERT(weight->ne[1] == channels);
    GGML_ASSERT(silu_dst->ne[0] == channels && silu_dst->ne[1] == n_t && silu_dst->ne[2] == n_s);

    cudaStream_t stream = ctx.stream();
    constexpr int threads = 256;
    static const int requested_split = [] {
        const char * value = std::getenv("GGML_CUDA_SSM_SPLIT_T");
        return value ? std::atoi(value) : 8;
    }();
    auto launch_conv = [&](auto split, auto body_type, auto output_type) {
        constexpr int split_n_t = decltype(split)::value;
        using body_t = decltype(body_type);
        using out_t  = decltype(output_type);
        const dim3 conv_blocks(n_s, (channels + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
        ssm_conv_split_input<body_t, out_t, 4, split_n_t><<<conv_blocks, threads, 0, stream>>>(
            static_cast<const float *>(prefix->data),
            static_cast<const body_t *>(body_transposed->data),
            static_cast<const float *>(weight->data),
            static_cast<out_t *>(silu_dst->data),
            channels, n_t,
            prefix->nb[2] / sizeof(float),
            body_transposed->nb[2] / sizeof(float),
            n_t * channels);
    };
    auto launch_for_type = [&](auto body_type, auto output_type) {
        if (requested_split == 32) {
            launch_conv(std::integral_constant<int, 32>{}, body_type, output_type);
        } else if (requested_split == 16) {
            launch_conv(std::integral_constant<int, 16>{}, body_type, output_type);
        } else {
            launch_conv(std::integral_constant<int, 8>{}, body_type, output_type);
        }
    };
    if (body_bf16) {
        if (output_bf16) {
            launch_for_type(nv_bfloat16{}, nv_bfloat16{});
        } else {
            launch_for_type(nv_bfloat16{}, float{});
        }
    } else {
        if (output_bf16) {
            launch_for_type(float{}, nv_bfloat16{});
        } else {
            launch_for_type(float{}, float{});
        }
    }

    if (tail_start < concat_dst->ne[0]) {
        const int64_t tail_cols = concat_dst->ne[0] - tail_start;
        const int64_t elems_per_seq = tail_cols * channels;
        const dim3 tail_blocks((elems_per_seq + threads - 1) / threads, n_s, 1);
        auto launch_tail = [&](auto body_type) {
            using body_t = decltype(body_type);
            concat_split_input_tail<body_t><<<tail_blocks, threads, 0, stream>>>(
                static_cast<const float *>(prefix->data),
                static_cast<const body_t *>(body_transposed->data),
                static_cast<float *>(concat_dst->data),
                prefix->ne[0], body_transposed->ne[0], channels, tail_start,
                prefix->nb[2] / sizeof(float),
                body_transposed->nb[2] / sizeof(float),
                concat_dst->nb[2] / sizeof(float));
        };
        if (body_bf16) {
            launch_tail(nv_bfloat16{});
        } else {
            launch_tail(float{});
        }
    }
}
