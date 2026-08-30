#include "hc-stream-mean.cuh"

static __global__ void hc_stream_mean_f32(
        const float * stream0, const float * stream1,
        const float * stream2, const float * stream3,
        float * dst, int64_t n_embd,
        int64_t stride1, int64_t stride2, int64_t stride3) {
    const int64_t e = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_embd) {
        return;
    }

    const int64_t t = blockIdx.y;
    const int64_t i = e + n_embd*t;
    // Preserve the three ADD stores followed by SCALE, including every RN boundary.
    const float sum01  = __fadd_rn(stream0[i], stream1[e + t*stride1]);
    const float sum012 = __fadd_rn(sum01,      stream2[e + t*stride2]);
    const float sum    = __fadd_rn(sum012,     stream3[e + t*stride3]);
    // GGML_OP_SCALE uses fma(scale, value, bias). Preserve its +0.0f bias,
    // including the signed-zero result, rather than reducing this to a mul.
    dst[i] = __fmaf_rn(0.25f, sum, 0.0f);
}

void ggml_cuda_op_hc_stream_mean_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * const streams[4], ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));
    GGML_ASSERT(streams[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(streams[0]));
    for (int i = 1; i < 4; ++i) {
        GGML_ASSERT(streams[i]->type == GGML_TYPE_F32 && streams[i]->nb[0] == sizeof(float));
    }

    ggml_cuda_set_device(ctx.device);
    cudaStream_t stream = ctx.stream();
    constexpr int block_size = 256;
    const dim3 grid((dst->ne[0] + block_size - 1)/block_size, dst->ne[1], 1);
    hc_stream_mean_f32<<<grid, block_size, 0, stream>>>(
        (const float *) streams[0]->data,
        (const float *) streams[1]->data,
        (const float *) streams[2]->data,
        (const float *) streams[3]->data,
        (float *) dst->data, dst->ne[0],
        streams[1]->nb[1]/sizeof(float),
        streams[2]->nb[1]/sizeof(float),
        streams[3]->nb[1]/sizeof(float));
}
