#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_params(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_hc_combine_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * residual, const ggml_tensor * block,
        const ggml_tensor * repeated, const ggml_tensor * inject,
        ggml_tensor * dst, bool use_repeated_block);
