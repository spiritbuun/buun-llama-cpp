#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
void ggml_cuda_op_ssm_conv_split_input(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * prefix,
        const ggml_tensor * body_transposed,
        ggml_tensor * concat_dst,
        const ggml_tensor * weight,
        ggml_tensor * silu_dst,
        int64_t tail_start,
        bool body_bf16 = false);
void ggml_cuda_op_ssm_conv_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
