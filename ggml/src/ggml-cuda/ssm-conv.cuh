#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
void ggml_cuda_op_conv_state_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * prefix, const ggml_tensor * body, ggml_tensor * dst, ggml_tensor * state);
void ggml_cuda_op_ssm_conv_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
