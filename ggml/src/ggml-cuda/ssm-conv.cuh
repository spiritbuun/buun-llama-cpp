#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
// Writes CONCAT(prefix, body) and the new saved prefix; at decode width it can
// also evaluate the following SSM_CONV + SiLU (conv_weight/silu non-null).
void ggml_cuda_op_conv_state_concat(ggml_backend_cuda_context & ctx, const ggml_tensor * prefix, const ggml_tensor * body,
        ggml_tensor * dst, ggml_tensor * state, const ggml_tensor * conv_weight = nullptr, ggml_tensor * silu = nullptr);
void ggml_cuda_op_ssm_conv_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
