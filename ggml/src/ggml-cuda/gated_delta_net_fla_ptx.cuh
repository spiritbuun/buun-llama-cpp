#pragma once

#include "common.cuh"

bool ggml_cuda_gdn_fla_ptx_supported(
        int cc, bool kda, bool keep_rs, int64_t S_v, int64_t H, int64_t H_k,
        int64_t n_tokens, int64_t n_seqs);

void ggml_cuda_gdn_fla_ptx(
        ggml_backend_cuda_context & ctx, int cc,
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state_in,
        float * dst, float * state_out,
        int64_t n_tokens,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        const void * compact_conv_bf16 = nullptr,
        float l2_eps = -1.0f,
        const float * rms_weight = nullptr, const float * rms_gate = nullptr,
        bool rms_gate_bf16 = false,
        float * rms_output = nullptr, bool rms_output_bf16 = false,
        bool rms_output_int8 = false, float * rms_output_scale = nullptr,
        float rms_eps = 0.0f);
