#pragma once

#include "turbo-tcq-alpha.cuh"

// fattn-mma-f16.cuh must be included before this header (provides flash_attn_ext_f16,
// launch_fattn, fattn_kernel_t, and the MMA helper functions).

// Fused MMA-native turbo flash attention: reads raw turbo-quantized K/V directly,
// dequants into half2 shmem tiles inside the attention loop. No intermediate fp16 buffers.
// Uses the same flash_attn_ext_f16 kernel with type_K/type_V template params.

template <int DKQ, int DV, int ncols1, int ncols2, ggml_type type_K, ggml_type type_V>
void ggml_cuda_flash_attn_ext_mma_turbo_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV = dst;
    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    constexpr int ncols = ncols1 * ncols2;

    const int  nthreads       = ggml_cuda_fattn_mma_get_nthreads      (DKQ, DV, ncols, cc);
    const int  nbatch_fa      = ggml_cuda_fattn_mma_get_nbatch_fa_typed(DKQ, DV, ncols, cc, type_K, type_V);
    const int  nbatch_K2      = ggml_cuda_fattn_mma_get_nbatch_K2     (DKQ, DV, ncols, cc);
    const int  nbatch_V2      = ggml_cuda_fattn_mma_get_nbatch_V2     (DKQ, DV, ncols, cc);
    const int  nbatch_combine = ggml_cuda_fattn_mma_get_nbatch_combine(DKQ, DV, ncols, cc);
    const bool Q_in_reg       = ggml_cuda_fattn_mma_get_Q_in_reg      (DKQ, DV, ncols, cc);

    // Turbo forces nstages=0: cp.async can't do ALU dequant, so tiles load synchronously.
    // With nstages=0, tile_K and tile_V share the same shmem region (overlap).
    constexpr int nstages = 0;

    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const int warp_size_host = ggml_cuda_info().devices[ctx.device].warp_size;
    const int nwarps         = nthreads / warp_size_host;

    constexpr bool V_is_K_view = false; // Turbo K/V are separate tensors.

    const size_t nbytes_shared_KV_1stage = nbatch_fa            * std::max(nbatch_K2 + 4,  nbatch_V2 + 4) * sizeof(half2);
    const size_t nbytes_shared_Q         = ncols                * (DKQ/2 + 4)                             * sizeof(half2);
    const size_t nbytes_shared_mask      = ncols1               * (nbatch_fa/2 + 4)                       * sizeof(half2);
    const size_t nbytes_shared_combine   = nwarps*cols_per_warp * (nbatch_combine + 4)                    * sizeof(half2);

    const size_t nbytes_shared_KV = nbytes_shared_KV_1stage; // nstages=0 → 1-stage layout

    const size_t nbytes_shared_total = std::max(nbytes_shared_combine, Q_in_reg ?
        std::max(nbytes_shared_Q,  nbytes_shared_KV + nbytes_shared_mask) :
                 nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

    float logit_softcap;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

#if defined(GGML_USE_HIP)
    using fattn_kernel_ptr_t = const void*;
#else
    using fattn_kernel_ptr_t = fattn_kernel_t;
#endif // defined(GGML_USE_HIP)
    fattn_kernel_t fattn_kernel;
    if (logit_softcap == 0.0f) {
        constexpr bool use_logit_softcap = false;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
#if defined(GGML_USE_HIP)
            // RDNA/ROCm: a D=256 fused kernel needs >32KB dynamic LDS. It launches fine eagerly,
            // but HIP graph CAPTURE rejects the launch unless the max-dynamic-shared attribute is
            // raised first. The static guard runs this once, on the first (eager, pre-capture)
            // launch. Non-fatal: clear any error if a given ROCm build rejects the attribute.
            (void) cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total);
            (void) cudaGetLastError();
#else
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    } else {
        constexpr bool use_logit_softcap = true;
        fattn_kernel = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, use_logit_softcap, V_is_K_view, false, type_K, type_V>;

#if !defined(GGML_USE_MUSA)
        static bool shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
        if (!shared_memory_limit_raised[id]) {
#if defined(GGML_USE_HIP)
            // RDNA/ROCm: a D=256 fused kernel needs >32KB dynamic LDS. It launches fine eagerly,
            // but HIP graph CAPTURE rejects the launch unless the max-dynamic-shared attribute is
            // raised first. The static guard runs this once, on the first (eager, pre-capture)
            // launch. Non-fatal: clear any error if a given ROCm build rejects the attribute.
            (void) cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total);
            (void) cudaGetLastError();
#else
            CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<fattn_kernel_ptr_t>(fattn_kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared_total));
#endif
            shared_memory_limit_raised[id] = true;
        }
#endif // !defined(GGML_USE_MUSA)
    }

    // Set TCQ constants in THIS compilation unit's __constant__ memory before kernel launch.
    // Each template instance .cu file has its own static __constant__ copies.
    turbo_vanilla_cb_load_fattn();  // TURBO_CB_T2/3/4/8 override, self-guarded per device
    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ || type_V == GGML_TYPE_TURBO3_TCQ) {
        static bool cb3_loaded = false;
        if (!cb3_loaded) {
            cb3_loaded = true;
            turbo_tcq_load_kv_decode();
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO2_TCQ || type_V == GGML_TYPE_TURBO2_TCQ) {
        static bool cb2_loaded = false;
        if (!cb2_loaded) {
            cb2_loaded = true;
            turbo2_tcq_load_kv_decode();
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO1_TCQ || type_V == GGML_TYPE_TURBO1_TCQ) {
        static bool cb1_loaded = false;
        static int cb1_hot = -1; if (cb1_hot < 0) cb1_hot = getenv("TURBO1_TCQ_HOTSWAP") ? 1 : 0;
        if (!cb1_loaded || cb1_hot) {
            cb1_loaded = true;
            turbo1_tcq_load_kv_encode();
        }
    }
    if constexpr (type_K == GGML_TYPE_TURBO3_TCQ || type_K == GGML_TYPE_TURBO2_TCQ || type_K == GGML_TYPE_TURBO1_TCQ ||
                  type_V == GGML_TYPE_TURBO3_TCQ || type_V == GGML_TYPE_TURBO2_TCQ || type_V == GGML_TYPE_TURBO1_TCQ) {
        const ggml_tensor * V = dst->src[2];
        const int64_t n_kv = V->ne[1] > 0 ? V->ne[1] : 1;
        const float ln_ctx = logf((float)n_kv);

        // V alpha: context-adaptive unless env var override
        float alpha_v = 1.0f;
        static float alpha_v_static = -1.0f;
        if (alpha_v_static < 0.0f) {
            alpha_v_static = 0.0f;
            const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V");
            if (s) {
                char * end;
                float a = strtof(s, &end);
                if (end != s && a > 0.0f && a < 10.0f) alpha_v_static = a;
            }
        }
        if (alpha_v_static > 0.0f) {
            alpha_v = alpha_v_static;
        } else if constexpr (type_V == GGML_TYPE_TURBO3_TCQ) {
            alpha_v = TURBO_TCQ_ALPHA_V_T3;
        } else if constexpr (type_V == GGML_TYPE_TURBO2_TCQ) {
            alpha_v = TURBO_TCQ_ALPHA_V_T2;
        } else if constexpr (type_V == GGML_TYPE_TURBO1_TCQ) {
            alpha_v = TURBO_TCQ_ALPHA_V_T1;
        }
        // Push alpha to this TU's __constant__ ONCE per device, skipping the memcpy on repeat
        // launches. alpha_v is constant for this template instance, so re-pushing it every call is
        // redundant — and a synchronous cudaMemcpyToSymbol during CUDA/HIP graph capture is illegal
        // (ROCm: "operation would make the legacy stream depend on a capturing blocking stream").
        // The first (eager, pre-capture) launch sets it; captured launches skip. Matches the
        // static cbN_loaded codebook guards above.
        static float alpha_v_pushed[GGML_CUDA_MAX_DEVICES];
        static bool  alpha_v_have  [GGML_CUDA_MAX_DEVICES] = {};
        if (!alpha_v_have[id] || alpha_v_pushed[id] != alpha_v) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha_v, sizeof(float)));
            alpha_v_pushed[id] = alpha_v;
            alpha_v_have[id]   = true;
        }

        // K alpha: static (default 1.0, env var override)
        static float alpha_k = -1.0f;
        if (alpha_k < 0.0f) {
            alpha_k = 1.0f;
            const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_K");
            if (s) {
                char * end;
                float a = strtof(s, &end);
                if (end != s && a > 0.0f && a < 10.0f) alpha_k = a;
            }
        }
        static float alpha_k_pushed[GGML_CUDA_MAX_DEVICES];
        static bool  alpha_k_have  [GGML_CUDA_MAX_DEVICES] = {};
        if (!alpha_k_have[id] || alpha_k_pushed[id] != alpha_k) {
            CUDA_CHECK(cudaMemcpyToSymbol(d_tcq_decode_alpha_k_fattn, &alpha_k, sizeof(float)));
            alpha_k_pushed[id] = alpha_k;
            alpha_k_have[id]   = true;
        }
    }

    // need_f16_K=false, need_f16_V=false: raw turbo data passes through to kernel.
    launch_fattn<DV, ncols1, ncols2>
        (ctx, dst, fattn_kernel, nwarps, nbytes_shared_total, nbatch_fa, false, false, true, warp_size_host);
}


#define DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, ncols1, ncols2, tK, tV)                                  \
    template void ggml_cuda_flash_attn_ext_mma_turbo_case                                            \
    <DKQ, DV, ncols1, ncols2, tK, tV>(ggml_backend_cuda_context & ctx, ggml_tensor * dst)            \

// Fused Turbo attention handles at most four query tokens. These are the nine
// tile shapes selected by the NVIDIA and RDNA decode paths.
#define DECL_FATTN_MMA_TURBO_ALL(DKQ, DV, tK, tV) \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  8, 1, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  4, 2, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  2, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  1, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV, 16, 1, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  8, 2, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  4, 4, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  2, 8, tK, tV); \
    extern DECL_FATTN_MMA_TURBO_CASE(DKQ, DV,  4, 8, tK, tV); \

DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
// Asymmetric "q6 sweet spot" (6.124 bpw): turbo8 K + turbo4 V, D=256 only.
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO4_0)
// Asymmetric pairs for the dynamic VBR degrade ladder, D=256 only. The priced degrade orders
// move K and V of a layer independently and are NOT banded — a live mixed layer can straddle up
// to 4 rungs (q27 holds K=t8:V=t3 across a multi-step cursor range; g31 holds K=t8:V=t1 across
// long ranges). Only the ADJACENT-tier subset gets fused instances (compile-time budget); wider
// straddles fall to the materialize path (measured -13-15% tg32 @ d8192 when uniform).
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO8_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO4_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO2_TCQ)
// f16<->t8 entry band: VBR enters at f16 and the FIRST band every layer crosses is f16->t8, which
// the adjacent-tier set above skipped (it started at t8). A straddled layer there (K or V still f16)
// otherwise falls to the materialize path — the penalty GROWS with context (-50% @ d64k measured),
// exactly the regime the dynamic controller churns through as context grows. f16 side needs no
// dequant and (as K) no WHT rotation, so the fused kernel is strictly cheaper than matched t8:t8.
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO8_0, GGML_TYPE_F16)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_F16,      GGML_TYPE_TURBO8_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
DECL_FATTN_MMA_TURBO_ALL(128, 128, GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
DECL_FATTN_MMA_TURBO_ALL(256, 256, GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
