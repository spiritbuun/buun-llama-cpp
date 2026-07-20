#include "common.cuh"
#include "turbo-tcq-alpha.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-mma-turbo.cuh"
#include "fattn-tile.cuh"
#include "fattn-vec.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"

#include <atomic>
#include <sys/stat.h>
#include <vector>


// InnerQ: update the fattn-side inverse scale array from host (all devices)
void turbo_innerq_update_fattn_scales(const float * scale_inv) {
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, scale_inv, 128 * sizeof(float));
    }
    cudaSetDevice(cur_device);
}

void turbo_innerq_init_fattn() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, ones, sizeof(ones));
    }
    cudaSetDevice(cur_device);
}

// Q² calibration: host-side management
static int q_calibrate_state = 0; // 0=off, 1=collecting, 2=done

void turbo_q_calibrate_init() {
    const char * env = getenv("TURBO_Q_CALIBRATE");
    if (!env || atoi(env) != 1) return;

    double zeros[128] = {};
    int zero = 0, one = 1;
    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_channel_sq_fattn, zeros, sizeof(zeros));
        cudaMemcpyToSymbol(d_q_channel_count_fattn, &zero, sizeof(zero));
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &one, sizeof(one));
    }
    cudaSetDevice(cur_device);
    q_calibrate_state = 1;
    fprintf(stderr, "TURBO_Q_CALIBRATE: collecting per-position Q² statistics\n");
}

void turbo_q_calibrate_finalize() {
    if (q_calibrate_state != 1) return;

    int cur_device;
    cudaGetDevice(&cur_device);
    int device_count;
    cudaGetDeviceCount(&device_count);

    int zero = 0;
    for (int id = 0; id < device_count; id++) {
        cudaSetDevice(id);
        cudaMemcpyToSymbol(d_q_calibrate_fattn, &zero, sizeof(zero));
    }
    cudaSetDevice(cur_device);

    double sq[128];
    int count;
    cudaMemcpyFromSymbol(sq, d_q_channel_sq_fattn, sizeof(sq));
    cudaMemcpyFromSymbol(&count, d_q_channel_count_fattn, sizeof(count));

    if (count == 0) {
        fprintf(stderr, "TURBO_Q_CALIBRATE: no Q vectors seen, skipping\n");
        q_calibrate_state = 2;
        return;
    }

    // Compute E[Q²] per position and save as 128 float32 values
    float weights[128];
    fprintf(stderr, "TURBO_Q_CALIBRATE: %d Q groups accumulated\n", count);
    double total = 0;
    for (int i = 0; i < 128; i++) {
        weights[i] = (float)(sq[i] / count);
        total += weights[i];
    }
    float mean = (float)(total / 128.0);
    float maxw = 0, minw = 1e30f;
    for (int i = 0; i < 128; i++) {
        if (weights[i] > maxw) maxw = weights[i];
        if (weights[i] < minw) minw = weights[i];
    }
    fprintf(stderr, "  E[Q²] mean=%.6f min=%.6f max=%.6f ratio=%.2f\n",
            mean, minw, maxw, maxw / (minw > 1e-10f ? minw : 1e-10f));

    const char * path = "/tmp/q_weights.bin";
    FILE * fp = fopen(path, "wb");
    if (fp) {
        fwrite(weights, sizeof(float), 128, fp);
        fclose(fp);
        fprintf(stderr, "  Saved Q weights to %s (128 floats)\n", path);
    }

    q_calibrate_state = 2;
}

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if constexpr (ncols2 <= 16) {
        if (Q->ne[1] <= 16/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 32/ncols2 || (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING) ||
            (GGML_CUDA_CC_IS_AMD(cc) && DKQ > 256)) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // Edge cases like no mask, ALiBi, unpadded K/V, or misaligned addresses for large data transfers
    //     are put into the template specialization without GQA optimizations.
    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    // On Volta the GQA optimizations aren't as impactful vs. minimizing wasted compute:
    if (cc == GGML_CUDA_CC_VOLTA) {
        if (use_gqa_opt && gqa_ratio % 8 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
            return;
        }

        if (use_gqa_opt && gqa_ratio % 4 == 0) {
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
            return;
        }

        if constexpr (DKQ <= 256) {
            if (use_gqa_opt && gqa_ratio % 2 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
                return;
            }

            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
            return;
        } else {
            GGML_ABORT("fatal error");
        }
    }

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    if constexpr (DKQ <= 256) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
    } else {
        GGML_ABORT("fatal error");
    }
}

// Turbo MMA fused dispatch: ncols1 selection (mirrors f16 version but calls turbo case).
template <int DKQ, int DV, int ncols2, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (turing_mma_available(cc) && Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 8/ncols2, ncols2, type_K, type_V>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 16/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    // Turing (sm_75) is capped at ncols=32 — the kernel has NO_DEVICE_CODE for ncols>32.
    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 32/ncols2, ncols2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_case<DKQ, DV, 64/ncols2, ncols2, type_K, type_V>(ctx, dst);
}

// Turbo MMA fused dispatch: ncols2 selection based on GQA ratio.
template <int DKQ, int DV, ggml_type type_K, ggml_type type_V>
static void ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    bool use_gqa_opt = mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                use_gqa_opt = false;
                break;
            }
        }
    }

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio > 4) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 8, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 2) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 4, type_K, type_V>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 2, type_K, type_V>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols1<DKQ, DV, 1, type_K, type_V>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 192: {
            // MiMo-V2.5 / V2.5-Pro / V2-Flash: gqa_ratio is 8 (SWA) or 16 (full attn)
            GGML_ASSERT(V->ne[0] == 128);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));
            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);
            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128, 16>(ctx, dst);
            } else {
                GGML_ASSERT(gqa_ratio % 8 == 0);
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<192, 128,  8>(ctx, dst);
            }
        } break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 320:
            // For Mistral Small 4, go straight to the ncols1 switch (ncols2=32-only build).
            GGML_ASSERT(V->ne[0] == 256);
            {
                float max_bias = 0.0f;
                memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

                const bool use_gqa_opt = mask && max_bias == 0.0f;
                GGML_ASSERT(use_gqa_opt);
                GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
                const int gqa_ratio = Q->ne[2] / K->ne[2];
                GGML_ASSERT(gqa_ratio % 32 == 0);

                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<320, 256, 32>(ctx, dst);
            }
            break;
        case 512:
            GGML_ASSERT(V->ne[0] == 512);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<512, 512>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            if (gqa_ratio == 20) { // GLM 4.7 Flash
                if (cc >= GGML_CUDA_CC_DGX_SPARK) {
                    if (Q->ne[1] <= 8) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_BLACKWELL) {
                    if (Q->ne[1] <= 4 && K->ne[1] >= 65536) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 4) {
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                if (cc >= GGML_CUDA_CC_TURING) {
                    if (Q->ne[1] <= 4) {
                        if (K->ne[1] <= 16384) {
                            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
                            break;
                        }
                        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 32>(ctx, dst);
                        break;
                    }
                    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
                    break;
                }
                // Volta:
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 4>(ctx, dst);
            } else if (gqa_ratio % 16 == 0) {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512,  4>(ctx, dst);
            }
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// Context-adaptive V alpha: logarithmic scaling based on current KV occupancy.
// 3-bit: alpha = 1.1484 - 0.01443 * ln(n_kv), calibrated on Qwen3.5-27B decode-time KLD sweeps
//   at 8K/16K/32K. Optima: 8K→1.020, 16K→1.005, 32K→1.000.
// 2-bit: alpha = 0.8865 + 0.0195 * ln(n_kv), calibrated on fine-grained 0.005-step decode-time
//   KLD sweeps at 2K/7K/16K/32K on Qwen3.5-27B. Optima: 2K→1.030, 7K→1.065, 16K→1.090, 32K→1.075.
// Override with TURBO_TCQ_DECODE_ALPHA_V env var to force a static alpha (disables adaptive).
static float d_tcq_decode_alpha_v_static = 0.0f; // 0 = use adaptive, >0 = static override
static float d_tcq_decode_alpha_k = 1.0f;       // K decode alpha, static (default 1.0)
static bool d_tcq_decode_alpha_loaded = false;

static inline float tcq_compute_alpha_v(ggml_type v_type, int64_t n_kv) {
    if (d_tcq_decode_alpha_v_static > 0.0f) return d_tcq_decode_alpha_v_static;
    if (n_kv < 1) n_kv = 1;
    if (v_type == GGML_TYPE_TURBO3_TCQ) {
        // Flat optimum for the coord-descent codebook (K=iter374/V=iter500). The retrained
        // codebook removed the depth-dependence — a flat alpha beats the old adaptive curve.
        return TURBO_TCQ_ALPHA_V_T3;
    } else if (v_type == GGML_TYPE_TURBO2_TCQ) {
        // Flat optimum for the coord-descent codebook (K=iter195/V=iter208).
        return TURBO_TCQ_ALPHA_V_T2;
    } else if (v_type == GGML_TYPE_TURBO1_TCQ) {
        // 1-bit wants a higher decode alpha than 2/3-bit (monotone in bits: t3=1.02, t2=1.06).
        // KLD-panel confirmed 2026-07-05 (native-path sweep, q27, 8k/16k/32k x {builtin,
        // finalist s11/s5} x alpha 1.06-1.30): every completed series has an INTERIOR minimum
        // at 1.26 (1.30 turns up), same-top flips agree; ~2% median better than the old 1.22.
        // The earlier ragged-harness ladder said <=1.14 — instrument != deployment; only the
        // native fattn path is authoritative for this constant.
        return TURBO_TCQ_ALPHA_V_T1;
    }
    return 1.0f;
}

static void load_tcq_decode_alpha(int device) {
    static bool loaded[GGML_CUDA_MAX_DEVICES] = {};
    if (loaded[device]) return;
    loaded[device] = true;
    if (!d_tcq_decode_alpha_loaded) {
        d_tcq_decode_alpha_loaded = true;
        const char * sk = getenv("TURBO_TCQ_DECODE_ALPHA_K");
        if (sk) {
            char * end;
            errno = 0;
            float a = strtof(sk, &end);
            if (end != sk && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_k = a;
                fprintf(stderr, "TCQ decode: K alpha=%.4f\n", a);
            }
        }
        const char * s = getenv("TURBO_TCQ_DECODE_ALPHA_V");
        if (s) {
            char * end;
            errno = 0;
            float a = strtof(s, &end);
            if (end != s && errno == 0 && a > 0.0f && a < 10.0f) {
                d_tcq_decode_alpha_v_static = a;
            }
        }
    }
    if (d_tcq_decode_alpha_v_static > 0.0f) {
        cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &d_tcq_decode_alpha_v_static, sizeof(float));
        if (device == 0) fprintf(stderr, "TCQ decode: V alpha=%.4f (static override)\n", d_tcq_decode_alpha_v_static);
    } else {
        if (device == 0) fprintf(stderr, "TCQ decode: context-adaptive V alpha enabled\n");
    }
}

// === Turbo prefill: bulk dequant to fp16 + MMA attention ===
// During prefill (Q->ne[1] > 1), dequantize turbo K/V to fp16 temp buffers
// and use the fast MMA kernel instead of the slower vec kernel.

static __global__ void k_turbo2_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2;
    const int j_in_blk = j % QK_TURBO2;
    const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const float val = d_turbo_centroids_2bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3, const int is_v = 0) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3;
    const int j_in_blk = j % QK_TURBO3;
    const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
    const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
#if TURBO3_SKEW_EXP >= 2
    const float val = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)]
                    * (is_v ? d_turbo3_ss_v_fattn[j % 128] : d_turbo3_ss_k_fattn[j % 128]) * norm;
#else
    GGML_UNUSED(is_v);
    const float val = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * norm;
#endif

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo3_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x; // element index within row (0..ne0-1)
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO3_TCQ;
    const int t = j % QK_TURBO3_TCQ; // element index within 128-element block
    const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 9-bit state from bitstream at bit offset t*3
    const int bit_pos = t * 3;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0x1FF;
    const float val = (is_v ? d_turbo3_tcq_codebook_v_fattn : d_turbo3_tcq_codebook_fattn)[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo2_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO2_TCQ;
    const int t = j % QK_TURBO2_TCQ;
    const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;

    // Sliding window decode: read 8-bit state from bitstream at bit offset t*2
    const int bit_pos = t * 2;
    const int byte_idx = bit_pos / 8;
    const int bit_off = bit_pos % 8;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0xFF;
    const float val = (is_v ? d_turbo2_tcq_codebook_v_fattn : d_turbo2_tcq_codebook_fattn)[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

static __global__ void k_turbo4_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO4;
    const int j_in_blk = j % QK_TURBO4;
    const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const uint8_t idx = (j_in_blk & 1) ? (blk->qs[j_in_blk / 2] >> 4) : (blk->qs[j_in_blk / 2] & 0xF);
    const float val = d_turbo_centroids_4bit_fattn[idx] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo3 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element FWHT groups
// (each group spans 4 turbo3 storage blocks of 32 elements; all 4 blocks share the same norm).
// In-place 128-point inverse FWHT spread across 128 threads, one element per thread.
// h=1..16 use intra-warp __shfl_xor_sync (no smem, no syncthreads).
// h=32 and h=64 cross warp boundaries, so we fall back to smem with syncs.
// Caller supplies a __shared__ float[128] scratch buffer.
static __device__ __forceinline__ float fwht128_butterfly_inplace(float val, float * smem) {
    const int tid = threadIdx.x;

    // Intra-warp passes: shuffle xor with stride h, no smem, no sync.
    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFFULL, val, h);
        val = (tid & h) ? (other - val) : (val + other);
    }

    // h=32 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 32) ? (smem[tid - 32] - val) : (val + smem[tid + 32]);
    __syncthreads();

    // h=64 (cross-warp within block, smem needed)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 64) ? (smem[tid - 64] - val) : (val + smem[tid + 64]);
    __syncthreads();

    return val;
}

// Pair-pack two adjacent threads' fp16 outputs into a single 32-bit half2 store.
// Halves the global store count vs. one __float2half per thread.
static __device__ __forceinline__ void fwht128_store_half(
        float val, half * dst_base) {
    const int tid = threadIdx.x;
    const float neighbor = __shfl_xor_sync(0xFFFFFFFFULL, val, 1);
    if ((tid & 1) == 0) {
        const half2 packed = __floats2half2_rn(val, neighbor);
        *((half2 *)(dst_base + tid)) = packed;
    }
}

static __global__ void k_turbo3_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    // ne0 in elements, FWHT group = 128 elements = 4 turbo3 blocks of 32
    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO3; // 4

    for (int g = 0; g < n_groups; g++) {
        // Element index within the FWHT group
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO3;  // 0..3
        const int j_in_blk  = j_in_grp % QK_TURBO3;   // 0..31

        const block_turbo3_0 * blk = (const block_turbo3_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);   // same for all 4 blocks in this group

        const uint8_t low2 = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const uint8_t hi1  = (blk->signs[j_in_blk / 8] >> (j_in_blk % 8)) & 0x1;
#if TURBO3_SKEW_EXP >= 2
        const float c = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)] * d_turbo3_ss_k_fattn[tid];
#else
        const float c = d_turbo_centroids_3bit_fattn[low2 | (hi1 << 2)];
#endif

        // Inverse FWHT: 5 intra-warp shfl passes + 2 cross-warp smem passes.
        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        // Normalize, apply signs1, undo InnerQ scaling, multiply by norm, cast to fp16
        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo4 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element turbo4 blocks.
static __global__ void k_turbo4_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO4);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo4_0 * blk = (const block_turbo4_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (tid & 1) ? (blk->qs[tid / 2] >> 4) : (blk->qs[tid / 2] & 0xF);

        float val = fwht128_butterfly_inplace(d_turbo_centroids_4bit_fattn[idx] * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo8 V dequant: simple centroid×norm (rotated domain, V un-rotation done at graph level).
static __global__ void k_turbo8_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx  = j / QK_TURBO8;
    const int j_in_blk = j % QK_TURBO8;
    const block_turbo8_0 * blk = (const block_turbo8_0 *)src_row + blk_idx;

    const float norm = __half2float(blk->norm);
    const float val = d_turbo_centroids_8bit_fattn[blk->qs[j_in_blk]] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo8 K dequant with inverse FWHT: produces K in original (unrotated) domain
// so Q does NOT need pre-rotation. 128 threads per block, loops over 128-element turbo8 blocks.
static __global__ void k_turbo8_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO8);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo8_0 * blk = (const block_turbo8_0 *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);

        float val = fwht128_butterfly_inplace(d_turbo_centroids_8bit_fattn[blk->qs[tid]] * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}




// turbo1_tcq decode codebooks (256-state, separate K/V). Cold-start = turbo2_tcq decode anchor;
// TURBO1_TCQ_CB_K / _V (?? TURBO1_TCQ_CB) override. MUST match the encode-side symbols.
// Baked-in turbo1_tcq decode codebooks — MUST be bit-identical to the encode-side arrays in
// turbo-quant-cuda.cuh (K = r1K_tail_s5/iter300, V = r1V_tail_s11/iter300, box-2 2026-07-04 finalists).
static __constant__ float d_turbo1_tcq_cb_fattn[256]   = {  // K decode
    -0.153683484f, +0.0932431519f, -0.0284221377f, -0.0213323738f, -0.0918863863f, -0.0834758952f, +0.0542073846f, +0.0542073846f,
    -0.128483802f, -0.0949096978f, +0.0436351784f, +0.0444926955f, -0.177975088f, +0.132742643f, -0.0365675129f, -0.00663438998f,
    +0.0619110875f, +0.0619110875f, -0.176081076f, -0.0222552121f, -0.0675581396f, -0.0675581396f, +0.0479601882f, +0.0701695681f,
    +0.0259720646f, +0.0664564744f, +0.0030253143f, +0.167568222f, -0.0674274117f, -0.0624906905f, +0.0593295284f, +0.0716510117f,
    +0.00550817279f, +0.00762284314f, +0.123896576f, +0.14741008f, -0.0566166341f, -0.0527233072f, -0.132352635f, +0.00985434558f,
    -0.0692158043f, -0.0692158043f, +0.054607287f, +0.054607287f, +0.13614215f, +0.19078669f, +0.00156285148f, +0.00209119194f,
    -0.188667893f, -0.0470676497f, -0.116242573f, -0.0126988571f, +0.00894579757f, +0.150882706f, +0.0449662991f, +0.0545337871f,
    +0.0826691911f, +0.0982875451f, -0.0603181981f, -0.0603181981f, +0.0555567369f, +0.0555567369f, -0.110769883f, -0.0770125017f,
    -0.0724448115f, -0.0724448115f, +0.0754566267f, +0.0754566267f, -0.11218828f, -0.0535744801f, +0.0575145707f, +0.0575145707f,
    +0.119879454f, +0.152365059f, -0.0423310474f, -0.0423310474f, -0.034485355f, -0.034485355f, -0.0241647139f, +0.0743408874f,
    +0.0647695139f, +0.0677786916f, -0.0567673519f, -0.0567673519f, +0.0939702094f, +0.142256647f, -0.0456033908f, -0.0456033908f,
    +0.0414441153f, +0.0414441153f, -0.116412848f, -0.0563932136f, +0.0463487506f, +0.118106365f, -0.147341222f, -0.0630168244f,
    -0.0628485903f, -0.0628485903f, +0.0281294063f, +0.0281294063f, +0.157495975f, +0.157495975f, -0.0421977639f, -0.0421977639f,
    +0.0308154635f, +0.0308154635f, -0.0292343032f, -0.0292343032f, -0.125520885f, -0.125520885f, +0.0718865544f, +0.0718865544f,
    +0.0689959824f, +0.0689959824f, -0.081269294f, -0.081269294f, -0.0909179002f, -0.0909179002f, +0.0190163292f, +0.173722342f,
    -0.0671347827f, -0.0671347827f, +0.0800654069f, +0.0800654069f, +0.0816404521f, +0.0816404521f, -0.0624779612f, -0.0624779612f,
    -0.0195930507f, -0.0195930507f, +0.138596535f, +0.138596535f, +0.0451605618f, +0.0459208488f, -0.109351009f, -0.109351009f,
    +0.0349098034f, +0.0685105473f, -0.124482363f, -0.124482363f, -0.0191877317f, -0.0191877317f, +0.0745684579f, +0.186214373f,
    -0.0670050532f, -0.0248501189f, +0.00614000158f, +0.127290398f, +0.0596978553f, +0.0726318583f, -0.0574334636f, -0.0574334636f,
    -0.0218063109f, -0.0218063109f, -0.176524386f, +0.0025834043f, +0.0332166888f, +0.0332166888f, -0.0316858068f, -0.0316858068f,
    -0.16282168f, -0.160597652f, -0.0151454788f, -0.000276063627f, +0.00815063529f, +0.00815063529f, +0.0220920816f, +0.197730258f,
    +0.0771044195f, +0.0771044195f, -0.0734175146f, -0.0734175146f, +0.00178386632f, +0.0152899409f, -0.15694806f, -0.15694806f,
    -0.00759414956f, +0.0315586925f, +0.0653701797f, +0.163669571f, -0.144907251f, +0.00212525437f, -0.048506137f, -0.0196365472f,
    -0.0620619431f, -0.0620619431f, +0.0959848315f, +0.0959848315f, -0.073461324f, -0.0689062104f, +0.0647197068f, +0.0647197068f,
    +0.0735893846f, +0.0735893846f, -0.0616040565f, -0.0610223711f, +0.0637794361f, +0.0996553302f, -0.0651129931f, -0.0651129931f,
    -0.0469228663f, -0.0452931263f, +0.113321088f, +0.113321088f, +0.0841016918f, +0.0841016918f, -0.194901869f, -0.0528107621f,
    -0.0684252232f, -0.0684252232f, +0.0722951591f, +0.0722951591f, -0.042255573f, -0.042255573f, +0.118550412f, +0.118550412f,
    -0.139325529f, -0.0234075822f, +0.0242734887f, +0.0253605284f, -0.146457925f, -0.0543592274f, +0.0258371048f, +0.0723357424f,
    +0.0426197089f, +0.0518061668f, -0.121822171f, -0.121822171f, -0.0367406197f, -0.0367406197f, +0.0920659974f, +0.0920659974f,
    -0.105211377f, -0.0984823778f, +0.0756842345f, +0.0756842345f, +0.0151738664f, +0.0239403863f, -0.0424663946f, -0.0277856514f,
    -0.0687293857f, -0.0687293857f, +0.0550689697f, +0.0550689697f, +0.0518884808f, +0.0518884808f, -0.163815364f, -0.0187361594f,
    +0.06937702f, +0.105992272f, -0.0624594353f, -0.0624594353f, -0.0639030039f, -0.0639030039f, +0.11936225f, +0.11936225f,
};
static __constant__ float d_turbo1_tcq_cb_v_fattn[256] = {  // V decode
    -0.119549245f, -0.119549245f, +0.04884452f, +0.04884452f, +0.112512514f, +0.112512514f, -0.0753565952f, -0.0124196028f,
    -0.0542244501f, -0.0454595797f, +0.113282755f, +0.113282755f, -0.0478299409f, -0.00588486949f, -0.156662762f, +0.0177723076f,
    +0.0502630547f, +0.0751912221f, -0.0732487515f, -0.0363520756f, -0.0878284127f, -0.0833693445f, +0.0655872673f, +0.0655872673f,
    -0.177922264f, +0.147200137f, -0.00281362329f, -0.00281362329f, -0.0236557927f, -0.00629976951f, +0.0262224339f, +0.0262224339f,
    -0.122524828f, -0.121980354f, +0.0355546959f, +0.0355546959f, -0.0892947987f, -0.0783565268f, +0.0634978563f, +0.0634978563f,
    +0.100714654f, +0.116031229f, -0.0382292345f, -0.0382292345f, +0.0700735897f, +0.0700735897f, -0.0744145215f, -0.0744145215f,
    +0.0285345633f, +0.0285345633f, -0.137101039f, -0.0669855997f, +0.0462302938f, +0.0955360904f, -0.0378546566f, -0.0378546566f,
    -0.00299687427f, +0.0354520716f, +0.158731595f, +0.159036174f, -0.00417822506f, -0.00417822506f, -0.169665679f, -0.101989843f,
    +0.0885970294f, +0.174134374f, -0.0419115685f, -0.0374047682f, -0.0866785944f, -0.0649861544f, +0.0656950325f, +0.0656950325f,
    -0.161553636f, -0.145003155f, +0.0255606212f, +0.0260413494f, -0.184533074f, -0.0552854389f, +0.0322727412f, +0.0322727412f,
    -0.0823699087f, -0.0823699087f, +0.0553851724f, +0.0553851724f, +0.0966504067f, +0.0966504067f, -0.0494570695f, -0.0411219671f,
    -0.0437131599f, -0.0437131599f, +0.124873698f, +0.166997984f, +0.0439662524f, +0.0628898293f, -0.169980094f, -0.0406379513f,
    +0.0272835698f, +0.0272835698f, -0.102720782f, -0.0547375381f, -0.104257204f, -0.0429750308f, +0.0265545268f, +0.171791926f,
    -0.0853084773f, -0.0853084773f, +0.0476404428f, +0.0476404428f, +0.0900258943f, +0.0900258943f, -0.054219868f, -0.054219868f,
    -0.0479668193f, -0.0429475419f, +0.131989121f, +0.131989121f, +0.0447537825f, +0.0447537825f, -0.0831433907f, -0.078851141f,
    +0.0472813472f, +0.0472813472f, -0.120939665f, -0.120939665f, -0.146120414f, -0.0148609718f, +0.0725838169f, +0.0725838169f,
    +0.0397478864f, +0.0397478864f, -0.0799675956f, -0.0799675956f, -0.0573435687f, -0.037379805f, +0.0547020584f, +0.169056475f,
    +0.0825577676f, +0.0825577676f, -0.0486915782f, -0.0486915782f, +0.0708708912f, +0.164926067f, -0.0399483964f, -0.0399483964f,
    -0.0932579115f, -0.066275239f, +0.0614030398f, +0.119138196f, +0.0609519258f, +0.0609519258f, -0.087209031f, -0.0788605884f,
    -0.0137468651f, -0.0137468651f, -0.170376986f, +0.0652351379f, +0.0180841498f, +0.0180841498f, -0.0360591412f, -0.0360591412f,
    +0.032522019f, +0.032522019f, -0.157200888f, -0.157200888f, +0.0732334927f, +0.0732334927f, -0.0809531361f, -0.0801029429f,
    -0.0435965359f, -0.0396991074f, +0.124063686f, +0.124063686f, -0.06837219f, -0.06837219f, +0.0630521029f, +0.0643705279f,
    -0.0663868934f, +0.186983824f, +0.00707759568f, +0.0534422286f, -0.0711288378f, -0.0526081622f, +0.0910681486f, +0.0910681486f,
    -0.181568101f, -0.0869468525f, -0.00884238817f, -0.00884238817f, +0.116818011f, +0.116818011f, +0.00558634428f, +0.0387513824f,
    -0.0515811332f, -0.0300675891f, +0.0699146464f, +0.0728352666f, +0.048193451f, +0.0833275244f, -0.0958049595f, -0.0958049595f,
    +0.0172887873f, +0.0292257778f, -0.148785338f, -0.148785338f, -0.000765698263f, +0.0509745441f, +0.0231456049f, +0.172527939f,
    +0.0573678799f, +0.0582933389f, -0.0823828951f, -0.0823828951f, -0.0570469573f, -0.0570469573f, +0.0781005844f, +0.111567542f,
    +0.10178373f, +0.10178373f, -0.0412749536f, -0.0412749536f, -0.0953622833f, -0.0633069724f, +0.0198892411f, +0.124936409f,
    -0.0825958773f, +0.186329335f, +0.0161208287f, +0.0615422241f, +0.0401945598f, +0.0468487293f, -0.164305717f, -0.0166916698f,
    +0.0545916557f, +0.0545916557f, -0.0917867944f, -0.0917867944f, -0.0585476533f, -0.0585476533f, +0.0977485031f, +0.0977485031f,
    +0.0931819826f, +0.0931819826f, -0.0435980707f, -0.0435980707f, -0.0865197331f, -0.0865197331f, +0.0579762831f, +0.0579762831f,
    -0.0325509906f, -0.0325509906f, +0.0279718302f, +0.0279718302f, +0.0218467899f, +0.18708235f, -0.0869560316f, -0.085878171f,
};
static void turbo1_tcq_load_cb_fattn() {
    auto load_file = [](const char * p, float * out) -> bool {
        FILE * f = fopen(p, "rb"); if (!f) return false;
        bool ok = fread(out, sizeof(float), 256, f) == 256; fclose(f); return ok;
    };
    auto file_mtime = [](const char * p) -> long { struct stat st; return (p && stat(p, &st) == 0) ? (long)st.st_mtime : 0; };
    int dev = 0; cudaGetDevice(&dev);
    static int hot = -1; if (hot < 0) hot = getenv("TURBO1_TCQ_HOTSWAP") ? 1 : 0;
    static bool init[GGML_CUDA_MAX_DEVICES] = {};
    static long mk[GGML_CUDA_MAX_DEVICES] = {}, mv[GGML_CUDA_MAX_DEVICES] = {};
    const char * cb = getenv("TURBO1_TCQ_CB");
    const char * kp = getenv("TURBO1_TCQ_CB_K"); if (!kp) kp = cb;
    const char * vp = getenv("TURBO1_TCQ_CB_V"); if (!vp) vp = cb;
    const bool first = !init[dev];
    const long nk = (hot || first) ? file_mtime(kp) : mk[dev];
    const long nv = (hot || first) ? file_mtime(vp) : mv[dev];
    const bool do_k = first || (hot && nk != mk[dev]);
    const bool do_v = first || (hot && nv != mv[dev]);
    if (!do_k && !do_v) return;
    float buf[256];
    // cold-start default = baked-in best codebook (seed5/iter135 K, seed1/iter490 V); only env overrides it.
    if (do_k && kp && load_file(kp, buf)) { cudaMemcpyToSymbol(d_turbo1_tcq_cb_fattn,   buf, 256*sizeof(float)); mk[dev] = nk; }
    if (do_v && vp && load_file(vp, buf)) { cudaMemcpyToSymbol(d_turbo1_tcq_cb_v_fattn, buf, 256*sizeof(float)); mv[dev] = nv; }
    init[dev] = true;
    if (first)
        fprintf(stderr, "TCQ1 decode: K/V codebooks (K=%s V=%s) hotswap=%d\n", kp?kp:"baked-in", vp?vp:"baked-in", hot);
}

// turbo1_tcq dequant: trellis-state decode of rotated coords, then per-row inverse FWHT → original
// domain (K and V). state_tid = read_8_bits(qs, tid*1); recon = norm * cb[state]. is_v picks K/V book.
static __global__ void k_turbo1_tcq_dequant_f16(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];
    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    const float * cb = is_v ? d_turbo1_tcq_cb_v_fattn : d_turbo1_tcq_cb_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    const int n_blocks = (int)(ne0 / QK_TURBO1_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo1_tcq * blk = (const block_turbo1_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm);
        const int bit_pos = tid;                 // k=1: bit offset = tid
        const int byte_idx = bit_pos >> 3;
        const int bit_off  = bit_pos & 7;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        const float recon = norm * alpha * cb[state];

        float val = fwht128_butterfly_inplace(recon * s2[tid], smem);
        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid];
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo1_tcq dequant, ROTATED domain: emits the stored trellis coefficients (norm*alpha*cb)
// directly, mirroring k_turbo2_tcq_dequant_f16 — the graph-level ggml_turbo_wht(inverse) on
// the attention output performs the un-rotation (turbo2/3_tcq contract). One thread/element.
static __global__ void k_turbo1_tcq_dequant_f16_rot(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int blk_idx = j / QK_TURBO1_TCQ;
    const int t = j % QK_TURBO1_TCQ;
    const block_turbo1_tcq * blk = (const block_turbo1_tcq *)src_row + blk_idx;

    const float norm = __half2float(blk->norm) * alpha;
    const int byte_idx = t >> 3;
    const int bit_off  = t & 7;
    const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
    const int state = (raw >> bit_off) & 0xFF;
    const float val = (is_v ? d_turbo1_tcq_cb_v_fattn : d_turbo1_tcq_cb_fattn)[state] * norm;

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// turbo2 K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element FWHT groups (each group spans
// 4 turbo2 storage blocks of 32 elements; all 4 blocks share the same norm).
static __global__ void k_turbo2_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_groups = (int)(ne0 / 128);
    constexpr int blocks_per_group = 128 / QK_TURBO2; // 4

    for (int g = 0; g < n_groups; g++) {
        const int j_in_grp = tid;            // 0..127
        const int blk_in_grp = j_in_grp / QK_TURBO2;  // 0..3
        const int j_in_blk  = j_in_grp % QK_TURBO2;   // 0..31

        const block_turbo2_0 * blk = (const block_turbo2_0 *)src_row + g * blocks_per_group + blk_in_grp;
        const float norm = __half2float(blk->norm);

        const uint8_t idx = (blk->qs[j_in_blk / 4] >> ((j_in_blk % 4) * 2)) & 0x3;
        const float c = d_turbo_centroids_2bit_fattn[idx];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + g * 128);
        __syncthreads();
    }
}

// turbo3_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
// 128 threads per block, loops over 128-element TCQ blocks (1 block per FWHT group).
static __global__ void k_turbo3_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v = 0) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO3_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo3_tcq * blk = (const block_turbo3_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 9-bit state from bitstream at bit offset tid*3
        const int bit_pos = tid * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        const float c = (is_v ? d_turbo3_tcq_codebook_v_fattn : d_turbo3_tcq_codebook_fattn)[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// turbo2_tcq K dequant with inverse FWHT: produces K in original (unrotated) domain.
static __global__ void k_turbo2_tcq_dequant_f16_inv_fwht(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3,
        const float alpha, const int is_v = 0) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int tid = threadIdx.x;

    const char * src_row = src + strm * nb3 + head * nb2 + row * nb1;
    const int64_t dst_base = strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0;

    __shared__ float smem[128];

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;
    constexpr float inv_sqrt_128 = 0.08838834764831845f;

    const int n_blocks = (int)(ne0 / QK_TURBO2_TCQ);

    for (int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
        const block_turbo2_tcq * blk = (const block_turbo2_tcq *)src_row + blk_idx;
        const float norm = __half2float(blk->norm) * alpha;

        // Sliding window decode: read 8-bit state from bitstream at bit offset tid*2
        const int bit_pos = tid * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        const float c = (is_v ? d_turbo2_tcq_codebook_v_fattn : d_turbo2_tcq_codebook_fattn)[state];

        float val = fwht128_butterfly_inplace(c * s2[tid], smem);

        val = val * inv_sqrt_128 * s1[tid] * d_innerq_channel_scale_inv_fattn[tid] * norm;
        fwht128_store_half(val, dst + dst_base + blk_idx * 128);
        __syncthreads();
    }
}

// q8_0 K dequant to f16 in TKHE layout, matching the turbo K dequant kernels.
// Used at D=512 when K=q8_0 paired with V=turbo: produces (F16, F16) for the FA dispatch
// and bypasses the (Q8_0, TURBO*) D=512 native VEC templates which have buggy SASS on
// sm_120 PTX-JIT for some K/V combos. Q8_0 is in original (unrotated) domain → output too.
// 1 thread per element, 1 block per (token, head, batch).
static __global__ void k_q8_0_dequant_f16_tkhe(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const char * src_row = src + strm * nb3 + row * nb1 + head * nb2;
    const int blk_idx = j / QK8_0;
    const int j_in_blk = j % QK8_0;
    const block_q8_0 * blk = (const block_q8_0 *)src_row + blk_idx;
    const float d = __half2float(blk->d);
    const float val = d * (float)blk->qs[j_in_blk];

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// bf16 K/V cast to f16 in the same TKHE layout as k_q8_0_dequant_f16_tkhe. Used when a bf16
// side is paired with a turbo side: the f16-only MMA/VEC kernels need an f16 input, and bf16 is
// already in the original (unrotated) domain so no rotation is applied. 1 thread per element.
static __global__ void k_bf16_to_f16_tkhe(
        const char * __restrict__ src, half * __restrict__ dst,
        const int64_t ne0, const int64_t ne1, const int64_t ne2,
        const size_t nb1, const size_t nb2, const size_t nb3) {
    const int64_t row  = blockIdx.x;
    const int64_t head = blockIdx.y;
    const int64_t strm = blockIdx.z;
    const int j = threadIdx.x;
    if (j >= ne0) return;

    const nv_bfloat16 * src_row = (const nv_bfloat16 *)(src + strm * nb3 + row * nb1 + head * nb2);
    const float val = __bfloat162float(src_row[j]);

    dst[strm * (ne1 * ne2 * ne0) + row * (ne2 * ne0) + head * ne0 + j] = __float2half(val);
}

// Persistent Q rotation buffer per device (shared between prefill and decode paths)
static float * q_rot_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  q_rot_buf_size[GGML_CUDA_MAX_DEVICES] = {};
// Address moves of fattn's persistent scratch — doc on the accessor in fattn.cuh
static unsigned long long fattn_scratch_epoch[GGML_CUDA_MAX_DEVICES] = {};

unsigned long long ggml_cuda_fattn_scratch_epoch(int device) {
    return fattn_scratch_epoch[device];
}

// Grow-only allocator for q_rot_buf. An address move MUST bump the scratch epoch — the buffer is
// not a graph node src, so the epoch is the only fence a captured CUDA graph has against it.
static void q_rot_buf_ensure(int device, size_t q_size) {
    if (q_size > q_rot_buf_size[device]) {
        if (q_rot_buf[device]) CUDA_CHECK(cudaFree(q_rot_buf[device]));
        CUDA_CHECK(cudaMalloc(&q_rot_buf[device], q_size));
        q_rot_buf_size[device] = q_size;
        fattn_scratch_epoch[device]++;
    }
}

// Persistent K/V fp16 dequant buffers per device (shared between prefill and decode paths).
// These are the cudaMalloc FALLBACK, used only when the device has no VMM support; the primary
// path backs the scratch with a VMM pool (below) so its deep-context growth never needs a
// contiguous block — see kv_dequant_scratch().
static half * kv_dequant_k_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_k_buf_size[GGML_CUDA_MAX_DEVICES] = {};
static half * kv_dequant_v_buf[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_buf_size[GGML_CUDA_MAX_DEVICES] = {};

// VMM pools backing the same scratch (primary path). One contiguous VA reservation per
// (device,side); physical 2 MiB pages are mapped on demand as the attended range grows. Growth
// needs free PAGES (fungible, non-contiguous), never a contiguous BLOCK, so it cannot fail on
// fragmentation the way the cudaMalloc grow did (it OOM'd at the final 131072->262144 doubling
// on a deep context, ~890 MiB fragmented-free but no 500 MiB contiguous span). vmm_va tracks the
// current VA-reservation size so we only re-reserve (rare, O(log n)) when the need outgrows it.
static ggml_vbr_vmm_pool * kv_dequant_k_vmm[GGML_CUDA_MAX_DEVICES] = {};
static ggml_vbr_vmm_pool * kv_dequant_v_vmm[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_k_vmm_va[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_vmm_va[GGML_CUDA_MAX_DEVICES] = {};
// mapped high-water per (device, side): the scratch is grow-only and never partially unmapped,
// so `need <= hw` short-circuits the per-boundary reserve to one compare instead of a full
// chunk-set re-walk (reset on VA re-reservation, which frees the pool whole)
static size_t  kv_dequant_k_vmm_hw[GGML_CUDA_MAX_DEVICES] = {};
static size_t  kv_dequant_v_vmm_hw[GGML_CUDA_MAX_DEVICES] = {};

// === FWHT rotation kernels for pre-rotate-queries approach ===
// Forward rotation on Q before attention (both prefill and decode paths).
// One block per 128-element group, 128 threads per block.
static __global__ void k_turbo_fwht_forward(
        const float * __restrict__ src, float * __restrict__ dst,
        const int64_t n_elements) {
    const int64_t offset = blockIdx.x * 128;
    if (offset >= n_elements) return;

    const float * s1 = d_turbo_wht_signs1_fattn;
    const float * s2 = d_turbo_wht_signs2_fattn;

    __shared__ float buf[128];

    // InnerQ: apply inverse channel scale to Q before rotation
    float val = src[offset + threadIdx.x] * d_innerq_channel_scale_inv_fattn[threadIdx.x] * s1[threadIdx.x];

    val = fwht128_butterfly_inplace(val, buf);

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val = val * inv_sqrt_128 * s2[threadIdx.x];
    dst[offset + threadIdx.x] = val;

    // Q² calibration: accumulate per-position squared values
    if (d_q_calibrate_fattn) {
        atomicAdd_double(&d_q_channel_sq_fattn[threadIdx.x], (double)(val * val));
        if (threadIdx.x == 0) atomicAdd(&d_q_channel_count_fattn, 1);
    }
}

// ---- Dynamic VBR transcode, Stage 1 (read side) ----------------------------------------------
#include "vbr-transcode.cuh"

// f16 -> f32 with a uniform scale (set_rows needs F32 src0). The scale carries the V decode-alpha
// correction for cross-tier transcode (see vbr_dequant_turbo_to_f32).
static __global__ void k_vbr_f16_to_f32_scaled(const half * __restrict__ src, float * __restrict__ dst,
                                               int64_t n, float scale) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { dst[i] = __half2float(src[i]) * scale; }
}

// F16 source rows (nb1-strided) -> dense scratch_f16. The f16 entry tier stores plain
// original-domain rows (no rotation, no mean-sub), so the "dequant" is a gather; the encode
// tap then runs UNsuppressed for f16 sources (see vbr-transcode.cu).
static __global__ void k_vbr_f16_gather(const char * __restrict__ src, half * __restrict__ dst,
                                        const int64_t ne0, const int64_t n_rows, const size_t nb1) {
    const int64_t row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const half * s = (const half *)(src + (size_t) row * nb1);
    half       * d = dst + row * ne0;
    for (int64_t i = threadIdx.x; i < ne0; i += blockDim.x) {
        d[i] = s[i];
    }
}

// V decode-alpha for a tier (1.0 for non-TCQ / K). The runtime V decode multiplies recon by this.
// NOTE: the TCQ-type guard is load-bearing — tcq_compute_alpha_v returns the TURBO_TCQ_DECODE_ALPHA_V
// static override for ANY type, so calling it unguarded would apply a spurious alpha to turbo4/8.
static inline float vbr_alpha_v_decode(ggml_type t, int64_t n_kv) {
    return (t == GGML_TYPE_TURBO3_TCQ || t == GGML_TYPE_TURBO2_TCQ || t == GGML_TYPE_TURBO1_TCQ)
        ? tcq_compute_alpha_v(t, n_kv) : 1.0f;
}

// Dequant n_cells rows of a turbo KV tensor (type A) to ORIGINAL-domain dense f32 [n_cells, ne0],
// pre-scaled so re-encoding to type B then decoding B reproduces A's reconstruction. Lives in
// fattn.cu so the decode codebooks/InnerQ/alpha (static to this TU) are visible.
void vbr_dequant_turbo_to_f32(const char * src, ggml_type src_type, ggml_type type_B,
                              half * scratch_f16, float * dst_f32,
                              int64_t n_cells, int64_t ne0, size_t nb1,
                              bool is_v, int device, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 128 == 0); // turbo FWHT blocks are 128-wide; the dequant loops ne0/128 blocks per cell
    // Treat the tensor as [ne0, n_cells]: one block per cell, 128 threads, ne0/128 FWHT blocks/cell.
    // head/stream dims collapse to 1 (nb2/nb3 unused since blockIdx.y/z == 0).
    turbo_vanilla_cb_load_fattn();  // vanilla-book override for the transcode dequant kernels
    const dim3 grid((unsigned) n_cells, 1, 1);
    const int iv = is_v ? 1 : 0;
    // V-mean affine tap: this dequant emits the source's STORED domain — V - mu_V for the
    // tapped tiers (t4 and TCQ), FULL-domain for t8 (tap-gated at encode) and F16. The
    // transcode re-encode (vbr-transcode.cu) suppresses the encode tap exactly when the source
    // was tapped, so the destination tier always receives its expected domain. See
    // g_turbo_meansub_suppress.
    // load_tcq_decode_alpha + the natural decode alpha are shared by the TCQ cases (one-shot loader,
    // harmless for non-TCQ; alpha is unused by the turbo4/8 kernels which take no alpha arg).
    load_tcq_decode_alpha(device);
    const float alpha = is_v ? tcq_compute_alpha_v(src_type, n_cells) : d_tcq_decode_alpha_k;
    switch (src_type) {
        case GGML_TYPE_TURBO4_0:
            k_turbo4_dequant_f16_inv_fwht<<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1);
            break;
        case GGML_TYPE_TURBO8_0:
            k_turbo8_dequant_f16_inv_fwht<<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1);
            break;
        case GGML_TYPE_TURBO3_TCQ:
            turbo_tcq_load_kv_decode();          // idempotent codebook staging (K+V)
            k_turbo3_tcq_dequant_f16_inv_fwht<<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1, alpha, iv);
            break;
        case GGML_TYPE_TURBO2_TCQ:
            turbo2_tcq_load_kv_decode();
            k_turbo2_tcq_dequant_f16_inv_fwht<<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1, alpha, iv);
            break;
        case GGML_TYPE_TURBO1_TCQ:
            turbo1_tcq_load_cb_fattn();
            // NOTE (post rotated-V, 2026-07-02): the STORED bits never changed — only the decode
            // split moved (fused loader is LUT-only + graph un-rotates). This kernel still does
            // LUT + inverse FWHT -> ORIGINAL domain, which is exactly this function's contract.
            k_turbo1_tcq_dequant_f16<<<grid, 128, 0, stream>>>(
                src, scratch_f16, ne0, n_cells, 1, nb1, nb1, nb1, alpha, iv);
            break;
        case GGML_TYPE_F16:
            // f16 entry tier: plain original-domain rows; gather only (alpha unused: the f16
            // path has no TCQ decode alpha, and the epilogue's /alpha_v(B) still applies)
            k_vbr_f16_gather<<<grid, 128, 0, stream>>>(src, scratch_f16, ne0, n_cells, nb1);
            break;
        default:
            GGML_ABORT("vbr_dequant_turbo_to_f32: unsupported src_type %d", (int) src_type);
    }
    // f16 -> f32, dividing V by the TARGET tier's decode-alpha so the later decode(B)'s *alpha_v(B)
    // reproduces A's reconstruction. For A->A this makes the round-trip byte-exact; K is unaffected
    // (alpha 1.0). The dequant above already applied A's natural decode alpha.
    const float vscale = is_v ? 1.0f / vbr_alpha_v_decode(type_B, n_cells) : 1.0f;
    const int64_t n_elem  = n_cells * ne0;
    const int64_t threads = 256;
    const int64_t blocks  = (n_elem + threads - 1) / threads;
    k_vbr_f16_to_f32_scaled<<<(unsigned) blocks, (unsigned) threads, 0, stream>>>(scratch_f16, dst_f32, n_elem, vscale);
}

// Round n_kv up so the materialize buffers below grow in O(log n) amortized steps instead of
// reallocating every ubatch as a conversation's attended range creeps forward.
static int64_t next_pow2_i64(int64_t x) {
    int64_t n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

// Grow (or first-allocate) one side's f16 dequant scratch to hold >= need_bytes, and return its
// base device pointer. Primary path: a VMM pool (contiguous VA reserved once, physical 2 MiB
// pages mapped on demand) — fragmentation cannot fail its growth, and we map to the EXACT
// attended width so the ~2x pow2 over-allocation the cudaMalloc path carried is handed back to
// the VBR KV budget as free VRAM. Fallback (no VMM support): plain cudaMalloc, sized to next_pow2
// to keep reallocs O(log n).
//
// Capture safety: this only ever *grows* (maps pages / reallocs) when need_bytes increases, which
// requires the K/V view ne to change. A ggml CUDA-graph capture begins only after the ne has been
// unchanged for two consecutive calls (warmup), so every growth lands in an eager (non-capture)
// pass; during the capture pass need_bytes is stable and map() finds every chunk already mapped,
// issuing no cuMem*/cudaMalloc under stream capture. This mirrors the invariant the prior
// cudaMalloc-in-decode path relied on. A VA re-reservation changes the base pointer, but only on a
// growth (eager) pass, and the graph's per-node src-ne memcmp independently forces a re-capture
// that bakes in the new pointer — for the graph whose OWN K/V view grew. A co-resident context's
// captured graph on the same device sees no ne change, so address moves also bump
// fattn_scratch_epoch, which fences ALL captured graphs (see fattn.cuh).
static half * kv_dequant_scratch_try(int device, size_t need_bytes,
                                     ggml_vbr_vmm_pool ** vmm, size_t * vmm_va, size_t * vmm_hw,
                                     half ** cuda_buf, size_t * cuda_size) {
    // Steady-state fast-out: the watermark is 256-cell padded, so need_bytes is unchanged for
    // 255 of every 256 boundaries — skip the device set + chunk-set walk entirely.
    if (*vmm != nullptr && need_bytes <= *vmm_hw) {
        return (half *) ggml_backend_cuda_vmm_pool_base(*vmm);
    }
    ggml_cuda_set_device(device); // fresh maps memset on the current device's legacy stream;
                                  // the cudaMalloc fallback allocates on the current device
    if (ggml_backend_cuda_vmm_available(device)) {
        if (*vmm == nullptr || need_bytes > *vmm_va) {
            if (*vmm) {
                ggml_backend_cuda_vmm_pool_free(*vmm);
                *vmm = nullptr;
                *vmm_va = 0;
                *vmm_hw = 0;
                fattn_scratch_epoch[device]++; // old base dies with the reservation, even if
                                               // the re-reserve below fails
            }
            const size_t va = (size_t) next_pow2_i64((int64_t) need_bytes);
            *vmm = ggml_backend_cuda_vmm_pool_init(device, va);
            if (*vmm) {
                *vmm_va = va;
                fattn_scratch_epoch[device]++; // base moved with the new reservation
            }
        }
        if (*vmm) {
            if (!ggml_backend_cuda_vmm_pool_map(*vmm, 0, need_bytes)) {
                return nullptr; // physical exhaustion — caller decides (reserve fails
                                // recoverably; the in-graph wrapper below aborts)
            }
            *vmm_hw = need_bytes;
            return (half *) ggml_backend_cuda_vmm_pool_base(*vmm);
        }
        // VA reservation failed — fall through to the cudaMalloc path.
    }
    const size_t alloc = (size_t) next_pow2_i64((int64_t) need_bytes);
    if (alloc > *cuda_size) {
        if (*cuda_buf) {
            CUDA_CHECK(cudaFree(*cuda_buf));
            *cuda_buf  = nullptr;
            *cuda_size = 0;
            fattn_scratch_epoch[device]++; // old address gone even if the grow below fails
        }
        half * p = nullptr;
        if (cudaMalloc(&p, alloc) != cudaSuccess) {
            (void) cudaGetLastError(); // clear the OOM so later CUDA_CHECKs don't trip on it
            return nullptr;
        }
        *cuda_buf  = p;
        *cuda_size = alloc;
    }
    return *cuda_buf;
}

static half * kv_dequant_scratch(int device, size_t need_bytes,
                                 ggml_vbr_vmm_pool ** vmm, size_t * vmm_va, size_t * vmm_hw,
                                 half ** cuda_buf, size_t * cuda_size) {
    half * buf = kv_dequant_scratch_try(device, need_bytes, vmm, vmm_va, vmm_hw, cuda_buf, cuda_size);
    if (buf == nullptr) {
        // Expected unreachable: the boundary-time reserve below grows the scratch to the
        // watermark before every batch, so a mid-graph grow should always find its pages mapped.
        GGML_ABORT("VBR f16 dequant scratch: physical memory exhausted growing to %.1f MiB on "
                   "device %d (boundary reserve missed)", need_bytes / 1048576.0, device);
    }
    return buf;
}

// Boundary-time scratch reserve (ggml_vbr_backend_iface): grow both sides' f16 dequant scratch
// OUTSIDE graph execution, so the mid-graph growers above always find their pages mapped and
// physical exhaustion surfaces here — recoverably — instead of aborting mid-decode. Called by
// the KV cache at the llama_decode boundary with post-degrade-wave watermark sizes; also keeps
// every scratch address move in an eager pass by construction (CUDA-graph capture safety).
bool ggml_backend_cuda_kv_dequant_scratch_reserve(int device, size_t k_bytes, size_t v_bytes) {
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    if (k_bytes > 0 && kv_dequant_scratch_try(device, k_bytes,
                                              &kv_dequant_k_vmm[device], &kv_dequant_k_vmm_va[device], &kv_dequant_k_vmm_hw[device],
                                              &kv_dequant_k_buf[device], &kv_dequant_k_buf_size[device]) == nullptr) {
        return false;
    }
    if (v_bytes > 0 && kv_dequant_scratch_try(device, v_bytes,
                                              &kv_dequant_v_vmm[device], &kv_dequant_v_vmm_va[device], &kv_dequant_v_vmm_hw[device],
                                              &kv_dequant_v_buf[device], &kv_dequant_v_buf_size[device]) == nullptr) {
        return false;
    }
    return true;
}

static void ggml_cuda_turbo_prefill_attend(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    load_tcq_decode_alpha(ctx.device);
    cudaStream_t stream = ctx.stream();
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const bool turbo_k = ggml_is_turbo_kv_type(K->type);
    const bool turbo_v = ggml_is_turbo_kv_type(V->type);
    // Which sides materialize to f16: the single authoritative predicate (ggml-vbr.h). A mixed
    // q8_0/bf16 side must be dequanted too, or its raw bytes are handed to the f16-only MMA
    // kernel below and read as f16 → garbage; f16 sides need no conversion. This path is only
    // dispatched when at least one side is turbo, so the predicate's "q8_0/bf16 next to a turbo
    // partner" clause is exactly the old unconditional q8_0/bf16 dequant here.
    bool mat_k = false;
    bool mat_v = false;
    ggml_vbr_kv_dequant_sides(K->type, V->type, &mat_k, &mat_v);

    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    half * k_fp16 = nullptr;
    half * v_fp16 = nullptr;

    // Allocate and dequant K to fp16 (turbo2/3/4, q8_0, or bf16)
    if (mat_k) {
        // Size for the CURRENT attended KV range (this call's view), not the full root/padded
        // capacity: the dequant kernels below and the K_f16 strides further down both index
        // densely from 0 over exactly K->ne[1]/ne[2]/ne[3] — nothing here depends on where in
        // the root cache this view sits (front-loading a root-sized allocation starved the VBR
        // VRAM budget early and used to OOM mid-prefill on deep contexts). The VMM-backed path
        // maps to this exact width; the cudaMalloc fallback rounds up to next_pow2 internally.
        const size_t k_size = (size_t) K->ne[0] * (size_t) K->ne[1] * (size_t) K->ne[2] * (size_t) K->ne[3] * sizeof(half);
        k_fp16 = kv_dequant_scratch(device, k_size, &kv_dequant_k_vmm[device], &kv_dequant_k_vmm_va[device], &kv_dequant_k_vmm_hw[device],
                                    &kv_dequant_k_buf[device], &kv_dequant_k_buf_size[device]);
        dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
        if (K->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
            {
                static bool tcq_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq_hot_k = -1; if (tcq_hot_k < 0) tcq_hot_k = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq_fattn_k_cb_loaded[device] || tcq_hot_k) {
                    tcq_fattn_k_cb_loaded[device] = true;
                    turbo_tcq_load_kv_decode();
                }
            }
            // TCQ K quality is sensitive to where the FWHT roundtrip is rounded. Decode to the
            // original domain here, matching the decode-side dequant path, instead of rotating Q
            // and consuming rotated-domain K in the F16 MMA kernel.
            k_turbo3_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
            {
                static bool tcq2_fattn_k_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq2_hot_k = -1; if (tcq2_hot_k < 0) tcq2_hot_k = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq2_fattn_k_cb_loaded[device] || tcq2_hot_k) {
                    tcq2_fattn_k_cb_loaded[device] = true;
                    turbo2_tcq_load_kv_decode();
                }
            }
            k_turbo2_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
        } else if (K->type == GGML_TYPE_TURBO4_0) {
            // turbo4 K: inverse FWHT dequant → produces K in original domain (no Q rotation needed)
            k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO8_0) {
            // turbo8 K: inverse FWHT dequant → produces K in original domain (no Q rotation needed)
            k_turbo8_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else if (K->type == GGML_TYPE_TURBO1_TCQ) {
            k_turbo1_tcq_dequant_f16<<<grid_k, 128, 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k, 0);
        } else if (K->type == GGML_TYPE_BF16) {
            // bf16 K (mixed K/V): cast to f16 in original (unrotated) domain. Q stays unrotated.
            k_bf16_to_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        } else {
            // q8_0 K (mixed K/V): plain dequant to f16 in original (unrotated) domain. Q stays
            // unrotated (turbo_k is false → the Q-rotation block below is skipped).
            k_q8_0_dequant_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                (const char *)K->data, k_fp16, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
        }
    }

    // Allocate and dequant V to fp16 (turbo2/3/4, q8_0, or bf16)
    if (mat_v) {
        // Size for the CURRENT attended KV range (this call's view) — see the matching comment
        // on the K-side buffer above for why root-sizing was wrong and how the scratch is backed.
        const size_t v_size = (size_t) V->ne[0] * (size_t) V->ne[1] * (size_t) V->ne[2] * (size_t) V->ne[3] * sizeof(half);
        v_fp16 = kv_dequant_scratch(device, v_size, &kv_dequant_v_vmm[device], &kv_dequant_v_vmm_va[device], &kv_dequant_v_vmm_hw[device],
                                    &kv_dequant_v_buf[device], &kv_dequant_v_buf_size[device]);
        dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
        if (V->type == GGML_TYPE_TURBO2_0) {
            k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO3_0) {
            k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], 1);
        } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
            // Runtime codebook loading for 3-bit V decode (in case K is a different type)
            {
                static bool tcq_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq_hot_v = -1; if (tcq_hot_v < 0) tcq_hot_v = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq_fattn_v_cb_loaded[device] || tcq_hot_v) {
                    tcq_fattn_v_cb_loaded[device] = true;
                    turbo_tcq_load_kv_decode();
                }
            }
            k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
        } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
            // Runtime codebook loading for 2-bit V decode (in case K is a different type)
            {
                static bool tcq2_fattn_v_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
                static int tcq2_hot_v = -1; if (tcq2_hot_v < 0) tcq2_hot_v = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
                if (!tcq2_fattn_v_cb_loaded[device] || tcq2_hot_v) {
                    tcq2_fattn_v_cb_loaded[device] = true;
                    turbo2_tcq_load_kv_decode();
                }
            }
            k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
        } else if (V->type == GGML_TYPE_TURBO4_0) {
            k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO8_0) {
            k_turbo8_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else if (V->type == GGML_TYPE_TURBO1_TCQ) {
            // ROTATED-domain V (2026-07-02): graph-level ggml_turbo_wht(inverse) un-rotates the
            // attention output, same contract as turbo2/3_tcq. Must stay in lockstep with the
            // turbo_v_rotated set in llama-graph.cpp.
            k_turbo1_tcq_dequant_f16_rot<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
        } else if (V->type == GGML_TYPE_BF16) {
            // bf16 V (mixed K/V): cast to f16 in original domain. Not rotated, and the graph-level
            // inverse WHT is gated on V being a turbo type (llama-graph.cpp), so none is applied.
            k_bf16_to_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        } else {
            // q8_0 V (mixed K/V): plain dequant to f16 in original domain. q8_0 V is not rotated,
            // and the graph-level inverse WHT is gated on V being a turbo type (llama-graph.cpp),
            // so no spurious un-rotation is applied here.
            k_q8_0_dequant_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                (const char *)V->data, v_fp16, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
        }
    }

    // Create fp16 tensor copies on stack
    ggml_tensor K_f16 = *K;
    ggml_tensor V_f16 = *V;

    if (k_fp16) {
        K_f16.type = GGML_TYPE_F16;
        K_f16.data = k_fp16;
        K_f16.nb[0] = sizeof(half);
        K_f16.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        K_f16.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        K_f16.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
    }

    if (v_fp16) {
        V_f16.type = GGML_TYPE_F16;
        V_f16.data = v_fp16;
        V_f16.nb[0] = sizeof(half);
        V_f16.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
        V_f16.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
        V_f16.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
    }

    // Rotate Q for turbo pre-rotate-queries (only when K is in rotated space).
    // turbo4/turbo8 and TCQ K are dequanted via inverse FWHT -> original domain, so Q stays unrotated.
    const ggml_tensor * Q = dst->src[0];
    float * q_rotated = nullptr;
    if (turbo_k &&
            K->type != GGML_TYPE_TURBO4_0 &&
            K->type != GGML_TYPE_TURBO8_0 &&
            K->type != GGML_TYPE_TURBO3_TCQ &&
            K->type != GGML_TYPE_TURBO2_TCQ &&
            K->type != GGML_TYPE_TURBO1_TCQ &&
            Q->ne[0] % 128 == 0) {
        const size_t q_size = ggml_nelements(Q) * sizeof(float);
        q_rot_buf_ensure(device, q_size);
        q_rotated = q_rot_buf[device];
        const int64_t n_q_groups = ggml_nelements(Q) / 128;
        k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
            (const float *)Q->data, q_rotated, ggml_nelements(Q));
    }

    // Temporarily swap src pointers to fp16 K/V and rotated Q
    ggml_tensor * orig_q = dst->src[0];
    ggml_tensor * orig_k = dst->src[1];
    ggml_tensor * orig_v = dst->src[2];

    ggml_tensor Q_rot;
    if (q_rotated) {
        Q_rot = *Q;
        Q_rot.data = q_rotated;
        dst->src[0] = &Q_rot;
    }
    dst->src[1] = k_fp16 ? &K_f16 : orig_k;
    dst->src[2] = v_fp16 ? &V_f16 : orig_v;
    std::atomic_signal_fence(std::memory_order_seq_cst);

    // Dispatch to MMA kernel (sees rotated Q, fp16 K/V, uses tensor cores)
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);

    // Restore original tensor pointers
    dst->src[0] = orig_q;
    dst->src[1] = orig_k;
    dst->src[2] = orig_v;

    // K/V fp16 buffers are persistent (grow-only), no free needed
}

#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            ggml_cuda_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                        \
    }                                                                                                            \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \

#define FATTN_VEC_CASES_ALL_D_512(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

static void ggml_cuda_flash_attn_ext_vec(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)

    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO2_TCQ)
#else
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO3_TCQ)
    FATTN_VEC_CASES_ALL_D_512(GGML_TYPE_Q8_0,       GGML_TYPE_TURBO2_TCQ)
#endif // GGML_CUDA_FA_ALL_QUANTS

    GGML_ABORT("fatal error");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_WMMA_F16 = 300,
    BEST_FATTN_KERNEL_MMA_F16  = 400,
};

static bool ggml_cuda_fattn_kv_type_supported(ggml_type type) {
    // TurboQuant KV types (turbo2/3/4/8_0, turbo3/2/1_tcq) are handled by the fork's
    // turbo FA paths (VEC early-return / dequant-to-f16); without this FA would silently
    // report unsupported and llama.cpp would fall back to non-FA attention (garbage with turbo V).
    if (ggml_is_turbo_kv_type(type)) {
        return true;
    }
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
            return true;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
            return false;
#endif // GGML_CUDA_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

static best_fattn_kernel ggml_cuda_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef FLASH_ATTN_AVAILABLE
    GGML_UNUSED(device); GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// FLASH_ATTN_AVAILABLE

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    // The effective batch size for the kernel can be increased by gqa_ratio.
    // The kernel versions without this optimization are also used for ALiBi, if there is no mask, or if the KV cache is not padded,
    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    const int cc = ggml_cuda_info().devices[device].cc;

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 192:
            if (V->ne[0] != 128 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 8 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 320:
            if (V->ne[0] != 256 || !gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (gqa_ratio % 32 != 0) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

#ifndef GGML_CUDA_FA_ALL_QUANTS
    {
        auto is_turbo_type = [](ggml_type t) {
            return ggml_is_turbo_kv_type(t);
        };
        // Asymmetric turbo K/V (e.g. q8_0 K + turbo4 V) is handled by the turbo dequant-to-f16
        // FA paths (ggml_cuda_turbo_prefill_attend for prefill, do_decode_dequant for decode).
        // Without this exemption the upstream K!=V guard reports FA unsupported, so llama.cpp
        // falls back to the non-FA attention path, which mishandles turbo4 V (garbage output).
        if (K->type != V->type && !is_turbo_type(K->type) && !is_turbo_type(V->type)) {
            return BEST_FATTN_KERNEL_NONE;
        }
    }
#endif // GGML_CUDA_FA_ALL_QUANTS

    if (!ggml_cuda_fattn_kv_type_supported(K->type) || !ggml_cuda_fattn_kv_type_supported(V->type)) {
        return BEST_FATTN_KERNEL_NONE;
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // For small batch sizes the vector kernel may be preferable over the kernels optimized for large batch sizes:
    // TurboQuant: only the vec kernel has native turbo dequant support.
    if (ggml_is_turbo_kv_type(K->type) || ggml_is_turbo_kv_type(V->type)) {
        if (Q->ne[0] <= 512 && Q->ne[0] % 64 == 0)
            return BEST_FATTN_KERNEL_VEC;
        return BEST_FATTN_KERNEL_NONE;
    }

    // D=512: MMA/TILE templates don't support this head_dim, use VEC unconditionally
    if (Q->ne[0] == 512) {
        return BEST_FATTN_KERNEL_VEC;
    }

    // 192 satisfies % 64 == 0 but has no vec instance (DKQ != DV); force it onto the MMA path.
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % 64 == 0 && Q->ne[0] != 192 && K->ne[1] % FATTN_KQ_STRIDE == 0;

    // If Turing tensor cores are available, use them:
    if (turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel) {
            if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE && Q->ne[1] == 1 && Q->ne[3] == 1 && !(gqa_ratio > 4 && K->ne[1] >= 8192)) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            } else {
                if (cc >= GGML_CUDA_CC_ADA_LOVELACE) {
                    if (Q->ne[1] <= 2) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                } else {
                    if (Q->ne[1] == 1) {
                        return BEST_FATTN_KERNEL_VEC;
                    }
                }
            }
            if (!gqa_opt_applies && Q->ne[1] == 1) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    const int ncols2_max = Q->ne[0] == 320 ? 32 : ((Q->ne[0] == 576 || Q->ne[0] == 192) ? 16 : 8);
    int gqa_ratio_eff = 1;
    while (gqa_ratio % (2*gqa_ratio_eff) == 0 && gqa_ratio_eff < ncols2_max) {
        gqa_ratio_eff *= 2;
    }

    if (volta_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if (can_use_vector_kernel && Q->ne[1] * gqa_ratio_eff <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        if (Q->ne[1] * gqa_ratio_eff <= 16) {
            return BEST_FATTN_KERNEL_TILE; // On Volta tensor cores are only faster for sufficiently large matrices.
        }
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // Use the WMMA kernel if possible:
    if (ggml_cuda_should_use_wmma_fattn(cc) && K->ne[1] % FATTN_KQ_STRIDE == 0 && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[0] != 192 && Q->ne[0] != 512 && Q->ne[0] != 576) {
        if (can_use_vector_kernel && Q->ne[1] <= 2) {
            return BEST_FATTN_KERNEL_VEC;
        }
        return BEST_FATTN_KERNEL_WMMA_F16;
    }

    // AMD MFMA needs a certain minimum batch size to outscale the tile kernel for large head sizes.
    if ((amd_mfma_available(cc) && Q->ne[0] <= 256) && Q->ne[0] != 40 && Q->ne[0] != 72) {
        if ((Q->ne[0] <= 64 && Q->ne[1] * gqa_ratio_eff > 8)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 16)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
        if ((Q->ne[0] <= 256 && Q->ne[1] * gqa_ratio_eff > 64)) {
            return BEST_FATTN_KERNEL_MMA_F16;
        }
    }

    // AMD WMMA is always faster than the tile kernel if the full tile width of 16 can be utilized.
    if ((amd_wmma_available(cc) && gqa_opt_applies && Q->ne[0] <= 128) && Q->ne[0] != 40 && Q->ne[0] != 72 && Q->ne[1] * gqa_ratio_eff > 8) {
        return BEST_FATTN_KERNEL_MMA_F16;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    return BEST_FATTN_KERNEL_TILE;
}

// TURBO_CERT_DUMP_Q=<dir>: dump raw post-RoPE query vectors per (layer, head) as fp32 to
// <dir>/q_l%03d_h%02d_w%d.f32. Codec-independent; fires on every flash-attn node. Layer parsed
// from the "__fattn__-<il>" node name. Q src after permute(0,2,1,3): ne0=head_dim (contiguous),
// ne1=n_tokens, ne2=n_head. For product-aware codebook design: gives Sigma_Q per head offline.
static void cert_dump_q(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    static int qstate = 0; static char qdir[1024] = {0}; static int64_t qcap = 4096;
    static int64_t q_done[160][64] = {};
    if (qstate == 0) {
        const char * e = getenv("TURBO_CERT_DUMP_Q");
        if (!e || !e[0]) { qstate = -1; return; }
        strncpy(qdir, e, sizeof(qdir) - 1);
        if (const char * r = getenv("TURBO_CERT_DUMP_Q_ROWS")) qcap = atoll(r);
        qstate = 1;
        fprintf(stderr, "CERT_DUMP_Q: dir=%s cap=%lld rows per (layer,head)\n", qdir, (long long)qcap);
    }
    if (qstate == -1) return;
    const ggml_tensor * q = dst->src[0];
    if (!q || (q->type != GGML_TYPE_F32 && q->type != GGML_TYPE_F16)) return;
    const char * dash = strrchr(dst->name, '-');
    if (!dash) return;
    const int layer = atoi(dash + 1);
    if (layer < 0 || layer >= 160) return;
    const int64_t hd = q->ne[0], ntok = q->ne[1], nhead = q->ne[2];
    if (hd <= 0 || nhead <= 0 || nhead > 64) return;
    cudaStreamSynchronize(ctx.stream());
    std::vector<float> buf((size_t)hd);
    std::vector<char> raw((size_t)hd * (q->type == GGML_TYPE_F16 ? 2 : 4));
    for (int64_t h = 0; h < nhead; h++) {
        int64_t & done = q_done[layer][h];
        if (done >= qcap) continue;
        char path[1200];
        snprintf(path, sizeof(path), "%s/q_l%03d_h%02d_w%lld.f32", qdir, layer, (int)h, (long long)hd);
        FILE * f = fopen(path, done == 0 ? "wb" : "ab");
        if (!f) { qstate = -1; return; }
        for (int64_t t = 0; t < ntok && done < qcap; t++) {
            const char * src = (const char *)q->data + t * q->nb[1] + h * q->nb[2];
            cudaMemcpy(raw.data(), src, raw.size(), cudaMemcpyDeviceToHost);
            if (q->type == GGML_TYPE_F16) {
                const uint16_t * hh = (const uint16_t *)raw.data();
                for (int64_t j = 0; j < hd; j++) buf[j] = __half2float(*(const __half *)&hh[j]);
            } else {
                memcpy(buf.data(), raw.data(), (size_t)hd * 4);
            }
            fwrite(buf.data(), sizeof(float), (size_t)hd, f);
            done++;
        }
        fclose(f);
    }
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(K != nullptr);
    GGML_ASSERT(V != nullptr);

    const best_fattn_kernel kernel = ggml_cuda_get_best_fattn_kernel(device, dst);

    bool need_f16_K = false;
    bool need_f16_V = false;

    switch (kernel) {
        case BEST_FATTN_KERNEL_TILE:
        case BEST_FATTN_KERNEL_WMMA_F16:
        case BEST_FATTN_KERNEL_MMA_F16:
            need_f16_K = true;
            need_f16_V = true;
            break;
        case BEST_FATTN_KERNEL_VEC:
            need_f16_K = K->type == GGML_TYPE_F32;
            need_f16_V = V->type == GGML_TYPE_F32;
            break;
        case BEST_FATTN_KERNEL_NONE:
            break;
    }

    const ggml_cuda_flash_attn_ext_f16_extra_data f16_extra =
        ggml_cuda_flash_attn_ext_get_f16_extra_data(dst, need_f16_K, need_f16_V);

    return f16_extra.end - (uintptr_t) dst->data;
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    turbo_vanilla_cb_load_fattn();  // TURBO_CB_T2/3/4/8 vanilla-book override (this TU's copies)
    turbo1_tcq_load_cb_fattn();  // E7: turbo1_tcq K/V decode codebooks (TURBO1_TCQ_CB_K/_V, cold-start = turbo2 anchor)


    cert_dump_q(ctx, dst);

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // Turbo prefill: dequant to fp16 and use tensor core MMA for batched attention.
    // turbo4 K uses inverse FWHT during dequant — mixes centroids in float32 shmem before
    // fp16 cast, so precision is fine. turbo2/turbo3 use simple centroid×norm dequant.
    // Set TURBO_PREFILL_VEC=1 to force vec kernel for all turbo types (debug override).
    static const bool turbo_prefill_vec = [] {
        const char * e = getenv("TURBO_PREFILL_VEC");
        if (e) fprintf(stderr, "TURBO_PREFILL_VEC=%s: forcing vec prefill for turbo types\n", e);
        return e != nullptr;
    }();
    const bool turbo_kv = ggml_is_turbo_kv_type(K->type) || ggml_is_turbo_kv_type(V->type);

    // Fused MMA turbo: reads raw turbo bytes directly in the MMA kernel, no intermediate fp16 buffers.
    // Phase 1: turbo4_0 matched K/V at D=128. Set GGML_TURBO_MMA_FUSED=0 to disable.
    static const bool turbo_mma_fused = [] {
        const char * e = getenv("GGML_TURBO_MMA_FUSED");
        if (e && atoi(e) == 0) {
            fprintf(stderr, "GGML_TURBO_MMA_FUSED=0: fused turbo MMA kernel disabled\n");
            return false;
        }
        return true;
    }();
    // turbo1_tcq handling below; removed 1-bit variants have no instances (dequant-to-f16 codec only); exclude it so it
    // does not enter the fused path and fall through with no kernel launched.
    const bool turbo_matched = K->type == V->type && turbo_kv && K->type != GGML_TYPE_TURBO1_TCQ;
    // fused t1 decode is not InnerQ-scale-aware — under calibrated scales t1 must take the
    // materialize path (which uses the pushed symbols); see turbo-quant-cuda.cuh calibration
    extern bool g_turbo_innerq_calibrated;
    const bool t1_fused_ok = !g_turbo_innerq_calibrated;
    const bool turbo1_tcq_matched = K->type == GGML_TYPE_TURBO1_TCQ && V->type == GGML_TYPE_TURBO1_TCQ &&
                                    (Q->ne[0] == 128 || Q->ne[0] == 256) && t1_fused_ok;
    // Asymmetric fused pairs, D=256 only (dense Qwen geometry). Both sides stay WHT-rotated like
    // the matched path (Q pre-rotated below, V un-rotated at graph level). The set = the q6 sweet
    // spot (t8k/t4v) + every ADJACENT-TIER pair the dynamic VBR degrade ladder can create: the
    // controller moves K and V of a layer independently but bands complete in order, so a live
    // mixed layer only ever holds adjacent tiers. Uncovered pairs fall to the materialize path
    // (measured -13-15% tg32 @ d8192), which band transits would sit in for thousands of tokens.
    auto turbo_fused_asym_pair = [](ggml_type k, ggml_type v) {
        return (k == GGML_TYPE_TURBO8_0   && v == GGML_TYPE_TURBO4_0)   ||
               (k == GGML_TYPE_TURBO4_0   && v == GGML_TYPE_TURBO8_0)   ||
               (k == GGML_TYPE_TURBO4_0   && v == GGML_TYPE_TURBO3_TCQ) ||
               (k == GGML_TYPE_TURBO3_TCQ && v == GGML_TYPE_TURBO4_0)   ||
               (k == GGML_TYPE_TURBO3_TCQ && v == GGML_TYPE_TURBO2_TCQ) ||
               (k == GGML_TYPE_TURBO2_TCQ && v == GGML_TYPE_TURBO3_TCQ) ||
               (k == GGML_TYPE_TURBO2_TCQ && v == GGML_TYPE_TURBO1_TCQ) ||
               (k == GGML_TYPE_TURBO1_TCQ && v == GGML_TYPE_TURBO2_TCQ) ||
               // f16<->t8: the VBR entry band (f16 is the tier above t8). Was missing, so every
               // f16->t8 straddle ate the materialize path (grows to -50% @ d64k). f16 side is free.
               (k == GGML_TYPE_TURBO8_0   && v == GGML_TYPE_F16)        ||
               (k == GGML_TYPE_F16        && v == GGML_TYPE_TURBO8_0);
    };
    const bool turbo_fused_asym = turbo_fused_asym_pair(K->type, V->type) && Q->ne[0] == 256 &&
        (t1_fused_ok || (K->type != GGML_TYPE_TURBO1_TCQ && V->type != GGML_TYPE_TURBO1_TCQ));
    // TURBO_FUSED_PREFILL=1 (experiment knob): route BATCHED attention through the fused MMA path
    // (ncols1 instances up to 64 exist). ⚠ MEASURED A LOSS (2026-07-03, 27B/3090, pp512): ~neutral
    // at d0 but −6% (t8/t4) to −11% (t3/t1_tcq) at d8192 — re-decoding the K/V tile once per
    // 64-column block swamps the DRAM savings vs the materialize path's decode-once f16 round
    // trip, and the win the codecs DO get from materialize grows with depth. Keep default OFF;
    // the knob stays for future probing (e.g. if a shared-tile multi-column loader lands).
    static const int turbo_fused_prefill = [] {
        const char * e = getenv("TURBO_FUSED_PREFILL");
        return e ? atoi(e) : 0;
    }();
    if (turbo_mma_fused && (turbo_matched || turbo_fused_asym || turbo1_tcq_matched) && (Q->ne[1] <= 4 || turbo_fused_prefill) &&
        (Q->ne[0] == 128 || Q->ne[0] == 256) &&
        (turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) ||
         // AMD RDNA WMMA: trying D=128 AND D=256 (gemma) after lifting the upstream DKQ<=128 cap.
         amd_wmma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc))) {
        cudaStream_t stream = ctx.stream();
        int device;
        CUDA_CHECK(cudaGetDevice(&device));

        // Load TCQ codebooks for fused kernel (constant memory used by inline dequant)
        if (K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ) {
            static bool tcq3_fused_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq_hot_f = -1; if (tcq_hot_f < 0) tcq_hot_f = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq3_fused_loaded[device] || tcq_hot_f) {
                tcq3_fused_loaded[device] = true;
                turbo_tcq_load_kv_decode();
            }
        }
        if (K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
            static bool tcq2_fused_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq2_hot_f = -1; if (tcq2_hot_f < 0) tcq2_hot_f = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq2_fused_loaded[device] || tcq2_hot_f) {
                tcq2_fused_loaded[device] = true;
                turbo2_tcq_load_kv_decode();
            }
        }

        // Pre-rotate Q: ALL fused turbo K types (incl. turbo1_tcq since 2026-07-02) keep K in the
        // WHT-rotated domain — the stored trellis/LUT coeffs are consumed directly and Q is
        // rotated once to match. Only V is decoded to the original domain inside the loader.
        ggml_tensor Q_rot_fused;
        ggml_tensor * orig_q_fused = nullptr;
        // f16 K is stored in the ORIGINAL (unrotated) domain, so Q must NOT be WHT-rotated for it
        // (rotated-Q . unrotated-K would be wrong). Turbo K is stored rotated → rotate Q to match.
        // Only K drives Q rotation; V is always decoded to the original domain inside the loader.
        const bool fused_k_original_domain = (K->type == GGML_TYPE_F16);
        if (!fused_k_original_domain && Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            q_rot_buf_ensure(device, q_size);
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device], ggml_nelements(Q));
            Q_rot_fused = *Q;
            Q_rot_fused.data = q_rot_buf[device];
            orig_q_fused = dst->src[0];
            dst->src[0] = &Q_rot_fused;
        }

        // TCQ decode alpha for V dequant: the FUSED kernel reads the alpha from its own template
        // instance's __constant__ copy, which ggml_cuda_flash_attn_ext_mma_turbo_case sets (once
        // per device, from the constant TURBO_TCQ_ALPHA_V_T*). The old per-launch context-adaptive
        // cudaMemcpyToSymbol here targeted fattn.cu's SEPARATE TU copy — dead for the fused path,
        // and an unchecked sync memcpy that is illegal during CUDA/HIP graph capture (ROCm latches
        // it → "previous error during capture"). Keep only the once-guarded env/static setup.
        if (V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO1_TCQ) {
            load_tcq_decode_alpha(device);
        }

#define TURBO_FUSED_DISPATCH(tK, tV) \
        if (K->type == tK && V->type == tV) { \
            if (Q->ne[0] == 128) \
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<128, 128, tK, tV>(ctx, dst); \
            else \
                ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, tK, tV>(ctx, dst); \
        }
#define TURBO_FUSED_DISPATCH_ASYM(tK, tV) \
        if (K->type == tK && V->type == tV) { \
            ggml_cuda_flash_attn_ext_mma_turbo_switch_ncols2<256, 256, tK, tV>(ctx, dst); \
        }
        // Asymmetric pairs, D=256 only (no D=128 instances): the q6 sweet spot + the VBR
        // adjacent-tier set (see turbo_fused_asym_pair above).
        TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO4_0)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO8_0)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO3_TCQ)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO4_0)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO2_TCQ)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO3_TCQ)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO1_TCQ)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO2_TCQ)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_TURBO8_0,   GGML_TYPE_F16)
        else TURBO_FUSED_DISPATCH_ASYM(GGML_TYPE_F16,        GGML_TYPE_TURBO8_0)
#undef TURBO_FUSED_DISPATCH_ASYM
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO4_0,   GGML_TYPE_TURBO4_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO8_0,   GGML_TYPE_TURBO8_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO3_TCQ, GGML_TYPE_TURBO3_TCQ)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO2_TCQ, GGML_TYPE_TURBO2_TCQ)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO1_TCQ, GGML_TYPE_TURBO1_TCQ)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO3_0,   GGML_TYPE_TURBO3_0)
        else TURBO_FUSED_DISPATCH(GGML_TYPE_TURBO2_0,   GGML_TYPE_TURBO2_0)
#undef TURBO_FUSED_DISPATCH

        if (orig_q_fused) dst->src[0] = orig_q_fused;
        return;
    }

    if (turbo_kv && !turbo_prefill_vec && Q->ne[1] > 1 && Q->ne[0] <= 256 &&
        (turing_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc) ||
         // AMD RDNA WMMA: trying D=256 prefill too after lifting the upstream DKQ<=128 cap.
         amd_wmma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc))) {
        // Prefill path: turbo4 K uses inverse FWHT dequant (original domain, no Q rotation),
        // turbo2/3 K uses simple dequant (rotated domain, Q pre-rotated). V un-rotation at graph level.
        ggml_cuda_turbo_prefill_attend(ctx, dst);
    } else {
        load_tcq_decode_alpha(ctx.device);

        // Update VEC/materialize __constant__ alpha. Context-adaptive alpha would re-push every
        // token, but a synchronous cudaMemcpyToSymbol during graph capture is illegal (ROCm:
        // hipErrorStreamCaptureImplicit → the memcpy uses the legacy stream and invalidates the
        // whole capture). Push ONCE per device on the first (eager, pre-capture) call and freeze —
        // matching the fused path, which already uses a fixed per-instance alpha, and matching what
        // graph capture does anyway (the captured graph freezes whatever alpha was set at capture
        // time). gemma4 ISWA routes some D=256 layers through this branch during graph-captured
        // decode, so this MUST be capture-safe. (On NVIDIA those layers fuse, so this never fired.)
        static bool vec_alpha_pushed[GGML_CUDA_MAX_DEVICES] = {};
        if (d_tcq_decode_alpha_v_static == 0.0f && !vec_alpha_pushed[ctx.device] &&
            (V->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO2_TCQ)) {
            float alpha = tcq_compute_alpha_v(V->type, V->ne[1]);
            cudaMemcpyToSymbol(d_tcq_decode_alpha_v_fattn, &alpha, sizeof(float));
            vec_alpha_pushed[ctx.device] = true;
        }

        // Load runtime codebooks for TCQ types (needed by both dequant and native VEC paths)
        if (K->type == GGML_TYPE_TURBO3_TCQ || V->type == GGML_TYPE_TURBO3_TCQ) {
            static bool tcq3_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            if (!tcq3_cb_loaded[ctx.device]) {
                tcq3_cb_loaded[ctx.device] = true;
                turbo_tcq_load_kv_decode();
            }
        }
        if (K->type == GGML_TYPE_TURBO2_TCQ || V->type == GGML_TYPE_TURBO2_TCQ) {
            static bool tcq2_cb_loaded[GGML_CUDA_MAX_DEVICES] = {};
            static int tcq2_hot_d = -1; if (tcq2_hot_d < 0) tcq2_hot_d = getenv("TURBO_TCQ_HOTSWAP") ? 1 : 0;
            if (!tcq2_cb_loaded[ctx.device] || tcq2_hot_d) {
                tcq2_cb_loaded[ctx.device] = true;
                turbo2_tcq_load_kv_decode();
            }
        }

        cudaStream_t stream = ctx.stream();

        // Dequant turbo K/V to fp16 for decode: MMA tensor cores on fp16 beat VEC scalar
        // on turbo bits for dense models. Bandwidth savings from turbo's 3/16 footprint are
        // negligible relative to FFN compute (~1% slower native on Qwen3.5-27B, 3090).
        // Set GGML_TURBO_DECODE_NATIVE=1 to force native VEC path (may help bandwidth-limited configs).
        static const bool turbo_decode_native = (getenv("GGML_TURBO_DECODE_NATIVE") != nullptr);
        // Dequant turbo K/V to fp16 for D<=256 (any K/V combo), or D=512 only when BOTH
        // K and V are turbo (Gemma 4 ISWA global layers with K=V — Bug A2). Mixed q8_0+turbo
        // at D=512 stays on native VEC path because post-dequant q8_0 K + f16 V has no
        // working VEC template at D=512.
        const bool turbo_k_only = ggml_is_turbo_kv_type(K->type);
        const bool turbo_v_only = ggml_is_turbo_kv_type(V->type);
        // Mixed f16/q8_0 + turbo at D=512: dequant K (and turbo V) to f16 so FA dispatches as
        // F16/F16 D=512 (which exists). Without this, the native VEC templates needed are
        // either missing (F16↔turbo) or have buggy SASS on sm_120 PTX-JIT (Q8_0↔turbo4 etc).
        const bool k_is_f16_q8_or_turbo = (K->type == GGML_TYPE_F16) || (K->type == GGML_TYPE_Q8_0) || (K->type == GGML_TYPE_BF16) || turbo_k_only;
        const bool v_is_f16_q8_or_turbo = (V->type == GGML_TYPE_F16) || (V->type == GGML_TYPE_Q8_0) || (V->type == GGML_TYPE_BF16) || turbo_v_only;
        const bool both_dequantable_512 = k_is_f16_q8_or_turbo && v_is_f16_q8_or_turbo;
        // turbo8 has no native VEC kernel (no FATTN_VEC_CASES registration) — it is a
        // dequant-to-f16-only codec. GGML_TURBO_DECODE_NATIVE must not route it to the native
        // VEC path (would hit BEST_FATTN_KERNEL_VEC → unregistered template → GGML_ABORT), so
        // force dequant whenever turbo8 is on either side, regardless of the override. Other
        // turbo types do have native VEC kernels and honor the override.
        // turbo8 has no native VEC kernel (dequant-to-f16 only) — force dequant even
        // under GGML_TURBO_DECODE_NATIVE so we never route them to an unregistered VEC template.
        const bool turbo8_involved = K->type == GGML_TYPE_TURBO8_0 || V->type == GGML_TYPE_TURBO8_0 ||
                                     K->type == GGML_TYPE_TURBO1_TCQ || V->type == GGML_TYPE_TURBO1_TCQ;
        const bool do_decode_dequant = (!turbo_decode_native || turbo8_involved) && turbo_kv && (Q->ne[0] <= 256 || (Q->ne[0] <= 512 && both_dequantable_512));

        half * k_fp16_dec = nullptr;
        half * v_fp16_dec = nullptr;
        ggml_tensor K_f16_dec, V_f16_dec;
        ggml_tensor * orig_k_decode = nullptr;
        ggml_tensor * orig_v_decode = nullptr;

        int device_dec;
        CUDA_CHECK(cudaGetDevice(&device_dec));

        if (do_decode_dequant) {
            // Mixed q8_0 + turbo: when one side is turbo (dequanted to f16) and the other is
            // q8_0, the FA kernel gets a (Q8_0, F16) pair, which produces garbage at D<=256
            // (no correct mixed VEC combo). Dequant the q8_0 side to f16 too so the pair is
            // (F16, F16) — matching the symmetric turbo path that goes through the same f16 FA.
            // Base condition = the single authoritative predicate (ggml-vbr.h); decode
            // additionally dequants q8_0/bf16 at head dims > 256 (no D=512 mixed template).
            bool mat_k_dec = false;
            bool mat_v_dec = false;
            ggml_vbr_kv_dequant_sides(K->type, V->type, &mat_k_dec, &mat_v_dec);
            const bool k_needs_dequant = mat_k_dec || ((K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_BF16) && Q->ne[0] > 256);
            const bool v_needs_dequant = mat_v_dec || ((V->type == GGML_TYPE_Q8_0 || V->type == GGML_TYPE_BF16) && Q->ne[0] > 256);
            if (k_needs_dequant) {
                // Size for the CURRENT attended KV range (this call's view) — NOT the full
                // root/kv_size capacity. The dequant kernel below writes exactly K->ne[1] rows and
                // the K_f16_dec strides index densely over ne[1]/ne[2]/ne[3], so nothing here needs
                // the root capacity. Root-sizing (the old behaviour) materialized the full advertised
                // kv_size to f16 during decode, which under dynamic VBR starves the KV VRAM budget as
                // layers degrade and n_kv grows.
                //
                // This is the OOM site the VMM-backed scratch fixes: the prior cudaMalloc grow needed
                // a contiguous ~500 MiB block at the final 131072->262144 doubling, which fragmented
                // free VRAM could not satisfy. kv_dequant_scratch() maps non-contiguous physical pages
                // into a pre-reserved VA range instead, so growth can never fail on fragmentation, and
                // it maps to this exact width (not next_pow2) so the slack returns to the KV budget.
                // See kv_dequant_scratch() for the graph-capture-safety argument (growth only ever
                // happens in an eager pass; the capture pass sees a stable, already-mapped size).
                const size_t k_max_bytes = (size_t) K->ne[0] * (size_t) K->ne[1] * (size_t) K->ne[2] * (size_t) K->ne[3] * sizeof(half);
                k_fp16_dec = kv_dequant_scratch(device_dec, k_max_bytes, &kv_dequant_k_vmm[device_dec], &kv_dequant_k_vmm_va[device_dec], &kv_dequant_k_vmm_hw[device_dec],
                                                &kv_dequant_k_buf[device_dec], &kv_dequant_k_buf_size[device_dec]);
                // K dequant to fp16 in ORIGINAL (unrotated) domain via inverse FWHT.
                // All turbo K types use inv-FWHT kernels so K matches native f16/q8_0 layout
                // and Q stays unrotated. This mirrors the prefill path's encode→decode chain
                // and is the only path that works on Gemma 4 ISWA + K=V global layers.
                //
                // Bug #31 exception: K=turbo2 inv-FWHT decode produces correct values for V in
                // {turbo2, *_tcq} but a (still-undiagnosed) divergence with V in {turbo3, turbo4,
                // q8_0, f16} on Gemma 4 26B-A4B (degenerate single-token output, attention scores
                // collapse). The PREFILL path uses the rotated-domain kernel + Q rotation for
                // turbo2 K and works for every V type. Mirror that here for the failing V types.
                const bool k_t2_use_rotated = (K->type == GGML_TYPE_TURBO2_0) &&
                    (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 ||
                     V->type == GGML_TYPE_Q8_0    || V->type == GGML_TYPE_F16);
                const bool k_t3_use_rotated = (K->type == GGML_TYPE_TURBO3_0) &&
                    (V->type == GGML_TYPE_TURBO2_0);
                dim3 grid_k(K->ne[1], K->ne[2], K->ne[3]);
                if (K->type == GGML_TYPE_TURBO2_0 && k_t2_use_rotated) {
                    // Rotated-domain dequant: K stays in WHT-rotated space; Q is pre-rotated below.
                    k_turbo2_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO3_0 && k_t3_use_rotated) {
                    // Rotated-domain dequant for K=t3 + V=t2 (same Bug #31 pattern, V side).
                    k_turbo3_dequant_f16<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO3_0) {
                    k_turbo3_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO4_0) {
                    k_turbo4_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO8_0) {
                    k_turbo8_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_TURBO1_TCQ) {
                    k_turbo1_tcq_dequant_f16<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k, 0);
        } else if (K->type == GGML_TYPE_TURBO3_TCQ) {
                    k_turbo3_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_TURBO2_TCQ) {
                    k_turbo2_tcq_dequant_f16_inv_fwht<<<grid_k, 128, 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3], d_tcq_decode_alpha_k);
                } else if (K->type == GGML_TYPE_Q8_0) {
                    // Q8_0 K dequant: only fires at D=512 when V is turbo (no F16/Q8_0 D=512
                    // template, and (Q8_0, TURBO4_0) D=512 has buggy SASS on sm_120 PTX-JIT).
                    // Output goes into TKHE layout matching V_f16_dec → dispatches as F16/F16 D=512.
                    k_q8_0_dequant_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                } else if (K->type == GGML_TYPE_BF16) {
                    // bf16 K cast to f16 (mixed bf16-K + turbo-V) → dispatches as F16/F16.
                    k_bf16_to_f16_tkhe<<<grid_k, K->ne[0], 0, stream>>>(
                        (const char *)K->data, k_fp16_dec, K->ne[0], K->ne[1], K->ne[2], K->nb[1], K->nb[2], K->nb[3]);
                }
                K_f16_dec = *K;
                K_f16_dec.type = GGML_TYPE_F16;
                K_f16_dec.data = k_fp16_dec;
                K_f16_dec.nb[0] = sizeof(half);
                K_f16_dec.nb[1] = K->ne[0] * K->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
                K_f16_dec.nb[2] = K->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
                K_f16_dec.nb[3] = K->ne[0] * K->ne[1] * K->ne[2] * sizeof(half);
                orig_k_decode = dst->src[1];
                dst->src[1] = &K_f16_dec;
            }
            if (v_needs_dequant) {
                // Same exact-width, VMM-backed sizing as K above — see kv_dequant_scratch() for why
                // root/kv_size sizing was wrong and why growth is fragmentation-proof and capture-safe.
                // This V grow is the exact call (fattn.cu cudaMalloc of kv_dequant_v_buf) that OOM'd at
                // the 131072->262144 doubling; it now maps pages into pre-reserved VA instead.
                const size_t v_max_bytes = (size_t) V->ne[0] * (size_t) V->ne[1] * (size_t) V->ne[2] * (size_t) V->ne[3] * sizeof(half);
                v_fp16_dec = kv_dequant_scratch(device_dec, v_max_bytes, &kv_dequant_v_vmm[device_dec], &kv_dequant_v_vmm_va[device_dec], &kv_dequant_v_vmm_hw[device_dec],
                                                &kv_dequant_v_buf[device_dec], &kv_dequant_v_buf_size[device_dec]);
                // V dequant to fp16. All turbo V stays in rotated domain — the graph-level
                // ggml_turbo_wht inverse op (added in build_attn) un-rotates the attention output.
                dim3 grid_v(V->ne[1], V->ne[2], V->ne[3]);
                if (V->type == GGML_TYPE_TURBO2_0) {
                    k_turbo2_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO3_TCQ) {
                    k_turbo3_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
                } else if (V->type == GGML_TYPE_TURBO2_TCQ) {
                    k_turbo2_tcq_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
                } else if (V->type == GGML_TYPE_TURBO4_0) {
                    k_turbo4_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO8_0) {
                    k_turbo8_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_TURBO1_TCQ) {
                    // ROTATED-domain V — see the prefill site; graph un-rotation keyed on v->type.
                    k_turbo1_tcq_dequant_f16_rot<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], tcq_compute_alpha_v(V->type, V->ne[1]), 1);
        } else if (V->type == GGML_TYPE_TURBO3_0) {
                    k_turbo3_dequant_f16<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3], 1);
                } else if (V->type == GGML_TYPE_Q8_0) {
                    // Q8_0 V dequant: only fires at D=512 when K is turbo (mirror of K=Q8_0 path).
                    k_q8_0_dequant_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                } else if (V->type == GGML_TYPE_BF16) {
                    // bf16 V cast to f16 (mixed turbo-K + bf16-V) → dispatches as F16/F16.
                    k_bf16_to_f16_tkhe<<<grid_v, V->ne[0], 0, stream>>>(
                        (const char *)V->data, v_fp16_dec, V->ne[0], V->ne[1], V->ne[2], V->nb[1], V->nb[2], V->nb[3]);
                }
                V_f16_dec = *V;
                V_f16_dec.type = GGML_TYPE_F16;
                V_f16_dec.data = v_fp16_dec;
                V_f16_dec.nb[0] = sizeof(half);
                V_f16_dec.nb[1] = V->ne[0] * V->ne[2] * sizeof(half);  // row stride: head_dim * n_head_kv (matches native cache)
                V_f16_dec.nb[2] = V->ne[0] * sizeof(half);             // head stride: head_dim (matches native cache)
                V_f16_dec.nb[3] = V->ne[0] * V->ne[1] * V->ne[2] * sizeof(half);
                orig_v_decode = dst->src[2];
                dst->src[2] = &V_f16_dec;
                // Bug A1: nvcc 13 on sm_120a reorders these V_f16_dec.nb[*] stores past the FA
                // dispatcher → stale strides → <unused49> garbage. signal_fence is a pure
                // host-compiler barrier (zero machine instructions).
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        }

        // Pre-rotate Q for turbo K stored in rotated domain.
        // When do_decode_dequant fires, all turbo K types are dequanted via inv-FWHT into
        // ORIGINAL domain → Q stays unrotated. When decode dequant is skipped (D>256 or
        // GGML_TURBO_DECODE_NATIVE), turbo K is consumed by the native vec turbo dot product,
        // which expects a pre-rotated Q — so rotate Q in that case.
        ggml_tensor Q_rot_decode;
        ggml_tensor * orig_q_decode = nullptr;
        const bool turbo_k_any = (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0 || K->type == GGML_TYPE_TURBO8_0 || K->type == GGML_TYPE_TURBO3_TCQ || K->type == GGML_TYPE_TURBO2_TCQ);
        // Bug #31 exception: when K=turbo2/turbo3 dequant fell back to the rotated kernel (see K
        // dispatch above), K is in WHT-rotated space, not original space, so Q must be pre-rotated.
        const bool k_uses_rotated_path = do_decode_dequant && (
            ((K->type == GGML_TYPE_TURBO2_0) &&
             (V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0 ||
              V->type == GGML_TYPE_Q8_0    || V->type == GGML_TYPE_F16)) ||
            ((K->type == GGML_TYPE_TURBO3_0) && (V->type == GGML_TYPE_TURBO2_0)));
        const bool turbo_k_in_orig_domain = do_decode_dequant && turbo_k_any && !k_uses_rotated_path;
        if (turbo_k_any && !turbo_k_in_orig_domain && Q->ne[0] % 128 == 0) {
            const size_t q_size = ggml_nelements(Q) * sizeof(float);
            q_rot_buf_ensure(device_dec, q_size);
            const int64_t n_q_groups = ggml_nelements(Q) / 128;
            k_turbo_fwht_forward<<<(int)n_q_groups, 128, 0, stream>>>(
                (const float *)Q->data, q_rot_buf[device_dec], ggml_nelements(Q));
            Q_rot_decode = *Q;
            Q_rot_decode.data = q_rot_buf[device_dec];
            orig_q_decode = dst->src[0];
            dst->src[0] = &Q_rot_decode;
        }

        switch (ggml_cuda_get_best_fattn_kernel(ggml_cuda_get_device(), dst)) {
            case BEST_FATTN_KERNEL_NONE:
                GGML_ABORT("fatal error");
            case BEST_FATTN_KERNEL_TILE:
                ggml_cuda_flash_attn_ext_tile(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_VEC:
                ggml_cuda_flash_attn_ext_vec(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_WMMA_F16:
                ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
                break;
            case BEST_FATTN_KERNEL_MMA_F16:
                ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
                break;
        }

        if (orig_q_decode) dst->src[0] = orig_q_decode;
        if (orig_k_decode) dst->src[1] = orig_k_decode;
        if (orig_v_decode) dst->src[2] = orig_v_decode;
        // K/V fp16 buffers are persistent (grow-only), no free needed
    }

    // Output inverse rotation for turbo V types is handled at graph level
    // (ggml_turbo_wht op in llama-graph.cpp) to maintain CUDA graph compatibility.
}

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    return ggml_cuda_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}
