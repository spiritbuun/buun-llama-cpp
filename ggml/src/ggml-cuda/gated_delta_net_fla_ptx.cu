#include "gated_delta_net_fla_ptx.cuh"
#include "unary.cuh"

#if !defined(GGML_USE_HIP)

#include <algorithm>
#include <array>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

#if defined(GGML_CUDA_GDN_FLA_EMBEDDED)
extern "C" {
extern const unsigned char _binary_sm80_chunk_local_cumsum_scalar_kernel_cubin_start[];
extern const unsigned char _binary_sm80_chunk_scaled_dot_kkt_fwd_kernel_cubin_start[];
extern const unsigned char _binary_sm80_merge_16x16_to_64x64_inverse_kernel_cubin_start[];
extern const unsigned char _binary_sm80_recompute_w_u_fwd_kernel_cubin_start[];
extern const unsigned char _binary_sm80_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_cubin_start[];
extern const unsigned char _binary_sm80_chunk_fwd_kernel_o_cubin_start[];
extern const unsigned char _binary_chunk_local_cumsum_scalar_kernel_cubin_start[];
extern const unsigned char _binary_chunk_scaled_dot_kkt_fwd_kernel_cubin_start[];
extern const unsigned char _binary_merge_16x16_to_64x64_inverse_kernel_cubin_start[];
extern const unsigned char _binary_recompute_w_u_fwd_kernel_cubin_start[];
extern const unsigned char _binary_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_cubin_start[];
extern const unsigned char _binary_chunk_fwd_kernel_o_cubin_start[];
}
#endif

constexpr int GDN_H  = 48;
constexpr int GDN_HK = 16;
constexpr int GDN_D  = 128;
constexpr int GDN_BT = 64;

enum kernel_id {
    K_CUMSUM,
    K_KKT,
    K_SOLVE,
    K_RECOMPUTE,
    K_STATE,
    K_OUTPUT,
    K_COUNT,
};

struct fla_modules {
    std::array<CUmodule, K_COUNT> modules{};
    std::array<CUfunction, K_COUNT> funcs{};
};

static fla_modules & get_modules(int device, int cc) {
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    GGML_ASSERT(cc == 800 || cc == 860);
    const int arch = cc == 800 ? 0 : 1;
    static std::array<std::array<fla_modules, 2>, GGML_CUDA_MAX_DEVICES> results;
    static std::array<std::array<std::once_flag, 2>, GGML_CUDA_MAX_DEVICES> once;
    std::call_once(once[device][arch], [&, device, arch] {
        fla_modules & result = results[device][arch];
        const char * base = std::getenv("GGML_CUDA_GDN_FLA_PTX_DIR");
        const char * files[K_COUNT] = {
            "H3W6T2GDMYXPGO54W5AODQVEFALZB3UDMPQRWCWEU2FJYSZ5GQ4A/chunk_local_cumsum_scalar_kernel.cubin",
            "B5MQC7CJOTDIGTHSP7OH4PECU2KEQFK3VNIWF6X3QZL3WH2OVWIA/chunk_scaled_dot_kkt_fwd_kernel.cubin",
            "SFUYNKWFVNZIUNB5NTVQPU62JXWINQTNX6W3UGGK32F4LOK3FCGQ/merge_16x16_to_64x64_inverse_kernel.cubin",
            "2JB4A4YQ53PU5USKELNWHNKDB4JH73JPOQOAKBTMZSUMDBRKD4SA/recompute_w_u_fwd_kernel.cubin",
            "KWGM6XYTQFRD43OWP6FMNI62B5PBPDUBKUKUXOMYSITZPCLLMTGQ/chunk_gated_delta_rule_fwd_kernel_h_blockdim64.cubin",
            "Y7FG6IV4K2AH375UEKR7A4QGRBN6BQ56VXXDPUXWXBAN7WNMGRTQ/chunk_fwd_kernel_o.cubin",
        };
        const char * names[K_COUNT] = {
            "chunk_local_cumsum_scalar_kernel",
            "chunk_scaled_dot_kkt_fwd_kernel",
            "merge_16x16_to_64x64_inverse_kernel",
            "recompute_w_u_fwd_kernel",
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "chunk_fwd_kernel_o",
        };
        const int shared_sm80[K_COUNT] = { 8, 8192, 10240, 36864, 90632, 32768 };
        const int shared_sm86[K_COUNT] = { 8, 16384, 10240, 32768, 49412, 20480 };
        const int * shared = cc == 800 ? shared_sm80 : shared_sm86;
#if defined(GGML_CUDA_GDN_FLA_EMBEDDED)
        const void * embedded_sm80[K_COUNT] = {
            _binary_sm80_chunk_local_cumsum_scalar_kernel_cubin_start,
            _binary_sm80_chunk_scaled_dot_kkt_fwd_kernel_cubin_start,
            _binary_sm80_merge_16x16_to_64x64_inverse_kernel_cubin_start,
            _binary_sm80_recompute_w_u_fwd_kernel_cubin_start,
            _binary_sm80_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_cubin_start,
            _binary_sm80_chunk_fwd_kernel_o_cubin_start,
        };
        const void * embedded_sm86[K_COUNT] = {
            _binary_chunk_local_cumsum_scalar_kernel_cubin_start,
            _binary_chunk_scaled_dot_kkt_fwd_kernel_cubin_start,
            _binary_merge_16x16_to_64x64_inverse_kernel_cubin_start,
            _binary_recompute_w_u_fwd_kernel_cubin_start,
            _binary_chunk_gated_delta_rule_fwd_kernel_h_blockdim64_cubin_start,
            _binary_chunk_fwd_kernel_o_cubin_start,
        };
        const void * const * embedded = cc == 800 ? embedded_sm80 : embedded_sm86;
#endif
        for (int i = 0; i < K_COUNT; ++i) {
            if (base != nullptr) {
                const std::string path = std::string(base) + "/" + files[i];
                CU_CHECK(cuModuleLoad(&result.modules[i], path.c_str()));
            } else {
#if defined(GGML_CUDA_GDN_FLA_EMBEDDED)
                CU_CHECK(cuModuleLoadData(&result.modules[i], embedded[i]));
#else
                GGML_ABORT("FLA GDN kernels are unavailable in this build");
#endif
            }
            CU_CHECK(cuModuleGetFunction(&result.funcs[i], result.modules[i], names[i]));
            CU_CHECK(cuFuncSetAttribute(result.funcs[i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared[i]));
        }
    });
    return results[device][arch];
}

// The imported FLA kernels are compiled for the whole model's head layout (GDN_H v-heads, GDN_HK q/k-heads:
// strides are constexpr) but each kernel block handles the single head named by its grid y index. A device
// holding only H of the heads (tensor split: H = 3*HK, H <= GDN_H) therefore keeps the packed buffers in the
// GDN_H/GDN_HK-strided layout, fills only heads < H (the rest is never read) and launches grid.y = H.
// The FLA head order groups the three v-heads of one q/k-head; llama's native order is h_native = hk + HK*j.
__device__ __forceinline__ int64_t gdn_h_native(int64_t h, int HK) {
    return h / 3 + int64_t(HK) * (h % 3);
}

__global__ void pack_gdn_inputs_bf16(
        const float * q, const float * k, const float * v,
        nv_bfloat16 * qp, nv_bfloat16 * kp, nv_bfloat16 * vp,
        int n_tokens, int H, int HK,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_qk = int64_t(n_tokens) * HK * GDN_D;
    const int64_t n_v  = int64_t(n_tokens) * H  * GDN_D;
    if (i < n_qk) {
        const int64_t d = i % GDN_D;
        const int64_t h = (i / GDN_D) % HK;
        const int64_t t = i / (GDN_D * HK);
        const int64_t src = d + h * sq1 + t * sq2;
        const int64_t pi  = (t * GDN_HK + h) * GDN_D + d;
        qp[pi] = q[src];
        kp[pi] = k[src];
    }
    if (i < n_v) {
        const int64_t d = i % GDN_D;
        const int64_t h = (i / GDN_D) % H;
        const int64_t t = i / (GDN_D * H);
        vp[(t * GDN_H + h) * GDN_D + d] = v[d + gdn_h_native(h, HK) * sv1 + t * sv2];
    }
    GGML_UNUSED(sq3);
    GGML_UNUSED(sv3);
}

// Preserve llama.cpp's L2_NORM arithmetic exactly while eliding its F32
// destination. Eight warps normalize eight Q/K rows per CTA; the remaining
// CTAs copy V. All three outputs land directly in the BF16 layout consumed by
// the embedded FLA kernels.
__global__ void pack_gdn_inputs_l2_bf16(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state,
        nv_bfloat16 * qp, nv_bfloat16 * kp, nv_bfloat16 * vp,
        float * gp, float * betap, float * statep,
        int n_tokens, int H, int HK,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3, float eps) {
    constexpr int rows_per_block = 8;
    const int n_qk_rows = 2 * n_tokens * HK;
    const int norm_blocks = n_qk_rows / rows_per_block;

    if (blockIdx.x < norm_blocks) {
        const int warp = threadIdx.x / WARP_SIZE;
        const int lane = threadIdx.x % WARP_SIZE;
        const int row = int(blockIdx.x) * rows_per_block + warp;
        const bool is_k = row >= n_tokens * HK;
        const int qk_row = row - (is_k ? n_tokens * HK : 0);
        const int t = qk_row / HK;
        const int h = qk_row % HK;
        const float * src = is_k ? k : q;
        nv_bfloat16 * dst = is_k ? kp : qp;
        src += h * sq1 + t * sq2;
        dst += (int64_t(t) * GDN_HK + h) * GDN_D;

        float values[GDN_D / WARP_SIZE];
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < GDN_D / WARP_SIZE; ++j) {
            values[j] = src[lane + j * WARP_SIZE];
            sum += values[j] * values[j];
        }
        sum = warp_reduce_sum(sum);
        const float scale = rsqrtf(fmaxf(sum, eps * eps));
#pragma unroll
        for (int j = 0; j < GDN_D / WARP_SIZE; ++j) {
            dst[lane + j * WARP_SIZE] = scale * values[j];
        }
    } else {
        const int64_t i = (int64_t(blockIdx.x) - norm_blocks) * blockDim.x + threadIdx.x;
        const int64_t n_v = int64_t(n_tokens) * H * GDN_D;
        const int64_t vi = 4 * i;
        if (vi < n_v) {
            const int64_t d = vi % GDN_D;
            const int64_t h = (vi / GDN_D) % H;
            const int64_t t = vi / (GDN_D * H);
            const float4 values = *reinterpret_cast<const float4 *>(v + d + gdn_h_native(h, HK) * sv1 + t * sv2);
            nv_bfloat162 * dst = reinterpret_cast<nv_bfloat162 *>(vp + (t * GDN_H + h) * GDN_D + d);
            dst[0] = __floats2bfloat162_rn(values.x, values.y);
            dst[1] = __floats2bfloat162_rn(values.z, values.w);
        }
        const int64_t n_g = int64_t(n_tokens) * H;
        constexpr int64_t state_stride = int64_t(GDN_D) * GDN_D;
        const int64_t n_state = int64_t(H) * state_stride;
        if (i < n_g) {
            const int64_t h = i % H;
            const int64_t t = i / H;
            const int64_t h_native = gdn_h_native(h, HK);
            gp[t * GDN_H + h]    = g[t * H + h_native];
            betap[t * GDN_H + h] = beta[t * H + h_native];
        }
        if (i < n_state) {
            const int64_t inner = i % state_stride;
            const int64_t h = i / state_stride;
            statep[i] = state[gdn_h_native(h, HK) * state_stride + inner];
        }
    }
    GGML_UNUSED(sq3);
    GGML_UNUSED(sv3);
}

// The recurrent convolution can write its complete token-major Q/K/V result
// directly as BF16. Normalize Q/K and reorder V from that compact allocation,
// avoiding the otherwise transient F32 convolution tensor entirely.
__global__ void pack_gdn_compact_conv_l2_bf16(
        const nv_bfloat16 * conv,
        const float * g, const float * beta, const float * state,
        nv_bfloat16 * qp, nv_bfloat16 * kp, nv_bfloat16 * vp,
        float * gp, float * betap, float * statep,
        int n_tokens, int H, int HK, float eps) {
    constexpr int rows_per_block = 8;
    const int qk_channels = HK * GDN_D;
    const int channels = 2 * qk_channels + H * GDN_D;
    const int n_qk_rows = 2 * n_tokens * HK;
    const int norm_blocks = n_qk_rows / rows_per_block;

    if (blockIdx.x < norm_blocks) {
        const int warp = threadIdx.x / WARP_SIZE;
        const int lane = threadIdx.x % WARP_SIZE;
        const int row = int(blockIdx.x) * rows_per_block + warp;
        const bool is_k = row >= n_tokens * HK;
        const int qk_row = row - (is_k ? n_tokens * HK : 0);
        const int t = qk_row / HK;
        const int h = qk_row % HK;
        const nv_bfloat16 * src = conv + int64_t(t) * channels + h * GDN_D + (is_k ? qk_channels : 0);
        nv_bfloat16 * dst = (is_k ? kp : qp) + (int64_t(t) * GDN_HK + h) * GDN_D;

        float values[GDN_D / WARP_SIZE];
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < GDN_D / WARP_SIZE; ++j) {
            values[j] = __bfloat162float(src[lane + j * WARP_SIZE]);
            sum += values[j] * values[j];
        }
        sum = warp_reduce_sum(sum);
        const float scale = rsqrtf(fmaxf(sum, eps * eps));
#pragma unroll
        for (int j = 0; j < GDN_D / WARP_SIZE; ++j) {
            dst[lane + j * WARP_SIZE] = scale * values[j];
        }
    } else {
        const int64_t i = (int64_t(blockIdx.x) - norm_blocks) * blockDim.x + threadIdx.x;
        const int64_t n_v = int64_t(n_tokens) * H * GDN_D;
        const int64_t vi = 4 * i;
        if (vi < n_v) {
            const int64_t d = vi % GDN_D;
            const int64_t h = (vi / GDN_D) % H;
            const int64_t t = vi / (GDN_D * H);
            const nv_bfloat162 * src = reinterpret_cast<const nv_bfloat162 *>(
                conv + t * channels + 2 * qk_channels + gdn_h_native(h, HK) * GDN_D + d);
            nv_bfloat162 * dst = reinterpret_cast<nv_bfloat162 *>(vp + (t * GDN_H + h) * GDN_D + d);
            dst[0] = src[0];
            dst[1] = src[1];
        }
        const int64_t n_g = int64_t(n_tokens) * H;
        constexpr int64_t state_stride = int64_t(GDN_D) * GDN_D;
        const int64_t n_state = int64_t(H) * state_stride;
        if (i < n_g) {
            const int64_t h = i % H;
            const int64_t t = i / H;
            const int64_t h_native = gdn_h_native(h, HK);
            gp[t * GDN_H + h]    = g[t * H + h_native];
            betap[t * GDN_H + h] = beta[t * H + h_native];
        }
        if (i < n_state) {
            const int64_t inner = i % state_stride;
            const int64_t h = i / state_stride;
            statep[i] = state[gdn_h_native(h, HK) * state_stride + inner];
        }
    }
}

__global__ void pack_gdn_heads_f32(
        const float * g, const float * beta, const float * state,
        float * gp, float * betap, float * statep, int n_tokens, int H, int HK) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_g = int64_t(n_tokens) * H;
    constexpr int64_t state_stride = int64_t(GDN_D) * GDN_D;
    const int64_t n_state = int64_t(H) * state_stride;
    if (i < n_g) {
        const int64_t h = i % H;
        const int64_t t = i / H;
        const int64_t h_native = gdn_h_native(h, HK);
        gp[t * GDN_H + h]    = g[t * H + h_native];
        betap[t * GDN_H + h] = beta[t * H + h_native];
    }
    if (i < n_state) {
        const int64_t inner = i % state_stride;
        const int64_t h = i / state_stride;
        statep[i] = state[gdn_h_native(h, HK) * state_stride + inner];
    }
}

__global__ void unpack_gdn_heads_f32(
        const nv_bfloat16 * src, const float * state_src,
        float * dst, float * state_dst, int n_tokens, int H, int HK) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_v = int64_t(n_tokens) * H * GDN_D;
    constexpr int64_t state_stride = int64_t(GDN_D) * GDN_D;
    const int64_t n_state = int64_t(H) * state_stride;
    if (i < n_v) {
        const int64_t d = i % GDN_D;
        const int64_t h = (i / GDN_D) % H;
        const int64_t t = i / (GDN_D * H);
        dst[(t * H + gdn_h_native(h, HK)) * GDN_D + d] = src[(t * GDN_H + h) * GDN_D + d];
    }
    if (i < n_state) {
        const int64_t inner = i % state_stride;
        const int64_t h = i / state_stride;
        state_dst[gdn_h_native(h, HK) * state_stride + inner] = state_src[i];
    }
}

__global__ void unpack_gdn_state_f32(const float * src, float * dst, int H, int HK) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int64_t state_stride = int64_t(GDN_D) * GDN_D;
    const int64_t n_state = int64_t(H) * state_stride;
    if (i < n_state) {
        const int64_t inner = i % state_stride;
        const int64_t h = i / state_stride;
        dst[gdn_h_native(h, HK) * state_stride + inner] = src[i];
    }
}

template <bool fuse_gate, bool gate_bf16 = false, bool output_bf16 = false>
__global__ void unpack_gdn_rms_f32(
        const nv_bfloat16 * src, const float * weight, const float * gate, float * dst, float eps, int H, int HK) {
    const int64_t row_native = blockIdx.x;
    const int d = threadIdx.x;
    const int64_t t = row_native / H;
    const int64_t h_native = row_native % H;
    const int64_t h = (h_native % HK) * 3 + h_native / HK;
    const float x = __bfloat162float(src[(t * GDN_H + h) * GDN_D + d]);

    float sum = x * x;
    extern __shared__ float s_sum[];
    sum = block_reduce<block_reduce_method::SUM, GDN_D>(sum, s_sum);
    const float scale = rsqrtf(sum / GDN_D + eps);
    const int64_t output_idx = row_native * GDN_D + d;
    const float normalized = scale * x * weight[d];
    if constexpr (fuse_gate) {
        const float gate_value = gate_bf16
            ? __bfloat162float(reinterpret_cast<const nv_bfloat16 *>(gate)[output_idx])
            : gate[output_idx];
        const float result = ggml_cuda_op_silu_single(gate_value) * normalized;
        if constexpr (output_bf16) {
            reinterpret_cast<nv_bfloat16 *>(dst)[output_idx] = result;
        } else {
            dst[output_idx] = result;
        }
    } else {
        if constexpr (output_bf16) {
            reinterpret_cast<nv_bfloat16 *>(dst)[output_idx] = normalized;
        } else {
            dst[output_idx] = normalized;
        }
    }

}

__device__ __forceinline__ int8_t gdn_float_to_i8_rn(float value) {
    uint32_t result;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(result) : "f"(value));
    return reinterpret_cast<const int8_t &>(result);
}

// The following I8 projection uses one symmetric activation scale per token.
// One CTA owns a token: each warp normalizes two 128-wide heads, then the CTA
// reduces their maxima and writes the final packed activation in place.
template <bool gate_bf16>
__global__ void unpack_gdn_rms_i8(
        const nv_bfloat16 * src, const float * weight, const float * gate,
        int8_t * dst, float * scales, float eps, int H, int HK) {
    constexpr int warps_max = GDN_H / 2;
    const int warps = H / 2;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t t = blockIdx.x;
    nv_bfloat16 values[8];
    float local_max = 0.0f;

#pragma unroll
    for (int head_pass = 0; head_pass < 2; ++head_pass) {
        const int h_native = warp + head_pass * warps;
        const int h = (h_native % HK) * 3 + h_native / HK;
        float x[4];
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = lane * 4 + j;
            x[j] = __bfloat162float(src[(t * GDN_H + h) * GDN_D + d]);
            sum = fmaf(x[j], x[j], sum);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        sum = __shfl_sync(0xffffffff, sum, 0);
        const float norm_scale = rsqrtf(sum / GDN_D + eps);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = lane * 4 + j;
            const int64_t output_idx = (t * H + h_native) * GDN_D + d;
            const float gate_value = gate_bf16
                ? __bfloat162float(reinterpret_cast<const nv_bfloat16 *>(gate)[output_idx])
                : gate[output_idx];
            // Preserve the old path's BF16 materialization before row quantization.
            const nv_bfloat16 value = ggml_cuda_op_silu_single(gate_value) *
                (norm_scale * x[j] * weight[d]);
            values[head_pass * 4 + j] = value;
            local_max = fmaxf(local_max, fabsf(__bfloat162float(value)));
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    __shared__ float warp_max[warps_max];
    __shared__ float inverse;
    if (lane == 0) {
        warp_max[warp] = local_max;
    }
    __syncthreads();
    if (warp == 0) {
        float maximum = lane < warps ? warp_max[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        }
        if (lane == 0) {
            scales[t] = maximum / 127.0f;
            inverse = maximum == 0.0f ? 0.0f : 127.0f / maximum;
        }
    }
    __syncthreads();

#pragma unroll
    for (int head_pass = 0; head_pass < 2; ++head_pass) {
        const int h_native = warp + head_pass * warps;
        char4 packed;
        packed.x = gdn_float_to_i8_rn(__bfloat162float(values[head_pass * 4 + 0]) * inverse);
        packed.y = gdn_float_to_i8_rn(__bfloat162float(values[head_pass * 4 + 1]) * inverse);
        packed.z = gdn_float_to_i8_rn(__bfloat162float(values[head_pass * 4 + 2]) * inverse);
        packed.w = gdn_float_to_i8_rn(__bfloat162float(values[head_pass * 4 + 3]) * inverse);
        reinterpret_cast<char4 *>(dst + (t * H + h_native) * GDN_D)[lane] = packed;
    }
}

__global__ void init_gdn_varlen_metadata(
        int * cu_seqlens, int * chunk_indices, int64_t * chunk_offsets,
        int n_tokens, int n_chunks) {
    const int i = threadIdx.x;
    if (i < 2) {
        cu_seqlens[i]   = i * n_tokens;
        chunk_offsets[i] = int64_t(i) * n_chunks;
    }
    for (int chunk = i; chunk < n_chunks; chunk += blockDim.x) {
        chunk_indices[2 * chunk + 0] = 0;
        chunk_indices[2 * chunk + 1] = chunk;
    }
}

static void launch(CUfunction fn, dim3 grid, dim3 block, unsigned shared, CUstream stream, void ** args) {
    CU_CHECK(cuLaunchKernel(fn, grid.x, grid.y, grid.z, block.x, block.y, block.z,
                           shared, stream, args, nullptr));
}

} // namespace

bool ggml_cuda_gdn_fla_ptx_supported(
        int cc, bool kda, bool keep_rs, int64_t S_v, int64_t H, int64_t H_k,
        int64_t n_tokens, int64_t n_seqs) {
#if defined(GGML_CUDA_GDN_FLA_EMBEDDED)
    constexpr bool embedded_available = true;
#else
    constexpr bool embedded_available = false;
#endif
    static const bool external_available = std::getenv("GGML_CUDA_GDN_FLA_PTX_DIR") != nullptr;
    const bool available = embedded_available || external_available;
    // any head subset with the compiled v/qk ratio (3) and an even head count (the i8 unpack pairs heads per warp)
    return available && (cc == 800 || cc == 860) &&
           !kda && !keep_rs && S_v == GDN_D && H == 3 * H_k && H <= GDN_H && H % 2 == 0 &&
           n_tokens >= 512 && n_tokens % GDN_BT == 0 && n_seqs == 1;
}

void ggml_cuda_gdn_fla_ptx(
        ggml_backend_cuda_context & ctx, int cc,
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * state_in,
        float * dst, float * state_out,
        int64_t n_tokens, int64_t H64, int64_t HK64,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        const void * compact_conv_bf16,
        float l2_eps,
        const float * rms_weight, const float * rms_gate, bool rms_gate_bf16,
        float * rms_output, bool rms_output_bf16,
        bool rms_output_int8, float * rms_output_scale, float rms_eps) {
    cudaStream_t stream = ctx.stream();
    fla_modules & m = get_modules(ctx.device, cc);

    GGML_ASSERT(n_tokens > 0 && n_tokens % GDN_BT == 0 && n_tokens <= INT_MAX);
    GGML_ASSERT(H64 == 3 * HK64 && H64 <= GDN_H && H64 % 2 == 0);
    const int H  = int(H64);  // heads held by this device; buffers stay GDN_H/GDN_HK-strided (see gdn_h_native)
    const int HK = int(HK64);
    const int n_chunks         = int(n_tokens / GDN_BT);
    const int64_t n_qk         = n_tokens * GDN_HK * GDN_D;
    const int64_t n_v          = n_tokens * GDN_H  * GDN_D;
    const int64_t n_g          = n_tokens * GDN_H;
    const int64_t n_A          = n_g * GDN_BT;
    const int64_t n_h          = int64_t(n_chunks) * GDN_H * GDN_D * GDN_D;
    constexpr int64_t n_state = int64_t(GDN_H) * GDN_D * GDN_D;

    ggml_cuda_pool_alloc<nv_bfloat16> q_p(ctx.pool(), n_qk);
    ggml_cuda_pool_alloc<nv_bfloat16> k_p(ctx.pool(), n_qk);
    ggml_cuda_pool_alloc<nv_bfloat16> v_p(ctx.pool(), n_v);
    ggml_cuda_pool_alloc<float>       g_cum(ctx.pool(), n_g);
    ggml_cuda_pool_alloc<float>       A(ctx.pool(), n_A);
    ggml_cuda_pool_alloc<nv_bfloat16> Ai(ctx.pool(), n_A);
    ggml_cuda_pool_alloc<nv_bfloat16> w(ctx.pool(), n_v);
    ggml_cuda_pool_alloc<nv_bfloat16> u(ctx.pool(), n_v);
    ggml_cuda_pool_alloc<nv_bfloat16> v_new(ctx.pool(), n_v);
    ggml_cuda_pool_alloc<nv_bfloat16> h(ctx.pool(), n_h);
    ggml_cuda_pool_alloc<nv_bfloat16> out(ctx.pool(), n_v);
    ggml_cuda_pool_alloc<float>       g_p(ctx.pool(), n_g);
    ggml_cuda_pool_alloc<float>       beta_p(ctx.pool(), n_g);
    ggml_cuda_pool_alloc<float>       state_in_p(ctx.pool(), n_state);
    ggml_cuda_pool_alloc<float>       state_out_p(ctx.pool(), n_state);
    ggml_cuda_pool_alloc<int>         cu_seqlens(ctx.pool(), 2);
    ggml_cuda_pool_alloc<int>         chunk_indices(ctx.pool(), 2 * n_chunks);
    ggml_cuda_pool_alloc<int64_t>     chunk_offsets(ctx.pool(), 2);

    constexpr int threads = 256;
    // element counts of this device's heads (the packed buffers above are sized for GDN_H)
    const int64_t n_v_dev     = n_tokens * H * GDN_D;
    const int64_t n_state_dev = int64_t(H) * GDN_D * GDN_D;
    const int64_t n_g_dev     = n_tokens * H;
    if (compact_conv_bf16 != nullptr) {
        GGML_ASSERT(l2_eps >= 0.0f);
        constexpr int rows_per_block = 8;
        const int norm_blocks = 2 * int(n_tokens) * HK / rows_per_block;
        const int v_blocks = (n_v_dev / 4 + threads - 1) / threads; // n_v_dev/4 >= n_state_dev for n_tokens >= 512
        pack_gdn_compact_conv_l2_bf16<<<norm_blocks + v_blocks, threads, 0, stream>>>(
            static_cast<const nv_bfloat16 *>(compact_conv_bf16), g, beta, state_in,
            q_p.get(), k_p.get(), v_p.get(), g_p.get(), beta_p.get(), state_in_p.get(),
            int(n_tokens), H, HK, l2_eps);
    } else if (l2_eps >= 0.0f) {
        constexpr int rows_per_block = 8;
        const int norm_blocks = 2 * int(n_tokens) * HK / rows_per_block;
        const int v_blocks = (n_v_dev / 4 + threads - 1) / threads;
        pack_gdn_inputs_l2_bf16<<<norm_blocks + v_blocks, threads, 0, stream>>>(
            q, k, v, g, beta, state_in,
            q_p.get(), k_p.get(), v_p.get(), g_p.get(), beta_p.get(), state_in_p.get(),
            int(n_tokens), H, HK,
            sq1, sq2, sq3, sv1, sv2, sv3, l2_eps);
    } else {
        pack_gdn_inputs_bf16<<<(n_v_dev + threads - 1) / threads, threads, 0, stream>>>(
            q, k, v, q_p.get(), k_p.get(), v_p.get(), int(n_tokens), H, HK,
            sq1, sq2, sq3, sv1, sv2, sv3);
        pack_gdn_heads_f32<<<(std::max(n_state_dev, n_g_dev) + threads - 1) / threads, threads, 0, stream>>>(
            g, beta, state_in, g_p.get(), beta_p.get(), state_in_p.get(), int(n_tokens), H, HK);
    }
    CUDA_CHECK(cudaGetLastError());
    init_gdn_varlen_metadata<<<1, 256, 0, stream>>>(
        cu_seqlens.get(), chunk_indices.get(), chunk_offsets.get(), int(n_tokens), n_chunks);
    CUDA_CHECK(cudaGetLastError());
    void * null_ptr = nullptr;
    int T = int(n_tokens);
    float scale = 1.0f / sqrtf(float(GDN_D));
    CUstream cu_stream = (CUstream) stream;

    void * cumsum_args[] = { &g_p.ptr, &g_cum.ptr, &cu_seqlens.ptr, &chunk_indices.ptr, &T, &null_ptr, &null_ptr };
    const bool sm80 = cc == 800;
    const unsigned Hu = (unsigned) H; // grid.y names the head; only this device's heads are launched
    launch(m.funcs[K_CUMSUM], {(unsigned) n_chunks, Hu, 1}, {sm80 ? 128u : 256u, 1, 1}, 8, cu_stream, cumsum_args);
    void * kkt_args[] = { &k_p.ptr, &beta_p.ptr, &g_cum.ptr, &A.ptr, &cu_seqlens.ptr, &chunk_indices.ptr, &T, &null_ptr, &null_ptr };
    launch(m.funcs[K_KKT], {(unsigned) n_chunks, Hu, 1}, {sm80 ? 128u : 256u, 1, 1}, sm80 ? 8192 : 16384, cu_stream, kkt_args);
    CUDA_CHECK(cudaMemsetAsync(Ai.get(), 0, n_A * sizeof(nv_bfloat16), stream));
    void * solve_args[] = { &A.ptr, &Ai.ptr, &cu_seqlens.ptr, &chunk_indices.ptr, &T, &null_ptr, &null_ptr };
    launch(m.funcs[K_SOLVE], {(unsigned) n_chunks, Hu, 1}, {sm80 ? 64u : 128u, 1, 1}, 10240, cu_stream, solve_args);
    void * recompute_args[] = { &k_p.ptr, &v_p.ptr, &beta_p.ptr, &w.ptr, &u.ptr, &Ai.ptr, &g_cum.ptr,
                                &cu_seqlens.ptr, &chunk_indices.ptr, &T, &null_ptr, &null_ptr };
    launch(m.funcs[K_RECOMPUTE], {(unsigned) n_chunks, Hu, 1}, {sm80 ? 64u : 128u, 1, 1}, sm80 ? 36864 : 32768, cu_stream, recompute_args);
    void * state_args[] = { &k_p.ptr, &u.ptr, &w.ptr, &v_new.ptr, &g_cum.ptr, &h.ptr,
                            &state_in_p.ptr, &state_out_p.ptr, &cu_seqlens.ptr, &chunk_offsets.ptr, &T, &null_ptr, &null_ptr };
    launch(m.funcs[K_STATE], {2, Hu, 1}, {128, 1, 1}, sm80 ? 90632 : 49412, cu_stream, state_args);
    void * output_args[] = { &q_p.ptr, &k_p.ptr, &v_new.ptr, &h.ptr, &g_cum.ptr, &out.ptr,
                             &cu_seqlens.ptr, &chunk_indices.ptr, &scale, &T, &null_ptr, &null_ptr };
    launch(m.funcs[K_OUTPUT], {sm80 ? 1u : 4u, (unsigned) n_chunks, Hu}, {sm80 ? 128u : 64u, 1, 1}, sm80 ? 32768 : 20480, cu_stream, output_args);
    if (rms_output != nullptr) {
        GGML_ASSERT(rms_weight != nullptr);
        if (rms_output_int8) {
            GGML_ASSERT(rms_gate != nullptr && rms_output_scale != nullptr);
            const int block_size = (H / 2) * 32;
            if (rms_gate_bf16) {
                unpack_gdn_rms_i8<true><<<n_tokens, block_size, 0, stream>>>(
                    out.get(), rms_weight, rms_gate,
                    reinterpret_cast<int8_t *>(rms_output), rms_output_scale, rms_eps, H, HK);
            } else {
                unpack_gdn_rms_i8<false><<<n_tokens, block_size, 0, stream>>>(
                    out.get(), rms_weight, rms_gate,
                    reinterpret_cast<int8_t *>(rms_output), rms_output_scale, rms_eps, H, HK);
            }
        } else if (rms_gate != nullptr) {
            if (rms_gate_bf16) {
                if (rms_output_bf16) {
                    unpack_gdn_rms_f32<true, true, true><<<n_g_dev, GDN_D, 32 * sizeof(float), stream>>>(
                        out.get(), rms_weight, rms_gate, rms_output, rms_eps, H, HK);
                } else {
                    unpack_gdn_rms_f32<true, true><<<n_g_dev, GDN_D, 32 * sizeof(float), stream>>>(
                        out.get(), rms_weight, rms_gate, rms_output, rms_eps, H, HK);
                }
            } else {
                if (rms_output_bf16) {
                    unpack_gdn_rms_f32<true, false, true><<<n_g_dev, GDN_D, 32 * sizeof(float), stream>>>(
                        out.get(), rms_weight, rms_gate, rms_output, rms_eps, H, HK);
                } else {
                    unpack_gdn_rms_f32<true><<<n_g_dev, GDN_D, 32 * sizeof(float), stream>>>(
                        out.get(), rms_weight, rms_gate, rms_output, rms_eps, H, HK);
                }
            }
        } else {
            if (rms_output_bf16) {
                unpack_gdn_rms_f32<false, false, true><<<n_g_dev, GDN_D, 32 * sizeof(float), stream>>>(
                    out.get(), rms_weight, nullptr, rms_output, rms_eps, H, HK);
            } else {
                unpack_gdn_rms_f32<false><<<n_g_dev, GDN_D, 32 * sizeof(float), stream>>>(
                    out.get(), rms_weight, nullptr, rms_output, rms_eps, H, HK);
            }
        }
        unpack_gdn_state_f32<<<(n_state_dev + threads - 1) / threads, threads, 0, stream>>>(
            state_out_p.get(), state_out, H, HK);
    } else {
        unpack_gdn_heads_f32<<<(std::max(n_v_dev, n_state_dev) + threads - 1) / threads, threads, 0, stream>>>(
            out.get(), state_out_p.get(), dst, state_out, int(n_tokens), H, HK);
    }
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_gdn_fla_ptx_supported(int, bool, bool, int64_t, int64_t, int64_t, int64_t, int64_t) { return false; }
void ggml_cuda_gdn_fla_ptx(ggml_backend_cuda_context &, int, const float *, const float *, const float *,
        const float *, const float *, const float *, float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, const void *, float, const float *, const float *, bool, float *, bool,
        bool, float *, float) { GGML_ABORT("FLA PTX is CUDA-only"); }

#endif
