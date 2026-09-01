#include "mmvq.cuh"
#include "mmvq-tuning.h"
#include "moe-cache-mmv-tuning.h"
#include "quantize.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"
#if !defined(GGML_USE_HIP)
#include "humming-fp8.cuh"
#include "humming-fp8-block.cuh"
#include "humming-nvfp4.cuh"
#include "marlin-q4-a32.cuh"
#endif

#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>

struct ggml_cuda_mmvq_fusion_args_device : ggml_cuda_mm_fusion_args_device {
    float post_scale = 1.0f;
    bool  post_silu = false;
};

typedef float (*vec_dot_q_cuda_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs);

#if !defined(GGML_USE_HIP)
static nv_bfloat16 * ggml_cuda_humming_get_input(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src,
    size_t count,
    cudaStream_t stream);

static __global__ void dequantize_fp8_block_bf16_kernel(
        const uint8_t * weight,
        const float * scale,
        nv_bfloat16 * dst,
        int64_t k,
        int64_t n,
        int64_t batches) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= k * n * batches) {
        return;
    }
    const int64_t batch = index / (k * n);
    const int64_t local = index - batch * k * n;
    const int64_t row = local / k;
    const int64_t col = local - row * k;
    const int64_t n_blocks = n / 128;
    const int64_t all_n_blocks = n_blocks * batches;
    const float block_scale = scale[(col / 128) * all_n_blocks + batch * n_blocks + row / 128];
    dst[index] = __float2bfloat16(ggml_cuda_e4m3_to_fp32(weight[index]) * block_scale);
}

void ggml_cuda_dequantize_fp8_block_bf16(
        const ggml_tensor * weight,
        const ggml_tensor * scale,
        nv_bfloat16 * dst,
        cudaStream_t stream) {
    GGML_ASSERT(weight->type == GGML_TYPE_F8_E4M3 && scale->type == GGML_TYPE_F32);
    GGML_ASSERT(weight->ne[0] % 128 == 0 && weight->ne[1] % 128 == 0 && weight->ne[3] == 1);
    GGML_ASSERT(scale->ne[0] == weight->ne[1] * weight->ne[2] / 128 &&
                scale->ne[1] == weight->ne[0] / 128 && scale->ne[2] == 1 && scale->ne[3] == 1);
    const int64_t count = weight->ne[0] * weight->ne[1] * weight->ne[2];
    constexpr int threads = 256;
    dequantize_fp8_block_bf16_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const uint8_t *>(weight->data), static_cast<const float *>(scale->data),
        dst, weight->ne[0], weight->ne[1], weight->ne[2]);
}

bool ggml_cuda_mul_mat_marlin_q4_a32(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!ggml_cuda_marlin_q4_a32_enabled() || src0->type != GGML_TYPE_Q4_A32 ||
        ids != nullptr ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        dst->ne[2] != 1 || dst->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const ggml_tensor * gate = fusion != nullptr ? fusion->gate : nullptr;
    if (fusion != nullptr && (fusion->x_scale != nullptr || fusion->gate_scale != nullptr ||
            fusion->x_bias != nullptr || fusion->gate_bias != nullptr)) {
        return false;
    }
    if (gate != nullptr && (fusion->glu_op != GGML_GLU_OP_SWIGLU ||
            gate->type != GGML_TYPE_Q4_A32 || !ggml_are_same_shape(gate, src0) ||
            !ggml_are_same_stride(gate, src0) || !ggml_is_contiguous(gate))) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (src1->ne[0] != k || dst->ne[0] != n || dst->ne[1] != m ||
        !ggml_cuda_marlin_q4_a32_supports_shape(n, k, m, cc)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    if (!ggml_cuda_marlin_q4_a32_is_repacked(src0) ||
            (gate != nullptr && !ggml_cuda_marlin_q4_a32_is_repacked(gate))) {
        return false;
    }

    auto prepare_weight = [&](const ggml_tensor * weight) {
        ggml_cuda_marlin_q4_a32_layout entry;
        entry.weight_size = size_t(n) * k / 2;
        entry.scale_size  = size_t(n) * k / QG4_A32 * sizeof(nv_bfloat16);
        entry.zero_size   = size_t(n) * k / QG4_A32 / 2;
        entry.weight = weight->data;
        entry.scale  = static_cast<char *>(weight->data) + entry.weight_size;
        entry.zero   = static_cast<char *>(entry.scale) + entry.scale_size;
        return entry;
    };
    const ggml_cuda_marlin_q4_a32_layout entry = prepare_weight(src0);
    const ggml_cuda_marlin_q4_a32_layout gate_entry = gate != nullptr ? prepare_weight(gate) :
        ggml_cuda_marlin_q4_a32_layout{};

    auto & lock_storage = ctx.humming_fp8_locks[ctx.curr_stream_no];
    const size_t required_locks = ((m + 15) / 16) * ((n + 63) / 64);
    if (lock_storage.count < required_locks) {
        if (lock_storage.ptr != nullptr) {
            lock_storage.retired.push_back(lock_storage.ptr);
        }
        CUDA_CHECK(cudaMalloc(&lock_storage.ptr, required_locks * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(lock_storage.ptr, 0, required_locks * sizeof(int32_t), stream));
        lock_storage.count = required_locks;
    }

    nv_bfloat16 * input = ggml_cuda_humming_get_input(ctx, src1, size_t(m) * k, stream);
    ggml_cuda_pool_alloc<nv_bfloat16> output_scratch(ctx.pool());
    nv_bfloat16 * output = output_scratch.alloc(size_t(m) * n);
    ggml_cuda_pool_alloc<nv_bfloat16> gate_output(ctx.pool());
    if (gate != nullptr) {
        gate_output.alloc(size_t(m) * n);
    }
    ggml_cuda_marlin_q4_a32_launch(
        input, entry.weight, entry.scale, entry.zero, output, lock_storage.ptr,
        n, k, m, ctx.device, ggml_cuda_info().devices[ctx.device].nsm, stream);
    if (gate != nullptr) {
        ggml_cuda_marlin_q4_a32_launch(
            input, gate_entry.weight, gate_entry.scale, gate_entry.zero,
            gate_output.get(), lock_storage.ptr, n, k, m, ctx.device,
            ggml_cuda_info().devices[ctx.device].nsm, stream);
    }

    if (fusion != nullptr && fusion->residual != nullptr &&
            ggml_cuda_humming_finish_residual_rms(ctx, fusion, output, dst, n, m, stream)) {
        // The fused epilogue materializes the required F32 graph outputs and
        // preserves a BF16 normalized activation for following projections.
    } else if (gate != nullptr && fusion != nullptr && fusion->retain_bf16_output && n == 17408) {
        ggml_cuda_humming_fp8_swiglu_bf16(
            output, gate_output.get(), static_cast<nv_bfloat16 *>(dst->data), size_t(m) * n, stream);
        ctx.humming_bf16_activations.insert(dst);
    } else {
        ggml_cuda_humming_fp8_output_bf16_to_f32(
            output, gate != nullptr ? gate_output.get() : nullptr,
            static_cast<float *>(dst->data), size_t(m) * n, stream);
    }
    return true;
}

nv_bfloat16 * ggml_cuda_prepare_bf16_input(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        size_t count,
        cudaStream_t stream) {
    auto & storage = ctx.humming_inputs[ctx.curr_stream_no];
    if (storage.count < count) {
        if (storage.ptr != nullptr) {
            storage.retired.push_back(storage.ptr);
        }
        CUDA_CHECK(cudaMalloc(&storage.ptr, count * sizeof(nv_bfloat16)));
        storage.count = count;
        storage.source = nullptr;
    }
    if (storage.source != src || storage.source_count != count) {
        ggml_cuda_humming_fp8_input_f32_to_bf16(
            static_cast<const float *>(src->data), storage.ptr, count, stream);
        storage.source = src;
        storage.source_count = count;
    }
    return storage.ptr;
}

static nv_bfloat16 * ggml_cuda_humming_get_input(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        size_t count,
        cudaStream_t stream) {
    if (const nv_bfloat16 * cached = ggml_cuda_get_cached_bf16_input(ctx, src, count)) {
        return const_cast<nv_bfloat16 *>(cached);
    }
    return ggml_cuda_prepare_bf16_input(ctx, src, count, stream);
}

const nv_bfloat16 * ggml_cuda_get_cached_bf16_input(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        size_t count) {
    auto prepared = ctx.humming_prepared_activations.find(src);
    if (prepared != ctx.humming_prepared_activations.end() &&
            ctx.humming_prepared_active.count(src) != 0) {
        GGML_ASSERT(prepared->second.count >= count);
        return prepared->second.ptr;
    }
    if (ctx.consume_bf16_activation(src)) {
        return static_cast<nv_bfloat16 *>(src->data);
    }
    return nullptr;
}

bool ggml_cuda_humming_finish_residual_rms(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_mm_fusion_args_host * fusion,
        const nv_bfloat16 * output,
        ggml_tensor * dst,
        int64_t n,
        int64_t m,
        cudaStream_t stream) {
    if (fusion->residual == nullptr) {
        return false;
    }
    if (fusion->rms_weight == nullptr) {
        GGML_ASSERT(fusion->residual_out == nullptr);
        GGML_ASSERT(fusion->residual->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(fusion->residual) && ggml_is_contiguous(dst));
        GGML_ASSERT(ggml_nelements(fusion->residual) == m * n && ggml_nelements(dst) == m * n);
        ggml_cuda_humming_fp8_residual_add(
            output, static_cast<const float *>(fusion->residual->data),
            static_cast<float *>(dst->data), m * n, stream);
        return true;
    }
    GGML_ASSERT(fusion->residual_out != nullptr);
    GGML_ASSERT(fusion->residual->type == GGML_TYPE_F32 &&
                fusion->residual_out->type == GGML_TYPE_F32 &&
                fusion->rms_weight->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(fusion->residual) &&
                ggml_is_contiguous(fusion->residual_out) &&
                ggml_is_contiguous(fusion->rms_weight) && ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_nelements(fusion->residual) == m * n &&
                ggml_nelements(fusion->residual_out) == m * n &&
                ggml_nelements(fusion->rms_weight) == n && ggml_nelements(dst) == m * n);

    const size_t prepared_count = size_t(m) * size_t(n);
    auto result = ctx.humming_prepared_activations.try_emplace(dst);
    ggml_cuda_humming_prepared_activation & prepared = result.first->second;
    if (prepared.count < prepared_count) {
        if (prepared.ptr != nullptr) {
            prepared.retired.push_back(prepared.ptr);
        }
        prepared.count = prepared_count;
        CUDA_CHECK(cudaMalloc(&prepared.ptr, prepared.count * sizeof(nv_bfloat16)));
    }
    ctx.humming_prepared_active.insert(dst);

    ggml_cuda_humming_residual_rms_prepare(
        output,
        static_cast<const float *>(fusion->residual->data),
        static_cast<const float *>(fusion->rms_weight->data),
        static_cast<float *>(fusion->residual_out->data),
        fusion->materialize_rms_output ? static_cast<float *>(dst->data) : nullptr,
        prepared.ptr,
        n, m, fusion->rms_eps, stream);
    return true;
}

bool ggml_cuda_mul_mat_humming_nvfp4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!ggml_cuda_humming_nvfp4_enabled() || src0->type != GGML_TYPE_NVFP4 ||
        ids != nullptr || fusion == nullptr || fusion->x_scale == nullptr ||
        fusion->x_bias != nullptr || fusion->gate_bias != nullptr ||
        src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
        dst->ne[2] != 1 || dst->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_humming_nvfp4_supports_shape(n, k, m, cc) || src1->ne[0] != k ||
        dst->ne[0] != n || dst->ne[1] != m) {
        return false;
    }

    const ggml_tensor * x_scale = fusion->x_scale;
    if (x_scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(x_scale) || ggml_nelements(x_scale) != 1) {
        return false;
    }

    const ggml_tensor * gate = fusion->gate;
    const ggml_tensor * gate_scale = fusion->gate_scale;
    if (gate != nullptr) {
        if (fusion->glu_op != GGML_GLU_OP_SWIGLU || gate_scale == nullptr ||
            gate->type != GGML_TYPE_NVFP4 || !ggml_are_same_shape(gate, src0) ||
            !ggml_are_same_stride(gate, src0) || !ggml_is_contiguous(gate) ||
            gate_scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(gate_scale) ||
            ggml_nelements(gate_scale) != 1) {
            return false;
        }
    } else if (gate_scale != nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    auto & lock_storage = ctx.humming_fp8_locks[ctx.curr_stream_no];
    const size_t required_locks = ((m + 127) / 128) * ((n + 63) / 64);
    if (lock_storage.count < required_locks) {
        if (lock_storage.ptr != nullptr) {
            lock_storage.retired.push_back(lock_storage.ptr);
        }
        CUDA_CHECK(cudaMalloc(&lock_storage.ptr, required_locks * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(lock_storage.ptr, 0, required_locks * sizeof(int32_t), stream));
        lock_storage.count = required_locks;
    }

    ggml_cuda_pool_alloc<nv_bfloat16> output_scratch(ctx.pool());
    nv_bfloat16 * output = output_scratch.alloc(m * n);
    ggml_cuda_pool_alloc<nv_bfloat16> gate_output(ctx.pool());
    if (gate != nullptr) {
        gate_output.alloc(m * n);
    }
    nv_bfloat16 * input = ggml_cuda_humming_get_input(ctx, src1, m * k, stream);

    const int sms = ggml_cuda_info().devices[ctx.device].nsm;
    const size_t weight_size = size_t(n) * k / 2;
    const size_t scale_size = size_t(n) * k / QK_NVFP4_SUB;
    const bool inplace_repacked = ggml_cuda_humming_nvfp4_is_repacked(src0);
    if (inplace_repacked) {
        if (gate != nullptr && !ggml_cuda_humming_nvfp4_is_repacked(gate)) {
            return false;
        }
        ggml_cuda_humming_nvfp4_launch(
            input,
            src0->data,
            static_cast<const uint8_t *>(src0->data) + weight_size,
            static_cast<const float *>(x_scale->data),
            output, lock_storage.ptr, n, k, m, sms, stream);
        if (gate != nullptr) {
            ggml_cuda_humming_nvfp4_launch(
                input,
                gate->data,
                static_cast<const uint8_t *>(gate->data) + weight_size,
                static_cast<const float *>(gate_scale->data),
                gate_output.get(), lock_storage.ptr, n, k, m, sms, stream);
        }
        if (ggml_cuda_humming_finish_residual_rms(ctx, fusion, output, dst, n, m, stream)) {
            // The fused epilogue materializes both required F32 graph outputs
            // and retains a BF16 normalized activation for following Humming projections.
        } else if (gate != nullptr && fusion->retain_bf16_output && n == 17408) {
            ggml_cuda_humming_fp8_swiglu_bf16(
                output, gate_output.get(), static_cast<nv_bfloat16 *>(dst->data), m * n, stream);
            ctx.humming_bf16_activations.insert(dst);
        } else {
            ggml_cuda_humming_fp8_output_bf16_to_f32(
                output, gate ? gate_output.get() : nullptr, static_cast<float *>(dst->data), m * n, stream);
        }
        return true;
    }
    auto prepare_and_launch = [&](const ggml_tensor * weight, const ggml_tensor * scale, nv_bfloat16 * result) {
        // The canonical GGML layout remains the sole resident copy. Repack one
        // projection at a time into bounded stream-ordered scratch so this path
        // does not duplicate the model weights in VRAM.
        ggml_cuda_pool_alloc<uint8_t> original(ctx.pool(), weight_size);
        ggml_cuda_pool_alloc<uint8_t> repacked(ctx.pool(), weight_size);
        ggml_cuda_pool_alloc<uint8_t> reordered_scale(ctx.pool(), scale_size);
        ggml_cuda_humming_nvfp4_prepare(
            weight->data, original.get(), repacked.get(), reordered_scale.get(), n, k, stream);
        ggml_cuda_humming_nvfp4_launch(
            input, repacked.get(), reordered_scale.get(), static_cast<const float *>(scale->data),
            result, lock_storage.ptr, n, k, m, sms, stream);
    };
    prepare_and_launch(src0, x_scale, output);
    if (gate != nullptr) {
        prepare_and_launch(gate, gate_scale, gate_output.get());
    }
    if (ggml_cuda_humming_finish_residual_rms(ctx, fusion, output, dst, n, m, stream)) {
        // The fused epilogue materializes both the residual and normalized F32
        // tensors, plus a BF16 copy reused by every following Humming projection.
    } else if (gate != nullptr && fusion->retain_bf16_output && n == 17408) {
        ggml_cuda_humming_fp8_swiglu_bf16(
            output, gate_output.get(), static_cast<nv_bfloat16 *>(dst->data), m * n, stream);
        ctx.humming_bf16_activations.insert(dst);
    } else {
        ggml_cuda_humming_fp8_output_bf16_to_f32(
            output, gate ? gate_output.get() : nullptr, static_cast<float *>(dst->data), m * n, stream);
    }
    return true;
}

bool ggml_cuda_mul_mat_humming_fp8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * ids,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!ggml_cuda_humming_fp8_enabled()) {
        return false;
    }

    if (src0->type != GGML_TYPE_F8_E4M3 || !ggml_cuda_humming_fp8_is_repacked(src0) ||
        ids != nullptr || fusion == nullptr ||
        fusion->x_scale == nullptr || fusion->x_bias != nullptr || fusion->gate_bias != nullptr ||
        src1->ne[2] != 1 || src1->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
        dst->ne[2] != 1 || dst->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_humming_fp8_supports_shape(n, k, m, cc) || src1->ne[0] != k ||
        dst->ne[0] != n || dst->ne[1] != m) {
        return false;
    }

    const ggml_tensor * x_scale = fusion->x_scale;
    if (x_scale->type != GGML_TYPE_BF16 || !ggml_is_contiguous(x_scale) ||
        x_scale->ne[0] != n || x_scale->ne[1] != 1 || x_scale->ne[2] != 1 || x_scale->ne[3] != 1) {
        return false;
    }

    const ggml_tensor * gate = fusion->gate;
    const ggml_tensor * gate_scale = fusion->gate_scale;
    if (gate != nullptr) {
        if (fusion->glu_op != GGML_GLU_OP_SWIGLU || gate_scale == nullptr ||
            gate->type != GGML_TYPE_F8_E4M3 || !ggml_cuda_humming_fp8_is_repacked(gate) ||
            !ggml_are_same_shape(gate, src0) ||
            !ggml_are_same_stride(gate, src0) || !ggml_is_contiguous(gate) ||
            gate_scale->type != GGML_TYPE_BF16 || !ggml_is_contiguous(gate_scale) ||
            gate_scale->ne[0] != n || gate_scale->ne[1] != 1 ||
            gate_scale->ne[2] != 1 || gate_scale->ne[3] != 1) {
            return false;
        }
    } else if (gate_scale != nullptr) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    auto prepare_weight = [&](const ggml_tensor * weight, const ggml_tensor * scale) {
        auto [it, inserted] = ctx.humming_fp8_cache.try_emplace(weight->data);
        ggml_cuda_humming_fp8_cache_entry & entry = it->second;
        if (inserted) {
            entry.scale_size = ggml_nbytes(scale);
            entry.source_scale = scale->data;
            CUDA_CHECK(cudaMalloc(&entry.scale, entry.scale_size));
            ggml_cuda_humming_fp8_reorder_scale(
                static_cast<const nv_bfloat16 *>(scale->data),
                static_cast<nv_bfloat16 *>(entry.scale), n, stream);
        } else {
            GGML_ASSERT(entry.source_scale == scale->data);
            GGML_ASSERT(entry.scale_size == ggml_nbytes(scale));
        }
        return &entry;
    };

    ggml_cuda_humming_fp8_cache_entry * up_entry = prepare_weight(src0, x_scale);
    ggml_cuda_humming_fp8_cache_entry * gate_entry = gate ? prepare_weight(gate, gate_scale) : nullptr;

    auto & lock_storage = ctx.humming_fp8_locks[ctx.curr_stream_no];
    const size_t block_m = m <= 16 ? 16 : 128;
    const size_t required_locks = ((m + block_m - 1) / block_m) * ((n + 63) / 64);
    if (lock_storage.count < required_locks) {
        if (lock_storage.ptr != nullptr) {
            lock_storage.retired.push_back(lock_storage.ptr);
        }
        CUDA_CHECK(cudaMalloc(&lock_storage.ptr, required_locks * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(lock_storage.ptr, 0, required_locks * sizeof(int32_t), stream));
        lock_storage.count = required_locks;
    }
    int32_t * locks = lock_storage.ptr;

    ggml_cuda_pool_alloc<nv_bfloat16> output_scratch(ctx.pool());
    nv_bfloat16 * output = output_scratch.alloc(m * n);
    ggml_cuda_pool_alloc<nv_bfloat16> gate_output(ctx.pool());
    if (gate != nullptr) {
        gate_output.alloc(m * n);
    }
    nv_bfloat16 * input = ggml_cuda_humming_get_input(ctx, src1, m * k, stream);

    const int sms = ggml_cuda_info().devices[ctx.device].nsm;
    ggml_cuda_humming_fp8_launch(
        input, src0->data, static_cast<const nv_bfloat16 *>(up_entry->scale),
        output, locks, n, k, m, sms, stream);
    if (gate != nullptr) {
        ggml_cuda_humming_fp8_launch(
            input, gate->data, static_cast<const nv_bfloat16 *>(gate_entry->scale),
            gate_output.get(), locks, n, k, m, sms, stream);
    }
    if (ggml_cuda_humming_finish_residual_rms(ctx, fusion, output, dst, n, m, stream)) {
        // See the NVFP4 path above.
    } else if (gate != nullptr && fusion->retain_bf16_output && n == 17408) {
        ggml_cuda_humming_fp8_swiglu_bf16(
            output, gate_output.get(), static_cast<nv_bfloat16 *>(dst->data), m * n, stream);
        ctx.humming_bf16_activations.insert(dst);
    } else {
        ggml_cuda_humming_fp8_output_bf16_to_f32(
            output, gate ? gate_output.get() : nullptr,
            static_cast<float *>(dst->data), m * n, stream);
    }
    return true;
}

static bool ggml_cuda_mul_mat_humming_fp8_block_impl(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * scale,
        const ggml_tensor * gate,
        const ggml_tensor * gate_scale,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (!ggml_cuda_humming_fp8_enabled() || src0->type != GGML_TYPE_F8_E4M3 ||
        !ggml_cuda_humming_fp8_is_repacked(src0) || scale == nullptr ||
        scale->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        dst->ne[2] != 1 || dst->ne[3] != 1 ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(scale) || !ggml_is_contiguous(dst)) {
        return false;
    }

    if (gate != nullptr) {
        if (gate_scale == nullptr || gate->type != GGML_TYPE_F8_E4M3 ||
            !ggml_cuda_humming_fp8_is_repacked(gate) ||
            !ggml_are_same_shape(gate, src0) || !ggml_are_same_stride(gate, src0) ||
            !ggml_is_contiguous(gate) || gate_scale->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(gate_scale, scale) || !ggml_is_contiguous(gate_scale)) {
            return false;
        }
    } else if (gate_scale != nullptr) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = src1->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (k % 128 != 0 || n % 128 != 0 || src1->ne[0] != k ||
        scale->ne[0] != n / 128 || scale->ne[1] != k / 128 ||
        scale->ne[2] != 1 || scale->ne[3] != 1 ||
        dst->ne[0] != n || dst->ne[1] != m ||
        !ggml_cuda_humming_fp8_block_supports_shape(n, k, m, cc)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    auto & lock_storage = ctx.humming_fp8_locks[ctx.curr_stream_no];
    const size_t block_m = m <= 16 ? 16 : 128;
    const size_t required_locks = ((m + block_m - 1) / block_m) * ((n + 63) / 64);
    if (lock_storage.count < required_locks) {
        if (lock_storage.ptr != nullptr) {
            lock_storage.retired.push_back(lock_storage.ptr);
        }
        CUDA_CHECK(cudaMalloc(&lock_storage.ptr, required_locks * sizeof(int32_t)));
        CUDA_CHECK(cudaMemsetAsync(lock_storage.ptr, 0, required_locks * sizeof(int32_t), stream));
        lock_storage.count = required_locks;
    }

    ggml_cuda_pool_alloc<nv_bfloat16> output(ctx.pool(), m * n);
    ggml_cuda_pool_alloc<nv_bfloat16> gate_output(ctx.pool());
    if (gate != nullptr) {
        gate_output.alloc(m * n);
    }
    nv_bfloat16 * input = ggml_cuda_humming_get_input(ctx, src1, m * k, stream);
    const int sms = ggml_cuda_info().devices[ctx.device].nsm;
    ggml_cuda_humming_fp8_block_launch(
        input, src0->data, static_cast<const float *>(scale->data), output.get(),
        lock_storage.ptr, n, k, m, sms, stream);
    if (gate != nullptr) {
        ggml_cuda_humming_fp8_block_launch(
            input, gate->data, static_cast<const float *>(gate_scale->data), gate_output.get(),
            lock_storage.ptr, n, k, m, sms, stream);
    }
    if (fusion != nullptr && ggml_cuda_humming_finish_residual_rms(
            ctx, fusion, output.get(), dst, n, m, stream)) {
        // The fused epilogue materializes the required F32 graph outputs and
        // retains a BF16 normalized activation for following projections.
    } else if (gate != nullptr && fusion != nullptr && fusion->retain_bf16_output && n == 17408) {
        ggml_cuda_humming_fp8_swiglu_bf16(
            output.get(), gate_output.get(), static_cast<nv_bfloat16 *>(dst->data), m * n, stream);
        ctx.humming_bf16_activations.insert(dst);
    } else {
        ggml_cuda_humming_fp8_output_bf16_to_f32(
            output.get(), gate ? gate_output.get() : nullptr, static_cast<float *>(dst->data), m * n, stream);
    }
    return true;
}

bool ggml_cuda_mul_mat_humming_fp8_block(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    return ggml_cuda_mul_mat_humming_fp8_block_impl(
        ctx, src0, src1, dst->src[2], nullptr, nullptr, dst, nullptr);
}

bool ggml_cuda_mul_mat_humming_fp8_block_fused(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * scale,
        ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    if (fusion == nullptr || fusion->gate != nullptr || fusion->gate_scale != nullptr) {
        return false;
    }
    return ggml_cuda_mul_mat_humming_fp8_block_impl(
        ctx, src0, src1, scale, nullptr, nullptr, dst, fusion);
}

bool ggml_cuda_mul_mat_humming_fp8_block_swiglu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        bool retain_bf16_output) {
    static const bool enabled = std::getenv("GGML_CUDA_DISABLE_HUMMING_FP8_BLOCK_SWIGLU") == nullptr;
    if (!enabled) {
        return false;
    }
    if (up == nullptr || gate == nullptr || up->op != GGML_OP_MUL_MAT || gate->op != GGML_OP_MUL_MAT ||
        up->src[1] != src1 || gate->src[1] != src1 || up->src[2] == nullptr || gate->src[2] == nullptr ||
        ggml_get_glu_op(dst) != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.retain_bf16_output = retain_bf16_output;
    return ggml_cuda_mul_mat_humming_fp8_block_impl(
        ctx, up->src[0], src1, up->src[2], gate->src[0], gate->src[2], dst, &fusion);
}
#endif

// Raw E4M3 weights have no embedded block metadata, but MMVQ consumes weights
// in units aligned with one Q8_1 activation block. ggml_cuda_mul_mat_vec_q()
// converts byte strides to these logical 32-value blocks before launch.
template<>
struct ggml_cuda_type_traits<GGML_TYPE_F8_E4M3> {
    static constexpr int qk = QK8_1;
    static constexpr int qr = 1;
    static constexpr int qi = QK8_1/16;
};

#define VDR_F8_E4M3_Q8_1_MMVQ 1

static __device__ __forceinline__ float e4m3fn_to_fp32_fast(const uint8_t x) {
    const uint32_t sign = uint32_t(x & 0x80) << 24;
    const uint32_t exp  = (x >> 3) & 0x0f;
    const uint32_t man  = x & 0x07;

    if (exp != 0) {
        if (exp == 0x0f && man == 0x07) {
            return NAN;
        }
        return __uint_as_float(sign | ((exp + 120) << 23) | (man << 20));
    }
    if (man == 0) {
        return __uint_as_float(sign);
    }

    const uint32_t p = 31 - __clz(man);
    return __uint_as_float(sign | ((p + 118) << 23) | ((man - (1U << p)) << (23 - p)));
}

static __device__ __forceinline__ float vec_dot_f8_e4m3_q8_1(
        const void * __restrict__ vf8,
        const block_q8_1 * __restrict__ bq8_1,
        const int32_t & kbx,
        const int32_t & iqs) {
    const uint4 packed = ((const uint4 *) vf8)[kbx*(QK8_1/16) + iqs];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t word = i < 4 ? packed.x : i < 8 ? packed.y : i < 12 ? packed.z : packed.w;
        const uint8_t value = uint8_t(word >> (8*(i & 3)));
        sum = fmaf(e4m3fn_to_fp32_fast(value), float(bq8_1->qs[16*iqs + i]), sum);
    }
    return sum * __low2float(bq8_1->ds);
}

static __device__ __forceinline__ float vec_dot_f8_e4m3_q8_1_lut(
        const void * __restrict__ vf8,
        const block_q8_1 * __restrict__ bq8_1,
        const int32_t & kbx,
        const int32_t & iqs,
        const half * __restrict__ lut) {
    const uint4 packed = ((const uint4 *) vf8)[kbx*(QK8_1/16) + iqs];
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t word = i < 4 ? packed.x : i < 8 ? packed.y : i < 12 ? packed.z : packed.w;
        const uint8_t value = uint8_t(word >> (8*(i & 3)));
        sum = fmaf(__half2float(lut[value]), float(bq8_1->qs[16*iqs + i]), sum);
    }
    return sum * __low2float(bq8_1->ds);
}

#ifdef GGML_CUDA_Q8_MMV_TEST_INSTRUMENTATION
static std::atomic<uint64_t> g_q8_0_mmv_test_baseline{0};
static std::atomic<uint64_t> g_q8_0_mmv_test_nwarps_2{0};

void ggml_cuda_q8_0_mmv_test_stats_reset() {
    g_q8_0_mmv_test_baseline.store(0, std::memory_order_relaxed);
    g_q8_0_mmv_test_nwarps_2.store(0, std::memory_order_relaxed);
}

void ggml_cuda_q8_0_mmv_test_stats_get(uint64_t * baseline, uint64_t * nwarps_2) {
    *baseline = g_q8_0_mmv_test_baseline.load(std::memory_order_relaxed);
    *nwarps_2 = g_q8_0_mmv_test_nwarps_2.load(std::memory_order_relaxed);
}

int ggml_cuda_q8_0_mmv_test_compute_capability() {
    return ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
}
#endif

#ifdef GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION
static std::atomic<uint64_t> g_moe_cache_flat_hits_factor_1{0};
static std::atomic<uint64_t> g_moe_cache_flat_hits_factor_2{0};

void ggml_cuda_moe_cache_flat_hits_test_stats_reset() {
    g_moe_cache_flat_hits_factor_1.store(0, std::memory_order_relaxed);
    g_moe_cache_flat_hits_factor_2.store(0, std::memory_order_relaxed);
}

void ggml_cuda_moe_cache_flat_hits_test_stats_get(uint64_t * factor_1, uint64_t * factor_2) {
    *factor_1 = g_moe_cache_flat_hits_factor_1.load(std::memory_order_relaxed);
    *factor_2 = g_moe_cache_flat_hits_factor_2.load(std::memory_order_relaxed);
}

#endif

static void ggml_cuda_moe_cache_flat_hits_record(ggml_cuda_moe_cache_flat_hits_path path) {
#ifdef GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION
    switch (path) {
        case ggml_cuda_moe_cache_flat_hits_path::factor_1:
            g_moe_cache_flat_hits_factor_1.fetch_add(1, std::memory_order_relaxed); break;
        case ggml_cuda_moe_cache_flat_hits_path::factor_2:
            g_moe_cache_flat_hits_factor_2.fetch_add(1, std::memory_order_relaxed); break;
    }
#else
    GGML_UNUSED(path);
#endif
}

#ifdef GGML_CUDA_Q8_POST_SILU_TEST_INSTRUMENTATION
static std::atomic<uint64_t> g_q8_post_silu_test_tuned_dispatches{0};

void ggml_cuda_q8_post_silu_test_kernel_stats_reset() {
    g_q8_post_silu_test_tuned_dispatches.store(0, std::memory_order_relaxed);
}

void ggml_cuda_q8_post_silu_test_kernel_stats_get(uint64_t * tuned_post_dispatches) {
    *tuned_post_dispatches = g_q8_post_silu_test_tuned_dispatches.load(std::memory_order_relaxed);
}

int ggml_cuda_q8_post_silu_test_compute_capability() {
    return ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
}
#endif

static constexpr __device__ vec_dot_q_cuda_t get_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F8_E4M3: return vec_dot_f8_e4m3_q8_1;
        case GGML_TYPE_Q1_0:    return vec_dot_q1_0_q8_1;
        case GGML_TYPE_Q2_0:    return vec_dot_q2_0_q8_1;
        case GGML_TYPE_Q2_0_G128: return vec_dot_q2_0_g128_q8_1;
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q4_A32:  return vec_dot_q4_a32_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q8_0_G128: return vec_dot_q8_0_g128_q8_1;
        case GGML_TYPE_MXFP4:   return vec_dot_mxfp4_q8_1;
        case GGML_TYPE_NVFP4:   return vec_dot_nvfp4_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        default:                return nullptr;
    }
}

static constexpr __host__ __device__ int get_vdr_mmvq(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F8_E4M3: return VDR_F8_E4M3_Q8_1_MMVQ;
        case GGML_TYPE_Q1_0:    return VDR_Q1_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0:    return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_0_G128: return VDR_Q2_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q4_A32:  return VDR_Q4_A32_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0_G128: return VDR_Q8_0_G128_Q8_1_MMVQ;
        case GGML_TYPE_MXFP4:   return VDR_MXFP4_Q8_1_MMVQ;
        case GGML_TYPE_NVFP4:   return VDR_NVFP4_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        default:                return 1;
    }
}

enum mmvq_parameter_table_id {
    MMVQ_PARAMETERS_GENERIC = 0,
    MMVQ_PARAMETERS_TURING,
    MMVQ_PARAMETERS_GCN,
    MMVQ_PARAMETERS_RDNA2,
    MMVQ_PARAMETERS_RDNA3_0,
    MMVQ_PARAMETERS_RDNA4,
    MMVQ_PARAMETERS_GB10
};

static constexpr __device__ mmvq_parameter_table_id get_device_table_id() {
#if defined(RDNA4)
    return MMVQ_PARAMETERS_RDNA4;
#elif defined(RDNA3_0)
    return MMVQ_PARAMETERS_RDNA3_0;
#elif defined(RDNA2) || defined(RDNA3_5)
    return MMVQ_PARAMETERS_RDNA2;
#elif defined(GCN) || defined(CDNA)
    return MMVQ_PARAMETERS_GCN;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING && __CUDA_ARCH__ < GGML_CUDA_CC_AMPERE
    return MMVQ_PARAMETERS_TURING;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK
    return MMVQ_PARAMETERS_GB10;
#else
    return MMVQ_PARAMETERS_GENERIC;
#endif
}

static __host__ mmvq_parameter_table_id get_device_table_id(int cc) {
    if (GGML_CUDA_CC_IS_RDNA4(cc)) {
        return MMVQ_PARAMETERS_RDNA4;
    }
    if (GGML_CUDA_CC_IS_RDNA3_0(cc)) {
        return MMVQ_PARAMETERS_RDNA3_0;
    }
    if (GGML_CUDA_CC_IS_RDNA2(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return MMVQ_PARAMETERS_RDNA2;
    }
    if (GGML_CUDA_CC_IS_GCN(cc) || GGML_CUDA_CC_IS_CDNA(cc)) {
        return MMVQ_PARAMETERS_GCN;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_TURING && ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_AMPERE) {
        return MMVQ_PARAMETERS_TURING;
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_DGX_SPARK) {
        return MMVQ_PARAMETERS_GB10;
    }
    return MMVQ_PARAMETERS_GENERIC;
}

// Per-architecture maximum batch size for which MMVQ should be used for MUL_MAT_ID.
// Returns a value <= MMVQ_MAX_BATCH_SIZE. Default is MMVQ_MAX_BATCH_SIZE.
// Check https://github.com/ggml-org/llama.cpp/pull/20905#issuecomment-4145835627 for details

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_pascal_older(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 4;
        case GGML_TYPE_NVFP4:   return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 6;
        case GGML_TYPE_Q4_1:    return 6;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_0:    return 6;
        case GGML_TYPE_Q5_1:    return 6;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_turing_plus(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 7;
        case GGML_TYPE_IQ3_S:   return 6;
        case GGML_TYPE_IQ3_XXS: return 7;
        case GGML_TYPE_MXFP4:   return 7;
        case GGML_TYPE_NVFP4:   return 8;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_gcn(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 5;
        case GGML_TYPE_IQ1_M:   return 5;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_Q2_K:    return 4;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 5;
        case GGML_TYPE_Q4_1:    return 5;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        case GGML_TYPE_Q8_0:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_cdna(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 5;
        case GGML_TYPE_IQ2_XS:  return 5;
        case GGML_TYPE_IQ2_XXS: return 5;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna1_rdna2(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_Q2_K:    return 7;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_K:    return 5;
        case GGML_TYPE_Q5_K:    return 6;
        case GGML_TYPE_Q6_K:    return 5;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna3(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 6;
        case GGML_TYPE_IQ1_M:   return 6;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 6;
        case GGML_TYPE_IQ4_XS:  return 6;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_K:    return 4;
        case GGML_TYPE_Q6_K:    return 4;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_rdna4(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:   return 7;
        case GGML_TYPE_IQ1_M:   return 7;
        case GGML_TYPE_IQ2_S:   return 4;
        case GGML_TYPE_IQ2_XS:  return 4;
        case GGML_TYPE_IQ2_XXS: return 4;
        case GGML_TYPE_IQ3_S:   return 4;
        case GGML_TYPE_IQ3_XXS: return 4;
        case GGML_TYPE_IQ4_NL:  return 7;
        case GGML_TYPE_IQ4_XS:  return 5;
        case GGML_TYPE_MXFP4:   return 5;
        case GGML_TYPE_NVFP4:   return 5;
        case GGML_TYPE_Q3_K:    return 4;
        case GGML_TYPE_Q4_0:    return 7;
        case GGML_TYPE_Q4_1:    return 7;
        case GGML_TYPE_Q4_K:    return 4;
        case GGML_TYPE_Q5_0:    return 7;
        case GGML_TYPE_Q5_1:    return 7;
        case GGML_TYPE_Q5_K:    return 5;
        case GGML_TYPE_Q6_K:    return 5;
        case GGML_TYPE_Q8_0:    return 7;
        default:                return MMVQ_MAX_BATCH_SIZE;
    }
}

// Host function: returns the max batch size for the current arch+type at runtime.
int get_mmvq_mmid_max_batch(ggml_type type, int cc) {
    // NVIDIA: Volta, Ada Lovelace, and Blackwell always use MMVQ for MUL_MAT_ID.
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {
            return MMVQ_MAX_BATCH_SIZE;
        }
        if (cc >= GGML_CUDA_CC_TURING) {
            return get_mmvq_mmid_max_batch_turing_plus(type);
        }
        return get_mmvq_mmid_max_batch_pascal_older(type);
    }

    // AMD
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        if (GGML_CUDA_CC_IS_RDNA4(cc)) {
            return get_mmvq_mmid_max_batch_rdna4(type);
        }
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            return get_mmvq_mmid_max_batch_rdna3(type);
        }
        if (GGML_CUDA_CC_IS_RDNA1(cc) || GGML_CUDA_CC_IS_RDNA2(cc)) {
            return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
        }
        if (GGML_CUDA_CC_IS_CDNA(cc)) {
            return get_mmvq_mmid_max_batch_cdna(type);
        }
        if (GGML_CUDA_CC_IS_GCN(cc)) {
            return get_mmvq_mmid_max_batch_gcn(type);
        }
    }
    return MMVQ_MAX_BATCH_SIZE;
}

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
    if (!ggml_is_quantized(type) && type != GGML_TYPE_F8_E4M3) {
        return false;
    }
    // k-quants cost more to decode and mvq redoes that per column, so MMQ wins sooner.
    // Only list quant-types MMQ supports, others would fall back to cuBLAS.
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_ADA_LOVELACE) {
        switch (type) { // tuned on RTX 4090
            case GGML_TYPE_Q2_K:
                return ne11 <= 4;
            case GGML_TYPE_Q3_K:
                return ne11 <= 6;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_BLACKWELL) {
        switch (type) { // tuned on RTX 5090
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 5;
            case GGML_TYPE_Q6_K:
                return ne11 <= 7;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc == GGML_CUDA_CC_DGX_SPARK) {
        switch (type) { // tuned on DGX Spark GB10
            case GGML_TYPE_Q2_K:
                return ne11 <= 6;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    if (GGML_CUDA_CC_IS_CDNA(cc)) {
        if (GGML_CUDA_CC_IS_CDNA1(cc)) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q5_1:
                    return ne11 <= 7;
                case GGML_TYPE_Q8_0:
                    return ne11 <= 6;
                case GGML_TYPE_Q2_K:
                    return ne11 <= 4;
                case GGML_TYPE_Q3_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q4_K:
                    return ne11 <= 2;
                case GGML_TYPE_Q5_K:
                    return ne11 <= 3;
                case GGML_TYPE_Q6_K:
                    return ne11 <= 4;
                case GGML_TYPE_IQ1_S:
                    return ne11 <= 5;
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ4_XS:
                    return ne11 <= 6;
                default:
                    return ne11 <= MMVQ_MAX_BATCH_SIZE;
            }
        }
        switch (type) { // tuned for CDNA2
            case GGML_TYPE_Q2_K:
                return ne11 <= 5;
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
                return ne11 <= 3;
            case GGML_TYPE_Q6_K:
                return ne11 <= 5;
            default:
                return ne11 <= MMVQ_MAX_BATCH_SIZE;
        }
    }
    return ne11 <= MMVQ_MAX_BATCH_SIZE;
}

// Device constexpr: returns the max batch size for the current arch+type at compile time.
template <ggml_type type>
static constexpr __device__ int get_mmvq_mmid_max_batch_for_device() {
#if defined(RDNA4)
    return get_mmvq_mmid_max_batch_rdna4(type);
#elif defined(RDNA3)
    return get_mmvq_mmid_max_batch_rdna3(type);
#elif defined(RDNA2) || defined(RDNA1)
    return get_mmvq_mmid_max_batch_rdna1_rdna2(type);
#elif defined(CDNA)
    return get_mmvq_mmid_max_batch_cdna(type);
#elif defined(GCN)
    return get_mmvq_mmid_max_batch_gcn(type);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_VOLTA || __CUDA_ARCH__ >= GGML_CUDA_CC_ADA_LOVELACE)
    return MMVQ_MAX_BATCH_SIZE;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
    return get_mmvq_mmid_max_batch_turing_plus(type);
#else
    return get_mmvq_mmid_max_batch_pascal_older(type);
#endif
}

static constexpr __host__ __device__ int calc_nwarps(ggml_type type, int ncols_dst, mmvq_parameter_table_id table_id, bool small_k = false, bool halve_iters = false) {
    if (table_id == MMVQ_PARAMETERS_GENERIC) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    } else if (table_id == MMVQ_PARAMETERS_GCN) {
        switch (ncols_dst) {
            case 1:
            case 2:
            case 3:
            case 4:
                return 2;
            case 5:
            case 6:
            case 7:
            case 8:
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_RDNA4) {
        // nwarps=8 benefits types with simple vec_dot on RDNA4 (ncols_dst=1).
        // Types with complex vec_dot (Q3_K, IQ2_*, IQ3_*) regress due to register
        // pressure and lookup table contention at higher thread counts.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {
        // RDNA3 (W7900): stricter whitelist than RDNA4.
        // Q2_K / Q5_K / IQ4_XS regress in full quant sweeps.
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return 8;
                case GGML_TYPE_Q6_K:
                    return 2;
                case GGML_TYPE_IQ4_NL:
                    return 8;
                default:
                    return 1;
            }
        }
        return 1;
    }
    if (table_id == MMVQ_PARAMETERS_TURING) {
        if (ncols_dst == 1) {
            switch (type) {
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return 2;
                default:
                    return 4;
            }
        }
        switch (ncols_dst) {
            case 2:
            case 3:
            case 4:
                return 4;
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    if (table_id == MMVQ_PARAMETERS_GB10) {
        const int generic = calc_nwarps(type, ncols_dst, MMVQ_PARAMETERS_GENERIC);
        // Only worth the wider block when it actually retires the K loop in half the trips (Observation)
        if (ncols_dst == 1 && !small_k && halve_iters) {
            switch (type) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_IQ4_NL:
                    return 2 * generic;
                default:
                    break;
            }
        }
        return generic;
    }
    return 1;
}

static constexpr __host__ __device__ int calc_rows_per_block(int ncols_dst, int table_id, bool small_k = false, int nwarps = 1) {
    if (table_id == MMVQ_PARAMETERS_GENERIC || table_id == MMVQ_PARAMETERS_GCN || table_id == MMVQ_PARAMETERS_TURING || table_id == MMVQ_PARAMETERS_GB10) {
        switch (ncols_dst) {
            case 1:
                return small_k ? nwarps : 1;
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 8:
                return 2;
            default:
                return 1;
        }
    }
    return 1;
}

bool ggml_cuda_q8_0_mmv_post_silu_supported(int cc, int64_t ncols_x) {
    if (cc != 860 || ncols_x <= 0 || ncols_x % QK8_0 != 0) {
        return false;
    }
    constexpr int qk  = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qk;
    constexpr int qi  = ggml_cuda_type_traits<GGML_TYPE_Q8_0>::qi;
    constexpr int vdr = get_vdr_mmvq(GGML_TYPE_Q8_0);
    const mmvq_parameter_table_id table_id = get_device_table_id(cc);
    const int nwarps = calc_nwarps(GGML_TYPE_Q8_0, 1, table_id);
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_iter_1warp = vdr * WARP_SIZE / qi;
    const bool small_k = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
    return ggml_cuda_select_q8_0_mmv_path({
        cc, 1, false, ggml_cuda_q8_0_mmv_fusion_kind::post_only, small_k, false,
    }) == ggml_cuda_q8_0_mmv_path::nwarps_2;
}

template <ggml_type type, int ncols_dst, bool has_fusion, bool small_k = false, bool halve_iters = false,
          int nwarps_override = 0, bool has_post_silu = false>
__launch_bounds__((nwarps_override > 0 ? nwarps_override : calc_nwarps(type, ncols_dst, get_device_table_id(), small_k, halve_iters))*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q(
        const void * vx_ptr, const void * vy_ptr, const int32_t * ids_ptr, const ggml_cuda_mmvq_fusion_args_device fusion, float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const uint32_t ids_stride) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr mmvq_parameter_table_id table_id = get_device_table_id();
    constexpr int nwarps = nwarps_override > 0 ? nwarps_override : calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    const     int tid = warp_size*threadIdx.y + threadIdx.x;
    const     int row0 = rows_per_cuda_block*blockIdx.x;
    const     int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps*warp_size / qi;

    __shared__ half f8_lut[type == GGML_TYPE_F8_E4M3 ? 256 : 1];
    if constexpr (type == GGML_TYPE_F8_E4M3) {
        for (int i = tid; i < 256; i += nwarps*warp_size) {
            f8_lut[i] = __float2half(e4m3fn_to_fp32_fast(uint8_t(i)));
        }
        __syncthreads();
    }

    const uint32_t channel_dst = blockIdx.y;

    uint32_t channel_x;
    uint32_t channel_y;
    uint32_t sample_dst;

    ggml_cuda_pdl_sync();
    channel_x  = ncols_dst == 1 && ids ? ids[channel_dst]                     : fastdiv(channel_dst, channel_ratio);
    channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
    sample_dst = blockIdx.z;

    const uint32_t sample_x    = fastdiv(sample_dst, sample_ratio);
    const uint32_t sample_y    = sample_dst;

    bool use_gate = false;
    bool use_bias = false;
    bool use_gate_bias = false;
    bool use_scale = false;
    bool use_gate_scale = false;
    [[maybe_unused]] const void * vgate = nullptr;
    const float * x_bias = nullptr;
    const float * gate_bias = nullptr;
    const void * x_scale = nullptr;
    const void * gate_scale = nullptr;
    const float * residual = nullptr;
    ggml_glu_op active_glu;

    if constexpr (has_fusion) {
        use_gate      = fusion.gate      != nullptr;
        use_bias      = fusion.x_bias    != nullptr;
        use_gate_bias = fusion.gate_bias != nullptr && use_gate;
        vgate         = fusion.gate;
        x_bias        = (const float *) fusion.x_bias;
        gate_bias     = (const float *) fusion.gate_bias;
        active_glu    = fusion.glu_op;
        if constexpr (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_F8_E4M3) {
            use_scale      = fusion.x_scale    != nullptr;
            use_gate_scale = fusion.gate_scale != nullptr && use_gate;
        x_scale        = fusion.x_scale;
        gate_scale     = fusion.gate_scale;
        residual       = fusion.residual;
        }
    }


    [[maybe_unused]] float x_biases[ncols_dst]    = { 0.0f };
    [[maybe_unused]] float gate_biases[ncols_dst] = { 0.0f };
    [[maybe_unused]] float x_scales = 1.0f;
    [[maybe_unused]] float gate_scales = 1.0f;
    if constexpr (has_fusion) {
        // 1. Hide latency by prefetching bias, gates and scales here
        // 2. load only on threads that won't die after partial sum calculation
        const uint32_t channel_bias = ids ? channel_x : channel_dst;
        if (threadIdx.x < rows_per_cuda_block && threadIdx.y == 0 &&
            (rows_per_cuda_block == 1 || uint32_t(row0 + threadIdx.x) < stride_col_dst)) {
            if (use_bias) {
                x_bias = x_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    x_biases[j] = x_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if (use_gate_bias) {
                gate_bias = gate_bias + sample_dst * stride_sample_dst + channel_bias * stride_channel_dst + row0;
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    gate_biases[j] = gate_bias[j * stride_col_dst + threadIdx.x];
                }
            }
            if constexpr (type == GGML_TYPE_NVFP4) {
                if (use_scale) {
                    x_scales = ((const float *) x_scale)[ids ? channel_x : 0];
                }
                if (use_gate_scale) {
                    gate_scales = ((const float *) gate_scale)[ids ? channel_x : 0];
                }
            } else if constexpr (type == GGML_TYPE_F8_E4M3) {
                if (use_scale) {
                    x_scales = __bfloat162float(((const nv_bfloat16 *) x_scale)[row0 + threadIdx.x]);
                }
                if (use_gate_scale) {
                    gate_scales = __bfloat162float(((const nv_bfloat16 *) gate_scale)[row0 + threadIdx.x]);
                }
            }
        }
    }

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};
    float tmp_gate[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const block_q8_1 * y = ((const block_q8_1 *) vy) + sample_y*stride_sample_y + channel_y*stride_channel_y;
    const int kbx_offset = sample_x*stride_sample_x + channel_x*stride_channel_x + row0*stride_row_x;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1); // y block index that aligns with kbx

        // x block quant index when casting the quants to int
        const int kqs = vdr * (tid % (qi/vdr));

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                if constexpr (type == GGML_TYPE_F8_E4M3) {
                    tmp[j][i] += vec_dot_f8_e4m3_q8_1_lut(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs, f8_lut);
                } else {
                    tmp[j][i] += vec_dot_q_cuda(
                        vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                }
                if constexpr (has_fusion) {
                    if (use_gate) {
                        if constexpr (type == GGML_TYPE_F8_E4M3) {
                            tmp_gate[j][i] += vec_dot_f8_e4m3_q8_1_lut(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs, f8_lut);
                        } else {
                            tmp_gate[j][i] += vec_dot_q_cuda(
                                vgate, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs);
                        }
                    }
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps-1 > 0 ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];
    [[maybe_unused]] __shared__ float tmp_shared_gate[(has_fusion && (nwarps-1 > 0)) ? nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_shared_gate[threadIdx.y-1][j][i][threadIdx.x] = tmp_gate[j][i];
                    }
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

    dst += sample_dst*stride_sample_dst + channel_dst*stride_channel_dst + row0;

    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
                if constexpr (has_fusion) {
                    if (use_gate) {
                        tmp_gate[j][i] += tmp_shared_gate[l][j][i][threadIdx.x];
                    }
                }
            }
            tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);
            if constexpr (has_fusion) {
                if (use_gate) {
                    tmp_gate[j][i] = warp_reduce_sum<warp_size>(tmp_gate[j][i]);
                }
            }

            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                float result = tmp[j][i];
                if constexpr (has_fusion) {
                    if constexpr (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_F8_E4M3) {
                        result *= x_scales;
                    }
                    result += x_biases[j];
                    if (residual != nullptr) {
                        result += residual[sample_dst*stride_sample_dst +
                                           channel_dst*stride_channel_dst +
                                           j*stride_col_dst + row0 + i];
                    }
                    if (use_gate) {
                        float gate_value = tmp_gate[j][i];
                        if constexpr (type == GGML_TYPE_NVFP4 || type == GGML_TYPE_F8_E4M3) {
                            gate_value *= gate_scales;
                        }
                        gate_value += gate_biases[j];
                        switch (active_glu) {
                            case GGML_GLU_OP_SWIGLU:
                                result *= ggml_cuda_op_silu_single(gate_value);
                                break;
                            case GGML_GLU_OP_GEGLU:
                                result *= ggml_cuda_op_gelu_single(gate_value);
                                break;
                            case GGML_GLU_OP_SWIGLU_OAI:
                                result = ggml_cuda_op_swiglu_oai_single(gate_value, result);
                                break;
                            default:
                                result = result * gate_value;
                                break;
                        }
                    }
                }
                if constexpr (has_post_silu) {
                    result = __fmaf_rn(fusion.post_scale, result, 0.0f);
                    result = ggml_cuda_op_silu_single(result);
                }
                dst[j*stride_col_dst + i] = result;
            }
        }
    }

    if constexpr (!has_fusion) {
        GGML_UNUSED_VARS(use_gate, use_bias, use_gate_bias, use_scale, use_gate_scale, active_glu, gate_bias, x_bias, x_scale, gate_scale, residual, tmp_gate);
    }
    if constexpr (type != GGML_TYPE_NVFP4 && type != GGML_TYPE_F8_E4M3) {
        GGML_UNUSED_VARS(use_scale, use_gate_scale, x_scale, gate_scale, x_scales, gate_scales);
    }
}

// Dedicated MoE multi-token kernel.
// Grid: (ceil(nrows_x / c_rows_per_block), nchannels_dst)
// Block: (warp_size, ncols_dst) - each warp handles one token independently.
// No shared memory reduction needed since each warp works alone.
template <ggml_type type, int c_rows_per_block, bool has_fusion, bool has_clamp, bool flat_hits = false>
__launch_bounds__((flat_hits ? 2 : get_mmvq_mmid_max_batch_for_device<type>())*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mul_mat_vec_q_moe(
        const void * vx_ptr, const void * vy_ptr,
        const int32_t * ids_ptr, const int32_t * act_ids_ptr,
        const void * gate_ptr, const int32_t * gate_ids_ptr,
        float * dst_ptr,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const float up_min, const float up_max, const float gate_min, const float gate_max) {
    const void    * GGML_CUDA_RESTRICT vx  = vx_ptr;
    const void    * GGML_CUDA_RESTRICT vy  = vy_ptr;
    const int32_t * GGML_CUDA_RESTRICT ids = ids_ptr;
    const int32_t * GGML_CUDA_RESTRICT act_ids = act_ids_ptr;
    const void    * GGML_CUDA_RESTRICT gate = gate_ptr;
    const int32_t * GGML_CUDA_RESTRICT gate_ids = gate_ids_ptr;
    float         * GGML_CUDA_RESTRICT dst = dst_ptr;

    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type);

    uint32_t token_idx;
    uint32_t channel_dst;
    uint32_t route_idx;
    if constexpr (flat_hits) {
        token_idx = 0;
        route_idx = blockIdx.y*blockDim.y + threadIdx.y;
        channel_dst = route_idx;
        if (route_idx >= ncols_dst) {
            return;
        }
    } else {
        token_idx = threadIdx.y;
        channel_dst = blockIdx.y;
        route_idx = channel_dst + token_idx*ids_stride;
        if (token_idx >= ncols_dst) {
            return;
        }
    }
    const int      row0        = c_rows_per_block*blockIdx.x;
    const int      blocks_per_row_x = ncols_x / qk;
    constexpr int  blocks_per_iter  = vdr * warp_size / qi;

    ggml_cuda_pdl_sync();
    const uint32_t channel_x = ids[route_idx];
    const uint32_t channel_gate = has_fusion
        ? gate_ids[route_idx] : 0;
    const uint32_t channel_y = act_ids
        ? act_ids[route_idx]
        : fastmodulo(channel_dst, nchannels_y);

    const block_q8_1 * y = ((const block_q8_1 *) vy) + channel_y*stride_channel_y + token_idx*stride_col_y;
    const int kbx_offset  = channel_x*stride_channel_x + row0*stride_row_x;
    const int gate_kbx_offset = channel_gate*stride_channel_x + row0*stride_row_x;

    // partial sum for each thread
    float tmp[c_rows_per_block] = {0.0f};
    float tmp_gate[c_rows_per_block] = {0.0f};

    for (int kbx = threadIdx.x / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk/QK8_1);
        const int kqs = vdr * (threadIdx.x % (qi/vdr));

#pragma unroll
        for (int i = 0; i < c_rows_per_block; ++i) {
            if (uint32_t(row0 + i) < nrows_x) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + i*stride_row_x + kbx, kqs);
                if constexpr (has_fusion) {
                    tmp_gate[i] += vec_dot_q_cuda(gate, &y[kby], gate_kbx_offset + i*stride_row_x + kbx, kqs);
                }
            }
        }
    }

    ggml_cuda_pdl_lc();

    // Warp-level reduction only - no shared memory needed
#pragma unroll
    for (int i = 0; i < c_rows_per_block; ++i) {
        tmp[i] = warp_reduce_sum<warp_size>(tmp[i]);
        if constexpr (has_fusion) {
            tmp_gate[i] = warp_reduce_sum<warp_size>(tmp_gate[i]);
        }
    }

    // Write results
    if (threadIdx.x < c_rows_per_block && (c_rows_per_block == 1 || uint32_t(row0 + threadIdx.x) < nrows_x)) {
        float result = tmp[threadIdx.x];
        if constexpr (has_fusion) {
            float gate_value = tmp_gate[threadIdx.x];
            if constexpr (has_clamp) {
                result = fmaxf(fminf(result, up_max), up_min);
                gate_value = fmaxf(fminf(gate_value, gate_max), gate_min);
            }
            result *= ggml_cuda_op_silu_single(gate_value);
        }
        if constexpr (flat_hits) {
            dst[route_idx*stride_channel_dst + row0 + threadIdx.x] = result;
        } else {
            dst[channel_dst*stride_channel_dst + token_idx*stride_col_dst + row0 + threadIdx.x] = result;
        }
    }

    if constexpr (!has_clamp) {
        GGML_UNUSED_VARS(up_min, up_max, gate_min, gate_max);
    }
}

template<ggml_type type>
static std::pair<dim3, dim3> calc_launch_params(
        const int ncols_dst, const int nrows_x, const int nchannels_dst, const int nsamples_or_ntokens,
        const int warp_size, const mmvq_parameter_table_id table_id, const bool small_k = false, const bool halve_iters = false) {
    const int nwarps = calc_nwarps(type, ncols_dst, table_id, small_k, halve_iters);
    const int rpb = calc_rows_per_block(ncols_dst, table_id, small_k, nwarps);
    const int64_t nblocks = (nrows_x + rpb - 1) / rpb;
    const dim3 block_nums(nblocks, nchannels_dst, nsamples_or_ntokens);
    const dim3 block_dims(warp_size, nwarps, 1);
    return {block_nums, block_dims};
}

static bool mmvq_has_existing_fusion(const ggml_cuda_mmvq_fusion_args_device & fusion) {
    return fusion.gate != nullptr || fusion.x_bias != nullptr || fusion.gate_bias != nullptr ||
           fusion.x_scale != nullptr || fusion.gate_scale != nullptr || fusion.residual != nullptr;
}

template<ggml_type type, int c_ncols_dst, bool small_k = false, bool halve_iters = false, int nwarps_override = 0>
static void mul_mat_vec_q_switch_fusion(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mmvq_fusion_args_device fusion, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst, const uint3 channel_ratio, const uint32_t stride_channel_x,
        const uint32_t stride_channel_y, const uint32_t stride_channel_dst, const uint3 sample_ratio,
        const uint32_t stride_sample_x, const uint32_t stride_sample_y, const uint32_t stride_sample_dst,
        const dim3 & block_nums, const dim3 & block_dims, const int nbytes_shared,
        const uint32_t ids_stride, cudaStream_t stream) {

    const bool has_existing_fusion = mmvq_has_existing_fusion(fusion);
    if constexpr (c_ncols_dst == 1) {
        if (has_existing_fusion) {
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
            ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, true, small_k, halve_iters, nwarps_override>, launch_params,
                 vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
            return;
        }
    }

    GGML_ASSERT(!has_existing_fusion && !fusion.post_silu && "fusion only supported for ncols_dst=1");

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, nbytes_shared, stream);
    ggml_cuda_kernel_launch(mul_mat_vec_q<type, c_ncols_dst, false, small_k, halve_iters, nwarps_override>, launch_params,
        vx, vy, ids, fusion, dst, ncols_x, nchannels_y, stride_row_x, stride_col_y, stride_col_dst,
        channel_ratio, stride_channel_x, stride_channel_y, stride_channel_dst,
        sample_ratio, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride);
}

template <ggml_type type, bool has_fusion = false, bool has_clamp = false>
static void mul_mat_vec_q_moe_launch(
        const void * vx, const void * vy, const int32_t * ids,
        const int32_t * act_ids, const void * gate,
        const int32_t * gate_ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x, const uint32_t stride_col_y, const uint32_t stride_col_dst,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t ncols_dst, const uint32_t ids_stride,
        const int warp_size, const int nchannels_dst,
        const float up_min, const float up_max, const float gate_min,
        const float gate_max, cudaStream_t stream) {

    constexpr int rows_per_block = 2; // 2 gives best perf based on tuning
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, nchannels_dst);
    const dim3 block_dims(warp_size, ncols_dst);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, has_fusion, has_clamp>, launch_params,
        vx, vy, ids, act_ids, gate, gate_ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, stride_col_y, stride_col_dst,
        stride_channel_x, stride_channel_y, stride_channel_dst,
        ncols_dst, ids_stride, up_min, up_max, gate_min, gate_max);
}

template <ggml_type type, bool has_fusion = false, bool has_clamp = false>
static void mul_mat_vec_q_moe_cache_launch(
        const void * vx, const void * vy, const int32_t * ids,
        const int32_t * act_ids, const void * gate,
        const int32_t * gate_ids, float * dst,
        const uint32_t ncols_x, const uint3 nchannels_y, const uint32_t nrows_x,
        const uint32_t stride_row_x,
        const uint32_t stride_channel_x, const uint32_t stride_channel_y, const uint32_t stride_channel_dst,
        const uint32_t n_hits, const int warp_size,
        const float up_min, const float up_max, const float gate_min,
        const float gate_max, cudaStream_t stream) {

    constexpr int rows_per_block = 2;
    const int device = ggml_cuda_get_device();
    const auto flat_hits = ggml_cuda_select_moe_cache_flat_hits(
        ggml_cuda_info().devices[device].cc, type, n_hits);
    if (flat_hits.path == ggml_cuda_moe_cache_flat_hits_path::factor_1) {
        mul_mat_vec_q_moe_launch<type, has_fusion, has_clamp>(
            vx, vy, ids, act_ids, gate, gate_ids, dst, ncols_x, nchannels_y, nrows_x,
            stride_row_x, 0, 0, stride_channel_x, stride_channel_y, stride_channel_dst,
            1, n_hits, warp_size, n_hits, up_min, up_max, gate_min, gate_max, stream);
        ggml_cuda_moe_cache_flat_hits_record(flat_hits.path);
        return;
    }
    const int64_t nblocks_rows = (nrows_x + rows_per_block - 1) / rows_per_block;
    const dim3 block_nums(nblocks_rows, (n_hits + flat_hits.factor - 1) / flat_hits.factor);
    const dim3 block_dims(warp_size, flat_hits.factor);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);

    ggml_cuda_kernel_launch(mul_mat_vec_q_moe<type, rows_per_block, has_fusion, has_clamp, true>, launch_params,
        vx, vy, ids, act_ids, gate, gate_ids, dst, ncols_x, nchannels_y, nrows_x,
        stride_row_x, 0, 0, stride_channel_x, stride_channel_y, stride_channel_dst,
        n_hits, n_hits, up_min, up_max, gate_min, gate_max);
    ggml_cuda_moe_cache_flat_hits_record(flat_hits.path);
}

template <ggml_type type>
static void mul_mat_vec_q_switch_ncols_dst(
        const void * vx, const void * vy, const int32_t * ids, const ggml_cuda_mmvq_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, bool allow_small_k) {

    GGML_ASSERT(ncols_x % ggml_blck_size(type) == 0);
    GGML_ASSERT(ncols_dst <= MMVQ_MAX_BATCH_SIZE);

    const uint3 nchannels_y_fd   = ids ? init_fastdiv_values(nchannels_y) : make_uint3(0, 0, 0);
    const uint3 channel_ratio_fd = ids ? make_uint3(0, 0, 0)              : init_fastdiv_values(nchannels_dst / nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(nsamples_dst  / nsamples_x);

    const int device = ggml_cuda_get_device();
    const int                     cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const mmvq_parameter_table_id table_id  = get_device_table_id(cc);

    const bool has_ids = ids != nullptr;

    // How the K loop divides up at the baseline block width, both decisions below use these.
    constexpr int qk                    = ggml_cuda_type_traits<type>::qk;
    constexpr int qi                    = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr                   = get_vdr_mmvq(type);
    const int     blocks_per_row_x      = ncols_x / qk;
    const int     blocks_per_iter_1warp = vdr * warp_size / qi;

    const auto should_use_small_k = [&](int c_ncols_dst) {
        // When K is small, increase rows_per_block to match nwarps so each warp has more work to do
        // Trigger when the full thread block covers all K blocks in a single loop iteration and few threads remain idle.
        const int  nwarps = calc_nwarps(type, c_ncols_dst, table_id);
        bool       use    = nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;

        constexpr std::array<ggml_type, 2> iq_slow_turing = {
            GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S,
        };
        constexpr std::array<ggml_type, 8> iq_slow_other = {
            GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,   GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,   GGML_TYPE_IQ4_XS,
        };
        constexpr std::array<ggml_type, 3> slow_pascal = {
            GGML_TYPE_IQ3_S,
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
        };

        const bool is_nvidia_turing_plus  = GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
        const bool is_nvidia_pascal_older = GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_VOLTA;

        if (is_nvidia_turing_plus) {
            if (ncols_dst == 1 &&
                    std::find(iq_slow_turing.begin(), iq_slow_turing.end(), type) != iq_slow_turing.end()) {
                use = false;
            }
        } else if ((ncols_dst == 1 && std::find(iq_slow_other.begin(), iq_slow_other.end(), type) != iq_slow_other.end()) ||
                (is_nvidia_pascal_older && std::find(slow_pascal.begin(), slow_pascal.end(), type) != slow_pascal.end()) ||
                GGML_CUDA_CC_IS_RDNA(cc)) {
            use = false;
        }

        return use;
    };

    // Whether doubling nwarps pays off on the ncols_dst == 1 path, where K sets the K loop trip count.
    const auto should_halve_iters = [&] {
        if (table_id != MMVQ_PARAMETERS_GB10) {
            return false;
        }

        // Expert rows are gathered per token, so a wider block adds reduction work without reuse.
        if (has_ids) {
            return false;
        }

        const int blocks_per_iter = calc_nwarps(type, 1, table_id) * blocks_per_iter_1warp;
        const int iters           = (blocks_per_row_x + blocks_per_iter - 1) /  blocks_per_iter;
        const int iters_wide      = (blocks_per_row_x + blocks_per_iter * 2 - 1) / (blocks_per_iter * 2);

        // An odd trip count leaves half the wider block idle for its last iteration, that tail is
        // only affordable once the loop is long enough to dilute it to an eighth of the work (observation).
        const int idle = iters_wide * 2 - iters;

        return idle * 8 <= iters_wide * 2;
    };

    if (has_ids && ncols_dst > 1) {
        // Multi-token MUL_MAT_ID path - dedicated MoE kernel
        mul_mat_vec_q_moe_launch<type>(
            vx, vy, ids, nullptr, nullptr, nullptr, dst,
            ncols_x, nchannels_y_fd, nrows_x,
            stride_row_x, stride_col_y, stride_col_dst,
            stride_channel_x, stride_channel_y, stride_channel_dst,
            ncols_dst, ids_stride, warp_size, nchannels_dst,
            0.0f, 0.0f, 0.0f, 0.0f, stream);
        return;
    }

    switch (ncols_dst) {
        case 1: {
            // static, else MSVC lambda capture breaks the constexpr uses below
            static constexpr int c_ncols_dst = 1;
            const bool has_existing_fusion = mmvq_has_existing_fusion(fusion);

            if constexpr (type == GGML_TYPE_IQ3_XXS) {
                // Blackwell has enough grid-level parallelism when several indexed experts are active.
                // A single warp avoids redundant per-block setup and cross-warp reduction in this case.
                if (has_ids && nchannels_dst >= 8 && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_BLACKWELL) {
                    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, 1, 1);
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, 1>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y,
                        stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y,
                        stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y,
                        stride_sample_dst, block_nums, block_dims, 0, ids_stride, stream);
                    break;
                }
            }

            // Tag types keep the flags compile-time, so __launch_bounds__ matches what is launched.
            const auto launch = [&](auto small_k_tag, auto halve_iters_tag) {
                constexpr bool c_small_k = decltype(small_k_tag)::value;
                // Types the table does not promote would compile a second, identical kernel.
                constexpr bool c_promoted =
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, true) !=
                    calc_nwarps(type, c_ncols_dst, MMVQ_PARAMETERS_GB10, false, false);

                constexpr bool c_halve_iters = decltype(halve_iters_tag)::value && c_promoted;

                const std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst,
                                                                              nsamples_dst, warp_size, table_id, c_small_k, c_halve_iters);
                mul_mat_vec_q_switch_fusion<type, c_ncols_dst, c_small_k, c_halve_iters>(
                    vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                    channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst, sample_ratio_fd,
                    stride_sample_x, stride_sample_y, stride_sample_dst, dims.first, dims.second, 0, ids_stride,
                    stream);
            };

            const bool use_small_k  = allow_small_k && should_use_small_k(c_ncols_dst);
            const bool halve_iters = !use_small_k && should_halve_iters();

            if constexpr (type == GGML_TYPE_Q8_0) {
                const ggml_cuda_q8_0_mmv_fusion_kind fusion_kind = fusion.post_silu
                    ? ggml_cuda_q8_0_mmv_fusion_kind::post_only
                    : has_existing_fusion ? ggml_cuda_q8_0_mmv_fusion_kind::existing
                                          : ggml_cuda_q8_0_mmv_fusion_kind::none;
                const ggml_cuda_q8_0_mmv_path path = ggml_cuda_select_q8_0_mmv_path({
                    cc, c_ncols_dst, has_ids, fusion_kind, use_small_k, halve_iters,
                });
                if (fusion.post_silu) {
                    GGML_ASSERT(ggml_cuda_q8_0_mmv_post_silu_supported(cc, ncols_x) &&
                        path == ggml_cuda_q8_0_mmv_path::nwarps_2 && !has_existing_fusion);
                    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, 2, 1);
                    const ggml_cuda_kernel_launch_params launch_params =
                        ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
                    ggml_cuda_kernel_launch(
                        mul_mat_vec_q<GGML_TYPE_Q8_0, 1, false, false, false, 2, true>, launch_params,
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y,
                        stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y,
                        stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y,
                        stride_sample_dst, ids_stride);
#ifdef GGML_CUDA_Q8_MMV_TEST_INSTRUMENTATION
                    g_q8_0_mmv_test_nwarps_2.fetch_add(1, std::memory_order_relaxed);
#endif
#ifdef GGML_CUDA_Q8_POST_SILU_TEST_INSTRUMENTATION
                    g_q8_post_silu_test_tuned_dispatches.fetch_add(1, std::memory_order_relaxed);
#endif
                    break;
                }
                if (path == ggml_cuda_q8_0_mmv_path::nwarps_2) {
                    const dim3 block_nums(nrows_x, nchannels_dst, nsamples_dst);
                    const dim3 block_dims(warp_size, 2, 1);
                    mul_mat_vec_q_switch_fusion<type, c_ncols_dst, false, false, 2>(
                        vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y,
                        stride_col_dst, channel_ratio_fd, stride_channel_x, stride_channel_y,
                        stride_channel_dst, sample_ratio_fd, stride_sample_x, stride_sample_y,
                        stride_sample_dst, block_nums, block_dims, 0, ids_stride, stream);
#ifdef GGML_CUDA_Q8_MMV_TEST_INSTRUMENTATION
                    g_q8_0_mmv_test_nwarps_2.fetch_add(1, std::memory_order_relaxed);
#endif
                    break;
                }
#ifdef GGML_CUDA_Q8_MMV_TEST_INSTRUMENTATION
                g_q8_0_mmv_test_baseline.fetch_add(1, std::memory_order_relaxed);
#endif
            }

            if (use_small_k) {
                launch(std::true_type{},  std::false_type{});
            } else if (halve_iters) {
                launch(std::false_type{}, std::true_type{});
            } else {
                launch(std::false_type{}, std::false_type{});
            }
        } break;
        case 2: {
            constexpr int c_ncols_dst = 2;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 3: {
            constexpr int c_ncols_dst = 3;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 4: {
            constexpr int c_ncols_dst = 4;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 5: {
            constexpr int c_ncols_dst = 5;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 6: {
            constexpr int c_ncols_dst = 6;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 7: {
            constexpr int c_ncols_dst = 7;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        case 8: {
            constexpr int c_ncols_dst = 8;
            std::pair<dim3, dim3> dims = calc_launch_params<type>(c_ncols_dst, nrows_x, nchannels_dst, nsamples_dst, warp_size, table_id);
            mul_mat_vec_q_switch_fusion<type, c_ncols_dst>(vx, vy, ids, fusion, dst, ncols_x, nchannels_y_fd, stride_row_x, stride_col_y, stride_col_dst,
                 channel_ratio_fd, stride_channel_x, stride_channel_y, stride_channel_dst,
                 sample_ratio_fd, stride_sample_x, stride_sample_y, stride_sample_dst,
                 dims.first, dims.second, 0, ids_stride, stream);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}
static void mul_mat_vec_q_switch_type(
        const void * vx, const ggml_type type_x, const void * vy, const int32_t * ids, const ggml_cuda_mmvq_fusion_args_device fusion, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst,
        const int nchannels_x, const int nchannels_y, const int nchannels_dst,
        const int stride_channel_x, const int stride_channel_y, const int stride_channel_dst,
        const int nsamples_x, const int nsamples_dst, const int stride_sample_x, const int stride_sample_y, const int stride_sample_dst,
        const int ids_stride, cudaStream_t stream, bool allow_small_k = true) {
    switch (type_x) {
        case GGML_TYPE_F8_E4M3:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_F8_E4M3>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q1_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q1_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q2_0_G128:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_0_G128>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q4_A32:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_A32>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_1>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q8_0_G128:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q8_0_G128>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_MXFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_MXFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_NVFP4:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_NVFP4>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q2_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q3_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q4_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q5_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_Q6_K>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ2_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_XXS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ1_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ1_M:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ1_M>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_NL>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ4_XS>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_switch_ncols_dst<GGML_TYPE_IQ3_S>
                (vx, vy, ids, fusion, dst, ncols_x, nrows_x, ncols_dst, stride_row_x, stride_col_y, stride_col_dst,
                 nchannels_x, nchannels_y, nchannels_dst, stride_channel_x, stride_channel_y, stride_channel_dst,
                 nsamples_x, nsamples_dst, stride_sample_x, stride_sample_y, stride_sample_dst, ids_stride, stream, allow_small_k);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_mul_mat_vec_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion, float post_scale, bool post_silu) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    GGML_ASSERT(!ids || ne12 <= MMVQ_MAX_BATCH_SIZE);

#if !defined(GGML_USE_HIP)
    if (ggml_cuda_mul_mat_humming_fp8(ctx, src0, src1, ids, dst, fusion)) {
        return;
    }
#endif

    const float   * src1_d =       (const float   *) src1->data;
    const int32_t *  ids_d = ids ? (const int32_t *)  ids->data : nullptr;
    float         *  dst_d =       (float         *)  dst->data;

    ggml_cuda_mmvq_fusion_args_device fusion_local{};

    if (post_silu) {
        const bool has_existing_fusion = fusion &&
            (fusion->x_bias || fusion->gate || fusion->gate_bias || fusion->x_scale || fusion->gate_scale);
        const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
        GGML_ASSERT(src0->type == GGML_TYPE_Q8_0 && !ids && dst->ne[1] == 1 && cc == 860);
        GGML_ASSERT(!has_existing_fusion);
    }

    if (fusion) {
        GGML_ASSERT( !ids || dst->ne[2] == 1);
        GGML_ASSERT(  ids || dst->ne[1] == 1);
        // Scale fusion is only allowed for NVFP4 currently as the cost of checking this at run-time in the prologue is
        // non-negligible for some models such as gpt-oss-20b
        GGML_ASSERT((fusion->x_scale == nullptr && fusion->gate_scale == nullptr) ||
                    src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_F8_E4M3);

        if (fusion->x_bias) {
            GGML_ASSERT(fusion->x_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->x_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->x_bias->ne[1] == src0->ne[2]);
            fusion_local.x_bias = fusion->x_bias->data;
        }
        if (fusion->gate) {
            GGML_ASSERT(fusion->gate->type == src0->type && ggml_are_same_stride(fusion->gate, src0));
            fusion_local.gate = fusion->gate->data;
        }
        if (fusion->gate_bias) {
            GGML_ASSERT(fusion->gate_bias->type == GGML_TYPE_F32);
            GGML_ASSERT(fusion->gate_bias->ne[0] == dst->ne[0]);
            GGML_ASSERT(!ids || fusion->gate_bias->ne[1] == src0->ne[2]);
            fusion_local.gate_bias = fusion->gate_bias->data;
        }
        if (fusion->x_scale) {
            GGML_ASSERT(ggml_is_contiguous(fusion->x_scale));
            if (src0->type == GGML_TYPE_F8_E4M3) {
                GGML_ASSERT(!ids);
                GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_BF16);
                GGML_ASSERT(ggml_nelements(fusion->x_scale) == src0->ne[1]);
            } else {
                GGML_ASSERT(fusion->x_scale->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_nelements(fusion->x_scale) == (ids ? src0->ne[2] : 1));
            }
            fusion_local.x_scale = fusion->x_scale->data;
        }
        if (fusion->gate_scale) {
            GGML_ASSERT(ggml_is_contiguous(fusion->gate_scale));
            if (src0->type == GGML_TYPE_F8_E4M3) {
                GGML_ASSERT(!ids);
                GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_BF16);
                GGML_ASSERT(ggml_nelements(fusion->gate_scale) == src0->ne[1]);
            } else {
                GGML_ASSERT(fusion->gate_scale->type == GGML_TYPE_F32);
                GGML_ASSERT(ggml_nelements(fusion->gate_scale) == (ids ? src0->ne[2] : 1));
            }
            fusion_local.gate_scale = fusion->gate_scale->data;
        }
        if (fusion->residual) {
            GGML_ASSERT(!ids && fusion->residual->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(fusion->residual) &&
                        ggml_are_same_shape(fusion->residual, dst));
            fusion_local.residual = static_cast<const float *>(fusion->residual->data);
        }
        fusion_local.glu_op = fusion->glu_op;
    }
    fusion_local.post_scale = post_scale;
    fusion_local.post_silu  = post_silu;

    // If src0 is a temporary compute buffer, clear any potential padding.
    if (ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1);
    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }

    int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s11 = ne10_padded / QK8_1;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const int64_t s12 = ne11*s11;
    const int64_t s13 = ne12*s12;

    if (src0->type == GGML_TYPE_F8_E4M3) {
        GGML_ASSERT(ne00 % QK8_1 == 0);
        GGML_ASSERT(s01 % QK8_1 == 0 && s02 % QK8_1 == 0 && s03 % QK8_1 == 0);
        s01 /= QK8_1;
        s02 /= QK8_1;
        s03 /= QK8_1;
    }

    // For MUL_MAT_ID the memory layout is different than for MUL_MAT:
    const int64_t ncols_dst          = ids ? ne2  : ne1;
    const int64_t nchannels_y        = ids ? ne11 : ne12;
    const int64_t nchannels_dst      = ids ? ne1  : ne2;
    const int64_t stride_col_dst     = ids ? s2   : s1;
    const int64_t stride_col_y       = ids ? s12  : s11;
    const int64_t stride_channel_dst = ids ? s1   : s2;
    const int64_t stride_channel_y   = ids ? s11  : s12;

    const int64_t ids_stride = ids ? ids->nb[1] / ggml_type_size(ids->type) : 0;

    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
}

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;

    const int stride_row_x = ne00 / ggml_blck_size(src0->type);
    const int stride_col_y = src1_padded_row_size / QK8_1;

    ggml_cuda_mmvq_fusion_args_device fusion_local{};
    mul_mat_vec_q_switch_type(
        src0_dd_i, src0->type, src1_ddq_i, nullptr, fusion_local, dst_dd_i, ne00, row_diff, src1_ncols, stride_row_x, stride_col_y, nrows_dst,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, stream);

    GGML_UNUSED_VARS(src1, dst, src1_ddf_i, src1_ncols, src1_padded_row_size);
}

template <ggml_type type>
static void ggml_cuda_moe_cache_mmv_t(
        const void * pool, const char * act_q8,
        const int32_t * ids_dev, const int32_t * act_ids_dev,
        float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
        int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
        cudaStream_t stream) {
    const int64_t ts0 = ggml_type_size(type);
    const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
    const int64_t s01 = ggml_row_size(type, n_in) / ts0;
    const int64_t s02 = slot_stride_bytes / ts0;
    const int64_t s11 = ne10_padded / QK8_1;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    mul_mat_vec_q_moe_cache_launch<type>(
        pool, act_q8, ids_dev, act_ids_dev, nullptr, nullptr, dst_dev, n_in,
        init_fastdiv_values(act_rows), n_out,
        s01, s02, s11, n_out,
        n_hits, warp_size,
        0.0f, 0.0f, 0.0f, 0.0f, stream);

    GGML_UNUSED(n_slots);
}

ggml_cuda_moe_cache_mmv_path ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8,
    const int32_t * ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    bool force_dedicated, cudaStream_t stream) {

    const int64_t ts0 = ggml_type_size(type0);
    GGML_ASSERT(slot_stride_bytes % ts0 == 0);

    if (!act_ids_dev && !force_dedicated) {
        const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
        const int64_t s01 = ggml_row_size(type0, n_in) / ts0;
        const int64_t s02 = slot_stride_bytes / ts0;
        const int64_t s11 = ne10_padded / QK8_1;
        const int64_t s12 = act_rows * s11;
        ggml_cuda_mmvq_fusion_args_device fusion_local{};
        mul_mat_vec_q_switch_type(
            pool, type0, act_q8, ids_dev, fusion_local, dst_dev, n_in,
            n_out, 1, s01, s12, n_out,
            n_slots, act_rows, n_hits, s02, s11, n_out,
            1, 1, s02*n_slots, s12, n_out*n_hits, n_hits, stream,
            false);
        return ggml_cuda_moe_cache_mmv_path::generic;
    }

#define MOE_CACHE_MMV_CASE(type_name) \
        case type_name: \
            ggml_cuda_moe_cache_mmv_t<type_name>( \
                pool, act_q8, ids_dev, act_ids_dev, dst_dev, n_in, n_out, \
                n_slots, slot_stride_bytes, n_hits, act_rows, stream); \
            break
    switch (type0) {
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q1_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q2_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q4_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q4_1);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q5_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q5_1);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q8_0);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q8_0_G128);
        MOE_CACHE_MMV_CASE(GGML_TYPE_MXFP4);
        MOE_CACHE_MMV_CASE(GGML_TYPE_NVFP4);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q2_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q3_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q4_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q5_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_Q6_K);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ2_XXS);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ2_XS);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ2_S);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ3_XXS);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ3_S);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ1_S);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ1_M);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ4_NL);
        MOE_CACHE_MMV_CASE(GGML_TYPE_IQ4_XS);
        default:
            GGML_ABORT("unsupported MoE cache type");
    }
#undef MOE_CACHE_MMV_CASE
    return ggml_cuda_moe_cache_mmv_path::dedicated;
}

template <ggml_type type>
static void ggml_cuda_moe_cache_mmv_fused_t(
        const void * up_pool, const void * gate_pool, const char * act_q8,
        const int32_t * up_ids_dev, const int32_t * gate_ids_dev,
        const int32_t * act_ids_dev, float * dst_dev,
        int64_t n_in, int64_t n_out, int64_t slot_stride_bytes,
        int64_t n_hits, int64_t act_rows, float up_min, float up_max,
        float gate_min, float gate_max, cudaStream_t stream) {
    const int64_t ts0 = ggml_type_size(type);
    const int64_t ne10_padded = GGML_PAD(n_in, MATRIX_ROW_PADDING);
    const int64_t s01 = ggml_row_size(type, n_in) / ts0;
    const int64_t s02 = slot_stride_bytes / ts0;
    const int64_t s11 = ne10_padded / QK8_1;

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const float inf = std::numeric_limits<float>::infinity();
    if (up_min == -inf && up_max == inf &&
        gate_min == -inf && gate_max == inf) {
        mul_mat_vec_q_moe_cache_launch<type, true, false>(
            up_pool, act_q8, up_ids_dev, act_ids_dev,
            gate_pool, gate_ids_dev, dst_dev, n_in,
            init_fastdiv_values(act_rows), n_out,
            s01, s02, s11, n_out,
            n_hits, warp_size,
            up_min, up_max, gate_min, gate_max, stream);
    } else {
        mul_mat_vec_q_moe_cache_launch<type, true, true>(
            up_pool, act_q8, up_ids_dev, act_ids_dev,
            gate_pool, gate_ids_dev, dst_dev, n_in,
            init_fastdiv_values(act_rows), n_out,
            s01, s02, s11, n_out,
            n_hits, warp_size,
            up_min, up_max, gate_min, gate_max, stream);
    }
}

void ggml_cuda_moe_cache_mmv_fused(
        const void * up_pool, const void * gate_pool, ggml_type type0,
        const char * act_q8, const int32_t * up_ids_dev,
        const int32_t * gate_ids_dev, const int32_t * act_ids_dev,
        float * dst_dev, int64_t n_in, int64_t n_out,
        int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
        float up_min, float up_max, float gate_min, float gate_max,
        cudaStream_t stream) {
    const int64_t ts0 = ggml_type_size(type0);
    GGML_ASSERT(slot_stride_bytes % ts0 == 0);

#define MOE_CACHE_MMV_FUSED_CASE(type_name) \
        case type_name: \
            ggml_cuda_moe_cache_mmv_fused_t<type_name>( \
                up_pool, gate_pool, act_q8, up_ids_dev, gate_ids_dev, \
                act_ids_dev, dst_dev, n_in, n_out, slot_stride_bytes, \
                n_hits, act_rows, up_min, up_max, gate_min, gate_max, \
                stream); \
            break
    switch (type0) {
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q1_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q2_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q4_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q4_1);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q5_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q5_1);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q8_0);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q8_0_G128);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_MXFP4);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q2_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q3_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q4_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q5_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_Q6_K);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ2_XXS);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ2_XS);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ2_S);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ3_XXS);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ3_S);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ1_S);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ1_M);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ4_NL);
        MOE_CACHE_MMV_FUSED_CASE(GGML_TYPE_IQ4_XS);
        default:
            GGML_ABORT("unsupported fused MoE cache type");
    }
#undef MOE_CACHE_MMV_FUSED_CASE
}
