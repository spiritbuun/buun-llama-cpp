#include "ggml.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

enum class ggml_cuda_moe_cache_mmv_path {
    generic,
    dedicated,
};

ggml_cuda_moe_cache_mmv_path ggml_cuda_moe_cache_mmv(
    const void * pool, ggml_type type0, const char * act_q8,
    const int32_t * ids_dev, const int32_t * act_ids_dev,
    float * dst_dev, int64_t n_in, int64_t n_out, int64_t n_slots,
    int64_t slot_stride_bytes, int64_t n_hits, int64_t act_rows,
    bool force_dedicated, cudaStream_t stream);

#ifdef GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION
void ggml_cuda_moe_cache_flat_hits_test_stats_reset();
void ggml_cuda_moe_cache_flat_hits_test_stats_get(uint64_t * factor_1, uint64_t * factor_2);
#endif

namespace {

bool cuda_ok(cudaError_t status, const char * operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

template <typename T>
struct device_allocation {
    T * ptr = nullptr;

    explicit device_allocation(size_t count) {
        cudaMalloc((void **) &ptr, count*sizeof(T));
    }
    ~device_allocation() {
        cudaFree(ptr);
    }
    device_allocation(const device_allocation &) = delete;
    device_allocation & operator=(const device_allocation &) = delete;
};

bool run_type(ggml_type type, bool expect_flat, cudaStream_t stream) {
    constexpr int64_t n_in = 512;
    constexpr int64_t n_out = 19;
    constexpr int64_t n_slots = 12;
    constexpr int64_t act_rows = 10;
    constexpr float canary = -9876.5f;

    const ggml_type_traits * weight_traits = ggml_get_type_traits(type);
    const ggml_type_traits * q8_traits = ggml_get_type_traits(GGML_TYPE_Q8_1);
    if (!weight_traits->from_float_ref || !q8_traits->from_float_ref ||
        !weight_traits->to_float) {
        std::fprintf(stderr, "%s: missing Q8_1 reference traits\n", ggml_type_name(type));
        return false;
    }

    const size_t weight_row_size = ggml_row_size(type, n_in);
    const size_t slot_stride = weight_row_size*n_out;
    std::vector<unsigned char> pool(slot_stride*n_slots);
    std::vector<float> row(n_in);
    for (int64_t slot = 0; slot < n_slots; ++slot) {
        for (int64_t out = 0; out < n_out; ++out) {
            for (int64_t col = 0; col < n_in; ++col) {
                row[col] = 0.16f*std::sin(0.011f*(float)(col + 17*out + 31*slot)) +
                           0.07f*std::cos(0.019f*(float)(3*col + 5*out + 7*slot));
            }
            weight_traits->from_float_ref(
                row.data(), pool.data() + slot*slot_stride + out*weight_row_size, n_in);
        }
    }

    const size_t q8_row_size = ggml_row_size(GGML_TYPE_Q8_1, n_in);
    std::vector<unsigned char> activations(q8_row_size*act_rows);
    std::vector<float> activations_reference(n_in*act_rows);
    for (int64_t act = 0; act < act_rows; ++act) {
        for (int64_t col = 0; col < n_in; ++col) {
            row[col] = 0.31f*std::sin(0.013f*(float)(col + 23*act)) -
                       0.09f*std::cos(0.023f*(float)(2*col + 11*act));
            activations_reference[act*n_in + col] = row[col];
        }
        q8_traits->from_float_ref(row.data(), activations.data() + act*q8_row_size, n_in);
    }

    device_allocation<unsigned char> pool_d(pool.size());
    device_allocation<unsigned char> act_d(activations.size());
    device_allocation<int32_t> ids_d(10);
    device_allocation<int32_t> act_ids_d(10);
    device_allocation<float> dst_d(10*n_out + 2);
    if (!pool_d.ptr || !act_d.ptr || !ids_d.ptr || !act_ids_d.ptr || !dst_d.ptr) {
        std::fprintf(stderr, "%s: device allocation failed\n", ggml_type_name(type));
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(pool_d.ptr, pool.data(), pool.size(), cudaMemcpyHostToDevice, stream), "copy pool") ||
        !cuda_ok(cudaMemcpyAsync(act_d.ptr, activations.data(), activations.size(), cudaMemcpyHostToDevice, stream), "copy activations")) {
        return false;
    }

#ifdef GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION
    ggml_cuda_moe_cache_flat_hits_test_stats_reset();
#endif
    bool ok = true;
    for (int64_t n_hits = 1; n_hits <= 10 && ok; ++n_hits) {
        std::vector<int32_t> ids(n_hits);
        std::vector<int32_t> act_ids(n_hits);
        for (int64_t hit = 0; hit < n_hits; ++hit) {
            ids[hit] = (int32_t) ((7*hit + 3) % n_slots);
            act_ids[hit] = (int32_t) ((3*hit + 1) % act_rows);
        }
        std::vector<float> output(n_hits*n_out + 2, canary);
        if (!cuda_ok(cudaMemcpyAsync(ids_d.ptr, ids.data(), n_hits*sizeof(int32_t), cudaMemcpyHostToDevice, stream), "copy ids") ||
            !cuda_ok(cudaMemcpyAsync(act_ids_d.ptr, act_ids.data(), n_hits*sizeof(int32_t), cudaMemcpyHostToDevice, stream), "copy act ids") ||
            !cuda_ok(cudaMemcpyAsync(dst_d.ptr, output.data(), output.size()*sizeof(float), cudaMemcpyHostToDevice, stream), "copy canaries")) {
            return false;
        }

        const auto path = ggml_cuda_moe_cache_mmv(
            pool_d.ptr, type, (const char *) act_d.ptr, ids_d.ptr, act_ids_d.ptr,
            dst_d.ptr + 1, n_in, n_out, n_slots, slot_stride, n_hits, act_rows, true, stream);
        if (path != ggml_cuda_moe_cache_mmv_path::dedicated ||
            !cuda_ok(cudaMemcpyAsync(output.data(), dst_d.ptr, output.size()*sizeof(float), cudaMemcpyDeviceToHost, stream), "copy output") ||
            !cuda_ok(cudaStreamSynchronize(stream), "synchronize")) {
            return false;
        }
        if (output.front() != canary || output.back() != canary) {
            std::fprintf(stderr, "%s hits=%lld: output canary overwritten\n",
                ggml_type_name(type), (long long) n_hits);
            ok = false;
            break;
        }
        std::vector<float> weight_f32(n_in);
        for (int64_t hit = 0; hit < n_hits && ok; ++hit) {
            const float * activation_f32 = activations_reference.data() + act_ids[hit]*n_in;
            for (int64_t out = 0; out < n_out; ++out) {
                const void * x = pool.data() + ids[hit]*slot_stride + out*weight_row_size;
                weight_traits->to_float(x, weight_f32.data(), n_in);
                float expected = 0.0f;
                for (int64_t col = 0; col < n_in; ++col) {
                    expected += weight_f32[col]*activation_f32[col];
                }
                const float actual = output[1 + hit*n_out + out];
                const float tolerance = 3e-2f*std::max(1.0f, std::fabs(expected));
                if (!std::isfinite(actual) || std::fabs(actual - expected) > tolerance) {
                    std::fprintf(stderr,
                        "%s hits=%lld hit=%lld row=%lld: got %.9g expected %.9g tolerance %.9g\n",
                        ggml_type_name(type), (long long) n_hits, (long long) hit,
                        (long long) out, actual, expected, tolerance);
                    ok = false;
                    break;
                }
            }
        }
    }
#ifdef GGML_CUDA_MOE_CACHE_FLAT_HITS_TEST_INSTRUMENTATION
    uint64_t factor_1 = 0;
    uint64_t factor_2 = 0;
    ggml_cuda_moe_cache_flat_hits_test_stats_get(&factor_1, &factor_2);
    const bool receipt_ok = expect_flat
        ? factor_2 > 0 && factor_1 == 0
        : factor_1 > 0 && factor_2 == 0;
    if (!receipt_ok) {
        std::fprintf(stderr, "%s: expected %s path, got factor1=%llu factor2=%llu\n",
            ggml_type_name(type), expect_flat ? "factor2" : "factor1",
            (unsigned long long) factor_1, (unsigned long long) factor_2);
        ok = false;
    }
#else
    (void) expect_flat;
#endif
    return ok;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::puts("SKIP: CUDA device unavailable");
        return 0;
    }
    cudaDeviceProp props{};
    if (!cuda_ok(cudaGetDeviceProperties(&props, 0), "get device properties")) {
        return 1;
    }
    if (props.major*100 + props.minor*10 != 860) {
        std::printf("SKIP: flattened-hit specialization requires SM86 (found %d%d)\n", props.major, props.minor);
        return 0;
    }
    if (!cuda_ok(cudaSetDevice(0), "set device")) {
        return 1;
    }
    cudaStream_t stream = nullptr;
    if (!cuda_ok(cudaStreamCreate(&stream), "create stream")) {
        return 1;
    }
    const bool ok = run_type(GGML_TYPE_Q4_K, true, stream) &&
                    run_type(GGML_TYPE_Q5_1, true, stream) &&
                    run_type(GGML_TYPE_Q8_0, false, stream);
    cudaStreamDestroy(stream);
    std::printf("cache-flat-hits-numerical: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
