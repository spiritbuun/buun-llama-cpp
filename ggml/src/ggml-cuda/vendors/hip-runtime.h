#pragma once

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

// ROCm may pin/cache pageable transfer ranges. With mmap-backed weights or
// checkpoint buffers this can exhaust KFD's resident-system-memory allowance
// even while Linux reports free RAM. Keep such transfers on bounded staging;
// explicitly pinned buffers retain the normal asynchronous fast path.
inline hipError_t ggml_hip_memcpy_async(void * dst, const void * src, size_t size,
                                      hipMemcpyKind kind, hipStream_t stream) {
    constexpr size_t chunk_size = 1 << 20;
    if (!dst || !src || size <= chunk_size || (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost)) {
        return hipMemcpyAsync(dst, src, size, kind, stream);
    }

    const void * host = kind == hipMemcpyHostToDevice ? src : dst;
    hipPointerAttribute_t attributes{};
    hipError_t status = hipPointerGetAttributes(&attributes, host);
    if (status == hipSuccess && attributes.type != hipMemoryTypeUnregistered) {
        return hipMemcpyAsync(dst, src, size, kind, stream);
    }
    if (status != hipSuccess) {
        if (status != hipErrorInvalidValue) {
            return status;
        }
        (void) hipGetLastError(); // Older HIP versions reject unregistered host pointers.
    }

    hipStreamCaptureStatus capture;
    status = hipStreamIsCapturing(stream, &capture);
    if (status != hipSuccess) {
        return status;
    }
    if (capture != hipStreamCaptureStatusNone) {
        // Do not introduce host copies or synchronization into a captured graph.
        return hipMemcpyAsync(dst, src, size, kind, stream);
    }

    struct staging_buffer {
        void * data = nullptr;
        staging_buffer() {
            if (std::getenv("GGML_CUDA_NO_PINNED") == nullptr &&
                hipHostMalloc(&data, chunk_size, hipHostMallocPortable) != hipSuccess) {
                (void) hipGetLastError();
                data = nullptr;
            }
        }
        ~staging_buffer() {
            if (data) {
                (void) hipHostFree(data);
            }
        }
    };
    // Each chunk completes before reuse, including when this thread changes
    // devices/streams. Portable host memory is shared, not device-owned.
    thread_local staging_buffer staging;
    auto * target = static_cast<char *>(dst);
    const auto * source = static_cast<const char *>(src);
    for (size_t offset = 0; offset < size;) {
        const size_t bytes = std::min(chunk_size, size - offset);
        if (!staging.data) {
            // Respect GGML_CUDA_NO_PINNED and allocation-failure fallback. With
            // ROCm's default threshold these small copies use its own staging.
            status = hipMemcpyAsync(target + offset, source + offset, bytes, kind, stream);
        } else if (kind == hipMemcpyHostToDevice) {
            std::memcpy(staging.data, source + offset, bytes);
            status = hipMemcpyAsync(target + offset, staging.data, bytes, kind, stream);
        } else {
            status = hipMemcpyAsync(staging.data, source + offset, bytes, kind, stream);
        }
        if (status != hipSuccess) {
            return status;
        }
        if (staging.data) {
            status = hipStreamSynchronize(stream);
            if (status != hipSuccess) {
                return status;
            }
            if (kind == hipMemcpyDeviceToHost) {
                std::memcpy(target + offset, staging.data, bytes);
            }
        }
        offset += bytes;
    }
    return hipSuccess;
}

inline hipError_t ggml_hip_memcpy(void * dst, const void * src, size_t size, hipMemcpyKind kind) {
    if (kind != hipMemcpyHostToDevice && kind != hipMemcpyDeviceToHost) {
        return hipMemcpy(dst, src, size, kind);
    }
    const hipError_t status = ggml_hip_memcpy_async(dst, src, size, kind, nullptr);
    return status == hipSuccess ? hipStreamSynchronize(nullptr) : status;
}

inline hipError_t ggml_hip_prepare_kernel(const void * symbol) {
    // HIP 7.2's hipLaunchKernel treats a failed static lookup as a raw module
    // handle. A resource failure can consequently become an invalid-pointer
    // SIGSEGV. Resolve first so CUDA_CHECK reports an error instead.
    hipFuncAttributes attributes{};
    return hipFuncGetAttributes(&attributes, symbol);
}
