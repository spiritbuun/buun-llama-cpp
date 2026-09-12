#include "../ggml/src/ggml-cuda/vendors/hip-runtime.h"

#include <cstdio>
#include <vector>

#define CHECK_HIP(call) do { \
    const hipError_t result = (call); \
    if (result != hipSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(result)); \
        std::exit(1); \
    } \
} while (0)

static void not_a_kernel() {}

static void test_capture(bool pinned) {
    constexpr size_t size = (2 << 20) + 17;
    std::vector<unsigned char> pageable_input(pinned ? 0 : size);
    std::vector<unsigned char> pageable_output(pinned ? 0 : size);
    void * input = pageable_input.data();
    void * output = pageable_output.data();
    void * device = nullptr;
    hipStream_t stream;
    if (pinned) {
        CHECK_HIP(hipHostMalloc(&input, size, hipHostMallocPortable));
        CHECK_HIP(hipHostMalloc(&output, size, hipHostMallocPortable));
    }
    CHECK_HIP(hipMalloc(&device, size));
    CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipError_t native_status = hipSuccess;
    for (bool wrapped : { false, true }) {
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable;
        auto copy = wrapped ? ggml_hip_memcpy_async : hipMemcpyAsync;
        CHECK_HIP(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        hipError_t status = copy(device, input, size, hipMemcpyHostToDevice, stream);
        if (status == hipSuccess) {
            status = copy(output, device, size, hipMemcpyDeviceToHost, stream);
        }
        const hipError_t end_status = hipStreamEndCapture(stream, &graph);
        if (status == hipSuccess) {
            status = end_status;
        }
        if (!wrapped) {
            native_status = status;
        } else if (status != native_status) {
            std::fprintf(stderr, "capture status mismatch: pinned=%d native=%d wrapper=%d\n",
                         pinned, int(native_status), int(status));
            std::exit(1);
        }
        if (status != hipSuccess) {
            // Older runtimes may reject pageable capture. Preserve that result,
            // but do not silently accept allocation failures or other errors.
            if (pinned || (status != hipErrorNotSupported && status != hipErrorStreamCaptureUnsupported)) {
                CHECK_HIP(status);
            }
            if (graph) {
                CHECK_HIP(hipGraphDestroy(graph));
            }
            (void) hipGetLastError();
            continue;
        }
        CHECK_HIP(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        for (int pattern : { 0x5a, 0xa5 }) {
            std::memset(input, pattern, size);
            CHECK_HIP(hipGraphLaunch(executable, stream));
            CHECK_HIP(hipStreamSynchronize(stream));
            if (std::memcmp(input, output, size) != 0) {
                std::fprintf(stderr, "captured transfer mismatch: pinned=%d wrapped=%d\n", pinned, wrapped);
                std::exit(1);
            }
        }
        CHECK_HIP(hipGraphExecDestroy(executable));
        CHECK_HIP(hipGraphDestroy(graph));
    }
    std::printf("HIP capture: pinned=%d native=%s\n", pinned, hipGetErrorString(native_status));
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(device));
    if (pinned) {
        CHECK_HIP(hipHostFree(output));
        CHECK_HIP(hipHostFree(input));
    }
}

static void test_null_host(size_t size, bool synchronous) {
    void * device = nullptr;
    hipStream_t stream;
    CHECK_HIP(hipMalloc(&device, std::max(size, size_t(1))));
    CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    for (hipMemcpyKind kind : { hipMemcpyHostToDevice, hipMemcpyDeviceToHost }) {
        void * dst = kind == hipMemcpyHostToDevice ? device : nullptr;
        const void * src = kind == hipMemcpyHostToDevice ? nullptr : device;
        const hipError_t native = synchronous ? hipMemcpy(dst, src, size, kind) :
                                               hipMemcpyAsync(dst, src, size, kind, stream);
        (void) hipGetLastError();
        const hipError_t wrapped = synchronous ? ggml_hip_memcpy(dst, src, size, kind) :
                                                ggml_hip_memcpy_async(dst, src, size, kind, stream);
        (void) hipGetLastError();
        if (native != wrapped || (size != 0 && wrapped == hipSuccess)) {
            std::fprintf(stderr, "null transfer mismatch: size=%zu sync=%d kind=%d native=%d wrapper=%d\n",
                         size, synchronous, int(kind), int(native), int(wrapped));
            std::exit(1);
        }
    }
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(device));
}

static void test_transfers(size_t size, bool pinned, bool synchronous) {
    constexpr size_t padding = 31;
    const size_t capacity = size + 2 * padding;
    std::vector<unsigned char> input(capacity, 0x5a);
    std::vector<unsigned char> output(capacity, 0xa5);
    for (size_t i = 0; i < size; ++i) {
        input[padding + i] = static_cast<unsigned char>(i * 131 + i / 257);
    }
    void * host = nullptr;
    if (pinned) {
        CHECK_HIP(hipHostMalloc(&host, capacity, hipHostMallocPortable));
        std::memcpy(host, input.data(), capacity);
    }
    void * device = nullptr;
    void * device_copy = nullptr;
    hipStream_t stream;
    CHECK_HIP(hipMalloc(&device, capacity));
    CHECK_HIP(hipMalloc(&device_copy, capacity));
    CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    const void * source = static_cast<unsigned char *>(pinned ? host : input.data()) + padding;
    auto copy = [&](void * dst, const void * src, hipMemcpyKind kind) {
        if (synchronous) {
            CHECK_HIP(ggml_hip_memcpy(dst, src, size, kind));
        } else {
            CHECK_HIP(ggml_hip_memcpy_async(dst, src, size, kind, stream));
        }
    };
    copy(device, source, hipMemcpyHostToDevice);
    copy(device_copy, device, hipMemcpyDeviceToDevice);
    copy(output.data() + padding, device_copy, hipMemcpyDeviceToHost);
    CHECK_HIP(hipStreamSynchronize(stream));
    if (std::memcmp(input.data() + padding, output.data() + padding, size) != 0 ||
        !std::all_of(output.begin(), output.begin() + padding, [](unsigned char c) { return c == 0xa5; }) ||
        !std::all_of(output.end() - padding, output.end(), [](unsigned char c) { return c == 0xa5; })) {
        std::fprintf(stderr, "transfer mismatch: size=%zu pinned=%d sync=%d\n", size, pinned, synchronous);
        std::exit(1);
    }
    if (pinned) {
        copy(host, device_copy, hipMemcpyDeviceToHost);
        CHECK_HIP(hipStreamSynchronize(stream));
        if (std::memcmp(host, input.data() + padding, size) != 0) {
            std::fprintf(stderr, "pinned download mismatch\n");
            std::exit(1);
        }
        CHECK_HIP(hipHostFree(host));
    }
    CHECK_HIP(hipStreamDestroy(stream));
    CHECK_HIP(hipFree(device_copy));
    CHECK_HIP(hipFree(device));
}

int main() {
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::puts("SKIP: no HIP device");
        return 77;
    }
    // A host symbol must be rejected rather than reinterpreted as hipFunction_t.
    if (ggml_hip_prepare_kernel(reinterpret_cast<const void *>(&not_a_kernel)) == hipSuccess) {
        std::fprintf(stderr, "unregistered host symbol accepted as kernel\n");
        return 1;
    }
    (void) hipGetLastError();

    for (int device = 0; device < devices; ++device) {
        CHECK_HIP(hipSetDevice(device));
        test_capture(false);
        test_capture(true);
        for (size_t size : { size_t(0), size_t(1), size_t((1 << 20) - 1),
                            size_t(1 << 20), size_t((1 << 20) + 1), size_t((2 << 20) + 17) }) {
            test_null_host(size, false);
            test_null_host(size, true);
            for (bool pinned : { false, true }) {
                for (bool synchronous : { false, true }) {
                    test_transfers(size, pinned, synchronous);
                }
            }
        }
    }
    std::puts("HIP runtime boundary tests passed");
    return 0;
}
