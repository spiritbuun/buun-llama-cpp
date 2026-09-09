#pragma once

#include "../ggml-prefix-export.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

// Backend-owned staging outlives captured callbacks. Published checkpoint
// storage remains owned by the caller, never by this workspace.
struct ggml_cuda_prefix_export {
    struct range {
        ggml_prefix_export_range spec;
        const void * device;
        size_t staging;
        cudaEvent_t ready_event = nullptr;
        std::atomic<bool> ready{false};
    };
    std::vector<std::unique_ptr<range>> ranges;
    unsigned char * staging = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t joined = nullptr;
    cudaEvent_t done = nullptr;
    bool armed = false;
    bool scheduled = false;
    bool launched = false;
    size_t reads = 0;

    ~ggml_cuda_prefix_export() {
        if (stream) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        for (auto & r : ranges) {
            if (r->ready_event) CUDA_CHECK(cudaEventDestroy(r->ready_event));
        }
        if (joined) CUDA_CHECK(cudaEventDestroy(joined));
        if (done) CUDA_CHECK(cudaEventDestroy(done));
        if (staging) CUDA_CHECK(cudaFreeHost(staging));
    }

    static void CUDART_CB notify(void * opaque) {
        static_cast<range *>(opaque)->ready.store(true, std::memory_order_release);
    }

    bool matches(const ggml_tensor * node, const range & r) const {
        return node->op == GGML_OP_CPY && node->data == r.device &&
            ggml_nbytes(node) == r.spec.size && ggml_is_contiguous(node) &&
            (node->flags & GGML_TENSOR_FLAG_COMPUTE);
    }

    bool begin_graph(ggml_cgraph * graph) {
        scheduled = launched = false;
        if (!armed) return false;
        for (auto & r : ranges) {
            int writers = 0;
            for (int i = 0; i < graph->n_nodes; ++i) {
                writers += matches(graph->nodes[i], *r);
            }
            if (writers != 1) return false;
        }
        for (auto & r : ranges) r->ready.store(false, std::memory_order_release);
        scheduled = true;
        return true;
    }

    void after_nodes(ggml_cgraph * graph, int first, int last, cudaStream_t producer) {
        if (!scheduled) return;
        for (int i = first; i <= last; ++i) {
            if (graph->nodes[i]->op != GGML_OP_CPY) continue;
            for (auto & r : ranges) {
                if (!matches(graph->nodes[i], *r)) continue;
                CUDA_CHECK(cudaEventRecord(r->ready_event, producer));
                CUDA_CHECK(cudaStreamWaitEvent(stream, r->ready_event, 0));
                CUDA_CHECK(cudaMemcpyAsync(staging + r->staging, r->device,
                    r->spec.size, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaLaunchHostFunc(stream, notify, r.get()));
            }
        }
    }

    void join(cudaStream_t main) {
        if (!scheduled) return;
        CUDA_CHECK(cudaEventRecord(joined, stream));
        CUDA_CHECK(cudaStreamWaitEvent(main, joined, 0));
    }
};

static ggml_cuda_prefix_export * ggml_cuda_prefix_find(ggml_backend_cuda_context * ctx) {
    return ctx->prefix_export;
}

static void ggml_cuda_prefix_destroy(ggml_backend_cuda_context * ctx) {
    delete ctx->prefix_export;
    ctx->prefix_export = nullptr;
}

static bool ggml_cuda_prefix_prepare(ggml_backend_t backend,
        const ggml_prefix_export_range * specs, size_t count) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    if (ctx->external_capture || count == 0) return false;
    auto * existing = ggml_cuda_prefix_find(ctx);
    if (existing) {
        // Never relocate an address embedded in a cached CUDA graph. A changed
        // layout simply uses the original synchronous writer.
        if (existing->ranges.size() != count) return false;
        for (size_t i = 0; i < count; ++i) {
            const auto & old = existing->ranges[i];
            const auto & now = specs[i];
            if (old->spec.tensor != now.tensor || old->spec.offset != now.offset ||
                    old->spec.size != now.size || old->spec.destination != now.destination ||
                    old->device != static_cast<char *>(now.tensor->data) + now.offset) return false;
        }
        existing->armed = true;
        return true;
    }
    auto result = std::make_unique<ggml_cuda_prefix_export>();
    size_t bytes = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & s = specs[i];
        if (s.offset != 0 || s.size != ggml_nbytes(s.tensor) || !ggml_is_contiguous(s.tensor)) return false;
        auto r = std::make_unique<ggml_cuda_prefix_export::range>();
        r->spec = s;
        r->device = s.tensor->data;
        r->staging = bytes;
        bytes += s.size;
        result->ranges.push_back(std::move(r));
    }
    const auto allocation = cudaMallocHost(&result->staging, bytes);
    if (allocation != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    if (cudaStreamCreateWithFlags(&result->stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&result->joined, cudaEventDisableTiming) != cudaSuccess ||
            cudaEventCreateWithFlags(&result->done, cudaEventDisableTiming) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    for (auto & r : result->ranges) {
        if (cudaEventCreateWithFlags(&r->ready_event, cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            return false;
        }
    }
    result->armed = true;
    GGML_LOG_INFO("prefix export experiment: prepared %zu ranges, %zu pinned bytes\n", count, bytes);
    ctx->prefix_export = result.release();
    return true;
}

static bool ggml_cuda_prefix_read(ggml_backend_t backend, unsigned char * dst, size_t size) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    auto * state = ggml_cuda_prefix_find(ctx);
    if (!state || !state->launched) return false;
    ggml_cuda_set_device(ctx->device);
    std::vector<bool> copied(state->ranges.size(), false);
    size_t remaining = copied.size();
    auto next_query = std::chrono::steady_clock::now();
    while (remaining) {
        bool progress = false;
        for (size_t i = 0; i < copied.size(); ++i) {
            const auto & r = *state->ranges[i];
            if (copied[i] || !r.ready.load(std::memory_order_acquire)) continue;
            if (r.spec.destination > size || r.spec.size > size - r.spec.destination) return false;
            memcpy(dst + r.spec.destination, state->staging + r.staging, r.spec.size);
            copied[i] = true;
            --remaining;
            progress = true;
        }
        if (!remaining) break;
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_query) {
            next_query = now + std::chrono::milliseconds(1);
            const auto status = cudaEventQuery(state->done);
            if (status != cudaSuccess && status != cudaErrorNotReady) return false;
            if (status == cudaSuccess) {
                // Completion includes all notifications. Detect a missing export
                // rather than waiting forever; caller can use the original writer.
                for (size_t i = 0; i < copied.size(); ++i) {
                    if (!copied[i] && !state->ranges[i]->ready.load(std::memory_order_acquire)) return false;
                }
            }
        }
        if (!progress) std::this_thread::yield();
    }
    if (state->reads++ == 0) GGML_LOG_INFO("prefix export experiment: overlapped export completed\n");
    return true;
}

static void ggml_cuda_prefix_cancel(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    auto * state = ggml_cuda_prefix_find(ctx);
    if (!state) return;
    ggml_cuda_set_device(ctx->device);
    if (state->launched) CUDA_CHECK(cudaEventSynchronize(state->done));
    CUDA_CHECK(cudaStreamSynchronize(state->stream));
    state->armed = state->scheduled = state->launched = false;
}

static const ggml_prefix_export_iface * ggml_cuda_prefix_iface() {
    static const ggml_prefix_export_iface iface = {
        ggml_cuda_prefix_prepare, ggml_cuda_prefix_read, ggml_cuda_prefix_cancel,
    };
    return &iface;
}
