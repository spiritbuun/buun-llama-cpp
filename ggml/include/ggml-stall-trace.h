#pragma once

// Temporary issue #124 diagnostics. No device queries, waits, or model data.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct ggml_stall_trace {
    using clock = std::chrono::steady_clock;
    const char * phase;
    const void * owner;
    long long a, b, c;
    unsigned long long id = 0, tid = 0;
    clock::time_point start;

    ggml_stall_trace(const char * phase, const void * owner = nullptr,
                     long long a = -1, long long b = -1, long long c = -1) noexcept
        : phase(phase), owner(owner), a(a), b(b), c(c) {
        static const bool enabled = [] {
            const char * value = std::getenv("BUUN_STALL_TRACE");
            return value && value[0] == '1';
        }();
        if (!enabled) { return; }
        static std::atomic<unsigned long long> next {1};
        id = next.fetch_add(1, std::memory_order_relaxed);
#ifdef __linux__
        tid = static_cast<unsigned long long>(syscall(SYS_gettid));
#else
        tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
#endif
        start = clock::now();
        emit("begin", start);
    }
    ~ggml_stall_trace() { if (id) { emit("end", clock::now()); } }
    ggml_stall_trace(const ggml_stall_trace &) = delete;
    ggml_stall_trace & operator=(const ggml_stall_trace &) = delete;

    void emit(const char * event, clock::time_point now) const noexcept {
        std::fprintf(stderr,
            "STALL event=%s tid=%llu id=%llu phase=%s owner=%p a=%lld b=%lld c=%lld mono_us=%lld elapsed_us=%lld\n",
            event, tid, id, phase, owner, a, b, c,
            static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count()),
            static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(now-start).count()));
    }
};
