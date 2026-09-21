#include "ggml-vbr-diagnostic.h"

#include <array>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {
constexpr size_t capacity = 128; // per channel; mapping traffic cannot evict state events
struct entry {
    unsigned long long sequence = 0;
    long long time_us = 0;
    char message[1024] = {};
};
struct recorder {
    std::mutex mutex;
    unsigned long long sequence = 0;
    std::array<size_t, GGML_VBR_DIAG_CHANNEL_COUNT> counts{};
    std::array<std::array<entry, capacity>, GGML_VBR_DIAG_CHANNEL_COUNT> rings{};
};
recorder & history() {
    static recorder value;
    return value;
}
}

bool ggml_vbr_diag_enabled(void) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_VBR_DIAG");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

void ggml_vbr_diag_record(enum ggml_vbr_diag_channel channel, const char * format, ...) {
    if (!ggml_vbr_diag_enabled() || channel < 0 || channel >= GGML_VBR_DIAG_CHANNEL_COUNT) {
        return;
    }
    auto & h = history();
    std::lock_guard<std::mutex> lock(h.mutex);
    auto & e = h.rings[channel][h.counts[channel]++ % capacity];
    e.sequence = ++h.sequence;
    e.time_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    va_list args;
    va_start(args, format);
    const int written = std::vsnprintf(e.message, sizeof(e.message), format, args);
    va_end(args);
    if (written < 0) {
        std::snprintf(e.message, sizeof(e.message), "diagnostic_format_error");
    } else if (size_t(written) >= sizeof(e.message)) {
        constexpr char suffix[] = " [truncated]";
        std::memcpy(e.message + sizeof(e.message) - sizeof(suffix), suffix, sizeof(suffix));
    }
}

void ggml_vbr_diag_dump(const char * reason) {
    if (!ggml_vbr_diag_enabled()) {
        return;
    }
    auto & h = history();
    std::lock_guard<std::mutex> lock(h.mutex);
    FILE * out = stderr;
    std::fprintf(out, "VBR_DIAG_BEGIN version=1 reason=%s events_total=%llu capacity_per_channel=%zu\n",
                 reason, h.sequence, capacity);
    std::array<size_t, GGML_VBR_DIAG_CHANNEL_COUNT> cursor{};
    for (size_t c = 0; c < cursor.size(); ++c) {
        cursor[c] = h.counts[c] > capacity ? h.counts[c] - capacity : 0;
        std::fprintf(out, "VBR_DIAG_CHANNEL channel=%zu total=%zu overwritten=%zu\n", c, h.counts[c], cursor[c]);
    }
    // Merge the three ordered rings without allocating/sorting on the abort path.
    for (;;) {
        size_t selected = cursor.size();
        for (size_t c = 0; c < cursor.size(); ++c) {
            if (cursor[c] < h.counts[c] && (selected == cursor.size() ||
                h.rings[c][cursor[c] % capacity].sequence < h.rings[selected][cursor[selected] % capacity].sequence)) {
                selected = c;
            }
        }
        if (selected == cursor.size()) {
            break;
        }
        const auto & e = h.rings[selected][cursor[selected]++ % capacity];
        std::fprintf(out, "VBR_DIAG seq=%llu mono_us=%lld channel=%zu %s\n", e.sequence, e.time_us, selected, e.message);
    }
    std::fprintf(out, "VBR_DIAG_END\n");
    std::fflush(out);
}
