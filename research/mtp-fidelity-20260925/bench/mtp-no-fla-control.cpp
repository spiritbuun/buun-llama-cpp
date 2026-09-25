// Diagnostic LD_PRELOAD only. Never installed or linked into a serving build.
#include <cstdint>
#include <cstdio>
bool ggml_cuda_gdn_fla_ptx_supported(int, bool, bool, int64_t, int64_t, int64_t, int64_t, int64_t) {
    static const bool logged = [] { std::fputs("MTP_DIAGNOSTIC: FLA prefill bypassed\n", stderr); return true; }();
    (void) logged;
    return false;
}
