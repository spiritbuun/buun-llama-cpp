#include "ggml-vbr-diagnostic.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main(int argc, char ** argv) {
    const bool enabled = argc > 1 && std::strcmp(argv[1], "enabled") == 0;
    check(ggml_vbr_diag_enabled() == enabled, "environment gate");
    ggml_vbr_diag_record(GGML_VBR_DIAG_STATE, "checkpoint_fixture_preserved");
    for (int i = 0; i < 300; ++i) {
        ggml_vbr_diag_record(GGML_VBR_DIAG_MAP, "map_fixture=%d", i);
    }
    std::vector<std::thread> writers;
    for (int i = 0; i < 4; ++i) {
        writers.emplace_back([i] {
            for (int j = 0; j < 100; ++j) {
                ggml_vbr_diag_record(GGML_VBR_DIAG_PRESSURE, "concurrent_fixture=%d:%d", i, j);
            }
        });
    }
    for (auto & writer : writers) {
        writer.join();
    }
    const std::string oversized(2048, 'x');
    ggml_vbr_diag_record(GGML_VBR_DIAG_STATE, "%s", oversized.c_str());
    ggml_vbr_diag_record(GGML_VBR_DIAG_MAP, "simulated_failure operation=cuMemSetAccess result=600");
    // The CTest wrapper captures process stderr. No FILE* crosses a DLL/CRT
    // boundary in production or in this test.
    ggml_vbr_diag_dump("unit_fixture_not_a_gpu_reproduction");
    std::puts("PASS: diagnostic recorder (not a GPU failure reproduction)");
}
