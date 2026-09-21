#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Temporary issue #134 instrumentation. One recorder in ggml-base, shared by
// llama, the server, and backend DLLs. Never records tensor or prompt contents.
enum ggml_vbr_diag_channel {
    GGML_VBR_DIAG_MAP,
    GGML_VBR_DIAG_PRESSURE,
    GGML_VBR_DIAG_STATE,
    GGML_VBR_DIAG_CHANNEL_COUNT,
};

GGML_API bool ggml_vbr_diag_enabled(void);
GGML_API void ggml_vbr_diag_record(enum ggml_vbr_diag_channel channel, const char * format, ...)
    GGML_ATTRIBUTE_FORMAT(2, 3);
// Host-only dump to this module's stderr: no CUDA calls, no heap allocation,
// and no CRT FILE pointers crossing Windows DLL boundaries.
GGML_API void ggml_vbr_diag_dump(const char * reason);

#ifdef __cplusplus
}
#endif
