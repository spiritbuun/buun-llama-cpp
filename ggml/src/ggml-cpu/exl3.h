#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;
bool ggml_cpu_exl3_supports(const struct ggml_tensor * dst);
size_t ggml_cpu_exl3_work_size(const struct ggml_tensor * dst, int threads);
void ggml_cpu_exl3_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

#ifdef __cplusplus
}
#endif
