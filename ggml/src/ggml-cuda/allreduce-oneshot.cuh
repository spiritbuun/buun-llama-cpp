#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

// One-shot AllReduce for small F32 tensors across N devices through pinned host memory.
// Every rank writes its slice into its own host buffer, signals a per-launch token, waits for all
// peers, then reads the N-1 peer slices and sums them in place. All work runs on the ranks' compute
// streams and every launch derives its token from a device-side counter, so the sequence can be
// recorded into a CUDA graph and replayed. Intended for latency-bound messages (decode activations);
// larger tensors keep the NCCL/ring path.
struct ggml_cuda_ar_oneshot;

ggml_cuda_ar_oneshot * ggml_cuda_ar_oneshot_init(const int * devices, size_t n_devices, size_t max_bytes);
void                   ggml_cuda_ar_oneshot_free(ggml_cuda_ar_oneshot * st);
bool                   ggml_cuda_ar_oneshot_eligible(const ggml_cuda_ar_oneshot * st, ggml_tensor ** tensors);
void                   ggml_cuda_ar_oneshot_report_all();
bool                   ggml_cuda_ar_oneshot_allreduce(ggml_cuda_ar_oneshot * st, ggml_backend_t * backends, ggml_tensor ** tensors);
