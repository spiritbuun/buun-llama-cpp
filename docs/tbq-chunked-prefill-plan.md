# TBQ Chunked MMA Prefill — Design Plan

## Problem

TBQ prefill currently dequants the ENTIRE KV cache to f16 before calling MMA.
This creates an O(context) temp buffer that OOMs at ~40K tokens on 27B models
(11.2GB temp buffer at 65K) even though the TBQ KV cache itself only uses ~2.3GB.

The decode path already uses native vec_dot (zero temp), but prefill needs MMA
for tensor-core throughput on large Q batches.

## Current quick fix (this commit)

Route TBQ prefill through the vec kernel (same as decode). This works at any
context but is slower than MMA for large prefill batches (~2-3x slower at pp8K+).

## Ultimate fix: chunked dequant + MMA

### Architecture

```
for chunk_start in range(0, kv_len, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, kv_len)
    n_chunk = chunk_end - chunk_start

    // Dequant one chunk of K and V to small f16 buffers
    dequant_tbq_to_f16(K[chunk_start:chunk_end], K_tmp)  // CHUNK_SIZE * head_dim * 2 bytes
    dequant_tbq_to_f16(V[chunk_start:chunk_end], V_tmp)

    // Run MMA attention for this chunk
    // Returns partial: softmax_num[i], softmax_den[i], max_val[i]
    partial = MMA_attend(Q, K_tmp, V_tmp, mask[chunk_start:chunk_end])

    // Online softmax accumulation (FlashAttention trick)
    for each query position q:
        new_max = max(running_max[q], partial.max[q])
        scale_old = exp(running_max[q] - new_max)
        scale_new = exp(partial.max[q] - new_max)
        running_num[q] = running_num[q] * scale_old + partial.num[q] * scale_new
        running_den[q] = running_den[q] * scale_old + partial.den[q] * scale_new
        running_max[q] = new_max

    // Final output: running_num / running_den
```

### Temp memory: O(CHUNK_SIZE) instead of O(context)

CHUNK_SIZE = 4096 tokens:
- K_tmp: 4096 * 640 * 2 = 5MB
- V_tmp: 4096 * 640 * 2 = 5MB
- Accumulator: Q_len * head_dim * (num + den + max) = small
- Total: ~10MB vs 11.2GB (1000x reduction)

### Implementation steps

1. Add `ggml_cuda_tbq_chunked_prefill_attend()` function
2. Persistent chunk buffers (K_tmp, V_tmp) — same pattern as q_rot_buf
3. Online softmax accumulator in GPU shared memory
4. Modify MMA dispatch to accept KV range [start, end]
5. Loop over chunks, accumulate

### Complexity: HIGH

The MMA kernel (ggml_cuda_flash_attn_ext_mma_f16) doesn't support partial KV
ranges or returning partial softmax statistics. Modifying it requires touching
the core FA tiling logic.

Alternative: implement a custom MMA kernel that takes TBQ blocks directly and
dequants per-tile in shared memory. This is the "fused dequant + MMA" approach.

### Risk assessment

- Chunked MMA: medium risk (modifying FA internals), high reward (full MMA speed at any context)
- Fused dequant+MMA: high risk (new kernel), highest reward (no temp buffer at all)
- Vec kernel fallback: zero risk, already implemented, ~2-3x slower prefill

### Recommended path

1. Ship vec kernel fallback now (done)
2. Chunked MMA as next PR
3. Fused dequant+MMA as future optimization
