#pragma once

// FlashPrefill: block-sparse attention for speculative prefill scoring.
// Based on Fan et al (2026). SM80+ BF16 only, D_HEAD=128, BLOCK_SIZE=128.

struct FlashPrefillConfig {
    int block_size     = 128;
    int attention_sink = 2;     // always retain first N blocks
    int local_window   = 4;     // always retain last N blocks relative to query
    int last_n_full    = 2;     // last N query blocks attend to everything
    float alpha        = 0.12f; // threshold = alpha * max(block_scores)
};

struct FlashPrefillBuffers {
    void * mean_K;     // [n_kv_blocks, n_kv_heads, D_HEAD] bf16
    void * scores;     // [n_q_blocks, n_kv_blocks, n_heads] float
    void * score_max;  // [n_q_blocks, n_kv_blocks, n_heads] float
    int  * indices;    // [n_q_blocks, n_kv_blocks, n_heads] int  (compacted active block indices)
    int  * counts;     // [n_q_blocks, n_heads] int               (number of active blocks per query block)
};

// Allocate scratch buffers for FlashPrefill. Caller must cudaFree each pointer.
FlashPrefillBuffers flash_prefill_alloc(int seq_len, int n_heads, int n_kv_heads, int block_size);
void flash_prefill_free(FlashPrefillBuffers * bufs);

// Run full FlashPrefill pipeline: mean_K -> score -> select -> sparse forward.
// Q,K,V,O are all BF16, contiguous [seq_len, n_heads/n_kv_heads, D_HEAD].
// Q shape: [seq_len, n_heads, 128], K/V shape: [seq_len, n_kv_heads, 128].
// stream: cudaStream_t cast to void* for non-CUDA compilation units.
void flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V,
    void * O,
    int seq_len, int n_heads, int n_kv_heads, int d_head,
    float scale,
    const FlashPrefillConfig * cfg,
    FlashPrefillBuffers * bufs,
    void * stream);
