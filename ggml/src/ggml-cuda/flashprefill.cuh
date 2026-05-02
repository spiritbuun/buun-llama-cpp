#pragma once

struct FlashPrefillConfig {
    int block_size     = 128;
    int attention_sink = 2;
    int local_window   = 4;
    int last_n_full    = 2;
    float alpha        = 0.12f;
};

struct FlashPrefillBuffers {
    void * mean_K;
    void * scores;
    int  * indices;
    int  * counts;
};

FlashPrefillBuffers flash_prefill_alloc(int seq_len, int n_heads, int n_kv_heads, int block_size);
void flash_prefill_free(FlashPrefillBuffers * bufs);

void flash_prefill_forward_bf16(
    const void * Q, const void * K, const void * V,
    void * O,
    int seq_len, int n_heads, int n_kv_heads, int d_head,
    float scale,
    const FlashPrefillConfig * cfg,
    FlashPrefillBuffers * bufs,
    void * stream);
