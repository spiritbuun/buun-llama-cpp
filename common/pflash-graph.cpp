// PFlash scorer forward pass: Qwen3-0.6B with FlashPrefill attention.
// Runs the full transformer forward, then scores token importance via
// tail Q@K^T attention analysis across all layers.
//
// Current status: scaffold with placeholder scoring. The per-layer forward
// loop (Phase B) requires CUDA runtime calls that will be implemented when
// we move the graph builder to a .cu file or use proc_address dispatch.

#include "pflash-graph.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

static constexpr int N_LOOKAHEAD = 8;

pflash_scorer_result pflash_score(
		const std::vector<int32_t> & token_ids,
		const pflash_model & model,
		const FlashPrefillConfig & fp_cfg,
		int gpu_device) {

	const int S = (int)token_ids.size();

	pflash_scorer_result result;
	result.n_lookahead = N_LOOKAHEAD;
	result.seq_len = S;
	result.running_max.resize(N_LOOKAHEAD * S);

	fprintf(stderr, "pflash: scoring %d tokens across %d layers (placeholder — uniform random scores)\n",
		S, model.n_layers);

	// TODO: Full implementation requires:
	// 1. Embedding lookup (tok_embd)
	// 2. Per-layer: RMS norm -> Q/K/V proj -> QK norm -> RoPE -> FlashPrefill attn -> o_proj + residual -> FFN
	// 3. Save K per layer for tail scoring
	// 4. Tail scoring: Q_last[8] @ K_cache^T -> softmax -> max-over-heads -> accumulate across layers
	//
	// This requires CUDA runtime calls (cudaMalloc, kernel launches) which need either:
	// a) This file as a .cu (compiled by nvcc), or
	// b) Using proc_address dispatch (like cross-ring-interleave.cu), or
	// c) Using ggml backend API for all GPU ops
	//
	// For now, return placeholder scores so we can test the full pipeline end-to-end.
	// Placeholder: position-weighted scores that simulate attention importance
	// (higher scores for beginning and end, lower for middle — mimics real attention patterns)

	for (int n = 0; n < N_LOOKAHEAD; n++) {
		for (int j = 0; j < S; j++) {
			// U-shaped importance: tokens at beginning and end matter more
			float rel_pos = (float)j / (float)(S - 1); // 0..1
			float u_shape = 1.0f - 4.0f * (rel_pos - 0.5f) * (rel_pos - 0.5f); // 0 at ends, 1 at middle
			float importance = 1.0f - u_shape; // 1 at ends, 0 at middle
			// add some noise based on token ID to vary chunk scores
			float noise = (float)((token_ids[j] * 2654435761u) & 0xFFFF) / 65536.0f * 0.3f;
			result.running_max[n * S + j] = importance + noise;
		}
	}

	(void)fp_cfg;
	(void)gpu_device;

	return result;
}
