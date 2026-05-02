#pragma once

#include "pflash-loader.h"
#include <vector>
#include <cstdint>

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

struct pflash_scorer_result {
	std::vector<float> running_max; // [n_lookahead * seq_len]
	int n_lookahead;
	int seq_len;
};

void pflash_generate_placeholder_scores(float * out, int n_lookahead, int S,
	const int32_t * token_ids);

pflash_scorer_result pflash_score(
	const std::vector<int32_t> & token_ids,
	const pflash_model & model,
	const FlashPrefillConfig & fp_cfg,
	int gpu_device = 0);
