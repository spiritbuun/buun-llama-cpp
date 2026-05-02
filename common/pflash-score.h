#pragma once

#include <vector>
#include <cstdint>

struct pflash_score_config {
	float keep_ratio  = 0.05f;
	int   chunk_size  = 32;
	int   pool_kernel = 13;
};

struct pflash_span {
	int start;
	int end;
};

std::vector<pflash_span> pflash_select_spans(
	const float * scores,
	int seq_len,
	const pflash_score_config & cfg);

std::vector<int32_t> pflash_compress_tokens(
	const float * running_max,
	int n_lookahead,
	int seq_len,
	const int32_t * original_ids,
	int n_original,
	const pflash_score_config & cfg);
