#include "pflash-score.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>

static std::vector<float> aggregate_scores(
		const float * running_max, int n_lookahead, int seq_len) {
	std::vector<float> scores(seq_len, 0.0f);
	for (int j = 0; j < seq_len; j++) {
		float sum = 0.0f;
		for (int n = 0; n < n_lookahead; n++) {
			sum += running_max[n * seq_len + j];
		}
		scores[j] = sum / (float)n_lookahead;
	}
	return scores;
}

// prefix-sum sliding window for O(S) instead of O(S*kernel)
static std::vector<float> avgpool_smooth(
		const std::vector<float> & scores, int kernel) {
	const int S = (int)scores.size();
	const int half = kernel / 2;

	std::vector<float> prefix(S + 1, 0.0f);
	for (int j = 0; j < S; j++) {
		prefix[j + 1] = prefix[j] + scores[j];
	}

	std::vector<float> smooth(S);
	for (int j = 0; j < S; j++) {
		int lo = std::max(0, j - half);
		int hi = std::min(S - 1, j + half);
		smooth[j] = (prefix[hi + 1] - prefix[lo]) / (float)(hi - lo + 1);
	}
	return smooth;
}

static std::vector<float> score_chunks(
		const std::vector<float> & smooth, int chunk_size) {
	const int S = (int)smooth.size();
	const int n_chunks = (S + chunk_size - 1) / chunk_size;
	std::vector<float> chunk_scores(n_chunks);

	for (int c = 0; c < n_chunks; c++) {
		int start = c * chunk_size;
		int end = std::min(start + chunk_size, S);
		float sum = 0.0f;
		for (int j = start; j < end; j++) {
			sum += smooth[j];
		}
		chunk_scores[c] = sum / (float)(end - start);
	}
	return chunk_scores;
}

std::vector<pflash_span> pflash_select_spans(
		const float * scores,
		int seq_len,
		const pflash_score_config & cfg) {

	const int n_chunks = (seq_len + cfg.chunk_size - 1) / cfg.chunk_size;
	const int n_keep = std::max(1, (int)(n_chunks * cfg.keep_ratio));

	std::vector<float> score_vec(scores, scores + seq_len);
	auto smooth = avgpool_smooth(score_vec, cfg.pool_kernel);
	auto chunk_scores = score_chunks(smooth, cfg.chunk_size);

	std::vector<int> chunk_idx(n_chunks);
	std::iota(chunk_idx.begin(), chunk_idx.end(), 0);
	std::partial_sort(chunk_idx.begin(), chunk_idx.begin() + n_keep, chunk_idx.end(),
		[&](int a, int b) { return chunk_scores[a] > chunk_scores[b]; });

	std::vector<int> selected(chunk_idx.begin(), chunk_idx.begin() + n_keep);
	std::sort(selected.begin(), selected.end());

	std::vector<pflash_span> spans;
	for (int i = 0; i < (int)selected.size(); i++) {
		int start = selected[i] * cfg.chunk_size;
		int end = std::min(start + cfg.chunk_size, seq_len);

		if (!spans.empty() && spans.back().end >= start) {
			spans.back().end = end;
		} else {
			spans.push_back({start, end});
		}
	}

	return spans;
}

std::vector<int32_t> pflash_compress_tokens(
		const float * running_max,
		int n_lookahead,
		int seq_len,
		const int32_t * original_ids,
		int n_original,
		const pflash_score_config & cfg) {

	auto scores = aggregate_scores(running_max, n_lookahead, seq_len);
	auto spans = pflash_select_spans(scores.data(), seq_len, cfg);

	int total_kept = 0;
	for (const auto & sp : spans) {
		total_kept += sp.end - sp.start;
	}

	fprintf(stderr, "pflash: %d -> %d tokens (%.1f%% kept, %d spans)\n",
		seq_len, total_kept, 100.0f * total_kept / seq_len, (int)spans.size());

	std::vector<int32_t> compressed;
	compressed.reserve(total_kept);

	for (const auto & sp : spans) {
		for (int j = sp.start; j < sp.end && j < n_original; j++) {
			compressed.push_back(original_ids[j]);
		}
	}

	return compressed;
}
