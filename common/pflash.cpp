#include "pflash.h"
#include "pflash-loader.h"
#include "pflash-graph.h"
#include "pflash-score.h"
#include "common.h"

#include <cstdio>
#include <chrono>

pflash_config pflash_config::from_params(const common_params_speculative & sp) {
	pflash_config cfg;
	cfg.scorer_path = sp.pflash_scorer_path;
	cfg.min_tokens  = sp.pflash_min_tokens;
	cfg.keep_ratio  = sp.pflash_keep_ratio;
	cfg.alpha       = sp.pflash_alpha;
	return cfg;
}

bool pflash_enabled(const pflash_config & cfg) {
	return !cfg.scorer_path.empty();
}

std::vector<int32_t> pflash_compress(
		const std::vector<int32_t> & prompt_tokens,
		const pflash_config & cfg) {

	const int S = (int)prompt_tokens.size();

	if (S < cfg.min_tokens) {
		return prompt_tokens;
	}

	if (cfg.scorer_path.empty()) {
		fprintf(stderr, "pflash: no scorer model specified, skipping compression\n");
		return prompt_tokens;
	}

	auto t0 = std::chrono::high_resolution_clock::now();

	fprintf(stderr, "pflash: compressing %d tokens (keep_ratio=%.3f, alpha=%.3f)\n",
		S, cfg.keep_ratio, cfg.alpha);

	pflash_scorer_result scores;

	if (cfg.scorer_path == "test") {
		fprintf(stderr, "pflash: using placeholder scores (--pflash-scorer test)\n");
		scores.n_lookahead = 8;
		scores.seq_len = S;
		scores.running_max.resize(8 * S);
		pflash_generate_placeholder_scores(scores.running_max.data(), 8, S,
			prompt_tokens.data());
	} else {
		pflash_model scorer;
		if (pflash_model_load(scorer, cfg.scorer_path, cfg.gpu_device) != 0) {
			fprintf(stderr, "pflash: failed to load scorer, falling back to full prefill\n");
			return prompt_tokens;
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		fprintf(stderr, "pflash: scorer loaded in %.2fs\n",
			std::chrono::duration<float>(t1 - t0).count());

		FlashPrefillConfig fp_cfg;
		fp_cfg.alpha = cfg.alpha;

		scores = pflash_score(prompt_tokens, scorer, fp_cfg, cfg.gpu_device);

		auto t2 = std::chrono::high_resolution_clock::now();
		fprintf(stderr, "pflash: scoring done in %.2fs\n",
			std::chrono::duration<float>(t2 - t1).count());

		pflash_model_free(scorer);
	}

	auto t3 = std::chrono::high_resolution_clock::now();

	pflash_score_config score_cfg;
	score_cfg.keep_ratio = cfg.keep_ratio;

	auto compressed = pflash_compress_tokens(
		scores.running_max.data(),
		scores.n_lookahead,
		scores.seq_len,
		prompt_tokens.data(),
		S,
		score_cfg);

	auto t4 = std::chrono::high_resolution_clock::now();
	fprintf(stderr, "pflash: total %.2fs (select=%.2f) — %d -> %d tokens\n",
		std::chrono::duration<float>(t4 - t0).count(),
		std::chrono::duration<float>(t4 - t3).count(),
		S, (int)compressed.size());

	return compressed;
}
