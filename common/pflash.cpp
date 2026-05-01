// PFlash orchestrator: speculative prefill for long-context prompts.
// Ties together: FlashPrefill kernels, scorer model, token selection.

#include "pflash.h"
#include "pflash-loader.h"
#include "pflash-graph.h"
#include "pflash-score.h"

#include <cstdio>
#include <chrono>

bool pflash_enabled(const pflash_config & cfg) {
	return !cfg.scorer_path.empty();
}

std::vector<int32_t> pflash_compress(
		const std::vector<int32_t> & prompt_tokens,
		const pflash_config & cfg) {

	const int S = (int)prompt_tokens.size();

	// short prompts: pass through
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
		// placeholder mode: generate synthetic scores for pipeline testing
		fprintf(stderr, "pflash: using placeholder scores (--pflash-scorer test)\n");
		scores.n_lookahead = 8;
		scores.seq_len = S;
		scores.running_max.resize(8 * S);
		for (int n = 0; n < 8; n++) {
			for (int j = 0; j < S; j++) {
				float rel_pos = (float)j / (float)(S - 1);
				float u_shape = 1.0f - 4.0f * (rel_pos - 0.5f) * (rel_pos - 0.5f);
				float importance = 1.0f - u_shape;
				float noise = (float)((prompt_tokens[j] * 2654435761u) & 0xFFFF) / 65536.0f * 0.3f;
				scores.running_max[n * S + j] = importance + noise;
			}
		}
	} else {
		// Step 1: Load scorer model
		pflash_model scorer;
		if (pflash_model_load(scorer, cfg.scorer_path, cfg.gpu_device) != 0) {
			fprintf(stderr, "pflash: failed to load scorer, falling back to full prefill\n");
			return prompt_tokens;
		}

		auto t1 = std::chrono::high_resolution_clock::now();
		float dt_load = std::chrono::duration<float>(t1 - t0).count();
		fprintf(stderr, "pflash: scorer loaded in %.2fs\n", dt_load);

		// Step 2: Run scorer forward pass with FlashPrefill
		FlashPrefillConfig fp_cfg;
		fp_cfg.alpha = cfg.alpha;

		scores = pflash_score(prompt_tokens, scorer, fp_cfg, cfg.gpu_device);

		auto t2 = std::chrono::high_resolution_clock::now();
		float dt_score = std::chrono::duration<float>(t2 - t1).count();
		fprintf(stderr, "pflash: scoring done in %.2fs\n", dt_score);

		// Step 3: Free scorer
		pflash_model_free(scorer);
	}

	auto t3 = std::chrono::high_resolution_clock::now();

	// Step 4: Compress tokens
	pflash_score_config score_cfg;
	score_cfg.keep_ratio  = cfg.keep_ratio;

	auto compressed = pflash_compress_tokens(
		scores.running_max.data(),
		scores.n_lookahead,
		scores.seq_len,
		prompt_tokens.data(),
		S,
		score_cfg);

	auto t4 = std::chrono::high_resolution_clock::now();
	float dt_total = std::chrono::duration<float>(t4 - t0).count();
	fprintf(stderr, "pflash: total %.2fs (load=%.2f, score=%.2f, select=%.2f) — %d -> %d tokens\n",
		dt_total, dt_load, dt_score,
		std::chrono::duration<float>(t4 - t3).count(),
		S, (int)compressed.size());

	return compressed;
}
