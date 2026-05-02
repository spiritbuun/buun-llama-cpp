#include "pflash-graph.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static constexpr int N_LOOKAHEAD = 8;

void pflash_generate_placeholder_scores(float * out, int n_lookahead, int S,
		const int32_t * token_ids) {
	for (int n = 0; n < n_lookahead; n++) {
		for (int j = 0; j < S; j++) {
			float rel_pos = (float)j / (float)(S - 1);
			float u_shape = 1.0f - 4.0f * (rel_pos - 0.5f) * (rel_pos - 0.5f);
			float importance = 1.0f - u_shape;
			float noise = (float)((token_ids[j] * 2654435761u) & 0xFFFF) / 65536.0f * 0.3f;
			out[n * S + j] = importance + noise;
		}
	}
}

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

	fprintf(stderr, "pflash: scoring %d tokens across %d layers (placeholder)\n",
		S, model.n_layers);

	pflash_generate_placeholder_scores(result.running_max.data(), N_LOOKAHEAD, S,
		token_ids.data());

	(void)fp_cfg;
	(void)gpu_device;

	return result;
}
