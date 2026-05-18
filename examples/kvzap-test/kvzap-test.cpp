// KVzap-style importance scoring test: use target model's own hidden states
// to predict token importance (via L2 norm as training-free proxy).
// Usage: kvzap-test -m <target.gguf> [-f <file>] [--layer N] [-c ctx] [-ngl N]

#include "llama.h"
#include "common.h"
#include "pflash-score.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <chrono>

struct capture_data {
	int target_layer = 4;
	int n_embd = 0;
	int n_tokens_seen = 0;
	std::vector<float> norms; // per-token L2 norm at target_layer
	std::string target_name;  // tensor name to match
	bool capture_done = false;
	bool early_exit = false;       // if true, return false after capturing (skip remaining layers)
	bool batch_captured = false;   // set true after capturing current batch's norms
};

static bool kvzap_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
	auto * cap = (capture_data *)user_data;

	if (cap->capture_done) return false;

	// after capturing this batch's target tensor, skip remaining layers
	if (cap->early_exit && cap->batch_captured) {
		return false;
	}

	if (ask) {
		return (strcmp(t->name, cap->target_name.c_str()) == 0);
	}

	// tensor data ready — compute L2 norm per token
	const int64_t embd = t->ne[0];
	const int64_t n_tokens = t->ne[1];

	cap->n_embd = (int)embd;

	std::vector<float> buf(embd * n_tokens);
	ggml_backend_tensor_get(t, buf.data(), 0, embd * n_tokens * sizeof(float));

	for (int64_t i = 0; i < n_tokens; i++) {
		float sum_sq = 0.0f;
		const float * row = buf.data() + i * embd;
		for (int64_t j = 0; j < embd; j++) {
			sum_sq += row[j] * row[j];
		}
		cap->norms.push_back(sqrtf(sum_sq));
	}
	cap->n_tokens_seen += (int)n_tokens;
	cap->batch_captured = true;

	return true;
}

int main(int argc, char ** argv) {
	const char * model_path = nullptr;
	const char * prompt_file = nullptr;
	int target_layer = 4;
	int n_ctx = 32768;
	int n_gpu_layers = 99;
	int n_batch = 2048;

	bool early_exit = false;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "-m") && i+1 < argc) model_path = argv[++i];
		else if (!strcmp(argv[i], "-f") && i+1 < argc) prompt_file = argv[++i];
		else if (!strcmp(argv[i], "--layer") && i+1 < argc) target_layer = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-c") && i+1 < argc) n_ctx = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-ngl") && i+1 < argc) n_gpu_layers = atoi(argv[++i]);
		else if (!strcmp(argv[i], "-b") && i+1 < argc) n_batch = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--early-exit")) early_exit = true;
	}

	if (!model_path || !prompt_file) {
		fprintf(stderr, "Usage: kvzap-test -m <model.gguf> -f <file> [--layer N] [-c ctx] [-ngl N] [-b batch]\n");
		return 1;
	}

	// read prompt
	std::string prompt;
	{
		std::ifstream f(prompt_file);
		if (!f) { fprintf(stderr, "cannot open %s\n", prompt_file); return 1; }
		prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
	}

	// init model
	auto mparams = llama_model_default_params();
	mparams.n_gpu_layers = n_gpu_layers;
	auto * model = llama_model_load_from_file(model_path, mparams);
	if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

	const auto * vocab = llama_model_get_vocab(model);

	// tokenize
	std::vector<llama_token> tokens(prompt.size() + 256);
	int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
		tokens.data(), tokens.size(), true, true);
	if (n_tokens < 0) {
		tokens.resize(-n_tokens);
		n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
			tokens.data(), tokens.size(), true, true);
	}
	tokens.resize(n_tokens);
	fprintf(stderr, "tokenized: %d tokens\n", n_tokens);

	if (n_tokens > n_ctx) {
		fprintf(stderr, "warning: truncating to %d tokens\n", n_ctx);
		n_tokens = n_ctx;
		tokens.resize(n_tokens);
	}

	// setup capture
	capture_data cap;
	cap.target_layer = target_layer;
	cap.early_exit = early_exit;
	// tensor name format: "l%d-ffn_out" or "l%d-attn_out" — need to check what names exist
	// In llama.cpp, after layer N the output is named "l_out-%d" or similar
	// Let's try the FFN output tensor name pattern
	char namebuf[64];
	snprintf(namebuf, sizeof(namebuf), "l_out-%d", target_layer);
	cap.target_name = namebuf;

	// init context with eval callback
	auto cparams = llama_context_default_params();
	cparams.n_ctx = n_ctx;
	cparams.n_batch = n_batch;
	cparams.n_ubatch = n_batch;
	cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
	cparams.cb_eval = kvzap_eval_callback;
	cparams.cb_eval_user_data = &cap;

	auto * ctx = llama_init_from_model(model, cparams);
	if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

	// prefill
	fprintf(stderr, "prefilling %d tokens (capturing layer %d: '%s')...\n",
		n_tokens, target_layer, cap.target_name.c_str());

	auto t0 = std::chrono::high_resolution_clock::now();

	// process in batches using llama_batch_get_one
	for (int pos = 0; pos < n_tokens; pos += n_batch) {
		int n_eval = std::min(n_batch, n_tokens - pos);
		llama_batch batch = llama_batch_get_one(tokens.data() + pos, n_eval);

		cap.batch_captured = false;  // reset for each batch

		if (llama_decode(ctx, batch) != 0) {
			if (cap.early_exit && cap.batch_captured) {
				// early exit causes decode to "fail" — that's expected
				continue;
			}
			fprintf(stderr, "decode failed at pos %d\n", pos);
			break;
		}
	}

	auto t1 = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration<double>(t1 - t0).count();

	fprintf(stderr, "prefill done in %.2fs, captured %d token norms\n",
		elapsed, (int)cap.norms.size());

	if ((int)cap.norms.size() != n_tokens) {
		fprintf(stderr, "WARNING: captured %d norms but expected %d\n",
			(int)cap.norms.size(), n_tokens);
		if (cap.norms.empty()) {
			fprintf(stderr, "No norms captured — tensor name '%s' may not exist.\n", cap.target_name.c_str());
			fprintf(stderr, "Try running with LLAMA_LOG_LEVEL=4 to see tensor names.\n");

			// try alternate names
			fprintf(stderr, "Trying alternate name patterns...\n");
			llama_free(ctx);
			llama_model_free(model);
			return 1;
		}
	}

	// analyze scores
	int n_scores = (int)cap.norms.size();
	std::vector<float> & scores = cap.norms;

	// print top/bottom 20
	std::vector<int> idx(n_scores);
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(), [&](int a, int b) { return scores[a] > scores[b]; });

	fprintf(stderr, "\n=== Top 20 by hidden state norm (layer %d) ===\n", target_layer);
	for (int i = 0; i < std::min(20, n_scores); i++) {
		int j = idx[i];
		char tok_buf[128] = {};
		int len = llama_token_to_piece(vocab, tokens[j], tok_buf, sizeof(tok_buf)-1, 0, false);
		if (len > 0) tok_buf[len] = 0;
		fprintf(stderr, "  [%5d] norm=%.4f  token=%d  '%s'\n", j, scores[j], tokens[j], tok_buf);
	}

	fprintf(stderr, "\n=== Bottom 20 ===\n");
	for (int i = std::max(0, n_scores - 20); i < n_scores; i++) {
		int j = idx[i];
		char tok_buf[128] = {};
		int len = llama_token_to_piece(vocab, tokens[j], tok_buf, sizeof(tok_buf)-1, 0, false);
		if (len > 0) tok_buf[len] = 0;
		fprintf(stderr, "  [%5d] norm=%.4f  token=%d  '%s'\n", j, scores[j], tokens[j], tok_buf);
	}

	// find NIAH needle
	const char * needle_text = "CRYSTAL";
	std::string full_text;
	full_text.reserve(prompt.size());
	for (int j = 0; j < n_tokens; j++) {
		char buf[256] = {};
		int len = llama_token_to_piece(vocab, tokens[j], buf, sizeof(buf)-1, 0, false);
		if (len > 0) buf[len] = 0;
		full_text += buf;
	}

	size_t needle_pos = full_text.find(needle_text);
	if (needle_pos == std::string::npos) {
		fprintf(stderr, "\nNeedle '%s' not found in tokenized text\n", needle_text);
	} else {
		// find token index near needle position
		size_t char_pos = 0;
		int needle_token_start = -1;
		for (int j = 0; j < n_tokens && j < n_scores; j++) {
			char buf[256] = {};
			int len = llama_token_to_piece(vocab, tokens[j], buf, sizeof(buf)-1, 0, false);
			if (len > 0) buf[len] = 0;
			size_t next_pos = char_pos + strlen(buf);
			if (needle_token_start < 0 && next_pos > needle_pos) {
				needle_token_start = j;
			}
			char_pos = next_pos;
		}

		if (needle_token_start >= 0) {
			fprintf(stderr, "\n=== Needle tokens (starting at %d) ===\n", needle_token_start);
			float needle_max_score = 0.0f;
			for (int j = needle_token_start; j < std::min(needle_token_start + 20, n_scores); j++) {
				char buf[256] = {};
				int len = llama_token_to_piece(vocab, tokens[j], buf, sizeof(buf)-1, 0, false);
				if (len > 0) buf[len] = 0;
				// find rank
				int rank = 0;
				for (int k = 0; k < n_scores; k++) {
					if (scores[k] > scores[j]) rank++;
				}
				fprintf(stderr, "  [%5d] norm=%.4f  rank=%d/%d (top %.1f%%)  '%s'\n",
					j, scores[j], rank+1, n_scores, 100.0f*(rank+1)/n_scores, buf);
				needle_max_score = std::max(needle_max_score, scores[j]);
			}

			// what percentile is the needle?
			int above_count = 0;
			for (int k = 0; k < n_scores; k++) {
				if (scores[k] > needle_max_score) above_count++;
			}
			fprintf(stderr, "\nNeedle max norm: %.4f — top %.1f%% (rank %d/%d)\n",
				needle_max_score, 100.0f * (above_count + 1) / n_scores,
				above_count + 1, n_scores);
			fprintf(stderr, "Would survive at keep_ratio=%.3f\n",
				(float)(above_count + 1) / n_scores);
		}
	}

	// compute score distribution stats
	float mean = 0, var = 0;
	for (float s : scores) mean += s;
	mean /= n_scores;
	for (float s : scores) var += (s - mean) * (s - mean);
	var /= n_scores;
	fprintf(stderr, "\nScore distribution: mean=%.4f std=%.4f min=%.4f max=%.4f\n",
		mean, sqrtf(var), *std::min_element(scores.begin(), scores.end()),
		*std::max_element(scores.begin(), scores.end()));

	// run chunk selection using pflash_select_spans
	pflash_score_config score_cfg;
	score_cfg.keep_ratio = 0.05f;
	score_cfg.chunk_size = 32;

	auto spans = pflash_select_spans(scores.data(), n_scores, score_cfg);

	int total_kept = 0;
	for (const auto & sp : spans) total_kept += sp.end - sp.start;
	fprintf(stderr, "\n=== Chunk selection: %d/%d tokens kept (%.1f%%), %d spans ===\n",
		total_kept, n_scores, 100.0f * total_kept / n_scores, (int)spans.size());

	// check if needle survived
	if (needle_pos != std::string::npos) {
		size_t char_pos = 0;
		int needle_tok = -1;
		for (int j = 0; j < n_tokens; j++) {
			char buf[256] = {};
			int len = llama_token_to_piece(vocab, tokens[j], buf, sizeof(buf)-1, 0, false);
			if (len > 0) buf[len] = 0;
			size_t next = char_pos + strlen(buf);
			if (needle_tok < 0 && next > needle_pos) needle_tok = j;
			char_pos = next;
		}
		if (needle_tok >= 0) {
			bool found = false;
			for (const auto & sp : spans) {
				if (needle_tok >= sp.start && needle_tok < sp.end) { found = true; break; }
			}
			fprintf(stderr, "NIAH: needle %s in surviving spans (token %d)\n",
				found ? "FOUND" : "NOT FOUND", needle_tok);
		}
	}

	llama_free(ctx);
	llama_model_free(model);
	return 0;
}
