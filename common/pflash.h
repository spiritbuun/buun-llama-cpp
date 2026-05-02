#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct common_params_speculative;

struct pflash_config {
	std::string scorer_path;
	int   min_tokens    = 8192;
	float keep_ratio    = 0.05f;
	float alpha         = 0.12f;
	int   gpu_device    = 0;

	static pflash_config from_params(const common_params_speculative & sp);
};

std::vector<int32_t> pflash_compress(
	const std::vector<int32_t> & prompt_tokens,
	const pflash_config & cfg);

bool pflash_enabled(const pflash_config & cfg);
