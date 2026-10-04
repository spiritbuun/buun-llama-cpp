#include "ggml.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-mtp-adaptive.h"

#include <random>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

extern struct llama_sampler * llama_sampler_init_dry_testing(float dry_multiplier, float dry_base, int32_t dry_allowed_length, int32_t dry_penalty_last_n, const std::vector<std::vector<llama_token>>& seq_breakers);

static void dump(const llama_token_data_array * cur_p) {
    for (size_t i = 0; i < cur_p->size; i++) {
        printf("%d: %f (%f)\n", cur_p->data[i].id, cur_p->data[i].p, cur_p->data[i].logit);
    }
}

#define DUMP(__cur_p) do { printf("%s:%d (%s)\n", __FILE__, __LINE__, __func__); dump((__cur_p)); printf("-\n"); } while(0)

struct sampler_tester {
    sampler_tester(size_t n_vocab) {
        cur.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < (llama_token)n_vocab; token_id++) {
            const float logit = logf(token_id);
            cur.emplace_back(llama_token_data{token_id, logit, 0.0f});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    sampler_tester(const std::vector<float> & probs, const std::vector<float> & probs_expected) : probs_expected(probs_expected) {
        cur.reserve(probs.size());
        for (llama_token token_id = 0; token_id < (llama_token)probs.size(); token_id++) {
            const float logit = logf(probs[token_id]);
            cur.emplace_back(llama_token_data{token_id, logit, probs[token_id]});
        }

        cur_p = llama_token_data_array { cur.data(), cur.size(), -1, false };
    }

    void apply(llama_sampler * sampler) {
        llama_sampler_apply(sampler, &cur_p);
        llama_sampler_free(sampler);
    }

    void check() {
        GGML_ASSERT(cur_p.size == probs_expected.size());
        for (size_t i = 0; i < cur_p.size; i++) {
            GGML_ASSERT(fabs(cur_p.data[i].p - probs_expected[i]) < 1e-5);
        }
    }

    llama_token_data_array cur_p;

private:
    const std::vector<float> probs_expected;

    std::vector<llama_token_data> cur;
};

static llama_token sample_dist(llama_sampler * sampler, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < (llama_token) logits.size(); ++token_id) {
        cur.push_back({ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    GGML_ASSERT(cur_p.selected >= 0);
    GGML_ASSERT((size_t) cur_p.selected < cur_p.size);
    return cur_p.data[cur_p.selected].id;
}

static void test_dist_singleton_rng() {
    llama_sampler * singleton = llama_sampler_init_dist(4242);
    llama_sampler * control   = llama_sampler_init_dist(4242);

    sample_dist(singleton, { 0.0f });
    sample_dist(control,   { 0.0f, 0.0f });

    const std::vector<float> logits(256, 0.0f);
    for (int i = 0; i < 4; ++i) {
        GGML_ASSERT(sample_dist(singleton, logits) == sample_dist(control, logits));
    }

    llama_sampler_free(singleton);
    llama_sampler_free(control);
}

static void test_temp(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp(temp));
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_temp_ext(const std::vector<float> & probs, const std::vector<float> & probs_expected, float temp, float delta, float exponent) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_temp_ext(temp, delta, exponent));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_k(const std::vector<float> & probs, const std::vector<float> & probs_expected, int k) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_k(k));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_top_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_min_p(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_min_p(p, 0));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_xtc(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p, float t) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_xtc(p, t, 0, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_typical(const std::vector<float> & probs, const std::vector<float> & probs_expected, float p) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_typical(p, 0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_penalties(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & probs_expected, float repeat_penalty, float alpha_frequency, float alpha_presence
) {
    GGML_ASSERT(probs.size() == probs_expected.size());

    sampler_tester tester(probs, probs_expected);

    auto * sampler = llama_sampler_init_penalties((int32_t) probs.size(), (int32_t) last_tokens.size(), repeat_penalty, alpha_frequency, alpha_presence);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_dry(
    const std::vector<float> & probs, const std::vector<llama_token> & last_tokens,
    const std::vector<float> & expected_probs, float dry_multiplier, float dry_base,
    int dry_allowed_length, int dry_penalty_last_n,
    const std::vector<std::vector<llama_token>> & seq_breakers
) {
    GGML_ASSERT(probs.size() == expected_probs.size());

    sampler_tester tester(probs, expected_probs);

    auto * sampler = llama_sampler_init_dry_testing(dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers);

    for (size_t i = 0; i < last_tokens.size(); i++) {
        llama_sampler_accept(sampler, last_tokens[i]);
    }

    DUMP(&tester.cur_p);
    tester.apply(sampler);
    tester.apply(llama_sampler_init_dist(0));
    DUMP(&tester.cur_p);
    tester.check();
}

static void test_top_n_sigma(const std::vector<float> & probs, const std::vector<float> & probs_expected, int n) {
    sampler_tester tester(probs, probs_expected);

    DUMP(&tester.cur_p);
    tester.apply(llama_sampler_init_top_n_sigma(n));
    tester.apply(llama_sampler_init_dist (0));
    DUMP(&tester.cur_p);

    tester.check();
}

static void test_sampler_queue(const size_t n_vocab, const std::string & samplers_sequence, const int top_k, const float top_p, const float min_p
) {
    sampler_tester tester(n_vocab);

          llama_token min_token_id = 0;
    const llama_token max_token_id = n_vocab - 1;

    for (auto s : samplers_sequence) {
        switch (s) {
            case 'k': tester.apply(llama_sampler_init_top_k(top_k)); break;
            case 'y': GGML_ABORT("typical test not implemented");
            case 'p': tester.apply(llama_sampler_init_top_p(top_p, 1)); break;
            case 'm': tester.apply(llama_sampler_init_min_p(min_p, 1)); break;
            case 't': GGML_ABORT("temperature test not implemented");
            default : GGML_ABORT("Unknown sampler");
        }

        tester.apply(llama_sampler_init_dist(0));

        auto & cur_p = tester.cur_p;

        const int size = cur_p.size;

        if (s == 'k') {
            const int expected_size = std::min(size, top_k);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - top_k));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(cur_p.data[0].id == max_token_id);
            GGML_ASSERT(cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'p') {
            const int softmax_divisor = n_vocab * (n_vocab-1) / 2 - min_token_id * (min_token_id-1) / 2;
            const int softmax_numerator_target = ceilf(top_p * softmax_divisor);

                min_token_id  = n_vocab;
            int expected_size = 0;
            int cumsum        = 0;
            do { // do-while because always at least one token is sampled
                min_token_id--;
                expected_size++;

                cumsum += min_token_id;
            } while (cumsum < softmax_numerator_target);

            // token 0 has p == 0, need special consideration for cumsum because top_p immediately returns
            if (min_token_id == 1) {
                min_token_id--;
                expected_size += 1;
            }

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else if (s == 'm') {
            int expected_size = ceilf((1.0f - min_p) * n_vocab);
            expected_size = std::max(expected_size, 1);
            expected_size = std::min(expected_size, size);

            min_token_id = floorf(min_p * n_vocab);
            min_token_id = std::max(min_token_id, 1);
            min_token_id = std::max(min_token_id, (llama_token)(n_vocab - size));
            min_token_id = std::min(min_token_id, (llama_token)(n_vocab - 1));

            GGML_ASSERT(size == expected_size);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[0].id == max_token_id);
            GGML_ASSERT(!cur_p.sorted || cur_p.data[expected_size-1].id == min_token_id);
        } else {
            GGML_ABORT("fatal error");
        }
    }

    printf("Sampler queue %3s OK with n_vocab=%05zu top_k=%5d top_p=%f min_p=%f\n",
           samplers_sequence.c_str(), n_vocab, top_k, top_p, min_p);
}

static void bench(llama_sampler * cnstr, const char * cnstr_name, const std::vector<llama_token_data> & data, int n_iter) {
    std::vector<llama_token_data> cur(data.size());
    std::copy(data.begin(), data.end(), cur.begin());
    llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(cnstr, &cur_p);
    llama_sampler_reset(cnstr);
    const int64_t t_start = ggml_time_us();
    for (int i = 0; i < n_iter; i++) {
        std::copy(data.begin(), data.end(), cur.begin());
        llama_token_data_array cur_p = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(cnstr, &cur_p);
        llama_sampler_reset(cnstr);
    }
    const int64_t t_end = ggml_time_us();
    llama_sampler_free(cnstr);
    printf("%-43s: %8.3f us/iter\n", cnstr_name, (t_end - t_start) / (float)n_iter);
}

#define BENCH(__cnstr, __data, __n_iter) bench((__cnstr), #__cnstr, (__data), (__n_iter))

static void test_perf() {
    const int n_vocab = 1 << 17;

    std::vector<llama_token_data> data;

    data.reserve(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        const float logit = 2.0f*((double)(rand())/RAND_MAX - 0.5);
        data.emplace_back(llama_token_data{i, logit, 0.0f});
    }

    BENCH(llama_sampler_init_top_k  (40),                     data, 32);
    BENCH(llama_sampler_init_top_p  (0.8f, 1),                data, 32);
    BENCH(llama_sampler_init_min_p  (0.2f, 1),                data, 32);
    BENCH(llama_sampler_init_typical(0.5f, 1),                data, 32);
    BENCH(llama_sampler_init_xtc    (1.0f, 0.1f, 1, 1),       data, 32);
}

static llama_token_data_array make_distribution(
        std::vector<llama_token_data> & storage,
        const std::vector<float> &      probabilities) {
    storage.clear();
    storage.reserve(probabilities.size());
    for (llama_token id = 0; id < (llama_token) probabilities.size(); ++id) {
        storage.push_back({ id, 0.0f, probabilities[id] });
    }
    return { storage.data(), storage.size(), -1, false };
}

static void test_proposal_rows() {
    std::vector<llama_token_data> data = {{7, std::log(0.6f), 0}, {9, std::log(0.3f), 0}, {3, std::log(0.1f), 0}};
    llama_token_data_array row{data.data(), data.size(), -1, true};
    float q[3];
    GGML_ASSERT(common_sampler_proposal_row(row, 1, 1, 0, q) == 7);
    GGML_ASSERT(std::abs(q[0] - 0.6f) < 1e-6 && std::abs(q[2] - 0.1f) < 1e-6);
    GGML_ASSERT(common_sampler_proposal_row(row, 1, 0.8f, 0.99, q) == 9);
    GGML_ASSERT(std::abs(q[0] - 2.0f/3) < 1e-6 && q[2] == 0);
    GGML_ASSERT(common_sampler_proposal_row(row, 0.5f, 1, 0, q) == 7);
    GGML_ASSERT(std::abs(q[0] - 36.0f/46) < 1e-6);
    GGML_ASSERT(common_sampler_proposal_row(row, 1, 0.1f, 0.99, q) == 7 && q[0] == 1);
    GGML_ASSERT(common_sampler_proposal_row(row, 0, 1, 0, q) == LLAMA_TOKEN_NULL);
    GGML_ASSERT(common_sampler_proposal_row(row, 1, 1, 1, q) == LLAMA_TOKEN_NULL);
    data[0].logit = NAN;
    GGML_ASSERT(common_sampler_proposal_row(row, 1, 1, 0, q) == LLAMA_TOKEN_NULL);
    data[0].logit = std::log(0.6f);

    data[1].logit = data[2].logit = -INFINITY;
    GGML_ASSERT(common_sampler_proposal_row(row, 1, 1, 0.999, q) == 7);
    GGML_ASSERT(q[0] == 1 && q[1] == 0 && q[2] == 0);
    data[0].logit = -INFINITY;
    GGML_ASSERT(common_sampler_proposal_row(row, 1, 1, 0, q) == LLAMA_TOKEN_NULL);
    data[0].logit = std::log(0.6f);
    data[1].logit = std::log(0.3f);
    data[2].logit = std::log(0.1f);

    // Sample the production row builder, then verify with a different target p.
    // The emitted distribution must be p, not q or the distribution of matches.
    std::vector<llama_token_data> target = {{7, 0, 0.2f}, {9, 0, 0.5f}, {3, 0, 0.3f}};
    llama_token_data_array p{target.data(), target.size(), -1, false};
    const int32_t ids[] = {7, 9, 3};
    std::mt19937 rng(1897);
    auto uniform = [&]() { return std::generate_canonical<double, 53>(rng); };
    int counts[3] = {};
    constexpr int n = 100000;
    for (int i = 0; i < n; ++i) {
        llama_token token = common_sampler_proposal_row(row, 0.8f, 0.9f, uniform(), q);
        if (uniform() >= common_sampler_speculative_acceptance_probability(&p, token, ids, q, 3)) {
            token = common_sampler_speculative_sample_residual(&p, ids, q, 3, uniform());
        }
        for (int j = 0; j < 3; ++j) {
            counts[j] += token == ids[j];
        }
    }
    for (int j = 0; j < 3; ++j) {
        GGML_ASSERT(std::abs(double(counts[j])/n - target[j].p) < 0.006);
    }

    common_speculative_proposal proposal;
    proposal.selected = {7, 9, 3};
    proposal.q_covered_tokens = 3;
    proposal.seq_id = 2;
    proposal.exact_q = true;
    GGML_ASSERT(proposal.matching_prefix_size(2, {7, 9, 3}) == 3);
    GGML_ASSERT(proposal.matching_prefix_size(2, {7}) == 1);
    GGML_ASSERT(proposal.matching_prefix_size(2, {7, 9, 3, 4}) == 3);
    GGML_ASSERT(proposal.matching_prefix_size(2, {7, 4}) == 0);
    GGML_ASSERT(proposal.matching_prefix_size(1, {7}) == 0);
    GGML_ASSERT(proposal.matching_prefix_size(2, {}) == 0);
    const size_t capacity = proposal.selected.capacity();
    proposal.clear();
    GGML_ASSERT(proposal.matching_prefix_size(2, {7}) == 0);
    GGML_ASSERT(proposal.selected.capacity() == capacity);
}

static void test_speculative_coupling() {
    std::vector<llama_token_data> storage;

    {
        auto p = make_distribution(storage, { 0.2f, 0.8f });
        const int32_t q_ids[] = { 0, 1 };
        const float q[] = { 0.2f, 0.8f };
        GGML_ASSERT(common_sampler_speculative_acceptance_probability(&p, 0, q_ids, q, 2) == 1.0);
        GGML_ASSERT(common_sampler_speculative_acceptance_probability(&p, 1, q_ids, q, 2) == 1.0);
    }

    {
        auto p = make_distribution(storage, { 0.2f, 0.8f });
        const int32_t q_ids[] = { 0, 1 };
        const float q[] = { 0.4f, 0.6f };
        GGML_ASSERT(std::abs(common_sampler_speculative_acceptance_probability(&p, 0, q_ids, q, 2) - 0.5) < 1e-6);
        GGML_ASSERT(common_sampler_speculative_acceptance_probability(&p, 1, q_ids, q, 2) == 1.0);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 2, 0.0) == 1);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 2, 0.999999) == 1);
    }

    {
        // The supports need not overlap. A rejected q-only proposal samples p.
        auto p = make_distribution(storage, { 0.25f, 0.75f, 0.0f });
        const int32_t q_ids[] = { 2 };
        const float q[] = { 1.0f };
        GGML_ASSERT(common_sampler_speculative_acceptance_probability(&p, 2, q_ids, q, 1) == 0.0);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 1, 0.0) == 0);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 1, 0.249999) == 0);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 1, 0.25) == 1);
    }

    {
        // K=16 and sparse q IDs outside p exercise the same O(V + K) path.
        auto p = make_distribution(storage, { 0.4f, 0.3f, 0.2f, 0.1f });
        const int32_t q_ids[] = { 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 };
        const float q[] = { 0.1f, 0.4f, 0.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        GGML_ASSERT(common_sampler_speculative_acceptance_probability(&p, 0, q_ids, q, 16) == 1.0);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 16, 0.0) == 0);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 16, 0.500001) == 2);
        GGML_ASSERT(common_sampler_speculative_sample_residual(&p, q_ids, q, 16, 0.999999) == 3);
    }

    {
        auto p = make_distribution(storage, { 0.5f, 0.5f });
        const int32_t q_ids[] = { 0, 0 };
        const float q[] = { 0.5f, 0.5f };
        bool threw = false;
        try {
            common_sampler_speculative_sample_residual(&p, q_ids, q, 2, 0.5);
        } catch (const std::runtime_error &) {
            threw = true;
        }
        GGML_ASSERT(threw);
    }

    {
        // A sampled, truncated draft distribution need not share the target's
        // support. Verify the complete accept/residual mixture, not only each
        // helper in isolation. Token 3 exists only in q and token 2 only in p.
        auto p = make_distribution(storage, { 0.1f, 0.3f, 0.6f, 0.0f });
        const int32_t q_ids[] = { 0, 1, 3 };
        const float q[] = { 0.5f, 0.25f, 0.25f };
        std::mt19937 rng(20260915);
        std::discrete_distribution<int> propose(q, q + 3);
        int counts[4] = {};
        constexpr int n = 200000;
        for (int i = 0; i < n; ++i) {
            llama_token token = q_ids[propose(rng)];
            const double accept = common_sampler_speculative_acceptance_probability(&p, token, q_ids, q, 3);
            if (std::generate_canonical<double, 53>(rng) >= accept) {
                token = common_sampler_speculative_sample_residual(
                    &p, q_ids, q, 3, std::generate_canonical<double, 53>(rng));
            }
            GGML_ASSERT(token >= 0 && token < 4);
            ++counts[token];
        }
        const double expected[] = { 0.1, 0.3, 0.6, 0.0 };
        for (int i = 0; i < 4; ++i) {
            GGML_ASSERT(std::abs(double(counts[i]) / n - expected[i]) < 0.005);
        }
        GGML_ASSERT(counts[3] == 0);
    }
}

static void test_mtp_adaptive() {
    common_speculative_mtp_adaptive state(2, 3);
    auto cycles = [&](int n, int accepted) {
        for (int i = 0; i < n; ++i) {
            state.accept(state.depth(), accepted, false);
            GGML_ASSERT(state.depth() >= 2 && state.depth() <= 3);
        }
    };
    cycles(64, 3); // high-match code retains the full depth
    GGML_ASSERT(state.depth() == 3);
    cycles(16, 0);
    GGML_ASSERT(state.depth() == 2);
    cycles(7, 2);
    GGML_ASSERT(state.depth() == 2);
    cycles(1, 2); // prose -> code, recover without waiting for another request
    GGML_ASSERT(state.depth() == 3);
    cycles(16, 3);
    GGML_ASSERT(state.depth() == 3);

    cycles(16, 2); // perfect first two rows but an unhelpful third
    GGML_ASSERT(state.depth() == 2);
    cycles(255, 2); // not a phase change: do not repeatedly probe every 8 cycles
    GGML_ASSERT(state.depth() == 2);
    cycles(1, 2); // bounded periodic recovery, even without a new streak
    GGML_ASSERT(state.depth() == 3);
    cycles(16, 0);
    cycles(256, 0); // periodic recovery also works with no accepted proposals
    GGML_ASSERT(state.depth() == 3);

    for (int i = 0; i < 32; ++i) {
        state.accept(2, 0, false); // clipped draft
        state.accept(0, 0, false); // failed/duplicate carry refresh
        state.accept(3, 0, true);  // another implementation
        state.accept(3, 4, false); // invalid count
    }
    GGML_ASSERT(state.depth() == 3);
    cycles(15, 0);
    GGML_ASSERT(state.depth() == 3);
    cycles(1, 0);
    GGML_ASSERT(state.depth() == 2);
    state.begin(); // learned depth survives, but a new request can recover
    GGML_ASSERT(state.depth() == 2);
    cycles(8, 2);
    GGML_ASSERT(state.depth() == 3);

    for (int matched = 7; matched <= 8; ++matched) {
        state.reset();
        cycles(matched, 3);
        cycles(16 - matched, 0);
        GGML_ASSERT(state.depth() == (matched == 7 ? 2 : 3));
    }
    for (int prefix = 11; prefix <= 12; ++prefix) {
        state.reset();
        cycles(prefix, 2);
        cycles(16 - prefix, 0);
        cycles(8, 2);
        GGML_ASSERT(state.depth() == (prefix == 11 ? 3 : 2));
    }
    state.reset();
    cycles(15, 0);
    state.begin(); // partial probe cannot leak into the next request
    cycles(1, 0);
    GGML_ASSERT(state.depth() == 3);

    state.reset();
    cycles(16, 0);
    for (int i = 0; i < 255; ++i) {
        state.begin();
        cycles(1, 0);
        GGML_ASSERT(state.depth() == 2);
    }
    state.begin();
    cycles(1, 0); // even one-token requests cannot postpone periodic recovery
    GGML_ASSERT(state.depth() == 3);

    common_speculative_mtp_adaptive slot0(2, 3), slot1(2, 3);
    for (int i = 0; i < 16; ++i) {
        slot0.accept(3, 0, false);
        slot1.accept(3, 3, false);
    }
    GGML_ASSERT(slot0.depth() == 2 && slot1.depth() == 3);
    for (int i = 0; i < 8; ++i) {
        // Same prefix clamp as MTP's CopySpec-composition integration.
        const int drafted = slot0.depth();
        slot0.accept(drafted, std::min(3, drafted), false);
    }
    GGML_ASSERT(slot0.depth() == 3 && slot1.depth() == 3);

    for (int minimum = 0; minimum <= 3; ++minimum) {
        state = common_speculative_mtp_adaptive(minimum, 3);
        for (int i = 0; i < 1024; ++i) {
            if (i == 512) {
                state.reset();
            }
            int drafted = state.depth();
            if (drafted < minimum) {
                drafted = 0; // production minimum-size boundary
            }
            GGML_ASSERT(drafted > 0);
            state.accept(drafted, 0, false);
            if (minimum == 3) {
                GGML_ASSERT(state.depth() == 3);
            }
        }
    }
}

static void test_copyspec_owner() {
    // No model needed: unavailable model drafters are omitted, leaving CopySpec
    // to exercise the same factory ownership and per-sequence lifecycle.
    for (auto type : {COMMON_SPECULATIVE_TYPE_DRAFT_MTP, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}) {
        common_params_speculative params;
        params.types = {COMMON_SPECULATIVE_TYPE_COPYSPEC, type};
        params.copyspec_gamma = 2;
        params.n_max = params.draft.n_max = 3;
        common_speculative_ptr shared(common_speculative_init(params, uint32_t(2)));
        common_speculative_ptr local(common_speculative_init(params, (llama_context *) nullptr));
        GGML_ASSERT(shared && !local);

        llama_tokens prompts[2] = {{10, 11, 12, 13, 14, 15, 16, 17},
                                  {20, 21, 22, 23, 24, 25, 26, 27}};
        llama_tokens prefixes[2] = {{10}, {20}};
        llama_tokens drafts[2];
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            common_speculative_begin(shared.get(), seq, prompts[seq]);
            auto & dp = common_speculative_get_draft_params(shared.get(), seq);
            dp.drafting = true;
            dp.n_max = 3;
            dp.id_last = prompts[seq][1];
            dp.prompt = &prefixes[seq];
            dp.result = &drafts[seq];
        }
        // An inactive descriptor can outlive its result in the server. Even
        // with valid storage here, it must not receive a CopySpec extension.
        common_speculative_get_draft_params(shared.get(), 1).drafting = false;
        drafts[1] = {22};
        common_speculative_draft(shared.get());
        GGML_ASSERT(drafts[1] == llama_tokens({22}));
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            drafts[seq].clear();
            common_speculative_get_draft_params(shared.get(), seq).drafting = true;
        }
        common_speculative_draft(shared.get());
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            GGML_ASSERT(drafts[seq] == llama_tokens(prompts[seq].begin() + 2, prompts[seq].begin() + 5));
            GGML_ASSERT(common_speculative_get_proposal(shared.get(), seq) == nullptr);
            common_speculative_accept(shared.get(), seq, 3);
        }
    }

    // Standalone CopySpec retains its legacy slot-local owner.
    common_params_speculative params;
    params.types = {COMMON_SPECULATIVE_TYPE_COPYSPEC};
    common_speculative_ptr shared(common_speculative_init(params, uint32_t(2)));
    common_speculative_ptr local(common_speculative_init(params, (llama_context *) nullptr));
    GGML_ASSERT(!shared && local);
}

int main(void) {
    test_mtp_adaptive();
    ggml_time_init();

    test_copyspec_owner();
    test_speculative_coupling();
    test_proposal_rows();
    test_dist_singleton_rng();

    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);
    test_temp({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f, 0.0f, 1.0f);
    test_temp_ext({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.0f, 1.0f}, 0.0f, 0.0f, 1.0f);

    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 1);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 3);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, 4);
    test_top_k({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0);

    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {1.0f}, 0);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.571429f, 0.428571f}, 0.7f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.44444f, 0.33333f, 0.22222f}, 0.8f);
    test_top_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 1.0f);

    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f/1.0f, 0.2f/1.0f, 0.3f/1.0f, 0.4f/1.0f}, 0.24f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.26f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.2f/0.9f, 0.3f/0.9f, 0.4f/0.9f},            0.49f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.51f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.3f/0.7f, 0.4f/0.7f},                       0.74f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  0.76f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.00f);
    test_min_p({0.1f, 0.2f, 0.3f, 0.4f}, {0.4f/0.4f},                                  1.05f);

    printf("XTC should:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.1f},                                0.99f, 0.09f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.2f, 0.1f},                          0.99f, 0.19f);
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.3f, 0.2f, 0.1f},                    0.99f, 0.29f);

    printf("XTC should not:\n");
    test_xtc({0.4f, 0.3f, 0.2f, 0.1f},   {0.4f, 0.3f, 0.2f, 0.1f},              0.99f, 0.39f);

    test_typical({0.97f, 0.01f, 0.01f, 0.01f}, {0.97f},            0.5f);
    test_typical({0.4f, 0.2f, 0.2f, 0.2f},     {0.2f, 0.2f, 0.2f}, 0.5f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0}, {0, 0.25f, 0.25f, 0.25f, 0.25f},   50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2}, {0, 0, 0, 0.5f, 0.5f},       50.0f, 0.0f, 0.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0, 0, 0, 0.5f, 0.5f}, 50.0f, 0.0f, 0.0f);

    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0},             {0.000011f, 0.249997f, 0.249997f, 0.249997f, 0.249997f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2},       {0.000023f, 0.000023f, 0.000023f, 0.499966f, 0.499966f}, 1.0f, 5.0f, 5.0f);
    test_penalties({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 0}, {0.000000f, 0.000023f, 0.000023f, 0.499977f, 0.499977f}, 1.0f, 5.0f, 5.0f);


    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1}, {0.25f, 0.25f, 0.25f, 0.25f}, 1.0f, 1.1f, 2, 4, {});
    test_dry({0.25f, 0.25f, 0.25f, 0.25f}, {0, 1, 2, 0, 1}, {0.296923f, 0.296923f, 0.109232f, 0.296923f}, 1.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 2, 6, {{3}});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 0, 1}, {0.241818f, 0.241818f, 0.032727f, 0.241818f, 0.241818f}, 2.0f, 1.1f, 2, 5, {});
    test_dry({0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, {0, 1, 2, 3, 4, 0, 1}, {0.2f, 0.2f, 0.2f, 0.2f, 0.2f}, 1.0f, 1.1f, 4, 7, {});

    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.0f, 0.0f, 0.428571f, 0.571429f}, 1.00f);
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 0.00f); // top_n_sigma == 0 now represents a no-op rather than greedy decoding as of PR#13345
    test_top_n_sigma({0.1f, 0.2f, 0.3f, 0.4f}, {0.1f, 0.2f, 0.3f, 0.4f}, 3.00f);

    test_sampler_queue(10000, "k", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "k",     1, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0f, 1e-12);

    test_sampler_queue(10000, "k",   100, 1.0000f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.0003f, 1.0f);
    test_sampler_queue(10000, "p", 10000, 0.8000f, 1.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "m", 10000, 1.0000f, 0.1f);

    test_sampler_queue(10000, "kp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "km", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 9997.9f/9999.0f);
    test_sampler_queue(10000, "mp", 100, 0.8f, 0.1f);

    test_sampler_queue(10000, "kpm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "kmp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pkm", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "pmk", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mkp", 100, 0.8f, 0.1f);
    test_sampler_queue(10000, "mpk", 100, 0.8f, 0.1f);

    printf("OK\n");

    test_perf();

    return 0;
}
