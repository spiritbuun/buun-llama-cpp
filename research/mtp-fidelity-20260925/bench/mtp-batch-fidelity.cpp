// Teacher-forced target-only diagnostic: no proposal or acceptance machinery.
// Each arm starts from a fresh, identically-prefilled context. Compare repeated
// single-row anchors, verification-width batches, and recurrent rollback sizing.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

struct metrics {
    std::vector<double> kld;
    double max_abs = 0;
    int changed = 0, flips = 0;
    void add(const float * p, const float * q, int n) {
        double pm = -INFINITY, qm = -INFINITY;
        int pi = 0, qi = 0;
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(p[i]) || !std::isfinite(q[i])) {
                throw std::runtime_error("nonfinite logits");
            }
            if (p[i] > pm) { pm = p[i]; pi = i; }
            if (q[i] > qm) { qm = q[i]; qi = i; }
            max_abs = std::max(max_abs, std::abs(double(p[i]) - q[i]));
        }
        changed += std::memcmp(p, q, n * sizeof(float)) != 0;
        flips += pi != qi;
        if (pi != qi) {
            std::fprintf(stderr, "BATCH_FLIP row=%zu reference_token=%d candidate_token=%d reference_margin=%.9g candidate_margin=%.9g\n",
                         kld.size(), pi, qi, double(p[pi])-p[qi], double(q[qi])-q[pi]);
        }
        double ps = 0, qs = 0;
        for (int i = 0; i < n; ++i) { ps += std::exp(p[i] - pm); qs += std::exp(q[i] - qm); }
        const double correction = std::log(qs) - std::log(ps);
        double kl = 0;
        for (int i = 0; i < n; ++i) {
            kl += std::exp(p[i] - pm) / ps * ((p[i] - pm) - (q[i] - qm) + correction);
        }
        kld.push_back(std::max(0.0, kl));
    }
    void print(int rs, int seqs, int width, int repeat, const char * reference) {
        std::sort(kld.begin(), kld.end());
        double sum = 0;
        for (double x : kld) { sum += x; }
        std::printf("BATCH_FIDELITY {\"rs\":%d,\"seqs\":%d,\"width\":%d,\"repeat\":%d,\"reference\":\"%s\","
                    "\"rows\":%zu,\"changed\":%d,\"argmax_flips\":%d,\"max_abs\":%.12g,"
                    "\"mean_kld\":%.12g,\"median_kld\":%.12g,\"p99_kld\":%.12g,\"max_kld\":%.12g}\n",
                    rs, seqs, width, repeat, reference, kld.size(), changed, flips, max_abs,
                    sum/kld.size(), kld[kld.size()/2], kld[size_t(.99*(kld.size()-1))], kld.back());
        std::fflush(stdout);
    }
};

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) { return 1; }
    common_init();
    llama_backend_init();
    auto * model = llama_model_load_from_file(params.model.path.c_str(), common_model_params_to_llama(params));
    if (!model) { return 2; }
    const auto * vocab = llama_model_get_vocab(model);
    auto tokens = common_tokenize(vocab, params.prompt, true, true);
    if (const char * path = std::getenv("MTP_PROBE_TOKENS")) {
        std::ifstream input(path);
        tokens.clear();
        llama_token token;
        while (input >> token) { tokens.push_back(token); }
    }
    const int prefix = std::getenv("MTP_PROBE_PREFIX") ? std::atoi(std::getenv("MTP_PROBE_PREFIX")) : 512;
    const int rows = std::getenv("MTP_PROBE_ROWS") ? std::atoi(std::getenv("MTP_PROBE_ROWS")) : 128;
    const int nv = llama_vocab_n_tokens(vocab);
    if (prefix <= 0 || rows <= 0) { return 3; }
    if (tokens.size() < size_t(prefix + rows)) { return 3; }
    auto batch = llama_batch_init(params.n_batch, 0, 1);
    std::vector<float> baseline, rs_anchor, width_anchor;
    const bool profile = std::getenv("MTP_PROBE_PROFILE") != nullptr;
    for (int config : {0, 1, 2}) {
        if (profile && config != 0) { continue; }
        const int rs = config == 2 ? 3 : 0;
        const int seqs = config == 0 ? 1 : 2;
        for (int width : {1, 2, 4}) {
            if (profile && width == 2) { continue; }
            for (int repeat = 0; repeat < 2; ++repeat) {
                if (profile && repeat != 0) { continue; }
                auto cp = common_context_params_to_llama(params);
                cp.n_rs_seq = rs;
                cp.n_seq_max = seqs;
                auto * ctx = llama_init_from_model(model, cp);
                if (!ctx) { return 4; }
                for (int pos = 0; pos < prefix; pos += params.n_batch) {
                    common_batch_clear(batch);
                    const int count = std::min(params.n_batch, prefix - pos);
                    for (int i = 0; i < count; ++i) { common_batch_add(batch, tokens[pos+i], pos+i, {0}, false); }
                    if (llama_decode(ctx, batch)) { return 5; }
                }
                std::vector<float> current(size_t(rows)*nv);
                using profile_fn = int (*)();
                auto start = reinterpret_cast<profile_fn>(dlsym(RTLD_DEFAULT, "cudaProfilerStart"));
                auto stop = reinterpret_cast<profile_fn>(dlsym(RTLD_DEFAULT, "cudaProfilerStop"));
                if (profile && width == 4) {
                    llama_synchronize(ctx);
                    if (!start || !stop || start()) { return 7; }
                }
                for (int pos = 0; pos < rows; pos += width) {
                    common_batch_clear(batch);
                    const int count = std::min(width, rows-pos);
                    for (int i = 0; i < count; ++i) {
                        common_batch_add(batch, tokens[prefix+pos+i], prefix+pos+i, {0}, true);
                    }
                    if (llama_decode(ctx, batch)) { return 6; }
                    for (int i = 0; i < count; ++i) {
                        const float * logits = llama_get_logits_ith(ctx, i);
                        std::copy(logits, logits + nv, current.data()+size_t(pos+i)*nv);
                    }
                }
                if (profile && width == 4) {
                    llama_synchronize(ctx);
                    if (stop()) { return 8; }
                }
                if (baseline.empty()) { baseline = current; }
                if (width == 1 && repeat == 0) { rs_anchor = current; }
                if (repeat == 0) { width_anchor = current; }
                metrics global, sized, self;
                for (int row = 0; row < rows; ++row) {
                    const size_t off = size_t(row)*nv;
                    global.add(baseline.data()+off, current.data()+off, nv);
                    sized.add(rs_anchor.data()+off, current.data()+off, nv);
                    self.add(width_anchor.data()+off, current.data()+off, nv);
                }
                global.print(rs, seqs, width, repeat, "rs0-seq1-width1");
                sized.print(rs, seqs, width, repeat, "same-config-width1");
                self.print(rs, seqs, width, repeat, "same-shape");
                llama_free(ctx);
            }
        }
    }
    llama_batch_free(batch);
    llama_model_free(model);
    llama_backend_free();
}
